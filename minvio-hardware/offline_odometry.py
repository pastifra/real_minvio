import numpy as np
import scipy
import scipy.signal
import scipy.fft
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

import lab_prototype_rt
import utils


def _as_scalar(x):
    """Convert numpy scalar / 0-d array / 1-element array -> Python scalar."""
    a = np.asarray(x)
    if a.shape == ():
        return a.item()
    a = a.squeeze()
    if a.shape == ():
        return a.item()
    if a.size == 1:
        return a.reshape(()).item()
    raise ValueError(f"Expected scalar, got shape={a.shape}, dtype={a.dtype}")


def process_data_continuous_rt():
    # First mincam (used in original lightweight vision experiments)
    # ~15 us settling time with R227 = 100k
    #D = np.load("data/rt-data-2025-11-02-16-37-28.npz")

    # ~3 us settling time with R227 jumped
    D = np.load("data/rt-data-2025-11-12-11-26-46.npz")

    # Second mincam from Mike
    # ~3us settling time with R227 = 0
    #D = np.load("data/rt-data-2025-11-12-10-20-26.npz")

    P = D["P"].astype(np.float64)
    Fs = float(_as_scalar(D["Fs"]))
    num_pixels = int(_as_scalar(D["num_pixels"]))
    mincam_fps = float(_as_scalar(D["mincam_fps"]))
    pixel_avg_start = float(_as_scalar(D["pixel_avg_start"]))
    time_per_pixel = float(_as_scalar(D["time_per_pixel"]))
   

    t = np.arange(0, P.shape[1]) / Fs

    sample_duration = int(time_per_pixel * Fs) # Number of samples to read on each pixel
    average_idx = lab_prototype_rt.generate_avg_idx(
        sample_duration, Fs, num_pixels, pixel_avg_start, time_per_pixel)

    p_mean = P[:,average_idx].mean(2)

    fig, ax = plt.subplots(figsize=(6,3))
    #ax.plot(t * 1e6, P[0] * 1e3)
    ax.plot(t * 1e6, P[1] * 1e3)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Measurement (mV)")
    ax.set_title("Raw Measurements")
    ax.set_xlim(0, t.max() * 1e6)
    fig.tight_layout()
    fig.savefig("raw-measurement.pdf", pad_inches=0.1)
    fig.show()

    # Example parameters
    f0 = 60  # Frequency to be removed (Hz)
    Q = 60.0 # Quality factor (the higher, the narrower the notch)

    # Design the notch filter
    b_notch_60, a_notch_60 = scipy.signal.iirnotch(f0, Q, mincam_fps)
    b_notch_79, a_notch_79 = scipy.signal.iirnotch(79, Q, mincam_fps)

    w, h = scipy.signal.freqz(b_notch_60, a_notch_60, worN=10000, fs=mincam_fps)
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.xlim([0, 240])
    plt.title("IIR Notch Filter (60 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    
    # Parameters
    cutoff = 1000  # Desired cutoff frequency (Hz)
    numtaps = 201

    # Design FIR lowpass filter using a Hamming window
    b_lpf = scipy.signal.firwin(numtaps, cutoff, window="hamming", fs=mincam_fps)

    # Plot frequency response
    w, h = scipy.signal.freqz(b_lpf, worN=2048, fs=mincam_fps)
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title("IIR Notch Filter (79 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    p_mean_filtered = np.zeros_like(p_mean)
    for pi in range(p_mean.shape[1]):
        p_filtered = scipy.signal.filtfilt(b_notch_60, a_notch_60, p_mean[:,pi])
        p_filtered = scipy.signal.filtfilt(b_notch_79, a_notch_79, p_filtered)
        p_filtered = scipy.signal.filtfilt(b_lpf, [1], p_filtered)
        p_mean_filtered[:,pi] = p_filtered
    
    
    # Plot FFT of original and filtered signal for one signal
    X_orig = scipy.fft.fftshift(scipy.fft.fft(p_mean[:,0]))
    X_filtered = scipy.fft.fftshift(scipy.fft.fft(p_mean_filtered[:,0]))
    freqs = scipy.fft.fftshift(scipy.fft.fftfreq(p_mean.shape[0], d=1/mincam_fps))
    fig, ax = plt.subplots()
    ax.plot(freqs, 20 * np.log10(np.abs(X_orig)), label="Original")
    ax.plot(freqs, 20 * np.log10(np.abs(X_filtered)), label="Filtered")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("FFT of Pixel 0")
    ax.legend()
    fig.tight_layout()
    fig.show()

    # Plot a single pixel original vs filtered
    plot_D = 10
    t = np.arange(p_mean.shape[0]) / mincam_fps
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t[::plot_D], p_mean[::plot_D,0] * 1e3, label="Original")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measurement (mV)")
    ax.set_xlim(0, t.max())
    fig.tight_layout()
    fig.savefig("single-pixel-og.pdf", pad_inches=0.1)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t[::plot_D], p_mean[::plot_D,0] * 1e3, label="Original")
    ax.plot(t[::plot_D], p_mean_filtered[::plot_D,0] * 1e3, label="Filtered")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measurement (mV)")
    ax.set_xlim(0, t.max())
    ax.legend()
    fig.tight_layout()
    fig.savefig("single-pixel-og-filtered.pdf", pad_inches=0.1)
    
    
    # Plot 16 pixels
    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    for pi in range(4):
        for pj in range(4):
            ax[pi, pj].plot(t * 1e3, p_mean[:,pi*4 + pj] * 1e3, label="Original")
            ax[pi, pj].plot(t * 1e3, p_mean_filtered[:,pi*4 + pj] * 1e3, label="Filtered")
            ax[pi, pj].set_title(f"Pixel {pi*4 + pj}")
            ax[pi, pj].set_xlabel("Time (ms)")
            ax[pi, pj].set_ylabel("Measurement (mV)")
            ax[pi, pj].legend()

    fig.tight_layout()
    fig.show()

    print(p_mean_filtered[2000:2500:2].std(0) * 1e6)
    print()
    print("Average noise across all pixels: %.0f uV RMS" % (p_mean_filtered[2000:2500:2].std(0).mean() * 1e6))

    plt.show()


def run_offline_odometry(D):
    P = D["P"].astype(np.float64)
    Fs = float(_as_scalar(D["Fs"]))
    num_pixels = int(_as_scalar(D["num_pixels"]))
    mincam_fps = float(_as_scalar(D["mincam_fps"]))       # rate of p_mean samples (filter fs)
    pixel_avg_start = float(_as_scalar(D["pixel_avg_start"]))
    time_per_pixel = float(_as_scalar(D["time_per_pixel"]))

    t = np.arange(0, P.shape[1]) / Fs


    sample_duration = int(time_per_pixel * Fs) # Number of samples to read on each pixel
    average_idx = lab_prototype_rt.generate_avg_idx(
        sample_duration, Fs, num_pixels, pixel_avg_start, time_per_pixel)

    p_mean = P[:,average_idx].mean(2)

    fig, ax = plt.subplots(figsize=(6,3))
    #ax.plot(t * 1e6, P[0] * 1e3)
    ax.plot(t * 1e6, P[1] * 1e3)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Measurement (mV)")
    ax.set_title("Raw Measurements")
    ax.set_xlim(0, t.max() * 1e6)
    fig.tight_layout()
    fig.show()

    notch_filter_freqs = [60, 79, 120, 240, 360, 480, 600, 720, 840, 960, 333.9]  # Frequencies to be removed (Hz)
    Q = 30.0 # Quality factor (the higher, the narrower the notch)

    all_filters_sos = []

    for notch_filter_freq in notch_filter_freqs:
        b_notch, a_notch = scipy.signal.iirnotch(notch_filter_freq, Q, fs=mincam_fps)
        all_filters_sos.append(scipy.signal.tf2sos(b_notch, a_notch))


    # Design FIR lowpass filter using a Hamming window
    cutoff = 1000  # Desired cutoff frequency (Hz)
    numtaps = 201
    b_lpf = scipy.signal.firwin(numtaps, cutoff, window="hamming", fs=mincam_fps)
    all_filters_sos.append(scipy.signal.tf2sos(b_lpf, [1]))

    sos_combined = np.vstack(all_filters_sos)

    p_mean_filtered = np.zeros_like(p_mean)
    for pi in range(num_pixels):
        p_mean_filtered[:,pi] = scipy.signal.sosfilt(sos_combined, p_mean[:,pi])
        
    # Skip the first half second in the plot to avoid initial conditions with the notch filter
    start_i = int(mincam_fps * 1) 
    
    # Plot FFT of original and filtered signal for one signal
    # FFT on original and filtered signal, but skipping the first second (given by start_i)
    X_orig = scipy.fft.fftshift(scipy.fft.fft(p_mean[start_i:,0]))
    X_filtered = scipy.fft.fftshift(scipy.fft.fft(p_mean_filtered[start_i:,0]))
    freqs = scipy.fft.fftshift(scipy.fft.fftfreq(p_mean[start_i:].shape[0], d=1/mincam_fps))
    fig, ax = plt.subplots()
    ax.plot(freqs, 20 * np.log10(np.abs(X_orig)), label="Original")
    ax.plot(freqs, 20 * np.log10(np.abs(X_filtered)), label="Filtered")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("FFT of Pixel 0")
    ax.legend()
    fig.tight_layout()
    fig.show()

    # Plot a single pixel original vs filtered
    plot_D = 1
    t = np.arange(p_mean.shape[0]) / mincam_fps
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t[start_i::plot_D], p_mean[start_i::plot_D,0] * 1e3, label="Original")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measurement (mV)")
    ax.set_xlim(0, t.max())
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t[start_i::plot_D], p_mean[start_i::plot_D,1] * 1e3, label="Original")
    ax.plot(t[start_i::plot_D], p_mean_filtered[start_i::plot_D,1] * 1e3, label="Filtered")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measurement (mV)")
    ax.set_xlim(0, t.max())
    ax.legend()
    fig.tight_layout()
    fig.show()
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t[start_i::plot_D], p_mean_filtered[start_i::plot_D,0] * 1e3, label="Cos (+) Filtered")
    ax.plot(t[start_i::plot_D], p_mean_filtered[start_i::plot_D,1] * 1e3, label="Cos (-) Filtered")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measurement (mV)")
    ax.set_xlim(0, t.max())
    ax.legend()
    fig.tight_layout()
    fig.show()
    
    
    cos_differential = p_mean_filtered[:,0] - p_mean_filtered[:,1]
    sin_differential = p_mean_filtered[:,2] - p_mean_filtered[:,3]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
    axs[0].plot(t[start_i:] * 1e3, cos_differential[start_i:] * 1e3)
    axs[0].set_title("Cosine Differential Signal")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Measurement (mV)")

    axs[1].plot(t[start_i:] * 1e3, sin_differential[start_i:] * 1e3)
    axs[1].set_title("Sine Differential Signal")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Measurement (mV)")
    fig.tight_layout()
    fig.show()

    
    ## FFT of the differential signals
    X_cos_differential = scipy.fft.fftshift(scipy.fft.fft(cos_differential[start_i:]))
    X_sin_differential = scipy.fft.fftshift(scipy.fft.fft(sin_differential[start_i:]))

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
    fig.suptitle("FFT of Differential Signals")
    axs[0].plot(freqs, 20 * np.log10(np.abs(X_cos_differential)))
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Magnitude (dB)")
    axs[0].set_title("FFT of Cos Differential")
    axs[0].set_xlim(-1000, 1000)

    axs[1].plot(freqs, 20 * np.log10(np.abs(X_sin_differential)))
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude (dB)")
    axs[1].set_title("FFT of Sin Differential")
    axs[1].set_xlim(-1000, 1000)

    fig.tight_layout()
    fig.show()

    breakpoint()



if __name__ == "__main__":
    D = np.load("C:/pixi_ws/minvio-hardware/data/rt-data/005ms.npz")
    run_offline_odometry(D)