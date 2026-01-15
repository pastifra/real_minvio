from pathlib import Path
import signal
import argparse
from multiprocessing import Process, Queue, Event
import numpy as np
import matplotlib.pyplot as plt
import sys


import logging
module_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
import cv2
import time
from datetime import datetime
from typing import List



import nidaqmx
from nidaqmx import constants

from pypylon import pylon

import os

WINGET_FFMPEG_BIN = (
    r"C:\Users\franc\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.0.1-full_build\bin"
)

def _sanitize_path_for_ffmpeg():
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_libbin = os.path.join(conda_prefix, "Library", "bin").lower() if conda_prefix else ""
    parts = [p for p in os.environ.get("PATH", "").split(";") if p and p.lower() != conda_libbin]
    os.environ["PATH"] = WINGET_FFMPEG_BIN + ";" + ";".join(parts)

_sanitize_path_for_ffmpeg()

import skvideo
skvideo.setFFmpegPath(WINGET_FFMPEG_BIN)

import skvideo.io

import utils
from utils import CircularBuffer

vis_img_size = (500, 500)
EXIT = False

#17 is cos +
#18 is cos -
#22 is sin +
#23 is sin -
PIXEL_IDS = np.asarray([17, 18, 22, 23]) - 1 # Pixel indices, starting at 0
NUM_PIXELS = len(PIXEL_IDS) # Do not change
PIXEL_AVG_START = 4e-6
TIME_PER_PIXEL = 6e-6
MINCAM_FPS = 1 / (TIME_PER_PIXEL * NUM_PIXELS)
CAMERA_FPS = 30
GUI_FPS = 30 # Do not change
LIVE_PLOT_FPS = 2

def mincam_annotation_overlay(annotated_cell_ids=True):
    def annotate_cell(img_cell, cell_id):
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.5
        font_color = (192, 255, 192)
        font_thickness = 1

        if annotated_cell_ids:
            cv2.putText(
                img_cell, str(cell_id), (10, 30), font, font_scale, 
                font_color, font_thickness, cv2.LINE_AA)
        
        wall_thickness = 8
        wall_color = (50, 50, 50)
        img_cell[:wall_thickness,:,:] = wall_color
        img_cell[-wall_thickness:,:,:] = wall_color
        img_cell[:,:wall_thickness,:] = wall_color
        img_cell[:,-wall_thickness:,:] = wall_color

    img = np.zeros((*vis_img_size, 3), dtype=np.uint8)

    cell_size = (img.shape[0] // 5, img.shape[1] // 5)

    cell_ids = [13, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 12, 11, 10, 9, 
                8, 7, 6, 5, 4, 3, 2, 1]
    i = 0
    for r in range(5):
        for c in range(5):
            if r == 2 and c == 2:
                continue
            img_cell = img[(r*cell_size[0]):((r + 1)*cell_size[0]),
                            (c*cell_size[1]):((c + 1)*cell_size[1])]
            img_cell = annotate_cell(img_cell, cell_ids[i])
            i += 1
    
    mask = np.all(img != 0, 2)[:,:,None]
    return img, mask

def _overlay_readout(img, p):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    font_color = (50, 50, 192)
    font_thickness = 1
    cell_size = (img.shape[0] // 5, img.shape[1] // 5)

    for r in range(5):
        for c in range(5):
            if r == 2 and c == 2:
                continue
            img_cell = img[(r*cell_size[0]):((r + 1)*cell_size[0]),
                            (c*cell_size[1]):((c + 1)*cell_size[1])]
            cv2.putText(img_cell, "%.0f mV" % (p[r,c] * 1e3),
                (10, 55), font, font_scale, 
                font_color, font_thickness, cv2.LINE_AA)
    
    return img

def visualize_readout(p: np.ndarray, 
                      annotation_overlay: np.ndarray, 
                      annotation_mask: np.ndarray):
    assert len(p) == len(PIXEL_IDS)
    if len(p) < 24:
        p_all = np.zeros(24)
        p_all[PIXEL_IDS] = p
        p = p_all

    # Re-arrange consistent with PCB layout
    p = np.asarray([
        [p[13-1], p[24-1], p[23-1], p[22-1], p[21-1]],
        [p[20-1], p[19-1], p[18-1], p[17-1], p[16-1]],
        [p[15-1], p[14-1], 0, p[12-1], p[11-1]],
        [p[10-1], p[9-1], p[8-1], p[7-1], p[6-1]],
        [p[5-1], p[4-1], p[3-1], p[2-1], p[1-1]],
    ])
    max_val = 3.2
    colors = (np.minimum((p / max_val) * 255, 255)).astype(np.uint8)
    img = np.reshape(colors, (5, 5))
    img = cv2.resize(img, vis_img_size, interpolation=cv2.INTER_NEAREST)
    img = np.tile(img[:,:,None], (1, 1, 3))

    img = _overlay_readout(img, p)

    img = (img * ~annotation_mask) + (annotation_overlay * annotation_mask)

    img = cv2.resize(img, (img.shape[0]*2, img.shape[1]*2), 
                     interpolation=cv2.INTER_NEAREST)

    return img

def visualize_door_status(door_open: bool):
    bg_color = (0, 255, 0) if door_open else (0, 0, 255)
    img = np.tile(np.asarray(bg_color, dtype=np.uint8)[None,None,:], 
                  (512, vis_img_size[1], 1))
    
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 1

    cv2.putText(img, "Open" if door_open else "Close", 
                (20, 42), font, font_scale, font_color, 
                font_thickness, cv2.LINE_AA)

    return img

def visualize_pc(pc: float):
    bg_color = (0, 0, 0)
    img = np.tile(np.asarray(bg_color, dtype=np.uint8)[None,None,:], 
                  (512, vis_img_size[1], 1))
    
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 2
    font_color = (255, 255, 255)
    font_thickness = 1

    cv2.putText(img, "Count: %.2f" % pc, (20, 80), font, font_scale, 
                font_color, font_thickness, lineType=cv2.FILLED)

    return img

def configure_camera(cam: pylon.InstantCamera):
    # Load the User Set 1 user set
    cam.UserSetSelector.Value = "UserSet1"
    cam.UserSetLoad.Execute()

    cam.Width.Value = 1100
    cam.Height.Value = 1100
    cam.OffsetX.Value = 444
    cam.OffsetY.Value = 46

    cam.PixelFormat.Value = "RGB8"
    cam.Gain.Value = 10
    cam.ExposureTime.Value = 5000 # us
    cam.Gamma.Value = 1

    cam.AcquisitionMode.Value = "Continuous"

    cam.TriggerSelector.Value = "FrameStart"
    cam.TriggerMode.Value = "On"
    cam.TriggerSource.Value = "Line1"
    cam.TriggerActivation.Value = "RisingEdge"

    cam.BslDemosaicingMode.Value = "Manual"
    cam.BslDemosaicingMethod.Value = "Unilinear"

    cam.DeviceLinkThroughputLimitMode.Value = 'Off'

def camera_fn(camera_ready_event,
              duration: int, 
              exp_name: str | None=None):
    codec = "libx264"
    crfOut = 18
    ffmpeg_num_threads = 16
    writer = skvideo.io.FFmpegWriter(
        str(utils.get_data_path() / "video-data" / ("%s.mp4" % exp_name)),
        inputdict={'-r': str(CAMERA_FPS)},
        outputdict={'-vcodec': codec, 
                    '-crf': str(crfOut), 
                    '-threads': str(ffmpeg_num_threads),
                    '-r': str(CAMERA_FPS),
                    '-pix_fmt': 'yuv420p'})


    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    configure_camera(camera)
    camera.StartGrabbing()

    camera_ready_event.set()

    last_imshow_time = 0

    for i in range(duration):
        with camera.RetrieveResult(
            60000, pylon.TimeoutHandling_Return) as result:
            if not result.IsValid():
                print("Camera frame not valid")
                break

            if result.GrabSucceeded():
                img = result.Array

                writer.writeFrame(img)

                if (time.perf_counter() - last_imshow_time) >= (1 / GUI_FPS):
                    img_vis = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_RGB2BGR)
                    cv2.imshow("Camera", utils.gamma_correct(img_vis))
                    cv2.waitKey(1)
                    last_imshow_time = time.perf_counter()
            else:
                print("CameraGrab failed")
                continue
            
        if EXIT:
            print("Exit from global")
            break

    camera.StopGrabbing()
    camera.Close()
    writer.close()


class GaborHistory:
    def __init__(self, N_history=int(2e6), D=1):
        self.N_history = N_history
        self.cos_pos_buf = CircularBuffer(N_history, np.float32)
        self.cos_neg_buf = CircularBuffer(N_history, np.float32)
        self.sin_pos_buf = CircularBuffer(N_history, np.float32)
        self.sin_neg_buf = CircularBuffer(N_history, np.float32)

        self.D = D
    
    def push_batch(self, cos_pos, cos_neg, sin_pos, sin_neg):
        self.cos_pos_buf.push_batch(cos_pos)
        self.cos_neg_buf.push_batch(cos_neg)
        self.sin_pos_buf.push_batch(sin_pos)
        self.sin_neg_buf.push_batch(sin_neg)
    
    def get_plot_signals(self, mV=False):
        cos_pos = self.cos_pos_buf.unravel(pad_to_full_size=True)[::self.D]
        cos_neg = self.cos_neg_buf.unravel(pad_to_full_size=True)[::self.D]
        sin_pos = self.sin_pos_buf.unravel(pad_to_full_size=True)[::self.D]
        sin_neg = self.sin_neg_buf.unravel(pad_to_full_size=True)[::self.D]

        x_cos = cos_pos - cos_neg
        x_sin = sin_pos - sin_neg

        scale = 1e3 if mV else 1

        return {
            "cos_pos": cos_pos * scale,
            "cos_neg": cos_neg * scale,
            "sin_pos": sin_pos * scale,
            "sin_neg": sin_neg * scale,
            "x_cos": x_cos * scale,
            "x_sin": x_sin * scale
        }

def create_live_gabor_plot(gabor_history: GaborHistory, Fs):
    fig, axs = plt.subplots(figsize=(12, 8), ncols=2, nrows=3)
    fig.suptitle("Live Data")
    fig.show()

    # Create initial plots
    signals = gabor_history.get_plot_signals(mV=True)
    t = np.arange(0, gabor_history.N_history)[::gabor_history.D] / Fs
    for col, cos_or_sin in zip([0, 1], ["cos", "sin"]):
        axs[0,col].plot(t, signals["%s_pos" % cos_or_sin])
        axs[0,col].set_xlabel("Time (s)")
        axs[0,col].set_ylabel("Voltage (mV)")
        axs[0,col].set_title("%s (+)" % cos_or_sin.capitalize())

        axs[1,col].plot(t, signals["%s_neg" % cos_or_sin])
        axs[1,col].set_xlabel("Time (s)")
        axs[1,col].set_ylabel("Voltage (mV)")
        axs[1,col].set_title("%s (-)" % cos_or_sin.capitalize())

        axs[2,col].plot(t, signals["x_%s" % cos_or_sin])
        axs[2,col].set_xlabel("Time (s)")
        axs[2,col].set_ylabel("Voltage (mV)")
        axs[2,col].set_title("%s Differential" % cos_or_sin.capitalize())
    
    fig.tight_layout()

    return fig, axs

def update_live_gabor_plot(fig, axs, gabor_history: GaborHistory):
    signals = gabor_history.get_plot_signals(mV=True)

    for col, cos_or_sin in zip([0, 1], ["cos", "sin"]):
        axs[0,col].get_lines()[0].set_ydata(signals["%s_pos" % cos_or_sin])
        axs[0,col].set_ylim([0, signals["%s_pos" % cos_or_sin].max() * 1.1])

        axs[1,col].get_lines()[0].set_ydata(signals["%s_neg" % cos_or_sin])
        axs[1,col].set_ylim([0, signals["%s_neg" % cos_or_sin].max() * 1.1])

        axs[2,col].get_lines()[0].set_ydata(signals["x_%s" % cos_or_sin])
        axs[2,col].set_ylim([signals["x_%s" % cos_or_sin].min() * 1.1, 
                             signals["x_%s" % cos_or_sin].max() * 1.1])

    fig.canvas.draw()
    fig.canvas.flush_events()

def gui_fn(queue: Queue, 
           annotation_overlay: np.ndarray, 
           annotation_mask: np.ndarray, 
           duration,
           Fs: int,
           exp_name: str | None=None):
    if duration != np.inf:
        P = None

    last_imshow_time = 0
    i = 0
    while True:
        recvd = queue.get()

        p, p_mean_curr = recvd

        if p.ndim == 1:
            print("should not be here")
            p = p[None,:]
            p_mean_curr = p_mean_curr[None]

        if duration != np.inf:
            if P is None:
                P = np.zeros((duration, p.shape[1]))
                print("Allocating P: %.2f GB" % (P.size * P.itemsize / (1024**3)))
                print("P.shape", P.shape)

            if P[i:i+p.shape[0]].shape != p.shape:
                print("Padding P...This should only happen on the last iteration.")
                addl_size = p.shape[0] - P[i:i+p.shape[0]].shape[0]
                P = np.pad(P, ((0, addl_size), (0,0)))

            P[i:i+p.shape[0]] = p

        ## Update GUI At GUI_FPS
        if (time.perf_counter() - last_imshow_time) > (1 / GUI_FPS):
            img = visualize_readout(p_mean_curr[-1,:], annotation_overlay, annotation_mask)
            cv2.imshow("Mincam Readout", img)
            cv2.waitKey(1)

            last_imshow_time = time.perf_counter()

        i += p.shape[0]

        if EXIT or i >= duration:
            break

    if duration != np.inf:
        if exp_name is not None:
            save_name = utils.get_data_path() / "rt-data" / ("%s.npz" % exp_name)
        else:
            save_name = utils.get_data_path() / "rt-data" / \
                ("rt-data-%s.npz" % datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        np.savez(
            save_name, 
            P=P, 
            Fs=Fs,
            num_pixels=NUM_PIXELS,
            pixel_avg_start=PIXEL_AVG_START,
            time_per_pixel=TIME_PER_PIXEL,
            mincam_fps=MINCAM_FPS)


def plot_fn(queue: Queue, duration, Fs: int):
    gabor_history = GaborHistory(N_history=int(Fs), D=40)
    fig, axs = create_live_gabor_plot(gabor_history, Fs)

    last_plot_time = 0
    i = 0
    while True:
        recvd = queue.get()

        p, p_mean_curr = recvd

        if duration != np.inf:
            gabor_history.push_batch(p[2], p[0], p[3], p[1])

        ## Update plot at LIVE_PLOT_FPS
        if (time.perf_counter() - last_plot_time) > (1 / LIVE_PLOT_FPS):
            update_live_gabor_plot(fig, axs, gabor_history)
            last_plot_time = time.perf_counter()

        i += p.shape[0]

        if EXIT or i >= duration:
            break

    
def generate_single_readout_waveform(sample_duration, sleep_duration, pixel_ids):
    """
    sample_duration (samples)
    sleep_duration (samples)
    num_pixels
    """
    assert sleep_duration >= 0
    assert sample_duration >= 0

    num_pixels = len(pixel_ids)

    N = sample_duration * num_pixels + sleep_duration
    x = np.zeros(N, dtype=np.uint32)

    # Set address lines for the sample duration
    for i, pid in enumerate(pixel_ids):
        x[(i * sample_duration):((i+1)*sample_duration)] = pid
    
    return x


def add_cam_trigger_to_waveform(x, Fs, cam_trigger_pulse_length):
    N = len(x)
    num_replicas = int(MINCAM_FPS // np.gcd(int(MINCAM_FPS), CAMERA_FPS))
    new_duration_s = N / Fs * num_replicas

    num_camera_triggers = int(np.round(new_duration_s * CAMERA_FPS))

    x_full = np.tile(x, num_replicas)
    for i in range(num_camera_triggers):
        # Add the i'th camera trigger
        start_i = int(i * len(x_full) / num_camera_triggers)
        x_full[start_i:start_i+cam_trigger_pulse_length] |= 1 << 5

    return x_full, num_replicas


def generate_avg_idx(sample_duration, Fs, num_pixels, start_avg_s, end_avg_s):
    average_idx = None
    t = np.arange(sample_duration, dtype=np.int64) / Fs
    for i in range(num_pixels):
        idx = np.where((t >= start_avg_s) & (t < end_avg_s))[0] + i*sample_duration
        if average_idx is None:
            average_idx = np.zeros((num_pixels, len(idx)), dtype=np.int64)
        average_idx[i] = idx
    
    return average_idx

def run_rt_capture_continuous(queue_list: List[Queue],
                              duration: int, 
                              Fs: int,
                              read_cycles: int,
                              trigger_camera=False,
                              camera_ready_event=None):
    input_buffer_size = Fs * 30 # 10 second buffer

    analog_input_channel = "CAVE-DAQ/ai0"  # Replace with your specific channel name
    digital_output_channel = "CAVE-DAQ/port0/line0:%d" % (5 if trigger_camera else 4)

    sample_duration = int(TIME_PER_PIXEL * Fs) # Number of samples to read on each pixel
    # Time to sleep between each readout
    sleep_duration = np.maximum(
        int(Fs / MINCAM_FPS) - (sample_duration * NUM_PIXELS), 
        0)
    cam_trigger_pulse_length = int(10e-6 * Fs)
    
    assert duration % read_cycles == 0
    
    # Generate the digital waveform
    do_data = generate_single_readout_waveform(
        sample_duration, sleep_duration, PIXEL_IDS)
    
    # Trigger camera once every read cycles
    if trigger_camera:
        do_data, num_replicas = add_cam_trigger_to_waveform(
            do_data, Fs, cam_trigger_pulse_length)
        if num_replicas < read_cycles:
            do_data = np.tile(do_data, read_cycles // num_replicas)
        else:
            read_cycles = num_replicas
    else:
        # Repeat it read_cycles times
        do_data = np.tile(do_data, read_cycles)
    
    # Generate GUI averaging indices
    average_idx = generate_avg_idx(sample_duration, Fs, NUM_PIXELS, PIXEL_AVG_START, TIME_PER_PIXEL)

    # Create a Task for digital output
    with nidaqmx.Task() as digital_output_task:
        # Configure the digital output channel
        digital_output_task.do_channels.add_do_chan(digital_output_channel)
        # Configure timing for digital output
        digital_output_task.timing.cfg_samp_clk_timing(
            rate=Fs, sample_mode=constants.AcquisitionType.CONTINUOUS)
        digital_output_task.write(do_data)
        
        # Create a Task for analog input
        with nidaqmx.Task() as analog_input_task:
            # Add an analog input channel
            analog_input_task.ai_channels.add_ai_voltage_chan(
                analog_input_channel, 
                terminal_config=constants.TerminalConfiguration.DIFF,
                min_val=-5, max_val=5, units=constants.VoltageUnits.VOLTS)
            
            # Configure sample clock
            analog_input_task.timing.cfg_samp_clk_timing(
                rate=Fs, 
                source="do/SampleClock", 
                sample_mode=constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=int(np.ceil(input_buffer_size / len(do_data))) * \
                    len(do_data)
            )

            # Set this to False so that future read() calls block until 
            # we get all the samples
            analog_input_task.in_stream.read_all_avail_samp = False

            if camera_ready_event is not None:
                camera_ready_event.wait()

            # Start the analog input task
            analog_input_task.start()
            digital_output_task.start()
            
            # Main loop
            j = 0
            while True:
                read_data = np.asarray(
                    analog_input_task.read(
                    number_of_samples_per_channel=len(do_data),
                    timeout=10
                ))

                read_data = read_data.reshape(read_cycles, len(do_data) // read_cycles)

                if queue_list is not None:
                    avg_meas = read_data[:,average_idx].mean(2)
                    for q in queue_list:
                        q.put_nowait((read_data, avg_meas))

                j += read_cycles
                
                if EXIT or j >= duration:
                    break
                
def parse_args():
    parser = argparse.ArgumentParser(description="Real-time readout")
    return parser.parse_args()

def signal_handler(sig, frame):
    global EXIT
    EXIT = True

def run_realtime_readout():
    # Register SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    Fs = int(2e6) # Sampling rate

    #duration = int(MINCAM_FPS * 5)
    #duration = int(MINCAM_FPS * 60) * 20 # 20 minute capture
    duration = int(MINCAM_FPS * 60)
    
    read_cycles = 100
    duration = int(np.ceil(duration / read_cycles) * read_cycles)

    live_plot = False

    img, mask = mincam_annotation_overlay()

    queue_list = []

    ## Create the GUI process
    gui_queue = Queue()
    queue_list.append(gui_queue)
    gui_process = Process(
        target=gui_fn, 
        args=(gui_queue, img, mask, duration, Fs, args.name))
    gui_process.start()

    ## Create the plot process
    if live_plot:
        plot_queue = Queue()
        queue_list.append(plot_queue)
        plot_process = Process(
            target=plot_fn, 
            args=(plot_queue, duration, Fs))
        plot_process.start()

    run_rt_capture_continuous(
        queue_list, duration, Fs, read_cycles, trigger_camera=False)
    
    gui_process.join()
    if live_plot:
        plot_process.join()


def run_realtime_readout_and_video():
    # Register SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    video_name = utils.get_data_path() / "video-data" / "005ms.mp4"

    Fs = int(2e6) # Sampling rate

    read_cycles = 400

    duration = int(MINCAM_FPS * 10) # 10 second capture

    duration = int(np.ceil(duration / read_cycles) * read_cycles)
    camera_duration = int(np.floor(duration * CAMERA_FPS / MINCAM_FPS))

    img, mask = mincam_annotation_overlay()

    ## Create the camera process
    camera_ready_event = Event()
    camera_ready_event.clear()
    camera_process = Process(
        target=camera_fn,
        args=(camera_ready_event, camera_duration, args.name))
    camera_process.start()

    ## Create the gui process
    gui_queue = Queue()
    gui_process = Process(
        target=gui_fn, 
        args=(gui_queue, img, mask, duration, Fs, args.name))
    gui_process.start()

    ## Real-time capture
    run_rt_capture_continuous([gui_queue], duration, Fs, read_cycles,
                              trigger_camera=True, 
                              camera_ready_event=camera_ready_event)
    
    gui_process.join()
    camera_process.join()


def parse_args():
    args = argparse.ArgumentParser("Offline Odometry")
    args.add_argument("--name", "-n", type=str, required=False, default=None, help="Experiment name (without .npz)")
    return args.parse_args()

if __name__ == "__main__":
    #run_realtime_readout()
    run_realtime_readout_and_video()

