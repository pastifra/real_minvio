from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import scipy.signal


@dataclass(frozen=True)
class DaqSession:
    session_dir: Path
    meta_path: Path
    raw_path: Path
    do_waveform_path: Path

    fs_hz: int
    mincam_fps: float
    t0_wall_ns: int
    t1_wall_ns: Optional[int]

    num_pixels: int
    pixel_ids: np.ndarray

    time_per_pixel_s: float
    pixel_avg_start_s: float

    read_cycles: int
    sample_duration_samples: int
    sleep_duration_samples: int

    samples_written: int

    @property
    def cycle_len(self) -> int:
        return int(self.sample_duration_samples * self.num_pixels + self.sleep_duration_samples)

    @property
    def cycles_total(self) -> int:
        return int(self.samples_written // self.cycle_len)


def load_session(session_dir: Path) -> DaqSession:
    meta_path = session_dir / "daq_meta.json"
    raw_path = session_dir / "daq_raw_f32.bin"
    do_waveform_path = session_dir / "do_waveform.npy"

    meta = json.loads(meta_path.read_text())

    fs_hz = int(meta["fs_hz"])
    mincam_fps = float(meta["mincam_fps"])
    t0_wall_ns = int(meta["t0_wall_ns"])
    t1_wall_ns = meta.get("t1_wall_ns")
    t1_wall_ns = int(t1_wall_ns) if t1_wall_ns is not None else None

    num_pixels = int(meta["num_pixels"])
    pixel_ids = np.asarray(meta["pixel_ids"], dtype=np.int64)

    time_per_pixel_s = float(meta["time_per_pixel_s"])
    pixel_avg_start_s = float(meta["pixel_avg_start_s"])

    read_cycles = int(meta["read_cycles"])
    sample_duration_samples = int(meta["sample_duration_samples"])
    sleep_duration_samples = int(meta["sleep_duration_samples"])

    samples_written = int(meta["samples_written"])

    return DaqSession(
        session_dir=session_dir,
        meta_path=meta_path,
        raw_path=raw_path,
        do_waveform_path=do_waveform_path,
        fs_hz=fs_hz,
        mincam_fps=mincam_fps,
        t0_wall_ns=t0_wall_ns,
        t1_wall_ns=t1_wall_ns,
        num_pixels=num_pixels,
        pixel_ids=pixel_ids,
        time_per_pixel_s=time_per_pixel_s,
        pixel_avg_start_s=pixel_avg_start_s,
        read_cycles=read_cycles,
        sample_duration_samples=sample_duration_samples,
        sleep_duration_samples=sleep_duration_samples,
        samples_written=samples_written,
    )


def generate_avg_idx(sample_duration: int, fs: int, num_pixels: int, start_avg_s: float, end_avg_s: float) -> np.ndarray:
    average_idx = None
    t = np.arange(sample_duration, dtype=np.int64) / fs
    for i in range(num_pixels):
        idx = np.where((t >= start_avg_s) & (t < end_avg_s))[0] + i * sample_duration
        if average_idx is None:
            average_idx = np.zeros((num_pixels, len(idx)), dtype=np.int64)
        average_idx[i] = idx

    if average_idx is None:
        raise RuntimeError("average_idx not computed")

    return average_idx


def design_combined_sos(mincam_fps: float) -> np.ndarray:
    # Matches run_offline_odometry() in offline_odometry.py
    notch_filter_freqs = [60, 79, 120, 240, 360, 480, 600, 720, 840, 960]
    q = 30.0

    all_filters_sos = []
    for f0 in notch_filter_freqs:
        b, a = scipy.signal.iirnotch(f0, q, mincam_fps)
        all_filters_sos.append(scipy.signal.tf2sos(b, a))

    cutoff = 1000
    numtaps = 201
    b_lpf = scipy.signal.firwin(numtaps, cutoff, window="hamming", fs=mincam_fps)
    all_filters_sos.append(scipy.signal.tf2sos(b_lpf, [1]))

    sos = np.vstack(all_filters_sos)
    return sos


def iter_cycle_chunks(raw_mem: np.memmap, cycle_len: int, chunk_cycles: int) -> Iterable[tuple[int, np.ndarray]]:
    total_cycles = raw_mem.size // cycle_len
    for start in range(0, total_cycles, chunk_cycles):
        end = min(total_cycles, start + chunk_cycles)
        chunk = raw_mem[start * cycle_len: end * cycle_len]
        yield start, np.asarray(chunk).reshape(end - start, cycle_len)


def compute_p_mean_memmap(session: DaqSession, out_dir: Path, chunk_cycles: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    p_mean_path = out_dir / "p_mean_f32.npy"

    cycle_len = session.cycle_len
    total_cycles = session.cycles_total

    # Validate do_waveform length matches read_cycles * cycle_len
    if session.do_waveform_path.exists():
        do = np.load(session.do_waveform_path)
        expected = session.read_cycles * cycle_len
        if int(do.size) != int(expected):
            # Not fatal; but it likely means trigger_camera replicas changed read_cycles.
            # Still, cycle_len is authoritative for decoding.
            pass

    raw = np.memmap(session.raw_path, dtype=np.float32, mode="r", shape=(session.samples_written,))

    avg_idx = generate_avg_idx(
        session.sample_duration_samples,
        session.fs_hz,
        session.num_pixels,
        session.pixel_avg_start_s,
        session.time_per_pixel_s,
    )

    p_mean_mm = np.lib.format.open_memmap(
        p_mean_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_cycles, session.num_pixels),
    )

    for start, cycles in iter_cycle_chunks(raw, cycle_len, chunk_cycles):
        # cycles: (N, cycle_len)
        # cycles[:, avg_idx] -> (N, num_pixels, n_avg)
        means = cycles[:, avg_idx].mean(axis=2)
        p_mean_mm[start:start + cycles.shape[0], :] = means.astype(np.float32)

    p_mean_mm.flush()
    return p_mean_path


def sosfilt_memmap(
    x_path: Path,
    y_path: Path,
    sos: np.ndarray,
    chunk_rows: int,
) -> None:
    x = np.load(x_path, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError("expected 2D array for x")

    y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=x.shape)

    n_sections = sos.shape[0]
    zi = np.zeros((n_sections, 2), dtype=np.float64)

    # Filter each pixel independently to match offline_odometry.py
    for pi in range(x.shape[1]):
        zi_pi = zi.copy()
        for start in range(0, x.shape[0], chunk_rows):
            end = min(x.shape[0], start + chunk_rows)
            x_chunk = np.asarray(x[start:end, pi], dtype=np.float64)
            y_chunk, zi_pi = scipy.signal.sosfilt(sos, x_chunk, zi=zi_pi)
            y[start:end, pi] = y_chunk.astype(np.float32)

    y.flush()


def compute_differentials(filtered_path: Path, out_dir: Path) -> tuple[Path, Path]:
    filt = np.load(filtered_path, mmap_mode="r")
    if filt.shape[1] < 4:
        raise RuntimeError("expected >=4 pixels for cos/sin differential")

    out_dir.mkdir(parents=True, exist_ok=True)
    cos_path = out_dir / "cos_diff_f32.npy"
    sin_path = out_dir / "sin_diff_f32.npy"

    cos = np.lib.format.open_memmap(cos_path, mode="w+", dtype=np.float32, shape=(filt.shape[0],))
    sin = np.lib.format.open_memmap(sin_path, mode="w+", dtype=np.float32, shape=(filt.shape[0],))

    # Convention from offline_odometry.py:
    # cos_differential = filtered[:,2] - filtered[:,0]
    # sin_differential = filtered[:,3] - filtered[:,1]
    chunk = 1_000_000
    for start in range(0, filt.shape[0], chunk):
        end = min(filt.shape[0], start + chunk)
        f = np.asarray(filt[start:end, :], dtype=np.float32)
        cos[start:end] = f[:, 0] - f[:, 1]
        sin[start:end] = f[:, 2] - f[:, 3]

    cos.flush()
    sin.flush()
    return cos_path, sin_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline processing for DAQ sessions recorded by daq_recorder.py")

    parser.add_argument("--session", type=str, required=True, help="Path to DAQ session directory (contains daq_meta.json)")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: <session>/processed)")

    parser.add_argument("--chunk-cycles", type=int, default=250_000, help="Chunk size in cycles for p_mean extraction")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000, help="Chunk size in rows for filtering")

    parser.add_argument("--no-filter", action="store_true", help="Only compute p_mean, skip filtering")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session)
    session = load_session(session_dir)

    out_dir = Path(args.out) if args.out else (session_dir / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Session: {session_dir}")
    print(f"[INFO] Fs={session.fs_hz}Hz mincam_fps={session.mincam_fps:.3f} cycle_len={session.cycle_len} cycles_total={session.cycles_total}")

    p_mean_path = compute_p_mean_memmap(session, out_dir, chunk_cycles=int(args.chunk_cycles))
    print(f"[INFO] Wrote: {p_mean_path}")

    if args.no_filter:
        return 0

    sos = design_combined_sos(session.mincam_fps)
    filt_path = out_dir / "p_mean_filtered_f32.npy"
    sosfilt_memmap(p_mean_path, filt_path, sos=sos, chunk_rows=int(args.chunk_rows))
    print(f"[INFO] Wrote: {filt_path}")

    cos_path, sin_path = compute_differentials(filt_path, out_dir)
    print(f"[INFO] Wrote: {cos_path}")
    print(f"[INFO] Wrote: {sin_path}")

    # Write a small summary json for downstream scripts
    summary = {
        "session_dir": str(session_dir),
        "processed_dir": str(out_dir),
        "p_mean": str(p_mean_path),
        "p_mean_filtered": str(filt_path),
        "cos_diff": str(cos_path),
        "sin_diff": str(sin_path),
        "fs_hz": session.fs_hz,
        "mincam_fps": session.mincam_fps,
        "t0_wall_ns": session.t0_wall_ns,
        "t1_wall_ns": session.t1_wall_ns,
        "cycle_len": session.cycle_len,
        "cycles_total": session.cycles_total,
    }
    (out_dir / "processed_meta.json").write_text(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
