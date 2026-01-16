from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import nidaqmx
from nidaqmx import constants

try:
    from multiprocessing import Process, Event
except Exception:  # pragma: no cover
    Process = None  # type: ignore
    Event = None  # type: ignore

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


# ---- Defaults copied from lab_prototype_rt.py (keep consistent) ----
PIXEL_IDS = (np.asarray([17, 18, 22, 23]) - 1).astype(np.int64)
NUM_PIXELS = int(len(PIXEL_IDS))
PIXEL_AVG_START_S = 4e-6
TIME_PER_PIXEL_S = 6e-6
MINCAM_FPS = 1.0 / (TIME_PER_PIXEL_S * NUM_PIXELS)
CAMERA_FPS = 30


EXIT = False


def _signal_handler(_sig, _frame):
    global EXIT
    EXIT = True


def generate_single_readout_waveform(sample_duration: int, sleep_duration: int, pixel_ids: np.ndarray) -> np.ndarray:
    """Generate DO waveform that selects pixel address lines."""
    if sleep_duration < 0 or sample_duration < 0:
        raise ValueError("sample_duration and sleep_duration must be >= 0")

    n = int(sample_duration) * int(len(pixel_ids)) + int(sleep_duration)
    x = np.zeros(n, dtype=np.uint32)

    for i, pid in enumerate(pixel_ids):
        x[(i * sample_duration):((i + 1) * sample_duration)] = np.uint32(pid)

    return x


def add_cam_trigger_to_waveform(x: np.ndarray, fs: int, cam_trigger_pulse_length: int) -> tuple[np.ndarray, int]:
    """Bit 5 (line5) is camera trigger; matches lab_prototype_rt.py."""
    n = int(len(x))
    num_replicas = int(MINCAM_FPS // np.gcd(int(MINCAM_FPS), CAMERA_FPS))
    new_duration_s = (n / fs) * num_replicas
    num_camera_triggers = int(np.round(new_duration_s * CAMERA_FPS))

    x_full = np.tile(x, num_replicas)
    for i in range(num_camera_triggers):
        start_i = int(i * len(x_full) / num_camera_triggers)
        x_full[start_i:start_i + cam_trigger_pulse_length] |= (1 << 5)

    return x_full, num_replicas


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


@dataclass
class SessionMeta:
    created_utc: str
    session_dir: str

    # Timing
    fs_hz: int
    t0_wall_ns: int
    t1_wall_ns: Optional[int]

    # Channels / wiring
    analog_input_channel: str
    digital_output_channel: str
    trigger_camera: bool

    # Waveform parameters
    pixel_ids: list[int]
    num_pixels: int
    time_per_pixel_s: float
    pixel_avg_start_s: float
    mincam_fps: float
    read_cycles: int
    sample_duration_samples: int
    sleep_duration_samples: int
    cam_trigger_pulse_length_samples: int

    # Files
    raw_bin_path: str
    raw_dtype: str
    do_waveform_npy: str
    basler_mp4_path: Optional[str]

    # Counts
    samples_written: int


class RosSyncPublisher:
    """Optional: publish start/stop markers into ROS so rosbag can align DAQ file timebase."""

    def __init__(self, start_ns: int, session_dir: Path, topic_start: str, topic_stop: str):
        self._start_ns = int(start_ns)
        self._session_dir = session_dir
        self._topic_start = topic_start
        self._topic_stop = topic_stop

        self._ok = False
        self._node = None
        self._pub_start = None
        self._pub_stop = None

        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
            from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy  # type: ignore
            from std_msgs.msg import Int64  # type: ignore

            self._rclpy = rclpy
            self._Int64 = Int64

            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )

            rclpy.init(args=None)

            class _SyncNode(Node):
                pass

            self._node = _SyncNode("daq_sync_pub")
            self._pub_start = self._node.create_publisher(Int64, topic_start, qos)
            self._pub_stop = self._node.create_publisher(Int64, topic_stop, qos)
            self._ok = True
        except Exception:
            # Keep DAQ recording working even if ROS isn't available.
            self._ok = False

    @property
    def ok(self) -> bool:
        return self._ok

    def publish_start(self):
        if not self._ok:
            return

        msg = self._Int64()
        msg.data = int(self._start_ns)

        # Publish multiple times to survive slow discovery and ensure rosbag catches it.
        t_end = time.time() + 2.0
        while time.time() < t_end:
            self._pub_start.publish(msg)
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            time.sleep(0.05)

    def publish_stop(self, stop_ns: int):
        if not self._ok:
            return

        msg = self._Int64()
        msg.data = int(stop_ns)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            self._pub_stop.publish(msg)
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            time.sleep(0.05)

    def close(self):
        if not self._ok:
            return
        try:
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


def _project_root() -> Path:
    # minvio-hardware/daq_recorder.py -> workspace root is parent of minvio-hardware
    return Path(__file__).resolve().parents[1]


def _default_session_dir() -> Path:
    root = _project_root() / "data" / "daq_sessions"
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    return root / f"daq_{ts}"


def _configure_basler_camera(cam):
    # Copied from lab_prototype_rt.py (no changes intended)
    cam.UserSetSelector.Value = "UserSet1"
    cam.UserSetLoad.Execute()

    cam.Width.Value = 1100
    cam.Height.Value = 1100
    cam.OffsetX.Value = 444
    cam.OffsetY.Value = 46

    cam.PixelFormat.Value = "RGB8"
    cam.Gain.Value = 10
    cam.ExposureTime.Value = 5000  # us
    cam.Gamma.Value = 1

    cam.AcquisitionMode.Value = "Continuous"

    cam.TriggerSelector.Value = "FrameStart"
    cam.TriggerMode.Value = "On"
    cam.TriggerSource.Value = "Line1"
    cam.TriggerActivation.Value = "RisingEdge"

    cam.BslDemosaicingMode.Value = "Manual"
    cam.BslDemosaicingMethod.Value = "Unilinear"

    cam.DeviceLinkThroughputLimitMode.Value = 'Off'


def _basler_record_process(ready_event, stop_event, out_mp4_path: str):
    if pylon is None or skvideo is None:
        raise RuntimeError("Basler dependencies missing (pypylon and/or skvideo)")

    codec = "libx264"
    crf_out = 18
    ffmpeg_num_threads = 8
    writer = skvideo.io.FFmpegWriter(
        out_mp4_path,
        inputdict={'-r': str(CAMERA_FPS)},
        outputdict={
            '-vcodec': codec,
            '-crf': str(crf_out),
            '-threads': str(ffmpeg_num_threads),
            '-r': str(CAMERA_FPS),
            '-pix_fmt': 'yuv420p'
        },
    )

    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice()
    )
    camera.Open()
    _configure_basler_camera(camera)
    camera.StartGrabbing()

    ready_event.set()

    try:
        while not stop_event.is_set():
            with camera.RetrieveResult(60000, pylon.TimeoutHandling_Return) as result:
                if not result.IsValid():
                    continue
                if result.GrabSucceeded():
                    img = result.Array
                    writer.writeFrame(img)
    finally:
        try:
            camera.StopGrabbing()
        except Exception:
            pass
        try:
            camera.Close()
        except Exception:
            pass
        writer.close()


def run_daq_capture(
    *,
    session_dir: Path,
    duration_s: float,
    fs: int,
    read_cycles: int,
    analog_input_channel: str,
    device_name: str,
    trigger_camera: bool,
    record_basler: bool,
    ros_sync: bool,
    ros_topic_start: str,
    ros_topic_stop: str,
) -> int:
    session_dir.mkdir(parents=True, exist_ok=False)

    raw_bin_path = session_dir / "daq_raw_f32.bin"
    meta_path = session_dir / "daq_meta.json"
    do_waveform_path = session_dir / "do_waveform.npy"
    basler_mp4_path = session_dir / "basler.mp4"

    # Waveform (matching lab_prototype_rt.py)
    sample_duration = int(TIME_PER_PIXEL_S * fs)
    sleep_duration = int(max(int(fs / MINCAM_FPS) - (sample_duration * NUM_PIXELS), 0))
    cam_trigger_pulse_length = int(10e-6 * fs)

    do_data = generate_single_readout_waveform(sample_duration, sleep_duration, PIXEL_IDS)

    if trigger_camera:
        do_data, num_replicas = add_cam_trigger_to_waveform(do_data, fs, cam_trigger_pulse_length)
        # Make sure do_data covers read_cycles cycles as in the original code.
        if num_replicas < read_cycles:
            do_data = np.tile(do_data, read_cycles // num_replicas)
        else:
            read_cycles = num_replicas
    else:
        do_data = np.tile(do_data, read_cycles)

    np.save(do_waveform_path, do_data)

    # Precompute average_idx params for offline decoding (store in meta; compute index later)
    # average_idx itself depends on sample_duration; we store the parameters.

    # ROS sync marker
    t0_wall_ns = time.time_ns()
    ros_pub = None
    if ros_sync:
        ros_pub = RosSyncPublisher(
            start_ns=t0_wall_ns,
            session_dir=session_dir,
            topic_start=ros_topic_start,
            topic_stop=ros_topic_stop,
        )

    if ros_pub is not None and not ros_pub.ok:
        print("[WARN] ROS sync requested but rclpy/std_msgs not available; continuing without sync marker.")
        ros_pub = None

    # Save initial metadata early (so crashes still leave useful info)
    meta = SessionMeta(
        created_utc=datetime.utcnow().isoformat() + "Z",
        session_dir=str(session_dir),
        fs_hz=int(fs),
        t0_wall_ns=int(t0_wall_ns),
        t1_wall_ns=None,
        analog_input_channel=str(analog_input_channel),
        digital_output_channel=f"{device_name}/port0/line0:{5 if trigger_camera else 4}",
        trigger_camera=bool(trigger_camera),
        pixel_ids=[int(x) for x in PIXEL_IDS.tolist()],
        num_pixels=int(NUM_PIXELS),
        time_per_pixel_s=float(TIME_PER_PIXEL_S),
        pixel_avg_start_s=float(PIXEL_AVG_START_S),
        mincam_fps=float(MINCAM_FPS),
        read_cycles=int(read_cycles),
        sample_duration_samples=int(sample_duration),
        sleep_duration_samples=int(sleep_duration),
        cam_trigger_pulse_length_samples=int(cam_trigger_pulse_length),
        raw_bin_path=str(raw_bin_path),
        raw_dtype="float32",
        do_waveform_npy=str(do_waveform_path),
        basler_mp4_path=str(basler_mp4_path) if record_basler else None,
        samples_written=0,
    )

    meta_path.write_text(json.dumps(asdict(meta), indent=2))

    if ros_pub is not None:
        ros_pub.publish_start()

    # Optional Basler recording (triggered by DO waveform)
    basler_ready = None
    basler_stop = None
    basler_process = None
    if record_basler:
        if Process is None or Event is None:
            raise RuntimeError("multiprocessing is not available")
        if pylon is None or skvideo is None:
            raise RuntimeError("Basler recording requested but pypylon/skvideo not installed")
        if not trigger_camera:
            raise RuntimeError("Basler recording requested but --trigger-camera is not set")

        basler_ready = Event()
        basler_stop = Event()
        basler_ready.clear()
        basler_stop.clear()

        basler_process = Process(
            target=_basler_record_process,
            args=(basler_ready, basler_stop, str(basler_mp4_path)),
        )
        basler_process.start()
        # Ensure camera is ready before starting DAQ tasks
        basler_ready.wait(timeout=30.0)
        if not basler_ready.is_set():
            raise RuntimeError("Basler did not become ready within 30s")

    # Configure NI-DAQ tasks (DO drives the sample clock)
    digital_output_channel = f"{device_name}/port0/line0:{5 if trigger_camera else 4}"

    input_buffer_size = int(fs * 5)  # smaller than the original 30s; enough for safety

    samples_written = 0
    t_start = time.perf_counter()
    last_report_s = 0

    with open(raw_bin_path, "wb", buffering=1024 * 1024) as f:
        with nidaqmx.Task() as digital_output_task:
            digital_output_task.do_channels.add_do_chan(digital_output_channel)
            digital_output_task.timing.cfg_samp_clk_timing(
                rate=fs,
                sample_mode=constants.AcquisitionType.CONTINUOUS,
            )
            digital_output_task.write(do_data)

            with nidaqmx.Task() as analog_input_task:
                analog_input_task.ai_channels.add_ai_voltage_chan(
                    analog_input_channel,
                    terminal_config=constants.TerminalConfiguration.DIFF,
                    min_val=-5,
                    max_val=5,
                    units=constants.VoltageUnits.VOLTS,
                )

                analog_input_task.timing.cfg_samp_clk_timing(
                    rate=fs,
                    source="do/SampleClock",
                    sample_mode=constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan=int(np.ceil(input_buffer_size / len(do_data))) * len(do_data),
                )

                analog_input_task.in_stream.read_all_avail_samp = False

                analog_input_task.start()
                digital_output_task.start()

                while True:
                    if EXIT:
                        break

                    elapsed = time.perf_counter() - t_start
                    if duration_s > 0 and elapsed >= duration_s:
                        break

                    read_data = analog_input_task.read(
                        number_of_samples_per_channel=len(do_data),
                        timeout=10,
                    )

                    # Convert to float32 and write
                    arr = np.asarray(read_data, dtype=np.float32)
                    f.write(arr.tobytes(order="C"))
                    samples_written += int(arr.size)

                    # Lightweight progress log every ~1s
                    now_s = int(time.perf_counter() - t_start)
                    if now_s != last_report_s:
                        last_report_s = now_s
                        mb = (samples_written * 4) / (1024 * 1024)
                        print(f"[INFO] t={now_s:4d}s samples={samples_written} ({mb:.1f} MiB)")

    t1_wall_ns = time.time_ns()

    if basler_stop is not None:
        basler_stop.set()
    if basler_process is not None:
        basler_process.join(timeout=5.0)

    if ros_pub is not None:
        ros_pub.publish_stop(t1_wall_ns)
        ros_pub.close()

    # Update meta with end time and counts
    meta.t1_wall_ns = int(t1_wall_ns)
    meta.samples_written = int(samples_written)
    meta_path.write_text(json.dumps(asdict(meta), indent=2))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw NI-DAQ logger (2 MHz), compatible with lab_prototype_rt timing.")

    parser.add_argument("--duration-s", type=float, default=10.0, help="Record duration in seconds (default: 600). Use <=0 for until Ctrl+C")
    parser.add_argument("--fs", type=int, default=int(2e6), help="Sample rate in Hz (default: 2000000)")
    parser.add_argument("--read-cycles", type=int, default=400, help="Number of mincam cycles per read block (default: 400). Larger reduces Python overhead.")

    parser.add_argument("--device", type=str, default="CAVE-DAQ", help="NI device name (default: CAVE-DAQ)")
    parser.add_argument("--ai", type=str, default="CAVE-DAQ/ai0", help="Analog input channel (default: CAVE-DAQ/ai0)")

    parser.add_argument("--trigger-camera", action="store_true", help="Include camera trigger bit in DO waveform (bit 5 / line5)")
    parser.add_argument("--record-basler", action="store_true", help="Record Basler to MP4 (requires --trigger-camera)")

    parser.add_argument("--session-dir", type=str, default=None, help="Output session directory. Default: data/daq_sessions/daq_<timestamp>")

    parser.add_argument("--ros-sync", action="store_true", help="Publish /daq_sync/start_ns and /daq_sync/stop_ns (std_msgs/Int64) for rosbag alignment")
    parser.add_argument("--ros-topic-start", type=str, default="/daq_sync/start_ns", help="Topic for start marker")
    parser.add_argument("--ros-topic-stop", type=str, default="/daq_sync/stop_ns", help="Topic for stop marker")

    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    args = parse_args()

    session_dir = Path(args.session_dir) if args.session_dir else _default_session_dir()

    print(f"[INFO] Writing session to: {session_dir}")
    print(f"[INFO] fs={args.fs}Hz duration={args.duration_s}s read_cycles={args.read_cycles}")
    print(f"[INFO] ai={args.ai} device={args.device} trigger_camera={args.trigger_camera}")
    if args.record_basler:
        print("[INFO] Basler recording ON")
    if args.ros_sync:
        print(f"[INFO] ROS sync ON: {args.ros_topic_start} / {args.ros_topic_stop}")

    return run_daq_capture(
        session_dir=session_dir,
        duration_s=float(args.duration_s),
        fs=int(args.fs),
        read_cycles=int(args.read_cycles),
        analog_input_channel=str(args.ai),
        device_name=str(args.device),
        trigger_camera=bool(args.trigger_camera),
        record_basler=bool(args.record_basler),
        ros_sync=bool(args.ros_sync),
        ros_topic_start=str(args.ros_topic_start),
        ros_topic_stop=str(args.ros_topic_stop),
    )


if __name__ == "__main__":
    raise SystemExit(main())
