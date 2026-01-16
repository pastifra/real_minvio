# Synchronized Recording Runbook (Windows laptop)

Goal: record **Robot ROS topics + RealSense + NI-DAQ + Basler**, all aligned to the **Windows wall clock**.

This repo uses:
- ROS 2 Jazzy (Windows) via **Pixi** (see `pixi.toml`).
- RealSense D455 publisher: `realsense.py`.
- NI-DAQ raw logger + optional Basler MP4: `minvio-hardwareV2/daq_recorder.py` (run in a separate Conda env).
- ROS sync marker bridge (Pixi env): `minvio-hardwareV2/daq_sync_bridge.py`.

## Assumptions
- Windows clock is the reference time.
- Raspberry Pi clock is synchronized to Windows via chrony (so its published `header.stamp` is meaningful vs Windows).
- You **do not** use ROS simulated time.
- You are running commands from the workspace root: `C:\pixi_ws`.
  - Important: `pixi.toml` sets `FASTRTPS_DEFAULT_PROFILES_FILE` using `%cd%`, so the current directory must be the repo root.

---

## One-time setup

### 1) Pixi / ROS environment
From PowerShell:

```powershell
Set-Location C:\pixi_ws
pixi --version
pixi run ros2 --help
```

### 2) DAQ + Basler Conda environment
Create/activate a Conda env that has NI-DAQmx + Basler deps:

- Required for DAQ: `nidaqmx`, `numpy`
- Optional for Basler MP4: `pypylon`, `scikit-video`, and `ffmpeg` on PATH

Sanity check in the **DAQ env**:

```powershell
python -c "import nidaqmx, numpy as np; print('nidaqmx ok', nidaqmx.__version__)"
python -c "from pypylon import pylon; import skvideo.io; print('basler ok')"
ffmpeg -version
```

---

## Preflight sanity checks (do these before every recording)

### A) Clock sanity

#### Windows time service
```powershell
w32tm /query /status
Get-Date
```

What you want:
- `w32tm` shows a stable sync source and reasonable offset (ideally milliseconds).

#### Robot Pi chrony (run on the Pi)
```bash
chronyc tracking
chronyc sources -v
```

What you want:
- Small offset (ideally < ~5 ms; tens of ms may be usable but expect worse alignment).
- No large jitter spikes.

### B) ROS time source sanity (no sim time)
On Windows (Pixi env):

```powershell
Set-Location C:\pixi_ws
pixi run ros2 topic list | Select-String "^/clock$"  # should print nothing
```

Once `realsense.py` is running:

```powershell
pixi run ros2 param get /d455_camera_node use_sim_time
```

Expected: `False`.

### C) Network + discovery sanity
Make sure you can see the robot topics:

```powershell
pixi run ros2 node list
pixi run ros2 topic list
```

If the robot isn’t visible:
- confirm you’re on the robot AP Wi‑Fi
- confirm `ROS_DOMAIN_ID` matches across machines (if you use it)

### D) QoS sanity (critical)
For each topic you plan to record, check the publisher QoS:

```powershell
pixi run ros2 topic info -v /merged_odom
pixi run ros2 topic info -v /<robot_imu_topic>

pixi run ros2 topic info -v /d455_camera/color/image_raw
pixi run ros2 topic info -v /d455_camera/depth/image_rect_raw
pixi run ros2 topic info -v /d455_camera/gyro/sample
```

Rule of thumb:
- If a publisher is **BEST_EFFORT** and your subscriber/rosbag is **RELIABLE**, you can get "no messages".
- If things look dead, fix it by matching rosbag subscriber QoS to the publisher (see QoS override file below).

### E) TF sanity (for RViz / downstream consumers)
With RealSense running, check transforms exist:

```powershell
pixi run ros2 topic echo --once /tf_static
pixi run ros2 run tf2_ros tf2_echo base_link d455_camera_depth_optical_frame
```

---

## Recording procedure (copy/paste)

You’ll use **4 terminals**.

### Terminal 1 — RealSense node (Pixi)
```powershell
Set-Location C:\pixi_ws
pixi run python realsense.py
```

Notes:
- Default publishes images RELIABLE. If you need BEST_EFFORT images, set:

```powershell
pixi run ros2 param set /d455_camera_node reliable_images false
```

### Terminal 2 — Choose a run ID + session folder (PowerShell)
Create one timestamp string and reuse it for DAQ + rosbag:

```powershell
Set-Location C:\pixi_ws
$run = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$daqDir = Join-Path $PWD "data\daq_sessions\daq_$run"
$bagDir = Join-Path $PWD "data\rosbags\run_$run"
New-Item -ItemType Directory -Force -Path $daqDir | Out-Null
New-Item -ItemType Directory -Force -Path $bagDir | Out-Null

Write-Host "DAQ session: $daqDir"
Write-Host "Bag folder:  $bagDir"
```

### Terminal 3 — DAQ sync bridge (Pixi)
Start the bridge pointed at the DAQ session directory. It will publish `/daq_sync/start_ns` and `/daq_sync/stop_ns` when it sees them in `daq_meta.json`.

```powershell
Set-Location C:\pixi_ws
pixi run python minvio-hardwareV2\daq_sync_bridge.py --session-dir "$daqDir" --exit-after-stop
```

### Terminal 4 — rosbag record (Pixi)
Decide which robot topics you want. Example placeholders:
- `/merged_odom`
- `/<robot_imu_topic>`
- any other robot topics you need

Run rosbag:

```powershell
Set-Location C:\pixi_ws

# Edit the topics list to match your robot
pixi run ros2 bag record -o "$bagDir" `
  /tf /tf_static `
  /d455_camera/color/image_raw /d455_camera/color/camera_info `
  /d455_camera/depth/image_rect_raw /d455_camera/depth/camera_info `
  /d455_camera/accel/sample /d455_camera/gyro/sample `
  /daq_sync/start_ns /daq_sync/stop_ns `
  /merged_odom /<robot_imu_topic> `
  --qos-profile-overrides-path rosbag_qos_overrides.yaml
```

If you don’t need QoS overrides, you can omit the last line. If you see missing data, use overrides.

### Terminal 2 (continued) — start DAQ + Basler (Conda env)
In your DAQ Conda env, start capture using the session dir created above:

```powershell
Set-Location C:\pixi_ws

# Activate your DAQ env (example name)
conda activate daq

python .\minvio-hardwareV2\daq_recorder.py `
  --session-dir "$daqDir" `
  --duration-s 600 `
  --fs 2000000 `
  --read-cycles 400 `
  --device CAVE-DAQ `
  --ai CAVE-DAQ/ai0 `
  --trigger-camera `
  --record-basler
```

Notes:
- If you do NOT want Basler MP4, drop `--record-basler` and optionally `--trigger-camera`.
- Keep DAQ in Conda and ROS in Pixi; the sync bridge handles alignment.

---

## During recording: quick live sanity checks

### Check RealSense IMU arrival rate (Pixi)
This is more trustworthy than `ros2 topic hz` on Windows:

```powershell
Set-Location C:\pixi_ws
pixi run python hz_check_imu.py /d455_camera/gyro/sample
```

### Check DAQ sync marker shows up in ROS (Pixi)
```powershell
pixi run ros2 topic echo --once /daq_sync/start_ns
```

---

## After recording

### 1) Verify rosbag contains what you expect
```powershell
Set-Location C:\pixi_ws
pixi run ros2 bag info "$bagDir"
```

### 2) Verify DAQ session files exist
In `$daqDir` you should have:
- `daq_meta.json`
- `daq_raw_f32.bin`
- `do_waveform.npy`
- optionally `basler.mp4`

### 3) Offline processing (optional)
```powershell
Set-Location C:\pixi_ws

# Run this in an env that has numpy + scipy
python .\minvio-hardwareV2\daq_offline.py --session "$daqDir"
```

---

## QoS overrides for rosbag
Edit `rosbag_qos_overrides.yaml` to match your **publisher QoS** (use `ros2 topic info -v`).

If a topic is BEST_EFFORT, set rosbag subscription reliability to BEST_EFFORT.

---

## Troubleshooting checklist
- If topics appear "dead": run `pixi run ros2 topic info -v <topic>` and fix QoS mismatch.
- If RViz drops depth with "Message Filter queue is full": TF chain is missing (check `tf2_echo`).
- If discovery breaks after network changes: restart the affected nodes and ensure you launched from `C:\pixi_ws` so FastDDS profile is applied.
