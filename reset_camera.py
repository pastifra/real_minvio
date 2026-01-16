import pyrealsense2 as rs
import time

try:
    print("Searching for RealSense devices...")
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No devices found. Is it plugged in?")
        exit(1)

    dev = ctx.devices[0]
    print(f"Resetting {dev.get_info(rs.camera_info.name)}...")
    dev.hardware_reset()
    print("Reset command sent. Camera will disappear and reappear in ~5 seconds.")

except Exception as e:
    print(f"Error: {e}")