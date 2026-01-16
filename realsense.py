import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterAlreadyDeclaredException
from sensor_msgs.msg import Image, Imu, CameraInfo
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from builtin_interfaces.msg import Time as TimeMsg
import pyrealsense2 as rs
import numpy as np
import queue
import time
import threading
import traceback
import sys

# --- CONFIGURATION ---
CAM_X = -0.085
CAM_Y = 0.0
CAM_Z = 0.105
FACING_BACKWARDS = False
STREAM_W = 640
STREAM_H = 480
STREAM_FPS = 30

MAX_IMU_PER_TICK = 300
WATCHDOG_TIMEOUT_S = 2.0
PROCESS_TIMER_S = 0.02


class D455LocalFix(Node):
    def __init__(self):
        super().__init__('d455_camera_node')
        self.ns = 'd455_camera'
        self.running = True

        # Mode flags
        self._using_rs_frame_queue = False     # True if pipeline started with rs.frame_queue
        self._imu_published_in_capture_thread = False

        # Background capture thread
        self._capture_thread = None
        self._capture_stop = threading.Event()

        try:
            self.declare_parameter('use_sim_time', False)
        except ParameterAlreadyDeclaredException:
            pass
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, False)])

        self.declare_parameter('reliable_images', True)
        reliable_images = bool(self.get_parameter('reliable_images').value)

        self.video_count = 0
        self._accel_pub_count = 0
        self._gyro_pub_count = 0
        self._last_rate_print = time.time()
        self.last_print = time.time()

        self._last_any_frame_monotonic = time.monotonic()
        self._pipeline_lock = threading.Lock()

        # Keep Python queues small; we only ever want the latest video frames
        self.imu_queue: queue.Queue[tuple[rs.frame, int]] = queue.Queue(maxsize=2000)
        self.video_queue: queue.Queue[tuple[rs.frame, int]] = queue.Queue(maxsize=60)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliable_images else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2000,
        )

        self.color_pub = self.create_publisher(Image, f'/{self.ns}/color/image_raw', image_qos)
        self.color_info_pub = self.create_publisher(CameraInfo, f'/{self.ns}/color/camera_info', info_qos)
        self.depth_pub = self.create_publisher(Image, f'/{self.ns}/depth/image_rect_raw', image_qos)
        self.depth_info_pub = self.create_publisher(CameraInfo, f'/{self.ns}/depth/camera_info', info_qos)
        self.accel_pub = self.create_publisher(Imu, f'/{self.ns}/accel/sample', imu_qos)
        self.gyro_pub = self.create_publisher(Imu, f'/{self.ns}/gyro/sample', imu_qos)

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_tf()

        self.pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, STREAM_FPS)
        self._config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, STREAM_FPS)

        try:
            self._config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
            self._config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            self.get_logger().info("Requested IMU rates: accel=100Hz gyro=200Hz")
        except Exception as ex:
            self.get_logger().warning(f"Could not request explicit IMU rates ({ex}); using defaults")
            self._config.enable_stream(rs.stream.accel)
            self._config.enable_stream(rs.stream.gyro)

        # C++ frame queue to avoid Python callback/GIL issues
        self._rs_queue = rs.frame_queue(2048)

        self.get_logger().info(f"Configuring D455 ({STREAM_W}x{STREAM_H} @ {STREAM_FPS}FPS)...")
        self._start_pipeline()

        # Start capture thread if we're using the C++ queue
        if self._using_rs_frame_queue:
            self._imu_published_in_capture_thread = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

        self.get_logger().info("CAMERA LIVE. WATCH TERMINAL FOR FPS")
        self.create_timer(PROCESS_TIMER_S, self._process_queues_safe)

    def _start_pipeline(self):
        with self._pipeline_lock:
            try:
                # Preferred: no Python callback; frames go into C++ rs.frame_queue
                self.profile = self.pipeline.start(self._config, self._rs_queue)
                self._using_rs_frame_queue = True
            except Exception as ex:
                # Fallback: Python callback (less reliable on Windows/Python)
                self.get_logger().warning(
                    f"pipeline.start(config, frame_queue) failed, falling back to Python callback: {ex}"
                )
                self.profile = self.pipeline.start(self._config, self.driver_callback)
                self._using_rs_frame_queue = False

            # Log negotiated stream FPS
            try:
                color_sp = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
                depth_sp = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
                accel_sp = self.profile.get_stream(rs.stream.accel)
                gyro_sp = self.profile.get_stream(rs.stream.gyro)
                self.get_logger().info(
                    f"Negotiated FPS: color={color_sp.fps()} depth={depth_sp.fps()} "
                    f"accel={accel_sp.fps()} gyro={gyro_sp.fps()}"
                )
            except Exception as ex:
                self.get_logger().warning(f"Could not query negotiated FPS: {ex}")

            self.color_intrinsics = (
                self.profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            self.depth_intrinsics = (
                self.profile.get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )

            self._last_any_frame_monotonic = time.monotonic()

    def _restart_pipeline(self, reason: str):
        self.get_logger().warning(f"Restarting RealSense pipeline: {reason}")
        with self._pipeline_lock:
            self.running = False
            try:
                try:
                    self.pipeline.stop()
                except Exception:
                    self.get_logger().warning("pipeline.stop() raised:\n" + traceback.format_exc())

                self._drain_queue(self.imu_queue)
                self._drain_queue(self.video_queue)

                self.running = True
                self._start_pipeline()
            finally:
                self.running = True

    def _capture_loop(self):
        # Drain frames from the C++ queue; publish IMU immediately; enqueue video for timer thread.
        while (not self._capture_stop.is_set()) and rclpy.ok():
            if not self.running:
                time.sleep(0.01)
                continue
            try:
                frame = self._rs_queue.wait_for_frame(500)  # ms
            except Exception:
                # pipeline.stop/restart can cause transient errors here
                continue

            if frame is None:
                continue

            self._last_any_frame_monotonic = time.monotonic()
            ts_ns = time.time_ns()

            try:
                frame.keep()
            except Exception:
                pass

            try:
                if frame.is_motion_frame():
                    # Publish IMU immediately to minimize jitter
                    self.publish_imu(frame, self._ts_ns_to_time_msg(ts_ns))
                else:
                    self._put_bounded(self.video_queue, (frame, ts_ns))
            except Exception:
                self.get_logger().error("_capture_loop() exception:\n" + traceback.format_exc())

    @staticmethod
    def _drain_queue(q: queue.Queue):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return

    def publish_static_tf(self):
        # Publish TFs for every frame_id we put into message headers.
        # RViz will drop messages if it can't transform their frame into the Fixed Frame.
        now = self.get_clock().now().to_msg()

        def make_tf(parent: str, child: str):
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = CAM_X
            t.transform.translation.y = CAM_Y
            t.transform.translation.z = CAM_Z
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            return t

        transforms = [
            make_tf("base_link", f"{self.ns}_color_optical_frame"),
            make_tf("base_link", f"{self.ns}_depth_optical_frame"),
            make_tf("base_link", f"{self.ns}_accel_optical_frame"),
            make_tf("base_link", f"{self.ns}_gyro_optical_frame"),
        ]
        self.tf_broadcaster.sendTransform(transforms)

    # Fallback-only path (used when rs.frame_queue start isn't available)
    def driver_callback(self, frame):
        if not self.running:
            return
        try:
            self._last_any_frame_monotonic = time.monotonic()
            ts_ns = time.time_ns()
            try:
                frame.keep()
            except Exception:
                pass

            if frame.is_motion_frame():
                self._put_bounded(self.imu_queue, (frame, ts_ns))
            else:
                self._put_bounded(self.video_queue, (frame, ts_ns))
        except Exception:
            self.get_logger().error("driver_callback() exception:\n" + traceback.format_exc())

    @staticmethod
    def _put_bounded(q: queue.Queue, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _process_queues_safe(self):
        try:
            self.process_queues()
        except Exception:
            self.get_logger().error("process_queues() exception:\n" + traceback.format_exc())

    @staticmethod
    def _ts_ns_to_time_msg(ts_ns: int) -> TimeMsg:
        msg = TimeMsg()
        msg.sec = int(ts_ns // 1_000_000_000)
        msg.nanosec = int(ts_ns % 1_000_000_000)
        return msg

    def process_queues(self):
        if (time.monotonic() - self._last_any_frame_monotonic) > WATCHDOG_TIMEOUT_S:
            self._restart_pipeline(f"no frames for > {WATCHDOG_TIMEOUT_S}s")
            return

        # If we are NOT publishing IMU in the capture thread, drain IMU queue here (fallback mode)
        if not self._imu_published_in_capture_thread:
            imu_processed = 0
            while imu_processed < MAX_IMU_PER_TICK:
                try:
                    frame, ts_ns = self.imu_queue.get_nowait()
                except queue.Empty:
                    break
                self.publish_imu(frame, self._ts_ns_to_time_msg(ts_ns))
                imu_processed += 1

        # VIDEO: accept framesets OR single video frames
        latest_color = None
        latest_color_ts = None
        latest_depth = None
        latest_depth_ts = None
        while True:
            try:
                frame, ts_ns = self.video_queue.get_nowait()
            except queue.Empty:
                break

            if frame.is_frameset():
                fset = frame.as_frameset()
                cf = fset.get_color_frame()
                df = fset.get_depth_frame()
                if cf:
                    latest_color = cf
                    latest_color_ts = ts_ns
                if df:
                    latest_depth = df
                    latest_depth_ts = ts_ns
            elif frame.is_video_frame():
                st = frame.get_profile().stream_type()
                if st == rs.stream.color:
                    latest_color = frame.as_video_frame()
                    latest_color_ts = ts_ns
                elif st == rs.stream.depth:
                    latest_depth = frame.as_video_frame()
                    latest_depth_ts = ts_ns

        if latest_color is not None and latest_color_ts is not None:
            self.process_video(latest_color, None, self._ts_ns_to_time_msg(latest_color_ts))
            self.video_count += 1

        if latest_depth is not None and latest_depth_ts is not None:
            self.process_video(None, latest_depth, self._ts_ns_to_time_msg(latest_depth_ts))

        if time.time() - self.last_print > 1.0:
            now = time.time()
            dt = max(1e-6, now - self._last_rate_print)
            accel_hz = self._accel_pub_count / dt
            gyro_hz = self._gyro_pub_count / dt
            self._accel_pub_count = 0
            self._gyro_pub_count = 0
            self._last_rate_print = now

            self.get_logger().info(
                f"STATUS: Video={self.video_count} FPS | "
                f"accel_pub={accel_hz:.1f}Hz gyro_pub={gyro_hz:.1f}Hz | "
                f"imu_q={self.imu_queue.qsize()} video_q={self.video_queue.qsize()} | "
                f"rs_queue={'on' if self._using_rs_frame_queue else 'off'}"
            )
            self.video_count = 0
            self.last_print = time.time()

    # ...keep your publish_imu/process_video/numpy_to_image/get_camera_info unchanged...
    

    def publish_imu(self, frame, stamp_msg: TimeMsg):
        try:
            data = frame.as_motion_frame().get_motion_data()
            msg = Imu()
            msg.header.stamp = stamp_msg

            stream_name = frame.get_profile().stream_name()
            if "Accel" in stream_name:
                msg.header.frame_id = f"{self.ns}_accel_optical_frame"
                msg.linear_acceleration.x = float(data.x)
                msg.linear_acceleration.y = float(data.y)
                msg.linear_acceleration.z = float(data.z)
                self.accel_pub.publish(msg)
                self._accel_pub_count += 1
            elif "Gyro" in stream_name:
                msg.header.frame_id = f"{self.ns}_gyro_optical_frame"
                msg.angular_velocity.x = float(data.x)
                msg.angular_velocity.y = float(data.y)
                msg.angular_velocity.z = float(data.z)
                self.gyro_pub.publish(msg)
                self._gyro_pub_count += 1
        except Exception:
            self.get_logger().error("publish_imu() exception:\n" + traceback.format_exc())



    def process_video(self, color_frame, depth_frame, stamp_msg: TimeMsg):
        try:
            if color_frame:
                data = np.asanyarray(color_frame.get_data())
                data = np.ascontiguousarray(data)
                msg = self.numpy_to_image(data, 'bgr8', f"{self.ns}_color_optical_frame", stamp_msg)
                self.color_pub.publish(msg)
                self.color_info_pub.publish(
                    self.get_camera_info(self.color_intrinsics, f"{self.ns}_color_optical_frame", stamp_msg)
                )

            if depth_frame:
                data = np.asanyarray(depth_frame.get_data())
                data = np.ascontiguousarray(data)
                # Depth should be 16UC1 (uint16), not mono16, for most depth consumers.
                msg = self.numpy_to_image(data, '16UC1', f"{self.ns}_depth_optical_frame", stamp_msg)
                self.depth_pub.publish(msg)
                self.depth_info_pub.publish(
                    self.get_camera_info(self.depth_intrinsics, f"{self.ns}_depth_optical_frame", stamp_msg)
                )
        except Exception:
            self.get_logger().error("process_video() exception:\n" + traceback.format_exc())

    def numpy_to_image(self, data, encoding, frame_id, stamp_msg: TimeMsg):
        msg = Image()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = frame_id
        msg.height, msg.width = data.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0  # Windows/x86 is little-endian
        channels = (data.shape[2] if len(data.shape) > 2 else 1)
        msg.step = msg.width * data.itemsize * channels
        msg.data = data.tobytes()
        return msg

    def get_camera_info(self, intrinsics, frame_id, stamp_msg: TimeMsg):
        msg = CameraInfo()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = frame_id
        msg.width = intrinsics.width
        msg.height = intrinsics.height

        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        msg.k = [fx, 0.0, cx,
                 0.0, fy, cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0,
                 0.0, fy, cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return msg
    

def main(argv=None):
    rclpy.init(args=argv)
    node = D455LocalFix()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.running = False
            node._capture_stop.set()
            if node._capture_thread is not None:
                node._capture_thread.join(timeout=1.0)
            try:
                node.pipeline.stop()
            except Exception:
                node.get_logger().warning("pipeline.stop() during shutdown raised:\n" + traceback.format_exc())
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)