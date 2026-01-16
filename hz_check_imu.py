import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu

class HzCheck(Node):
    def __init__(self, topic: str):
        super().__init__("hz_check_imu")
        self._topic = topic
        self._count = 0
        self._t0 = time.perf_counter()
        self._last = None
        self._min_dt = None
        self._max_dt = None
        self._count_window = 0
        self._t_last_report = time.perf_counter()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2000,
        )

        self.create_subscription(Imu, topic, self._cb, qos)
        self.create_timer(1.0, self._report)

    def _cb(self, msg: Imu):
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            self._min_dt = dt if self._min_dt is None else min(self._min_dt, dt)
            self._max_dt = dt if self._max_dt is None else max(self._max_dt, dt)
        self._last = now
        self._count += 1
        self._count_window += 1

    def _report(self):
        now = time.perf_counter()
        dt_win = max(1e-9, now - self._t_last_report)
        hz_win = self._count_window / dt_win
        dt_total = max(1e-9, now - self._t0)
        hz_total = self._count / dt_total

        print(f"{self._topic}: win={hz_win:.1f} Hz, avg={hz_total:.1f} Hz (samples={self._count})")

        self._count_window = 0
        self._t_last_report = now
def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/d455_camera/accel/sample"
    rclpy.init()
    node = HzCheck(topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()