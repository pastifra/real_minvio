from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Int64


def _project_root() -> Path:
    # minvio-hardware/daq_sync_bridge.py -> workspace root is parent of minvio-hardware
    return Path(__file__).resolve().parents[1]


def _default_sessions_root() -> Path:
    return _project_root() / "data" / "daq_sessions"


def _read_meta(meta_path: Path) -> Optional[dict]:
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _find_latest_session(sessions_root: Path) -> Optional[Path]:
    if not sessions_root.exists():
        return None

    # Pick the newest directory that contains daq_meta.json
    candidates = []
    for p in sessions_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "daq_meta.json").exists():
            try:
                candidates.append((p.stat().st_mtime, p))
            except Exception:
                continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


class DaqSyncBridge(Node):
    def __init__(
        self,
        session_dir: Optional[Path],
        sessions_root: Path,
        topic_start: str,
        topic_stop: str,
        poll_s: float,
        publish_s: float,
        exit_after_stop: bool,
    ):
        super().__init__("daq_sync_bridge")

        self._session_dir = session_dir
        self._sessions_root = sessions_root
        self._topic_start = topic_start
        self._topic_stop = topic_stop
        self._poll_s = poll_s
        self._publish_s = publish_s
        self._exit_after_stop = exit_after_stop

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub_start = self.create_publisher(Int64, self._topic_start, qos)
        self._pub_stop = self.create_publisher(Int64, self._topic_stop, qos)

        self._published_start_ns: Optional[int] = None
        self._published_stop_ns: Optional[int] = None

        self._timer = self.create_timer(self._poll_s, self._tick)

    def _publish_for(self, publisher, value: int, seconds: float):
        msg = Int64()
        msg.data = int(value)
        t_end = time.time() + seconds
        while time.time() < t_end and rclpy.ok():
            publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

    def _tick(self):
        # 1) Decide session dir
        if self._session_dir is None:
            latest = _find_latest_session(self._sessions_root)
            if latest is None:
                self.get_logger().info(f"Waiting for DAQ session in {self._sessions_root} ...")
                return
            self._session_dir = latest
            self.get_logger().info(f"Using session: {self._session_dir}")

        meta_path = self._session_dir / "daq_meta.json"
        meta = _read_meta(meta_path)
        if meta is None:
            self.get_logger().info(f"Waiting for readable {meta_path} ...")
            return

        # 2) Publish start marker
        t0 = meta.get("t0_wall_ns")
        if t0 is not None:
            t0 = int(t0)
            if self._published_start_ns != t0:
                self.get_logger().info(f"Publishing start marker {t0} on {self._topic_start}")
                self._publish_for(self._pub_start, t0, self._publish_s)
                self._published_start_ns = t0

        # 3) Publish stop marker once it appears
        t1 = meta.get("t1_wall_ns")
        if t1 is not None:
            t1 = int(t1)
            if self._published_stop_ns != t1:
                self.get_logger().info(f"Publishing stop marker {t1} on {self._topic_stop}")
                self._publish_for(self._pub_stop, t1, min(1.0, self._publish_s))
                self._published_stop_ns = t1

                if self._exit_after_stop:
                    self.get_logger().info("Stop marker published; exiting.")
                    rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish DAQ start/stop wall-clock markers into ROS so rosbag can align with DAQ files."
    )

    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Path to a DAQ session folder (contains daq_meta.json). If omitted, uses newest under --sessions-root.",
    )
    parser.add_argument(
        "--sessions-root",
        type=str,
        default=str(_default_sessions_root()),
        help="Root directory containing DAQ session folders (default: data/daq_sessions)",
    )

    parser.add_argument("--topic-start", type=str, default="/daq_sync/start_ns")
    parser.add_argument("--topic-stop", type=str, default="/daq_sync/stop_ns")

    parser.add_argument("--poll-ms", type=int, default=250, help="How often to poll the meta file")
    parser.add_argument(
        "--publish-seconds",
        type=float,
        default=2.0,
        help="How long to repeatedly publish start marker (helps discovery/rosbag catch it)",
    )
    parser.add_argument(
        "--exit-after-stop",
        action="store_true",
        help="Exit automatically after publishing stop marker",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    session_dir = Path(args.session_dir) if args.session_dir else None
    sessions_root = Path(args.sessions_root)

    rclpy.init(args=None)
    node = DaqSyncBridge(
        session_dir=session_dir,
        sessions_root=sessions_root,
        topic_start=args.topic_start,
        topic_stop=args.topic_stop,
        poll_s=max(0.05, args.poll_ms / 1000.0),
        publish_s=max(0.1, float(args.publish_seconds)),
        exit_after_stop=bool(args.exit_after_stop),
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
