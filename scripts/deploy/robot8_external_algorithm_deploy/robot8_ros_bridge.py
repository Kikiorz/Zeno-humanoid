#!/usr/bin/env python3
"""Standalone 20 Hz ROS2 bridge for a local external Robot8 algorithm.

This program is intentionally model-agnostic.  It turns ROS observations into
one localhost worker request and wraps the worker's raw 23D action into the
Robot8 24D whole-body command.  It is dry-run by default: no robot command is
published unless ``--publish-commands`` is explicitly supplied.

Run it with ROS2's system Python, not a conda model environment:

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 robot8_ros_bridge.py --log-full-action
"""

from __future__ import annotations

import argparse
import math
import pickle
import signal
import socket
import struct
import time
from typing import Any, Callable, Mapping, Sequence

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Float64MultiArray
except ImportError as exc:
    raise SystemExit(
        "Run this bridge with ROS2 system Python. Example: "
        "source /opt/ros/humble/setup.bash && /usr/bin/python3 robot8_ros_bridge.py"
    ) from exc


ACTION_DIM = 23
COMMAND_DIM = 24
ACTION_FIELDS = (
    "torso_lift",
    "torso_waist",
    "head_pan",
    "head_tilt",
    *(f"left_arm_j{i}" for i in range(7)),
    *(f"right_arm_j{i}" for i in range(7)),
    "left_gripper",
    "right_gripper",
    "base_vx",
    "base_vy",
    "base_rotation",
)
assert len(ACTION_FIELDS) == ACTION_DIM

TORSO_FIELDS = ACTION_FIELDS[0:4]
LEFT_ARM_FIELDS = ACTION_FIELDS[4:11]
RIGHT_ARM_FIELDS = ACTION_FIELDS[11:18]
LEFT_GRIPPER_FIELDS = ACTION_FIELDS[18:19]
RIGHT_GRIPPER_FIELDS = ACTION_FIELDS[19:20]
REQUIRED_JOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torso", TORSO_FIELDS),
    ("left_arm", LEFT_ARM_FIELDS),
    ("right_arm", RIGHT_ARM_FIELDS),
    ("left_gripper", LEFT_GRIPPER_FIELDS),
    ("right_gripper", RIGHT_GRIPPER_FIELDS),
)
JOINT_NAME_ALIASES = {
    "head_pan": ("torso_head_pan",),
    "head_tilt": ("torso_head_tilt",),
    "left_gripper": ("left_arm_gripper",),
    "right_gripper": ("right_arm_gripper",),
}
CAMERA_NAMES = ("head_cam", "left_arm_cam", "right_arm_cam")
MAX_WORKER_RESPONSE_BYTES = 1 << 20


def _set_deadline_timeout(sock: socket.socket, deadline: float) -> None:
    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0.0:
        raise TimeoutError("worker request deadline exceeded")
    sock.settimeout(remaining_s)


def _recv_exact(sock: socket.socket, size: int, *, deadline: float) -> bytes:
    if size < 0:
        raise ValueError("negative socket read size")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        _set_deadline_timeout(sock, deadline)
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("worker socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(sock: socket.socket, message: Mapping[str, Any], *, deadline: float) -> None:
    payload = pickle.dumps(dict(message), protocol=pickle.HIGHEST_PROTOCOL)
    frame = memoryview(struct.pack("!I", len(payload)) + payload)
    while frame:
        _set_deadline_timeout(sock, deadline)
        sent = sock.send(frame)
        if sent <= 0:
            raise ConnectionError("worker socket closed while sending")
        frame = frame[sent:]


def _recv_message(sock: socket.socket, *, deadline: float) -> Mapping[str, Any]:
    size = struct.unpack("!I", _recv_exact(sock, 4, deadline=deadline))[0]
    if not 0 < size <= MAX_WORKER_RESPONSE_BYTES:
        raise ValueError(f"invalid worker response size {size}")
    response = pickle.loads(_recv_exact(sock, size, deadline=deadline))
    if not isinstance(response, Mapping):
        raise ValueError("worker response must be a mapping")
    return response


class WorkerClient:
    """Persistent trusted-local connection to the external algorithm worker."""

    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock: socket.socket | None = None

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _connect(self, deadline: float) -> socket.socket:
        if self.sock is not None:
            return self.sock
        # ``create_connection`` consumes only the remaining cycle budget.
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError("worker request deadline exceeded before connection")
        sock = socket.create_connection((self.host, self.port), timeout=remaining_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _set_deadline_timeout(sock, deadline)
        self.sock = sock
        return sock

    def request_action(self, state: list[float], images: dict[str, bytes]) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        try:
            sock = self._connect(deadline)
            _send_message(sock, {"state": state, "images": images}, deadline=deadline)
            return _recv_message(sock, deadline=deadline)
        except Exception:
            self.close()
            raise


def _extract_joint_positions(msg: JointState, fields: Sequence[str]) -> list[float] | None:
    if not msg.position:
        return None
    if msg.name:
        by_name = {name: float(value) for name, value in zip(msg.name, msg.position, strict=False)}
        values: list[float] = []
        for field in fields:
            names = (field, *JOINT_NAME_ALIASES.get(field, ()))
            value = next((by_name[name] for name in names if name in by_name), None)
            if value is None:
                return None
            values.append(value)
        return values
    if len(msg.position) < len(fields):
        return None
    return [float(value) for value in msg.position[: len(fields)]]


def _odom_velocity(msg: Odometry) -> list[float]:
    twist = msg.twist.twist
    return [float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)]


def _format_action(action: Sequence[float]) -> str:
    return ", ".join(
        f"{name}={float(value):.4f}" for name, value in zip(ACTION_FIELDS, action, strict=True)
    )


class Robot8Bridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("robot8_external_algorithm_bridge")
        self.publish_commands = bool(args.publish_commands)
        self.rate_hz = float(args.rate_hz)
        self.max_obs_age_s = float(args.max_obs_age_s)
        self.log_every_n = int(args.log_every_n)
        self.log_full_action = bool(args.log_full_action)
        self.control_mode = float(args.control_mode)
        self.worker = WorkerClient(args.worker_host, int(args.worker_port), float(args.worker_timeout_s))
        self.image_topics = {
            "head_cam": args.head_cam_topic,
            "left_arm_cam": args.left_arm_cam_topic,
            "right_arm_cam": args.right_arm_cam_topic,
        }
        self.joint_topics = {
            "torso": args.torso_state_topic,
            "left_arm": args.left_arm_state_topic,
            "right_arm": args.right_arm_state_topic,
            "left_gripper": args.left_gripper_state_topic,
            "right_gripper": args.right_gripper_state_topic,
        }
        self.odom_topic = args.odom_topic
        self.image_cache: dict[str, tuple[bytes, float]] = {}
        self.joint_cache: dict[str, tuple[JointState, float]] = {}
        self.odom_cache: tuple[Odometry, float] | None = None
        self.subscriptions_keepalive = []
        self.infer_count = 0
        self.fault_count = 0

        self.publisher = self.create_publisher(Float64MultiArray, args.cmd_topic, 10)
        for name, topic in self.image_topics.items():
            self.subscriptions_keepalive.append(
                self.create_subscription(CompressedImage, topic, self._image_callback(name), 10)
            )
        for name, topic in self.joint_topics.items():
            self.subscriptions_keepalive.append(
                self.create_subscription(JointState, topic, self._joint_callback(name), 10)
            )
        self.subscriptions_keepalive.append(
            self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._timer_callback)
        mode = "PUBLISH" if self.publish_commands else "DRY-RUN (no commands are published)"
        self.get_logger().info(
            f"mode={mode}; rate={self.rate_hz:.1f}Hz; worker={args.worker_host}:{args.worker_port}; "
            f"worker_deadline={args.worker_timeout_s:.3f}s; command_topic={args.cmd_topic}"
        )

    def destroy_node(self) -> bool:
        # Best effort: make shutdown leave an explicit idle command when this
        # bridge was actively publishing.
        if rclpy.ok() and self.publish_commands:
            self._publish_idle()
        self.worker.close()
        return super().destroy_node()

    def _image_callback(self, name: str) -> Callable[[CompressedImage], None]:
        def callback(msg: CompressedImage) -> None:
            self.image_cache[name] = (bytes(msg.data), time.monotonic())

        return callback

    def _joint_callback(self, name: str) -> Callable[[JointState], None]:
        def callback(msg: JointState) -> None:
            self.joint_cache[name] = (msg, time.monotonic())

        return callback

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom_cache = (msg, time.monotonic())

    def _cache_issues(self) -> list[str]:
        now = time.monotonic()
        issues: list[str] = []
        for name, topic in self.image_topics.items():
            cached = self.image_cache.get(name)
            if cached is None:
                issues.append(f"missing image {name} ({topic})")
            elif now - cached[1] > self.max_obs_age_s:
                issues.append(f"stale image {name} age={now - cached[1]:.3f}s")
        for name, topic in self.joint_topics.items():
            cached = self.joint_cache.get(name)
            if cached is None:
                issues.append(f"missing joint state {name} ({topic})")
            elif now - cached[1] > self.max_obs_age_s:
                issues.append(f"stale joint state {name} age={now - cached[1]:.3f}s")
        if self.odom_cache is None:
            issues.append(f"missing odometry ({self.odom_topic})")
        elif now - self.odom_cache[1] > self.max_obs_age_s:
            issues.append(f"stale odometry age={now - self.odom_cache[1]:.3f}s")
        return issues

    def _build_state(self) -> list[float] | None:
        state: list[float] = []
        for name, fields in REQUIRED_JOINTS:
            cached = self.joint_cache.get(name)
            if cached is None:
                return None
            values = _extract_joint_positions(cached[0], fields)
            if values is None:
                self.get_logger().warn(f"required names missing from {name} JointState: {list(fields)}")
                return None
            state.extend(values)
        if self.odom_cache is None:
            return None
        state.extend(_odom_velocity(self.odom_cache[0]))
        if len(state) != ACTION_DIM or not all(math.isfinite(value) for value in state):
            self.get_logger().warn("invalid state: expected 23 finite values")
            return None
        return state

    def _build_images(self) -> dict[str, bytes] | None:
        images: dict[str, bytes] = {}
        for name in CAMERA_NAMES:
            cached = self.image_cache.get(name)
            if cached is None:
                return None
            images[name] = cached[0]
        return images

    def _publish(self, action: Sequence[float], *, control_mode: float) -> None:
        values = [float(value) for value in action]
        if len(values) != ACTION_DIM or not all(math.isfinite(value) for value in values):
            raise ValueError("cannot publish an invalid action")
        if not self.publish_commands:
            return
        message = Float64MultiArray()
        message.data = [float(control_mode), *values]
        if len(message.data) != COMMAND_DIM:
            raise AssertionError("Robot8 command must have 24 values")
        self.publisher.publish(message)

    def _publish_idle(self) -> None:
        if self.publish_commands:
            self._publish([0.0] * ACTION_DIM, control_mode=0.0)

    def _fault(self, reason: str) -> None:
        self.fault_count += 1
        self.worker.close()
        self._publish_idle()
        if (self.fault_count - 1) % self.log_every_n == 0:
            self.get_logger().warn(reason)

    @staticmethod
    def _response_action(response: Mapping[str, Any]) -> list[float] | None:
        if not response.get("ok"):
            return None
        raw_action = response.get("action")
        if not isinstance(raw_action, (list, tuple)):
            return None
        try:
            action = [float(value) for value in raw_action]
        except (TypeError, ValueError):
            return None
        if len(action) != ACTION_DIM or not all(math.isfinite(value) for value in action):
            return None
        return action

    def _timer_callback(self) -> None:
        issues = self._cache_issues()
        if issues:
            self._fault("observation unavailable: " + "; ".join(issues))
            return
        state = self._build_state()
        images = self._build_images()
        if state is None or images is None:
            self._fault("could not build a valid Robot8 observation")
            return
        try:
            response = self.worker.request_action(state, images)
        except Exception as exc:
            self._fault(f"external worker request failed: {exc}")
            return
        action = self._response_action(response)
        if action is None:
            error = response.get("error", "invalid or non-finite action")
            self._fault(f"external worker returned no safe action: {error}")
            return
        # The timer waits synchronously for inference; reject a result if the
        # latest cached robot data became stale during that wait.
        issues = self._cache_issues()
        if issues:
            self._fault("observation became stale during inference: " + "; ".join(issues))
            return

        self._publish(action, control_mode=self.control_mode)
        self.fault_count = 0
        self.infer_count += 1
        if (self.infer_count - 1) % self.log_every_n == 0:
            latency = response.get("latency_s", float("nan"))
            try:
                latency_text = f"{float(latency):.3f}s"
            except (TypeError, ValueError):
                latency_text = "unknown"
            if self.log_full_action:
                self.get_logger().info(
                    f"action[{self.infer_count}] worker_latency={latency_text}: {_format_action(action)}"
                )
            else:
                self.get_logger().info(
                    f"action[{self.infer_count}] worker_latency={latency_text}; "
                    f"base=({action[20]:.3f},{action[21]:.3f},{action[22]:.3f})"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Robot8 20 Hz external-algorithm ROS bridge")
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=8768)
    parser.add_argument(
        "--worker-timeout-s",
        type=float,
        default=0.045,
        help="Hard local-worker deadline; must be <= 0.05 s to preserve the 20 Hz cycle.",
    )
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-obs-age-s", type=float, default=0.5)
    parser.add_argument("--publish-commands", action="store_true", help="Actually publish commands; default is dry-run.")
    parser.add_argument("--control-mode", type=float, default=1.0)
    parser.add_argument("--log-full-action", action="store_true")
    parser.add_argument("--log-every-n", type=int, default=20)
    parser.add_argument("--cmd-topic", default="/zeno/h1/auto/wholebody/cmd")
    parser.add_argument("--head-cam-topic", default="/zeno/h1/sensor/head_cam/image/compressed")
    parser.add_argument("--left-arm-cam-topic", default="/zeno/h1/sensor/left_arm_cam/image/compressed")
    parser.add_argument("--right-arm-cam-topic", default="/zeno/h1/sensor/right_arm_cam/image/compressed")
    parser.add_argument("--odom-topic", default="/zeno/h1/sensor/odom_raw")
    parser.add_argument("--torso-state-topic", default="/zeno/h1/wheelarm/torso/joint_state")
    parser.add_argument("--left-arm-state-topic", default="/zeno/h1/wheelarm/left_arm/joint_state")
    parser.add_argument("--right-arm-state-topic", default="/zeno/h1/wheelarm/right_arm/joint_state")
    parser.add_argument("--left-gripper-state-topic", default="/zeno/h1/left_gripper/joint_state")
    parser.add_argument("--right-gripper-state-topic", default="/zeno/h1/right_gripper/joint_state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_host != "127.0.0.1":
        raise SystemExit("This pickle bridge only supports a trusted local worker at 127.0.0.1")
    if not 1 <= args.worker_port <= 65535:
        raise SystemExit("--worker-port must be in [1, 65535]")
    if not math.isclose(args.rate_hz, 20.0, abs_tol=1e-6):
        raise SystemExit("This Robot8 deployment contract requires --rate-hz 20.0")
    if not (0.0 < args.worker_timeout_s <= 0.05):
        raise SystemExit("--worker-timeout-s must be in (0, 0.05]")
    if not (0.0 < args.max_obs_age_s <= 0.5):
        raise SystemExit("--max-obs-age-s must be in (0, 0.5]")
    if args.log_every_n < 1:
        raise SystemExit("--log-every-n must be positive")
    if not math.isfinite(args.control_mode):
        raise SystemExit("--control-mode must be finite")

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def stop_handler(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_handler)
    node = Robot8Bridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
