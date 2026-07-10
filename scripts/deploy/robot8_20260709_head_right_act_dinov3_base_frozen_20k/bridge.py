#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
import socket
import struct
import time
from typing import Any, Callable, Sequence

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Float64MultiArray
except ImportError as exc:
    raise SystemExit(
        "Run bridge.py with ROS2 system Python, not the conda model Python. "
        "Example: source /opt/ros/humble/setup.bash && /usr/bin/python3 bridge.py"
    ) from exc


ROBOT = "robot8"
ACTION_DIM = 23
COMMAND_DIM = 24

TORSO_FIELDS = ["torso_lift", "torso_waist", "head_pan", "head_tilt"]
LEFT_ARM_FIELDS = [f"left_arm_j{i}" for i in range(7)]
RIGHT_ARM_FIELDS = [f"right_arm_j{i}" for i in range(7)]
LEFT_GRIPPER_FIELDS = ["left_gripper"]
RIGHT_GRIPPER_FIELDS = ["right_gripper"]
BASE_FIELDS = ["base_vx", "base_vy", "base_rotation"]
ACTION_FIELDS = (
    TORSO_FIELDS
    + LEFT_ARM_FIELDS
    + RIGHT_ARM_FIELDS
    + LEFT_GRIPPER_FIELDS
    + RIGHT_GRIPPER_FIELDS
    + BASE_FIELDS
)

IMAGE_TOPIC_PARAMS = {
    "head_cam": "head_cam_topic",
    "right_arm_cam": "right_arm_cam_topic",
}
JOINT_TOPIC_PARAMS = {
    "torso": "torso_state_topic",
    "left_arm": "left_arm_state_topic",
    "right_arm": "right_arm_state_topic",
    "left_gripper": "left_gripper_state_topic",
    "right_gripper": "right_gripper_state_topic",
}
REQUIRED_JOINTS = {
    "torso": TORSO_FIELDS,
    "left_arm": LEFT_ARM_FIELDS,
    "right_arm": RIGHT_ARM_FIELDS,
    "left_gripper": LEFT_GRIPPER_FIELDS,
    "right_gripper": RIGHT_GRIPPER_FIELDS,
}
JOINT_NAME_ALIASES = {
    "head_pan": ["torso_head_pan"],
    "head_tilt": ["torso_head_tilt"],
    "left_gripper": ["left_arm_gripper"],
    "right_gripper": ["right_arm_gripper"],
}


def send_message(sock: socket.socket, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    size = struct.unpack("!I", recv_exact(sock, 4))[0]
    return pickle.loads(recv_exact(sock, size))


def extract_joint_positions(msg: JointState, fields: Sequence[str]) -> list[float] | None:
    if len(msg.position) == 0:
        return None
    if msg.name:
        by_name = {name: float(pos) for name, pos in zip(msg.name, msg.position, strict=False)}
        values = []
        for field in fields:
            names = [field, *JOINT_NAME_ALIASES.get(field, [])]
            value = next((by_name[name] for name in names if name in by_name), None)
            if value is None:
                return None
            values.append(value)
        return values
    if len(msg.position) < len(fields):
        return None
    return [float(value) for value in msg.position[: len(fields)]]


def odom_velocity(msg: Odometry) -> list[float]:
    twist = msg.twist.twist
    return [float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)]


def format_named_values(names: Sequence[str], values: Sequence[float]) -> str:
    return ", ".join(f"{name}={float(value):.4f}" for name, value in zip(names, values, strict=False))


class WorkerClient:
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

    def connect(self) -> socket.socket:
        if self.sock is not None:
            return self.sock
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock
        return sock

    def request_action(self, state: list[float], images: dict[str, bytes]) -> dict:
        try:
            sock = self.connect()
            send_message(sock, {"state": state, "images": images})
            return recv_message(sock)
        except Exception:
            self.close()
            raise


class HeadRightAutoCmdBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(f"{ROBOT}_20260709_head_right_auto_cmd_bridge")
        self.image_names = list(IMAGE_TOPIC_PARAMS)
        self.publish_commands = args.publish_commands
        self.publish_idle_on_stale = args.publish_idle_on_stale
        self.control_mode = args.control_mode
        self.rate_hz = args.rate_hz
        self.max_obs_age_s = args.max_obs_age_s
        self.log_every_n = max(1, args.log_every_n)
        self.log_full_action = args.log_full_action
        self.worker = WorkerClient(args.worker_host, args.worker_port, args.worker_timeout_s)
        self.topics = vars(args)

        self.image_cache: dict[str, tuple[bytes, float]] = {}
        self.joint_cache: dict[str, tuple[JointState, float]] = {}
        self.odom_cache: tuple[Odometry, float] | None = None
        self.subscriptions_keepalive = []
        self.infer_count = 0
        self.stale_count = 0
        self.worker_error_count = 0
        self.publisher = self.create_publisher(Float64MultiArray, args.cmd_topic, 10)
        self.create_topic_subscriptions()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)

        mode = "PUBLISH" if self.publish_commands else "DRY-RUN"
        self.get_logger().info(
            f"mode={mode}; worker={args.worker_host}:{args.worker_port}; "
            f"rate={self.rate_hz:.1f}Hz; cameras=head_cam,right_arm_cam; "
            f"state/action=23D full body including left arm; cmd_topic={args.cmd_topic}"
        )

    def destroy_node(self) -> bool:
        self.worker.close()
        return super().destroy_node()

    def create_topic_subscriptions(self) -> None:
        for name, arg_name in IMAGE_TOPIC_PARAMS.items():
            self.subscriptions_keepalive.append(
                self.create_subscription(CompressedImage, self.topics[arg_name], self.image_callback(name), 10)
            )
        for name, arg_name in JOINT_TOPIC_PARAMS.items():
            self.subscriptions_keepalive.append(
                self.create_subscription(JointState, self.topics[arg_name], self.joint_callback(name), 10)
            )
        self.subscriptions_keepalive.append(
            self.create_subscription(Odometry, self.topics["odom_topic"], self.odom_callback, 10)
        )

    def image_callback(self, name: str) -> Callable[[CompressedImage], None]:
        def callback(msg: CompressedImage) -> None:
            self.image_cache[name] = (bytes(msg.data), time.monotonic())

        return callback

    def joint_callback(self, name: str) -> Callable[[JointState], None]:
        def callback(msg: JointState) -> None:
            self.joint_cache[name] = (msg, time.monotonic())

        return callback

    def odom_callback(self, msg: Odometry) -> None:
        self.odom_cache = (msg, time.monotonic())

    def observation_cache_issues(self) -> list[str]:
        now = time.monotonic()
        issues = []
        for name, arg_name in IMAGE_TOPIC_PARAMS.items():
            cached = self.image_cache.get(name)
            if cached is None:
                issues.append(f"{name} missing topic={self.topics[arg_name]}")
            elif now - cached[1] > self.max_obs_age_s:
                issues.append(f"{name} stale age={now - cached[1]:.3f}s topic={self.topics[arg_name]}")
        for name, arg_name in JOINT_TOPIC_PARAMS.items():
            cached = self.joint_cache.get(name)
            if cached is None:
                issues.append(f"{name} missing topic={self.topics[arg_name]}")
            elif now - cached[1] > self.max_obs_age_s:
                issues.append(f"{name} stale age={now - cached[1]:.3f}s topic={self.topics[arg_name]}")
        if self.odom_cache is None:
            issues.append(f"odom missing topic={self.topics['odom_topic']}")
        elif now - self.odom_cache[1] > self.max_obs_age_s:
            issues.append(f"odom stale age={now - self.odom_cache[1]:.3f}s topic={self.topics['odom_topic']}")
        return issues

    def build_state(self) -> list[float] | None:
        state = []
        for name, fields in REQUIRED_JOINTS.items():
            cached = self.joint_cache.get(name)
            if cached is None:
                return None
            values = extract_joint_positions(cached[0], fields)
            if values is None:
                self.get_logger().warn(f"Missing joint fields for {name}: {fields}")
                return None
            state.extend(values)
        if self.odom_cache is None:
            return None
        state.extend(odom_velocity(self.odom_cache[0]))
        if len(state) != ACTION_DIM or any(not math.isfinite(v) for v in state):
            self.get_logger().warn(f"Invalid state length={len(state)}")
            return None
        return state

    def build_images(self) -> dict[str, bytes] | None:
        images = {}
        for name in self.image_names:
            cached = self.image_cache.get(name)
            if cached is None:
                return None
            images[name] = cached[0]
        return images

    def publish_command(self, action: Sequence[float], control_mode: float | None = None) -> None:
        command = [0.0] * COMMAND_DIM
        command[0] = self.control_mode if control_mode is None else float(control_mode)
        command[1:] = [float(value) for value in action]
        msg = Float64MultiArray()
        msg.data = command
        if self.publish_commands:
            self.publisher.publish(msg)

    def publish_idle(self) -> None:
        if self.publish_commands and self.publish_idle_on_stale:
            self.publish_command([0.0] * ACTION_DIM, control_mode=0.0)

    def timer_callback(self) -> None:
        issues = self.observation_cache_issues()
        if issues:
            self.stale_count += 1
            if self.stale_count % self.log_every_n == 1:
                self.get_logger().warn("Observation missing/stale: " + "; ".join(issues))
            self.publish_idle()
            return
        state = self.build_state()
        images = self.build_images()
        if state is None or images is None:
            self.publish_idle()
            return
        try:
            response = self.worker.request_action(state, images)
        except Exception as exc:
            self.worker_error_count += 1
            if self.worker_error_count % self.log_every_n == 1:
                self.get_logger().warn(f"Worker request failed: {exc}")
            self.publish_idle()
            return
        if not response.get("ok"):
            self.worker_error_count += 1
            if self.worker_error_count % self.log_every_n == 1:
                self.get_logger().warn(f"Worker inference failed: {response.get('error')}")
            self.publish_idle()
            return
        action = response["action"]
        if len(action) != ACTION_DIM or any(not math.isfinite(float(v)) for v in action):
            self.get_logger().warn("Worker returned invalid action")
            self.publish_idle()
            return
        self.stale_count = 0
        self.worker_error_count = 0
        self.publish_command(action)
        self.infer_count += 1
        if self.infer_count % self.log_every_n == 1:
            latency_s = float(response.get("latency_s", 0.0))
            if self.log_full_action:
                self.get_logger().info(
                    f"action[{self.infer_count}] latency={latency_s:.3f}s: "
                    f"{format_named_values(ACTION_FIELDS, action)}"
                )
            else:
                self.get_logger().info(
                    f"action[{self.infer_count}] latency={latency_s:.3f}s: "
                    f"{format_named_values(ACTION_FIELDS[:6], action[:6])}, ..."
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="robot8 2026-07-09 head+right-camera ROS2 bridge for ACT policy"
    )
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=8768)
    parser.add_argument("--worker-timeout-s", type=float, default=5.0)
    parser.add_argument("--publish-commands", action="store_true", help="Publish real robot commands. Default is dry-run.")
    parser.add_argument("--no-publish-idle-on-stale", dest="publish_idle_on_stale", action="store_false", default=True)
    parser.add_argument("--control-mode", type=float, default=1.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-obs-age-s", type=float, default=0.5)
    parser.add_argument("--log-every-n", type=int, default=20)
    parser.add_argument("--log-full-action", action="store_true")
    parser.add_argument("--cmd-topic", default="/zeno/h1/auto/wholebody/cmd")
    parser.add_argument("--head-cam-topic", default="/zeno/h1/sensor/head_cam/image/compressed")
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
    rclpy.init()
    node = HeadRightAutoCmdBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
