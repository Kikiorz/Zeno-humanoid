#!/usr/bin/env python3
from __future__ import annotations

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
        "Run this bridge with ROS2 system Python, not the conda model Python. "
        "Example: source /opt/ros/humble/setup.bash && /usr/bin/python3 "
        "scripts/deploy/human_new_pick_ros2_auto_cmd_bridge.py"
    ) from exc


ACTION_DIM = 23
COMMAND_DIM = 24

CMD_TOPIC = "/zeno/h1/auto/wholebody/cmd"
CAM_HEAD_TOPIC = "/zeno/h1/sensor/head_cam/image/compressed"
CAM_LEFT_ARM_TOPIC = "/zeno/h1/sensor/left_arm_cam/image/compressed"
CAM_RIGHT_ARM_TOPIC = "/zeno/h1/sensor/right_arm_cam/image/compressed"
ODOM_TOPIC = "/zeno/h1/sensor/odom_raw"
STATE_TORSO_TOPIC = "/zeno/h1/wheelarm/torso/joint_state"
STATE_LEFT_ARM_TOPIC = "/zeno/h1/wheelarm/left_arm/joint_state"
STATE_RIGHT_ARM_TOPIC = "/zeno/h1/wheelarm/right_arm/joint_state"
STATE_LEFT_GRIPPER_TOPIC = "/zeno/h1/left_gripper/joint_state"
STATE_RIGHT_GRIPPER_TOPIC = "/zeno/h1/right_gripper/joint_state"

TORSO_FIELDS = ["torso_lift", "torso_waist", "head_pan", "head_tilt"]
LEFT_ARM_FIELDS = [f"left_arm_j{i}" for i in range(7)]
RIGHT_ARM_FIELDS = [f"right_arm_j{i}" for i in range(7)]
LEFT_GRIPPER_FIELDS = ["left_gripper"]
RIGHT_GRIPPER_FIELDS = ["right_gripper"]
ACTION_FIELDS = (
    TORSO_FIELDS
    + LEFT_ARM_FIELDS
    + RIGHT_ARM_FIELDS
    + LEFT_GRIPPER_FIELDS
    + RIGHT_GRIPPER_FIELDS
    + ["base_vx", "base_vy", "base_rotation"]
)

JOINT_NAME_ALIASES = {
    "head_pan": ["torso_head_pan"],
    "head_tilt": ["torso_head_tilt"],
    "left_gripper": ["left_arm_gripper"],
    "right_gripper": ["right_arm_gripper"],
}

IMAGE_NAMES = ["head_cam", "left_arm_cam", "right_arm_cam"]
JOINT_NAMES = ["torso", "left_arm", "right_arm", "left_gripper", "right_gripper"]
IMAGE_TOPIC_PARAMS = {
    "head_cam": "head_cam_topic",
    "left_arm_cam": "left_arm_cam_topic",
    "right_arm_cam": "right_arm_cam_topic",
}
JOINT_TOPIC_PARAMS = {
    "torso": "torso_state_topic",
    "left_arm": "left_arm_state_topic",
    "right_arm": "right_arm_state_topic",
    "left_gripper": "left_gripper_state_topic",
    "right_gripper": "right_gripper_state_topic",
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
        by_name = {
            name: float(position) for name, position in zip(msg.name, msg.position, strict=False)
        }
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


class HumanNewPickAutoCmdBridge(Node):
    def __init__(self) -> None:
        super().__init__("human_new_pick_auto_cmd_bridge")

        self.declare_parameter("worker_host", "127.0.0.1")
        self.declare_parameter("worker_port", 8765)
        self.declare_parameter("worker_timeout_s", 1.0)
        self.declare_parameter("publish_commands", False)
        self.declare_parameter("publish_idle_on_stale", True)
        self.declare_parameter("control_mode", 1.0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("max_obs_age_s", 0.5)
        self.declare_parameter("log_every_n", 20)
        self.declare_parameter("cmd_topic", CMD_TOPIC)
        self.declare_parameter("head_cam_topic", CAM_HEAD_TOPIC)
        self.declare_parameter("left_arm_cam_topic", CAM_LEFT_ARM_TOPIC)
        self.declare_parameter("right_arm_cam_topic", CAM_RIGHT_ARM_TOPIC)
        self.declare_parameter("odom_topic", ODOM_TOPIC)
        self.declare_parameter("torso_state_topic", STATE_TORSO_TOPIC)
        self.declare_parameter("left_arm_state_topic", STATE_LEFT_ARM_TOPIC)
        self.declare_parameter("right_arm_state_topic", STATE_RIGHT_ARM_TOPIC)
        self.declare_parameter("left_gripper_state_topic", STATE_LEFT_GRIPPER_TOPIC)
        self.declare_parameter("right_gripper_state_topic", STATE_RIGHT_GRIPPER_TOPIC)

        self.publish_commands = bool(self.get_parameter("publish_commands").value)
        self.publish_idle_on_stale = bool(self.get_parameter("publish_idle_on_stale").value)
        self.control_mode = float(self.get_parameter("control_mode").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.max_obs_age_s = float(self.get_parameter("max_obs_age_s").value)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))
        self.worker = WorkerClient(
            host=str(self.get_parameter("worker_host").value),
            port=int(self.get_parameter("worker_port").value),
            timeout_s=float(self.get_parameter("worker_timeout_s").value),
        )

        self.image_cache: dict[str, tuple[bytes, float]] = {}
        self.joint_cache: dict[str, tuple[JointState, float]] = {}
        self.odom_cache: tuple[Odometry, float] | None = None
        self.subscriptions_keepalive = []
        self.infer_count = 0
        self.stale_count = 0
        self.worker_error_count = 0

        self.publisher = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("cmd_topic").value),
            10,
        )
        self.create_topic_subscriptions()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)

        mode = "PUBLISH" if self.publish_commands else "DRY-RUN"
        self.get_logger().info(
            f"bridge mode={mode}; worker={self.worker.host}:{self.worker.port}; "
            f"rate={self.rate_hz:.1f}Hz; cmd_topic={self.get_parameter('cmd_topic').value}"
        )

    def destroy_node(self) -> bool:
        self.worker.close()
        return super().destroy_node()

    def create_topic_subscriptions(self) -> None:
        for name, param_name in IMAGE_TOPIC_PARAMS.items():
            topic = str(self.get_parameter(param_name).value)
            self.subscriptions_keepalive.append(
                self.create_subscription(CompressedImage, topic, self.image_callback(name), 10)
            )

        for name, param_name in JOINT_TOPIC_PARAMS.items():
            topic = str(self.get_parameter(param_name).value)
            self.subscriptions_keepalive.append(
                self.create_subscription(JointState, topic, self.joint_callback(name), 10)
            )

        self.subscriptions_keepalive.append(
            self.create_subscription(
                Odometry,
                str(self.get_parameter("odom_topic").value),
                self.odom_callback,
                10,
            )
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

        for name, param_name in IMAGE_TOPIC_PARAMS.items():
            topic = str(self.get_parameter(param_name).value)
            cached = self.image_cache.get(name)
            if cached is None:
                issues.append(f"{name} missing topic={topic}")
                continue
            age_s = now - cached[1]
            if age_s > self.max_obs_age_s:
                issues.append(f"{name} stale age={age_s:.3f}s topic={topic}")

        for name, param_name in JOINT_TOPIC_PARAMS.items():
            topic = str(self.get_parameter(param_name).value)
            cached = self.joint_cache.get(name)
            if cached is None:
                issues.append(f"{name} missing topic={topic}")
                continue
            age_s = now - cached[1]
            if age_s > self.max_obs_age_s:
                issues.append(f"{name} stale age={age_s:.3f}s topic={topic}")

        odom_topic = str(self.get_parameter("odom_topic").value)
        if self.odom_cache is None:
            issues.append(f"odom missing topic={odom_topic}")
        else:
            age_s = now - self.odom_cache[1]
            if age_s > self.max_obs_age_s:
                issues.append(f"odom stale age={age_s:.3f}s topic={odom_topic}")

        return issues

    def cache_is_fresh(self) -> bool:
        return not self.observation_cache_issues()

    def build_state(self) -> list[float] | None:
        required = {
            "torso": TORSO_FIELDS,
            "left_arm": LEFT_ARM_FIELDS,
            "right_arm": RIGHT_ARM_FIELDS,
            "left_gripper": LEFT_GRIPPER_FIELDS,
            "right_gripper": RIGHT_GRIPPER_FIELDS,
        }
        state = []
        for name, fields in required.items():
            cached = self.joint_cache.get(name)
            if cached is None:
                return None
            values = extract_joint_positions(cached[0], fields)
            if values is None:
                topic = str(self.get_parameter(JOINT_TOPIC_PARAMS[name]).value)
                self.get_logger().warn(f"Missing joint fields for {name} topic={topic}: {fields}")
                return None
            state.extend(values)

        if self.odom_cache is None:
            return None
        state.extend(odom_velocity(self.odom_cache[0]))

        if len(state) != ACTION_DIM:
            self.get_logger().warn(f"Unexpected state length: {len(state)}")
            return None
        if any(not math.isfinite(value) for value in state):
            self.get_logger().warn("State contains NaN or Inf")
            return None
        return state

    def build_images(self) -> dict[str, bytes] | None:
        images = {}
        for name in IMAGE_NAMES:
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
        cache_issues = self.observation_cache_issues()
        if cache_issues:
            self.stale_count += 1
            if self.stale_count % self.log_every_n == 1:
                self.get_logger().warn("Observation cache missing/stale: " + "; ".join(cache_issues))
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
        if len(action) != ACTION_DIM or any(not math.isfinite(float(value)) for value in action):
            self.get_logger().warn("Worker returned invalid action")
            self.publish_idle()
            return

        self.stale_count = 0
        self.worker_error_count = 0
        self.publish_command(action)
        self.infer_count += 1
        if self.infer_count % self.log_every_n == 1:
            latency_s = float(response.get("latency_s", 0.0))
            sample = ", ".join(
                f"{name}={float(value):.4f}"
                for name, value in zip(ACTION_FIELDS[:6], action[:6], strict=False)
            )
            self.get_logger().info(
                f"action[{self.infer_count}] latency={latency_s:.3f}s: {sample}, ..."
            )


def main() -> None:
    rclpy.init()
    node = HumanNewPickAutoCmdBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
