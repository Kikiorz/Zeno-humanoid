#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
import signal
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
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
ACTION_FIELDS = (
    TORSO_FIELDS
    + LEFT_ARM_FIELDS
    + RIGHT_ARM_FIELDS
    + LEFT_GRIPPER_FIELDS
    + RIGHT_GRIPPER_FIELDS
    + ["base_vx", "base_vy", "base_rotation"]
)
ACTION_FIELD_TO_INDEX = {name: index for index, name in enumerate(ACTION_FIELDS)}
BASE_ACTION_INDICES = tuple(range(ACTION_DIM - 3, ACTION_DIM))
JOINT_NAME_ALIASES = {
    "head_pan": ["torso_head_pan"],
    "head_tilt": ["torso_head_tilt"],
    "left_gripper": ["left_arm_gripper"],
    "right_gripper": ["right_arm_gripper"],
}
DEFAULT_IMAGE_NAMES = ["head_cam", "left_arm_cam", "right_arm_cam"]
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
REQUIRED_JOINTS = {
    "torso": TORSO_FIELDS,
    "left_arm": LEFT_ARM_FIELDS,
    "right_arm": RIGHT_ARM_FIELDS,
    "left_gripper": LEFT_GRIPPER_FIELDS,
    "right_gripper": RIGHT_GRIPPER_FIELDS,
}
MAX_WORKER_RESPONSE_BYTES = 1 << 20


def _set_deadline_timeout(sock: socket.socket, deadline: float | None) -> None:
    if deadline is None:
        return
    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0.0:
        raise TimeoutError("worker request deadline exceeded")
    sock.settimeout(remaining_s)


def send_message(sock: socket.socket, message: Any, *, deadline: float | None = None) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    frame = memoryview(struct.pack("!I", len(payload)) + payload)
    while frame:
        _set_deadline_timeout(sock, deadline)
        sent = sock.send(frame)
        if sent <= 0:
            raise ConnectionError("socket closed while sending")
        frame = frame[sent:]


def recv_exact(sock: socket.socket, size: int, *, deadline: float | None = None) -> bytes:
    if size < 0:
        raise ValueError("socket read size must not be negative")
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        _set_deadline_timeout(sock, deadline)
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(
    sock: socket.socket,
    *,
    deadline: float | None = None,
    max_bytes: int = MAX_WORKER_RESPONSE_BYTES,
) -> Any:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    size = struct.unpack("!I", recv_exact(sock, 4, deadline=deadline))[0]
    if size > max_bytes:
        raise ValueError(f"worker response exceeds {max_bytes} byte limit: {size}")
    return pickle.loads(recv_exact(sock, size, deadline=deadline))


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


def parse_frozen_action_values(raw: str) -> dict[str, float]:
    """Parse fixed active-command values as ``field=value`` comma pairs."""
    values: dict[str, float] = {}
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "--frozen-action-values entries must use field=value, "
                f"got {item!r}"
            )
        field, raw_value = (part.strip() for part in item.split("=", maxsplit=1))
        if field not in ACTION_FIELD_TO_INDEX:
            supported = ",".join(ACTION_FIELDS)
            raise argparse.ArgumentTypeError(
                f"unsupported --frozen-action-values field {field!r}; supported: {supported}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid fixed action value for {field!r}: {raw_value!r}"
            ) from exc
        if not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"fixed action value for {field!r} must be finite")
        values[field] = value
    return values


def parse_positive_triplet(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse a positive ``vx,vy,wz`` limit triple for the V2 mapper."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("base limits must be exactly three comma-separated values: vx,vy,wz")
    try:
        parsed = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("base limits must be finite decimal values") from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("base limits must all be finite and positive")
    return parsed


class WorkerClient:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("worker timeout must be finite and positive")
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

    def connect(self, deadline: float) -> socket.socket:
        if self.sock is not None:
            return self.sock
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError("worker request deadline exceeded before connection")
        sock = socket.create_connection((self.host, self.port), timeout=min(self.timeout_s, remaining_s))
        _set_deadline_timeout(sock, deadline)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock
        return sock

    def request_action(self, state: list[float], images: dict[str, bytes]) -> dict:
        deadline = time.monotonic() + self.timeout_s
        try:
            sock = self.connect(deadline)
            send_message(sock, {"state": state, "images": images}, deadline=deadline)
            return recv_message(sock, deadline=deadline)
        except Exception:
            self.close()
            raise


class AutoCmdBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(f"{ROBOT}_auto_cmd_bridge")
        self.image_names = list(args.cameras)
        self.publish_commands = args.publish_commands
        self.publish_idle_on_stale = args.publish_idle_on_stale
        self.control_mode = args.control_mode
        self.rate_hz = args.rate_hz
        self.max_obs_age_s = args.max_obs_age_s
        self.log_every_n = max(1, args.log_every_n)
        self.log_full_action = args.log_full_action
        self.frozen_action_fields = tuple(args.frozen_fields)
        self.frozen_action_indices = tuple(
            ACTION_FIELD_TO_INDEX[field] for field in self.frozen_action_fields
        )
        self.fixed_action_values = {
            ACTION_FIELD_TO_INDEX[field]: value
            for field, value in (args.frozen_action_values or {}).items()
        }
        self.base_mapper: Any | None = None
        self.last_base_mapping: Any | None = None
        self.last_base_active_publish_at: float | None = None
        self.current_timer_started_at: float | None = None
        if args.require_base_command_mapper and not args.base_command_mapper:
            raise ValueError("this deployment requires a non-empty --base-command-mapper path")
        if args.base_command_mapper:
            # V2 labels describe desired *physical* base velocity.  Reject any
            # direct base override: it would silently defeat the feedback
            # mapper and turn the V2 checkpoint back into an unsafe raw-command
            # publisher.
            frozen_base = sorted(set(self.frozen_action_indices).intersection(BASE_ACTION_INDICES))
            fixed_base = sorted(set(self.fixed_action_values).intersection(BASE_ACTION_INDICES))
            if frozen_base or fixed_base:
                raise ValueError(
                    "--base-command-mapper cannot be combined with frozen/fixed base action fields"
                )
            if not math.isclose(self.rate_hz, 20.0, abs_tol=1e-6):
                raise ValueError("--base-command-mapper requires --rate-hz 20.0 (the fitted ARX sample period)")
            if not self.publish_idle_on_stale:
                raise ValueError("--base-command-mapper requires idle publishing on stale/faulted observations")
            if (
                not math.isfinite(args.base_mapper_max_cycle_s)
                or not 0.0 < args.base_mapper_max_cycle_s <= 0.15
            ):
                raise ValueError("--base-mapper-max-cycle-s must be in (0, 0.15] for the 20 Hz V2 mapper")
            if not math.isfinite(args.worker_timeout_s) or args.worker_timeout_s <= 0.0:
                raise ValueError("--base-command-mapper requires a finite positive --worker-timeout-s")
            if args.worker_timeout_s > args.base_mapper_max_cycle_s:
                raise ValueError(
                    "--base-command-mapper requires --worker-timeout-s no greater than "
                    "--base-mapper-max-cycle-s so faults idle quickly"
                )
            if (
                not math.isfinite(args.max_obs_age_s)
                or not 0.0 < args.max_obs_age_s <= args.base_mapper_max_cycle_s
            ):
                raise ValueError(
                    "--base-command-mapper requires --max-obs-age-s in "
                    "(0, --base-mapper-max-cycle-s]"
                )
            if not math.isfinite(args.base_mapper_max_condition) or args.base_mapper_max_condition <= 0.0:
                raise ValueError("--base-mapper-max-condition must be finite and positive")
            deploy_root = Path(__file__).resolve().parents[1]
            if str(deploy_root) not in sys.path:
                sys.path.insert(0, str(deploy_root))
            from robot8_v2_base_feedback import PhysicalDesiredBaseMapper

            self.base_mapper = PhysicalDesiredBaseMapper.from_json(
                args.base_command_mapper,
                feedback_gain=args.base_feedback_gain,
                desired_limits=args.base_desired_limits,
                command_limits=args.base_command_limits,
                command_slew_limits=args.base_command_slew_limits,
                max_condition_number=args.base_mapper_max_condition,
            )
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
            f"rate={self.rate_hz:.1f}Hz; cameras={','.join(self.image_names)}; "
            f"frozen_fields={','.join(self.frozen_action_fields) if self.frozen_action_fields else 'none'}; "
            f"frozen_action_values={args.frozen_action_values or 'none'}; "
            f"base_mapper={self.base_mapper.source if self.base_mapper is not None else 'none'}; "
            f"cmd_topic={args.cmd_topic}"
        )

    def destroy_node(self) -> bool:
        # Best effort: disable whole-body control before tearing down DDS.
        if rclpy.ok() and self.publish_commands:
            self.publish_command([0.0] * ACTION_DIM, control_mode=0.0)
        self.worker.close()
        return super().destroy_node()

    def create_topic_subscriptions(self) -> None:
        for name in self.image_names:
            arg_name = IMAGE_TOPIC_PARAMS[name]
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
        for name in self.image_names:
            arg_name = IMAGE_TOPIC_PARAMS[name]
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
        if len(state) != ACTION_DIM:
            self.get_logger().warn(f"Invalid state length={len(state)}")
            return None
        for index in self.frozen_action_indices:
            state[index] = 0.0
        if any(not math.isfinite(v) for v in state):
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

    def publish_command(self, action: Sequence[float], control_mode: float | None = None) -> bool:
        action_values = [float(value) for value in action]
        if len(action_values) != ACTION_DIM:
            raise ValueError(f"action length must be {ACTION_DIM}, got {len(action_values)}")
        active_command = control_mode is None
        for index in self.frozen_action_indices:
            action_values[index] = self.fixed_action_values.get(index, 0.0) if active_command else 0.0
        if active_command:
            for index, value in self.fixed_action_values.items():
                action_values[index] = value
        self.last_base_mapping = None
        if active_command and self.base_mapper is not None:
            now = time.monotonic()
            timer_age_s = (
                None
                if self.current_timer_started_at is None
                else now - self.current_timer_started_at
            )
            if timer_age_s is not None and timer_age_s > self.topics["base_mapper_max_cycle_s"]:
                self.get_logger().error(
                    "V2 base mapper rejected command: inference/control cycle exceeded "
                    f"{self.topics['base_mapper_max_cycle_s']:.3f}s ({timer_age_s:.3f}s)"
                )
                self.base_mapper.reset()
                self.last_base_active_publish_at = None
                return False
            if (
                self.last_base_active_publish_at is not None
                and now - self.last_base_active_publish_at > self.topics["base_mapper_max_cycle_s"]
            ):
                self.get_logger().error(
                    "V2 base mapper rejected command: active command period exceeded "
                    f"{self.topics['base_mapper_max_cycle_s']:.3f}s"
                )
                self.base_mapper.reset()
                self.last_base_active_publish_at = None
                return False
            if self.odom_cache is None:
                self.get_logger().error("V2 base mapper rejected command: odom is missing")
                return False
            odom_msg, odom_time = self.odom_cache
            odom_age_s = time.monotonic() - odom_time
            if odom_age_s > self.max_obs_age_s:
                self.get_logger().error(f"V2 base mapper rejected command: odom is stale ({odom_age_s:.3f}s)")
                return False
            try:
                mapping = self.base_mapper.map(
                    action_values[-3:], odom_velocity(odom_msg), 1.0 / self.rate_hz
                )
            except Exception as exc:
                self.get_logger().error(f"V2 base mapper rejected command: {exc}")
                return False
            action_values[-3:] = [float(value) for value in mapping.sent_command]
            self.last_base_mapping = mapping
            self.last_base_active_publish_at = now
        elif not active_command and self.base_mapper is not None:
            # A stale observation, worker error, or shutdown always causes
            # the next active V2 command to slew from zero again.
            self.base_mapper.reset()
            self.last_base_active_publish_at = None
        command = [0.0] * COMMAND_DIM
        command[0] = self.control_mode if control_mode is None else float(control_mode)
        command[1:] = action_values
        msg = Float64MultiArray()
        msg.data = command
        if self.publish_commands:
            self.publisher.publish(msg)
        return True

    def publish_idle(self) -> None:
        if self.publish_commands and self.publish_idle_on_stale:
            self.publish_command([0.0] * ACTION_DIM, control_mode=0.0)

    def timer_callback(self) -> None:
        self.current_timer_started_at = time.monotonic()
        issues = self.observation_cache_issues()
        if issues:
            self.stale_count += 1
            if (self.stale_count - 1) % self.log_every_n == 0:
                self.get_logger().warn("Observation missing/stale: " + "; ".join(issues))
            # Reconnecting makes the worker clear any queued ACT actions.
            self.worker.close()
            self.publish_idle()
            return
        state = self.build_state()
        images = self.build_images()
        if state is None or images is None:
            self.worker.close()
            self.publish_idle()
            return
        try:
            response = self.worker.request_action(state, images)
        except Exception as exc:
            self.worker_error_count += 1
            if (self.worker_error_count - 1) % self.log_every_n == 0:
                self.get_logger().warn(f"Worker request failed: {exc}")
            self.publish_idle()
            return
        if not isinstance(response, dict) or not response.get("ok"):
            self.worker_error_count += 1
            if (self.worker_error_count - 1) % self.log_every_n == 0:
                error = (
                    response.get("error")
                    if isinstance(response, dict)
                    else f"invalid response type {type(response)}"
                )
                self.get_logger().warn(f"Worker inference failed: {error}")
            self.worker.close()
            self.publish_idle()
            return
        action = response.get("action")
        try:
            action_values = (
                [float(value) for value in action]
                if isinstance(action, (list, tuple))
                else []
            )
        except (TypeError, ValueError):
            action_values = []
        if len(action_values) != ACTION_DIM:
            self.get_logger().warn("Worker returned invalid action")
            self.worker.close()
            self.publish_idle()
            return
        action = action_values
        for index in self.frozen_action_indices:
            action[index] = self.fixed_action_values.get(index, 0.0)
        for index, value in self.fixed_action_values.items():
            action[index] = value
        if any(not math.isfinite(value) for value in action):
            self.get_logger().warn("Worker returned invalid action")
            self.worker.close()
            self.publish_idle()
            return

        # The timer callback waits synchronously for inference, so cached ROS
        # observations can become stale while the worker is running.  Recheck
        # immediately before publishing and discard the result if that happened.
        issues = self.observation_cache_issues()
        if issues:
            self.stale_count += 1
            if (self.stale_count - 1) % self.log_every_n == 0:
                self.get_logger().warn(
                    "Observation became stale during inference: " + "; ".join(issues)
                )
            self.worker.close()
            self.publish_idle()
            return
        self.stale_count = 0
        self.worker_error_count = 0
        if not self.publish_command(action):
            self.worker.close()
            self.publish_idle()
            return
        self.infer_count += 1
        if (self.infer_count - 1) % self.log_every_n == 0:
            try:
                latency_s = float(response.get("latency_s", 0.0))
            except (TypeError, ValueError):
                latency_s = 0.0
            base_mapping_suffix = ""
            if self.last_base_mapping is not None:
                mapping = self.last_base_mapping
                base_mapping_suffix = (
                    "; base_mapper "
                    f"desired={','.join(f'{value:.4f}' for value in mapping.desired_limited)} "
                    f"measured={','.join(f'{value:.4f}' for value in mapping.measured)} "
                    f"raw_command={','.join(f'{value:.4f}' for value in mapping.raw_command)} "
                    f"command={','.join(f'{value:.4f}' for value in mapping.sent_command)} "
                    f"desired_clip={mapping.desired_saturated} "
                    f"command_clip={mapping.command_saturated} "
                    f"slew_clip={mapping.slew_saturated}"
                )
            if self.log_full_action:
                self.get_logger().info(
                    f"action[{self.infer_count}] latency={latency_s:.3f}s: "
                    f"{format_named_values(ACTION_FIELDS, action)}{base_mapping_suffix}"
                )
            else:
                self.get_logger().info(
                    f"action[{self.infer_count}] latency={latency_s:.3f}s: "
                    f"{format_named_values(ACTION_FIELDS[:6], action[:6])}, ...{base_mapping_suffix}"
                )


def parse_camera_names(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        names = [name.strip() for name in value.split(",") if name.strip()]
    else:
        names = list(value)
    if not names:
        raise argparse.ArgumentTypeError("at least one camera is required")
    unknown = [name for name in names if name not in IMAGE_TOPIC_PARAMS]
    if unknown:
        supported = ",".join(IMAGE_TOPIC_PARAMS)
        raise argparse.ArgumentTypeError(f"unsupported camera(s): {','.join(unknown)}; supported: {supported}")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("duplicate camera names are not allowed")
    return names


def parse_frozen_fields(value: str) -> tuple[str, ...]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in ACTION_FIELD_TO_INDEX]
    if unknown:
        supported = ",".join(ACTION_FIELDS)
        raise argparse.ArgumentTypeError(
            f"unsupported frozen field(s): {','.join(unknown)}; supported: {supported}"
        )
    return tuple(dict.fromkeys(names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{ROBOT} ROS2 bridge for ACT policy")
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=8768)
    parser.add_argument("--worker-timeout-s", type=float, default=1.0)
    parser.add_argument("--publish-commands", action="store_true", help="Publish real robot commands. Default is dry-run.")
    parser.add_argument("--no-publish-idle-on-stale", dest="publish_idle_on_stale", action="store_false", default=True)
    parser.add_argument("--control-mode", type=float, default=1.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-obs-age-s", type=float, default=0.5)
    parser.add_argument("--log-every-n", type=int, default=20)
    parser.add_argument("--log-full-action", action="store_true")
    parser.add_argument(
        "--frozen-fields",
        type=parse_frozen_fields,
        default=(),
        help="Comma-separated 23D state fields forced to zero before model inference; e.g. torso_lift,torso_waist",
    )
    parser.add_argument(
        "--frozen-action-values",
        type=parse_frozen_action_values,
        default=None,
        help=(
            "Fixed active command values for frozen/unmodeled fields as field=value pairs; "
            "e.g. torso_lift=-0.001,torso_waist=-0.065"
        ),
    )
    parser.add_argument(
        "--cameras",
        type=parse_camera_names,
        default=list(DEFAULT_IMAGE_NAMES),
        help="Comma-separated cameras to send to the worker: head_cam,left_arm_cam,right_arm_cam",
    )
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
    parser.add_argument(
        "--base-command-mapper",
        default=None,
        help=(
            "Path to V2 dynamics_feedback_mapper.json. When set, action[20:23] is interpreted as "
            "desired physical odom velocity and is feedback-mapped before publish."
        ),
    )
    parser.add_argument(
        "--require-base-command-mapper",
        action="store_true",
        help="Fail at startup unless --base-command-mapper resolves to a non-empty path.",
    )
    parser.add_argument(
        "--base-feedback-gain",
        type=float,
        default=None,
        help="Optional V2 physical-velocity feedback gain in (0,1]; default reads the mapper recommendation.",
    )
    parser.add_argument(
        "--base-desired-limits",
        type=parse_positive_triplet,
        default=(0.16, 0.16, 0.35),
        metavar="VX,VY,WZ",
        help="Absolute physical desired-velocity limits for a V2 base mapper (m/s,m/s,rad/s).",
    )
    parser.add_argument(
        "--base-command-limits",
        type=parse_positive_triplet,
        default=(0.15, 0.15, 0.30),
        metavar="VX,VY,WZ",
        help="Absolute low-level base command limits for a V2 base mapper.",
    )
    parser.add_argument(
        "--base-command-slew-limits",
        type=parse_positive_triplet,
        default=(0.50, 0.50, 1.00),
        metavar="VX,VY,WZ",
        help="Maximum low-level base command rate per second for a V2 base mapper.",
    )
    parser.add_argument(
        "--base-mapper-max-condition",
        type=float,
        default=50.0,
        help="Fail closed if the V2 command matrix condition number exceeds this positive limit.",
    )
    parser.add_argument(
        "--base-mapper-max-cycle-s",
        type=float,
        default=0.10,
        help=(
            "V2 only: maximum inference/control cycle and active-command period in seconds; "
            "must be <= 0.15 at the fitted 20 Hz sample period."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Keep the ROS context alive until our finally block so Ctrl-C/SIGTERM can
    # publish a best-effort idle command before DDS is torn down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def stop_handler(_signum, _frame) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_handler)
    node = AutoCmdBridge(args)
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
