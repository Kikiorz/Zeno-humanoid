#!/usr/bin/env python3
"""Replay a Zeno 23-D ``state`` array through the normal deploy command topic.

This is deliberately a single-file ROS2 tool: it does not load a model, start a
worker, or need images.  It reads only ``timestamp_s`` and ``state`` from an
NPZ and publishes the exact deploy ABI:

    /zeno/h1/auto/wholebody/cmd
    std_msgs/msg/Float64MultiArray
    [control_mode, state[0], ..., state[22]]

The source recording is normally about 30 Hz.  The output is linearly
resampled on a 20 Hz clock by default, matching the deployment bridge.  The
last three state values are replayed verbatim as base vx, vy, and yaw-rate, as
requested; they are recorded odometry velocities, not the original low-level
twist commands.

Safe usage (preview only, no ROS publisher is created):

    python3 replay_zeno_npz_state.py

Actual robot output (after sourcing ROS2):

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 replay_zeno_npz_state.py --publish --unsafe-raw-state-base

Use --npz to select another file.  --publish is intentionally explicit.  On
normal exit or Ctrl-C the script sends the deploy idle command ``[0.0] * 24``.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ACTION_DIM = 23
COMMAND_DIM = 24
UPPER_BODY_DIM = 20
DEFAULT_CMD_TOPIC = "/zeno/h1/auto/wholebody/cmd"
DEFAULT_NPZ = "/home/zeno-rp/2027icra/Data/replay/8.1DEMO-1_full_merged_smoothed.npz"
IDLE_PUBLISH_REPEATS = 3

TORSO_FIELDS = ["torso_lift", "torso_waist", "head_pan", "head_tilt"]
LEFT_ARM_FIELDS = [f"left_arm_j{i}" for i in range(7)]
RIGHT_ARM_FIELDS = [f"right_arm_j{i}" for i in range(7)]
ACTION_FIELDS = (
    TORSO_FIELDS
    + LEFT_ARM_FIELDS
    + RIGHT_ARM_FIELDS
    + ["left_gripper", "right_gripper", "base_vx", "base_vy", "base_rotation"]
)
JOINT_NAME_ALIASES = {
    "head_pan": ["torso_head_pan"],
    "head_tilt": ["torso_head_tilt"],
    "left_gripper": ["left_arm_gripper"],
    "right_gripper": ["right_arm_gripper"],
}
REQUIRED_JOINTS = {
    "torso": TORSO_FIELDS,
    "left_arm": LEFT_ARM_FIELDS,
    "right_arm": RIGHT_ARM_FIELDS,
    "left_gripper": ["left_gripper"],
    "right_gripper": ["right_gripper"],
}
JOINT_TOPIC_PARAMS = {
    "torso": "torso_state_topic",
    "left_arm": "left_arm_state_topic",
    "right_arm": "right_arm_state_topic",
    "left_gripper": "left_gripper_state_topic",
    "right_gripper": "right_gripper_state_topic",
}


@dataclass(frozen=True)
class SourceState:
    timestamps_s: np.ndarray
    states: np.ndarray
    path: Path


@dataclass(frozen=True)
class ReplayTrajectory:
    """Commands on the deployment-rate timeline.

    ``times_s`` remains on the original NPZ clock.  If optional transition
    frames are inserted, neighbouring values may share a timestamp because the
    inserted safety ramp intentionally lengthens wall-clock playback.
    """

    times_s: np.ndarray
    states: np.ndarray
    rate_hz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay NPZ state as 24-D Robot8 deployment commands: "
            "[control_mode, state_23d].  The NPZ action array is never used."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ,
        help="NPZ containing timestamp_s[N] and state[N,23].",
    )
    parser.add_argument("--cmd-topic", default=DEFAULT_CMD_TOPIC)
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=20.0,
        help="Deployment publish frequency; source states are timestamp-resampled to this rate.",
    )
    parser.add_argument(
        "--start-offset-s",
        type=float,
        default=0.0,
        help="Start this many seconds after the first NPZ timestamp.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional duration from --start-offset-s.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--control-mode", type=float, default=1.0)
    parser.add_argument(
        "--base-mode",
        choices=("state", "zero"),
        default="state",
        help=(
            "state: put state[20:23] into the command; zero: keep the base stopped "
            "while replaying state[0:20]."
        ),
    )
    parser.add_argument(
        "--unsafe-raw-state-base",
        action="store_true",
        help=(
            "Required with --publish --base-mode state. Acknowledges that state[20:23] "
            "are measured odometry velocities sent directly without the V3 feedback mapper."
        ),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Actually publish robot commands.  Default is a local dry-run preview only.",
    )
    parser.add_argument(
        "--dry-run-realtime",
        action="store_true",
        help="In dry-run, keep 20 Hz wall-clock timing instead of printing a quick preview.",
    )
    parser.add_argument(
        "--discovery-wait-s",
        type=float,
        default=1.0,
        help="Before the first active command, wait this long for DDS matching and state messages.",
    )
    parser.add_argument(
        "--allow-no-command-subscriber",
        action="store_true",
        help="Permit --publish even if no subscriber matches --cmd-topic after discovery wait.",
    )
    parser.add_argument(
        "--allow-unchecked-start",
        action="store_true",
        help="Skip the default live JointState-vs-first-command safety check.",
    )
    parser.add_argument(
        "--preflight-timeout-s",
        type=float,
        default=3.0,
        help="Maximum time to collect all live JointState inputs for the start-pose check.",
    )
    parser.add_argument(
        "--start-max-position-error",
        type=float,
        default=0.20,
        help=(
            "Fail live preflight when any of the 20 upper-body state values differs from "
            "the first replay command by more than this native-unit tolerance."
        ),
    )
    parser.add_argument("--torso-state-topic", default="/zeno/h1/wheelarm/torso/joint_state")
    parser.add_argument("--left-arm-state-topic", default="/zeno/h1/wheelarm/left_arm/joint_state")
    parser.add_argument("--right-arm-state-topic", default="/zeno/h1/wheelarm/right_arm/joint_state")
    parser.add_argument("--left-gripper-state-topic", default="/zeno/h1/left_gripper/joint_state")
    parser.add_argument("--right-gripper-state-topic", default="/zeno/h1/right_gripper/joint_state")
    parser.add_argument("--log-every-n", type=int, default=20)
    parser.add_argument(
        "--transition-s",
        type=float,
        default=0.0,
        help=(
            "For an upper-body jump larger than --transition-threshold-rad, insert a "
            "linear upper-body transition of this duration.  Zero preserves NPZ state exactly."
        ),
    )
    parser.add_argument(
        "--transition-threshold-rad",
        type=float,
        default=0.30,
        help="Upper-body per-frame jump that triggers optional --transition-s smoothing.",
    )
    return parser.parse_args()


def load_source_state(raw_path: str) -> SourceState:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("timestamp_s", "state") if name not in archive]
        if missing:
            raise ValueError(f"NPZ is missing required array(s): {', '.join(missing)}")
        timestamps_s = np.asarray(archive["timestamp_s"], dtype=np.float64)
        states = np.asarray(archive["state"], dtype=np.float64)

    if timestamps_s.ndim != 1:
        raise ValueError(f"timestamp_s must have shape [N], got {timestamps_s.shape}")
    if states.ndim != 2 or states.shape[1] != ACTION_DIM:
        raise ValueError(f"state must have shape [N,{ACTION_DIM}], got {states.shape}")
    if len(timestamps_s) != len(states) or len(states) < 2:
        raise ValueError(
            "timestamp_s and state must have the same length, with at least two frames: "
            f"timestamps={len(timestamps_s)}, states={len(states)}"
        )
    if not np.isfinite(timestamps_s).all() or not np.isfinite(states).all():
        raise ValueError("timestamp_s and state must contain only finite values")
    if np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError("timestamp_s must be strictly increasing")
    return SourceState(timestamps_s=timestamps_s, states=states, path=path)


def build_trajectory(
    source: SourceState,
    rate_hz: float,
    start_offset_s: float,
    duration_s: float | None,
    max_frames: int | None,
) -> ReplayTrajectory:
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("--rate-hz must be finite and positive")
    if not math.isfinite(start_offset_s) or start_offset_s < 0.0:
        raise ValueError("--start-offset-s must be finite and non-negative")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0.0):
        raise ValueError("--duration-s must be finite and positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    source_start_s = float(source.timestamps_s[0])
    source_end_s = float(source.timestamps_s[-1])
    start_s = source_start_s + start_offset_s
    if start_s > source_end_s:
        raise ValueError("--start-offset-s is after the end of the NPZ")
    end_s = source_end_s if duration_s is None else min(source_end_s, start_s + duration_s)
    if end_s < start_s:
        raise ValueError("no source time remains after applying start/duration")

    # A regular 20 Hz target clock follows the deployment bridge.  Round the
    # end *up* to the next deployment tick so the final source state is not
    # silently omitted when the recording ends between 20 Hz ticks.  np.interp
    # holds that last state for the fractional tail; action is never used.
    frame_count = int(math.ceil((end_s - start_s) * rate_hz - 1e-9)) + 1
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    times_s = start_s + np.arange(frame_count, dtype=np.float64) / rate_hz
    states = np.empty((frame_count, ACTION_DIM), dtype=np.float64)
    for index in range(ACTION_DIM):
        states[:, index] = np.interp(times_s, source.timestamps_s, source.states[:, index])
    return ReplayTrajectory(times_s=times_s, states=states, rate_hz=rate_hz)


def add_large_jump_transitions(
    trajectory: ReplayTrajectory,
    transition_s: float,
    threshold_rad: float,
) -> ReplayTrajectory:
    """Optionally insert ramps at large *upper-body* discontinuities.

    The default transition duration is zero, so normal use sends the resampled
    state faithfully.  When requested, inserted frames lengthen replay by the
    requested duration per discontinuity rather than hiding a jump inside one
    20 Hz command period.
    """

    if not math.isfinite(transition_s) or transition_s < 0.0:
        raise ValueError("--transition-s must be finite and non-negative")
    if not math.isfinite(threshold_rad) or threshold_rad <= 0.0:
        raise ValueError("--transition-threshold-rad must be finite and positive")
    if transition_s == 0.0 or len(trajectory.states) < 2:
        return trajectory

    transition_steps = max(1, int(math.ceil(transition_s * trajectory.rate_hz)))
    output_states: list[np.ndarray] = [trajectory.states[0]]
    output_times: list[float] = [float(trajectory.times_s[0])]
    for index in range(1, len(trajectory.states)):
        previous = output_states[-1]
        target = trajectory.states[index]
        max_upper_body_delta = float(np.max(np.abs(target[:UPPER_BODY_DIM] - previous[:UPPER_BODY_DIM])))
        if max_upper_body_delta > threshold_rad:
            for fraction in np.linspace(1.0 / transition_steps, 1.0, transition_steps):
                # Only positions are intentionally ramped.  The final three
                # dimensions are base velocities and remain the source target
                # for this tick instead of being silently altered by a joint
                # discontinuity workaround.
                ramped = target.copy()
                ramped[:UPPER_BODY_DIM] = (
                    previous[:UPPER_BODY_DIM]
                    + fraction * (target[:UPPER_BODY_DIM] - previous[:UPPER_BODY_DIM])
                )
                output_states.append(ramped)
                output_times.append(float(trajectory.times_s[index]))
        else:
            output_states.append(target)
            output_times.append(float(trajectory.times_s[index]))
    return ReplayTrajectory(
        times_s=np.asarray(output_times, dtype=np.float64),
        states=np.asarray(output_states, dtype=np.float64),
        rate_hz=trajectory.rate_hz,
    )


def zero_arm_plateaus(source: SourceState, minimum_frames: int = 20) -> list[tuple[int, int]]:
    """Find long exact-zero arm/gripper stretches for an explicit operator warning."""

    all_zero = np.all(source.states[:, 4:UPPER_BODY_DIM] == 0.0, axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([False], all_zero, [False])).astype(np.int8)))
    return [
        (int(start), int(end))
        for start, end in edges.reshape(-1, 2)
        if end - start >= minimum_frames
    ]


def max_upper_body_step(trajectory: ReplayTrajectory) -> tuple[int, float]:
    if len(trajectory.states) < 2:
        return 0, 0.0
    deltas = np.max(np.abs(np.diff(trajectory.states[:, :UPPER_BODY_DIM], axis=0)), axis=1)
    index = int(np.argmax(deltas)) + 1
    return index, float(deltas[index - 1])


def format_state(state: Sequence[float]) -> str:
    head = ", ".join(
        f"{name}={float(value):.4f}" for name, value in zip(ACTION_FIELDS[:6], state[:6], strict=True)
    )
    base = ", ".join(
        f"{name}={float(value):.4f}"
        for name, value in zip(ACTION_FIELDS[-3:], state[-3:], strict=True)
    )
    return f"{head}, ..., {base}"


def print_source_summary(
    source: SourceState,
    trajectory: ReplayTrajectory,
    cmd_topic: str,
    base_mode: str,
) -> None:
    native_dt_s = np.diff(source.timestamps_s)
    native_rate_hz = 1.0 / float(np.median(native_dt_s))
    print(
        f"[load] npz={source.path}\n"
        f"[load] state={source.states.shape}, source_duration="
        f"{source.timestamps_s[-1] - source.timestamps_s[0]:.3f}s, native_rate≈{native_rate_hz:.3f}Hz\n"
        f"[ready] output_frames={len(trajectory.states)}, rate={trajectory.rate_hz:g}Hz, "
        f"output_duration={(len(trajectory.states) - 1) / trajectory.rate_hz:.3f}s\n"
        f"[ready] message=[control_mode, state_23d], cmd_topic={cmd_topic}",
        flush=True,
    )
    for start, end in zero_arm_plateaus(source):
        print(
            "[warning] state itself has an all-zero arm/gripper stretch: "
            f"frames {start}:{end - 1}, source_time={source.timestamps_s[start]:.3f}.."
            f"{source.timestamps_s[end - 1]:.3f}s. It will be replayed as supplied.",
            flush=True,
        )
    jump_index, jump_rad = max_upper_body_step(trajectory)
    if jump_rad > 0.30:
        print(
            f"[warning] largest output upper-body step={jump_rad:.4f} rad at "
            f"source_time={trajectory.times_s[jump_index]:.3f}s. For hardware, consider "
            "--transition-s 0.5 (this explicitly changes timing, not the NPZ default).",
            flush=True,
        )
    if base_mode == "state":
        print(
            "[note] state[20:23] are replayed verbatim as base_vx/base_vy/base_rotation. "
            "They are recorded odometry velocities, not the original low-level twist action.",
            flush=True,
        )
    else:
        print("[note] --base-mode zero replaces state[20:23] with zero commands.", flush=True)


def extract_joint_positions(message: Any, fields: Sequence[str]) -> list[float] | None:
    """Match the same JointState name/alias behavior as the deployment bridge."""

    positions = [float(value) for value in message.position]
    if not positions:
        return None
    names = [str(name) for name in message.name]
    if names:
        by_name = {name: value for name, value in zip(names, positions)}
        result: list[float] = []
        for field in fields:
            aliases = [field, *JOINT_NAME_ALIASES.get(field, [])]
            value = next((by_name[name] for name in aliases if name in by_name), None)
            if value is None:
                return None
            result.append(value)
        return result
    if len(positions) < len(fields):
        return None
    return positions[: len(fields)]


def create_ros_publisher(args: argparse.Namespace) -> tuple[Any, Any]:
    """Create ROS objects only for --publish, so dry-run needs no ROS install."""

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python packages are unavailable. Source ROS first, for example: "
            "source /opt/ros/humble/setup.bash && /usr/bin/python3 replay_zeno_npz_state.py --publish"
        ) from exc

    class CommandPublisher(Node):
        def __init__(self) -> None:
            super().__init__("zeno_npz_state_replay")
            # Same QoS shortcut as the deployment bridge: KEEP_LAST depth 10,
            # reliable and volatile in ROS2 Humble's default QoS profile.
            self.publisher = self.create_publisher(Float64MultiArray, args.cmd_topic, 10)
            self.current_joint_positions: dict[str, list[float]] = {}
            self.subscriptions_keepalive = []
            for group, fields in REQUIRED_JOINTS.items():
                topic = getattr(args, JOINT_TOPIC_PARAMS[group])
                self.subscriptions_keepalive.append(
                    self.create_subscription(JointState, topic, self.joint_callback(group, fields), 10)
                )

        def joint_callback(self, group: str, fields: Sequence[str]) -> Any:
            def callback(message: Any) -> None:
                values = extract_joint_positions(message, fields)
                if values is not None and all(math.isfinite(value) for value in values):
                    self.current_joint_positions[group] = values

            return callback

        def current_upper_body(self) -> tuple[list[float] | None, list[str]]:
            values: list[float] = []
            missing: list[str] = []
            for group in REQUIRED_JOINTS:
                current = self.current_joint_positions.get(group)
                if current is None:
                    missing.append(group)
                else:
                    values.extend(current)
            return (values if not missing else None), missing

        def publish_command(self, values: Sequence[float], control_mode: float) -> None:
            if len(values) != ACTION_DIM:
                raise ValueError(f"state command must contain {ACTION_DIM} values, got {len(values)}")
            message = Float64MultiArray()
            message.data = [float(control_mode), *[float(value) for value in values]]
            if len(message.data) != COMMAND_DIM:
                raise RuntimeError(f"command must contain {COMMAND_DIM} values, got {len(message.data)}")
            self.publisher.publish(message)

    # Keep ROS alive for our finally block: the default rclpy signal handler
    # can shut the context down before an idle command can be delivered.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    return rclpy, CommandPublisher()


def spin_for(rclpy_module: Any, node: Any, duration_s: float) -> None:
    deadline_s = time.monotonic() + duration_s
    while True:
        remaining_s = deadline_s - time.monotonic()
        if remaining_s <= 0.0:
            return
        rclpy_module.spin_once(node, timeout_sec=min(0.05, remaining_s))


def preflight_publish(rclpy_module: Any, node: Any, first_state: Sequence[float], args: argparse.Namespace) -> None:
    """Wait for DDS/state input, then reject a mismatched absolute-pose start."""

    spin_for(rclpy_module, node, args.discovery_wait_s)
    subscriber_count = int(node.publisher.get_subscription_count())
    if subscriber_count == 0 and not args.allow_no_command_subscriber:
        raise RuntimeError(
            f"no subscriber matched command topic {args.cmd_topic!r} after "
            f"{args.discovery_wait_s:g}s; refusing to start. Use --allow-no-command-subscriber "
            "only for a non-robot transport test."
        )
    print(f"[preflight] matched command subscribers={subscriber_count}", flush=True)

    if args.allow_unchecked_start:
        print("[preflight] WARNING: skipped live start-pose check by explicit request.", flush=True)
        return

    deadline_s = time.monotonic() + args.preflight_timeout_s
    missing: list[str] = list(REQUIRED_JOINTS)
    while time.monotonic() < deadline_s:
        rclpy_module.spin_once(node, timeout_sec=min(0.05, max(0.0, deadline_s - time.monotonic())))
        current, missing = node.current_upper_body()
        if current is None:
            continue
        errors = np.abs(np.asarray(current, dtype=np.float64) - np.asarray(first_state[:UPPER_BODY_DIM]))
        error_index = int(np.argmax(errors))
        max_error = float(errors[error_index])
        if max_error > args.start_max_position_error:
            raise RuntimeError(
                "start-pose check failed: "
                f"{ACTION_FIELDS[error_index]} current={current[error_index]:.5f}, "
                f"first_replay={float(first_state[error_index]):.5f}, "
                f"abs_error={max_error:.5f} > --start-max-position-error "
                f"{args.start_max_position_error:.5f}. Move the robot to the first pose or "
                "use --allow-unchecked-start only after checking it is safe."
            )
        print(
            f"[preflight] start pose accepted; maximum upper-body error={max_error:.5f} "
            f"at {ACTION_FIELDS[error_index]}",
            flush=True,
        )
        return
    raise RuntimeError(
        "timed out waiting for valid JointState inputs for: " + ", ".join(missing) + ". "
        "Check the five deploy state topics or use --allow-unchecked-start only for a safe test."
    )


def publish_idle(rclpy_module: Any, node: Any) -> None:
    """Repeat the deploy idle command so a short DDS loss does not leave control active."""

    if not rclpy_module.ok():
        print("[publish] ROS context already stopped; could not send deploy idle command.", flush=True)
        return
    for repeat in range(IDLE_PUBLISH_REPEATS):
        node.publish_command([0.0] * ACTION_DIM, control_mode=0.0)
        rclpy_module.spin_once(node, timeout_sec=0.0)
        if repeat + 1 < IDLE_PUBLISH_REPEATS:
            time.sleep(0.05)


def command_from_state(state: np.ndarray, base_mode: str) -> np.ndarray:
    command = np.asarray(state, dtype=np.float64).copy()
    if base_mode == "zero":
        command[-3:] = 0.0
    return command


def replay(trajectory: ReplayTrajectory, args: argparse.Namespace) -> None:
    realtime = args.publish or args.dry_run_realtime
    period_s = 1.0 / trajectory.rate_hz
    rclpy_module: Any | None = None
    node: Any | None = None
    completed = False

    if args.publish:
        rclpy_module, node = create_ros_publisher(args)

    try:
        if node is not None:
            preflight_publish(rclpy_module, node, trajectory.states[0], args)
        start_wall_s = time.perf_counter()
        for index, state in enumerate(trajectory.states):
            command = command_from_state(state, args.base_mode)
            if node is not None:
                node.publish_command(command, args.control_mode)
                rclpy_module.spin_once(node, timeout_sec=0.0)

            if index % args.log_every_n == 0 or index == len(trajectory.states) - 1:
                mode = "publish" if args.publish else "dry-run"
                print(
                    f"[{mode}] frame={index}/{len(trajectory.states) - 1} "
                    f"source_time={trajectory.times_s[index]:.3f}s: {format_state(command)}",
                    flush=True,
                )

            if realtime and index + 1 < len(trajectory.states):
                deadline_s = start_wall_s + (index + 1) * period_s
                time.sleep(max(0.0, deadline_s - time.perf_counter()))
        completed = True
    finally:
        if node is not None:
            status = "after completion" if completed else "after interruption/failure"
            try:
                publish_idle(rclpy_module, node)
                print(f"[publish] sent {IDLE_PUBLISH_REPEATS} deploy idle command(s) {status}.", flush=True)
            finally:
                node.destroy_node()
                if rclpy_module.ok():
                    rclpy_module.shutdown()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.control_mode):
        raise SystemExit("--control-mode must be finite")
    if args.log_every_n <= 0:
        raise SystemExit("--log-every-n must be positive")
    if not math.isfinite(args.discovery_wait_s) or args.discovery_wait_s < 0.0:
        raise SystemExit("--discovery-wait-s must be finite and non-negative")
    if not math.isfinite(args.preflight_timeout_s) or args.preflight_timeout_s <= 0.0:
        raise SystemExit("--preflight-timeout-s must be finite and positive")
    if not math.isfinite(args.start_max_position_error) or args.start_max_position_error <= 0.0:
        raise SystemExit("--start-max-position-error must be finite and positive")
    if args.publish and args.base_mode == "state" and not args.unsafe_raw_state_base:
        raise SystemExit(
            "Refusing raw base replay: state[20:23] are odometry measurements, not low-level "
            "base commands. Use --unsafe-raw-state-base to explicitly acknowledge direct state-base "
            "publishing, or use --base-mode zero."
        )

    source = load_source_state(args.npz)
    trajectory = build_trajectory(
        source,
        rate_hz=args.rate_hz,
        start_offset_s=args.start_offset_s,
        duration_s=args.duration_s,
        max_frames=args.max_frames,
    )
    unsmoothed_frame_count = len(trajectory.states)
    trajectory = add_large_jump_transitions(
        trajectory,
        transition_s=args.transition_s,
        threshold_rad=args.transition_threshold_rad,
    )
    print_source_summary(source, trajectory, args.cmd_topic, args.base_mode)
    inserted_frame_count = len(trajectory.states) - unsmoothed_frame_count
    if inserted_frame_count:
        print(
            f"[ready] --transition-s inserted {inserted_frame_count} linear ramp frame(s); "
            f"wall-clock replay is {inserted_frame_count / trajectory.rate_hz:.3f}s longer.",
            flush=True,
        )
    if not args.publish:
        print("[ready] dry-run only; pass --publish to send robot commands.", flush=True)
    # Match the deployment bridge: turn SIGTERM into a normal interruption so
    # the replay finally block has one chance to publish the idle command.
    def stop_handler(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_handler)
    try:
        replay(trajectory, args)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
