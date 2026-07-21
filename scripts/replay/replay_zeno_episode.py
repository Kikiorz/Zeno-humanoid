#!/usr/bin/python3
from __future__ import annotations

import argparse
import bisect
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except ImportError as exc:
    raise SystemExit(
        "Run with ROS2 system Python, for example: "
        "source /opt/ros/humble/setup.bash && /usr/bin/python3 replay_zeno_episode.py"
    ) from exc

try:
    import rosbag2_py
except ImportError as exc:
    raise SystemExit(
        "rosbag2_py is not available. Install/source ROS2 Humble before running this script."
    ) from exc


ACTION_DIM = 23
COMMAND_DIM = 24

TWIST_CMD = "/zeno/h1/twist/cmd"
ACTION_TORSO = "/zeno/h1/wheelarm/torso/joint_cmd"
ACTION_LEFT_ARM = "/zeno/h1/wheelarm/left_arm/joint_cmd"
ACTION_RIGHT_ARM = "/zeno/h1/wheelarm/right_arm/joint_cmd"
ACTION_LEFT_GRIPPER = "/zeno/h1/left_gripper/joint_cmd"
ACTION_RIGHT_GRIPPER = "/zeno/h1/right_gripper/joint_cmd"
DEFAULT_CMD_TOPIC = "/zeno/h1/auto/wholebody/cmd"

TORSO_FIELDS = ["torso_lift", "torso_waist", "head_pan", "head_tilt"]
LEFT_ARM_FIELDS = [f"left_arm_j{i}" for i in range(7)]
RIGHT_ARM_FIELDS = [f"right_arm_j{i}" for i in range(7)]
LEFT_GRIPPER_FIELDS = ["left_gripper"]
RIGHT_GRIPPER_FIELDS = ["right_gripper"]
BASE_FIELDS = ["base_vx", "base_vy", "base_rotation"]
UPPER_BODY_DIM = ACTION_DIM - len(BASE_FIELDS)

# The scoped modes constrain one part of the robot. Full replay preserves every
# recorded action dimension, including both the upper body and base velocity.
MOTION_MODE_FULL = "full"
MOTION_MODE_BASE_FROZEN = "base-frozen"
MOTION_MODE_BASE_ONLY = "base-only"
MOTION_MODE_CHOICES = (MOTION_MODE_FULL, MOTION_MODE_BASE_FROZEN, MOTION_MODE_BASE_ONLY)
DEFAULT_SCOPED_FIRST_FRAME_HOLD_S = 2.0
DEFAULT_FULL_FIRST_FRAME_HOLD_S = 0.0
ACTION_FIELDS = (
    TORSO_FIELDS
    + LEFT_ARM_FIELDS
    + RIGHT_ARM_FIELDS
    + LEFT_GRIPPER_FIELDS
    + RIGHT_GRIPPER_FIELDS
    + BASE_FIELDS
)

JOINT_NAME_ALIASES = {
    "head_pan": ["torso_head_pan"],
    "head_tilt": ["torso_head_tilt"],
    "left_gripper": ["left_arm_gripper"],
    "right_gripper": ["right_arm_gripper"],
}

REQUIRED_TOPICS = {
    ACTION_TORSO: JointState,
    ACTION_LEFT_ARM: JointState,
    ACTION_RIGHT_ARM: JointState,
    ACTION_LEFT_GRIPPER: JointState,
    ACTION_RIGHT_GRIPPER: JointState,
    TWIST_CMD: Twist,
}


@dataclass(frozen=True)
class BagSource:
    uri: str
    storage_id: str
    display_path: Path
    is_mcap: bool


@dataclass
class TimedSeries:
    times: list[int]
    values: list[list[float]]

    def nearest(self, stamp_ns: int) -> list[float]:
        if not self.times:
            raise RuntimeError("cannot sample an empty series")
        idx = bisect.bisect_left(self.times, stamp_ns)
        if idx <= 0:
            return self.values[0]
        if idx >= len(self.times):
            return self.values[-1]
        before = self.times[idx - 1]
        after = self.times[idx]
        return self.values[idx] if abs(after - stamp_ns) < abs(stamp_ns - before) else self.values[idx - 1]


@dataclass
class ReplayTrajectory:
    start_ns: int
    end_ns: int
    fps: float
    actions: list[list[float]]


class CommandPublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("zeno_episode_replay")
        self.publisher = self.create_publisher(Float64MultiArray, topic, 10)

    def publish_command(self, action: Sequence[float], control_mode: float) -> None:
        if len(action) != ACTION_DIM:
            raise ValueError(f"action length {len(action)}, expected {ACTION_DIM}")
        msg = Float64MultiArray()
        msg.data = [float(control_mode), *[float(value) for value in action]]
        if len(msg.data) != COMMAND_DIM:
            raise ValueError(f"command length {len(msg.data)}, expected {COMMAND_DIM}")
        self.publisher.publish(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a Zeno H1 trajectory from ROS2 bag command topics by publishing "
            "Float64MultiArray commands to /zeno/h1/auto/wholebody/cmd."
        )
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="ROS2 bag directory, metadata.yaml path, or .mcap file path.",
    )
    parser.add_argument("--cmd-topic", default=DEFAULT_CMD_TOPIC)
    parser.add_argument(
        "--motion-mode",
        choices=MOTION_MODE_CHOICES,
        default=MOTION_MODE_BASE_FROZEN,
        help=(
            "full: replay the recorded upper body and base without constraints; "
            "base-frozen: replay upper body while commanding zero base velocity; "
            "base-only: hold the first upper-body command and replay only base velocity."
        ),
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--first-frame-hold-s",
        type=float,
        default=None,
        help=(
            "Seconds to hold the first frame's upper-body position before replay "
            "and after a completed replay. Base velocity is zero during these holds. "
            "Defaults to 0 for --motion-mode full and 2 for scoped modes."
        ),
    )
    parser.add_argument("--start-offset-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="Replay speed multiplier. 0.5 is half speed, 2.0 is double speed.",
    )
    parser.add_argument("--control-mode", type=float, default=1.0)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Actually publish commands. Default is dry-run preview only.",
    )
    parser.add_argument(
        "--dry-run-realtime",
        action="store_true",
        help="In dry-run mode, sleep according to fps instead of previewing quickly.",
    )
    parser.add_argument(
        "--idle-on-exit",
        dest="idle_on_exit",
        action="store_true",
        default=True,
        help="Publish control_mode=0 zero command on exit when --publish is used.",
    )
    parser.add_argument("--no-idle-on-exit", dest="idle_on_exit", action="store_false")
    parser.add_argument("--log-every-n", type=int, default=20)
    parser.add_argument(
        "--max-action-abs",
        type=float,
        default=None,
        help="Abort if any action component absolute value exceeds this limit.",
    )
    parser.add_argument(
        "--max-step-delta",
        type=float,
        default=None,
        help="Abort if any component changes by more than this between replay frames.",
    )
    parser.add_argument(
        "--allow-missing-topics",
        action="store_true",
        help="Fill missing command topics with zeros. Use only for diagnostics.",
    )
    return parser.parse_args()


def resolve_bag_source(raw_path: str) -> BagSource:
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"bag path does not exist: {path}")

    if path.is_file() and path.name == "metadata.yaml":
        return BagSource(uri=str(path.parent), storage_id="", display_path=path.parent, is_mcap=False)

    if path.is_file() and path.suffix == ".mcap":
        if (path.parent / "metadata.yaml").is_file():
            return BagSource(uri=str(path.parent), storage_id="", display_path=path, is_mcap=True)
        return BagSource(uri=str(path), storage_id="mcap", display_path=path, is_mcap=True)

    return BagSource(uri=str(path), storage_id="", display_path=path, is_mcap=(path / "metadata.yaml").is_file())


def open_reader(source: BagSource) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=source.uri, storage_id=source.storage_id),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
    except RuntimeError as exc:
        if source.is_mcap or "mcap" in str(exc).lower():
            raise RuntimeError(
                "Could not open MCAP bag. This machine appears to be missing the ROS2 MCAP storage plugin. "
                "Install it with: sudo apt install ros-humble-rosbag2-storage-mcap"
            ) from exc
        raise
    return reader


def set_topic_filter(reader: rosbag2_py.SequentialReader, topics: Sequence[str]) -> None:
    try:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    except Exception:
        # Older rosbag2_py builds may not expose filtering. Reading all messages is still correct.
        return


def extract_named_positions(msg: JointState, expected_names: Sequence[str]) -> list[float]:
    positions = [float(value) for value in msg.position]
    names = [str(name) for name in msg.name]

    if names:
        by_name = {name: float(pos) for name, pos in zip(names, positions)}
        values: list[float] = []
        missing: list[str] = []
        for field in expected_names:
            aliases = [field, *JOINT_NAME_ALIASES.get(field, [])]
            matched = next((name for name in aliases if name in by_name), None)
            if matched is None:
                missing.append(field)
            else:
                values.append(by_name[matched])
        if missing:
            raise ValueError(f"missing joint fields {missing}; message has names={names}")
        return values

    if len(positions) < len(expected_names):
        raise ValueError(f"expected {len(expected_names)} positions, got {len(positions)}")
    return positions[: len(expected_names)]


def extract_twist(msg: Twist) -> list[float]:
    return [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]


def decode_topic_value(topic: str, data: bytes) -> list[float]:
    msg_type = REQUIRED_TOPICS[topic]
    msg = deserialize_message(data, msg_type)
    if topic == ACTION_TORSO:
        return extract_named_positions(msg, TORSO_FIELDS)
    if topic == ACTION_LEFT_ARM:
        return extract_named_positions(msg, LEFT_ARM_FIELDS)
    if topic == ACTION_RIGHT_ARM:
        return extract_named_positions(msg, RIGHT_ARM_FIELDS)
    if topic == ACTION_LEFT_GRIPPER:
        return extract_named_positions(msg, LEFT_GRIPPER_FIELDS)
    if topic == ACTION_RIGHT_GRIPPER:
        return extract_named_positions(msg, RIGHT_GRIPPER_FIELDS)
    if topic == TWIST_CMD:
        return extract_twist(msg)
    raise KeyError(f"unsupported topic: {topic}")


def read_command_series(source: BagSource, allow_missing_topics: bool) -> dict[str, TimedSeries]:
    reader = open_reader(source)
    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    missing = [topic for topic in REQUIRED_TOPICS if topic not in topic_types]
    if missing and not allow_missing_topics:
        raise RuntimeError(f"bag is missing required action topic(s): {missing}")

    present_topics = [topic for topic in REQUIRED_TOPICS if topic in topic_types]
    set_topic_filter(reader, present_topics)

    series = {topic: TimedSeries(times=[], values=[]) for topic in present_topics}
    read_count = 0
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        if topic not in series:
            continue
        value = decode_topic_value(topic, data)
        series[topic].times.append(int(stamp_ns))
        series[topic].values.append(value)
        read_count += 1
        if read_count % 100000 == 0:
            print(f"[read] decoded {read_count} command messages...", flush=True)

    empty = [topic for topic, values in series.items() if not values.times]
    if empty and not allow_missing_topics:
        raise RuntimeError(f"bag has no messages for required action topic(s): {empty}")

    if allow_missing_topics:
        present_series = [values for values in series.values() if values.times]
        if not present_series:
            raise RuntimeError("bag has no usable command messages")
        fill_start = max(values.times[0] for values in present_series)
        fill_end = min(values.times[-1] for values in present_series)
        for topic in REQUIRED_TOPICS:
            if topic in series:
                continue
            series[topic] = TimedSeries(
                times=[fill_start, fill_end],
                values=[zero_value_for_topic(topic), zero_value_for_topic(topic)],
            )

    return series


def zero_value_for_topic(topic: str) -> list[float]:
    if topic == ACTION_TORSO:
        return [0.0] * len(TORSO_FIELDS)
    if topic == ACTION_LEFT_ARM:
        return [0.0] * len(LEFT_ARM_FIELDS)
    if topic == ACTION_RIGHT_ARM:
        return [0.0] * len(RIGHT_ARM_FIELDS)
    if topic == ACTION_LEFT_GRIPPER:
        return [0.0]
    if topic == ACTION_RIGHT_GRIPPER:
        return [0.0]
    if topic == TWIST_CMD:
        return [0.0, 0.0, 0.0]
    raise KeyError(topic)


def finite_action(action: Sequence[float]) -> bool:
    return len(action) == ACTION_DIM and all(math.isfinite(float(value)) for value in action)


def build_action(series: dict[str, TimedSeries], stamp_ns: int) -> list[float]:
    action = (
        series[ACTION_TORSO].nearest(stamp_ns)
        + series[ACTION_LEFT_ARM].nearest(stamp_ns)
        + series[ACTION_RIGHT_ARM].nearest(stamp_ns)
        + series[ACTION_LEFT_GRIPPER].nearest(stamp_ns)
        + series[ACTION_RIGHT_GRIPPER].nearest(stamp_ns)
        + series[TWIST_CMD].nearest(stamp_ns)
    )
    if not finite_action(action):
        raise ValueError(f"invalid action at stamp {stamp_ns}: len={len(action)} action={action}")
    return action


def build_trajectory(
    series: dict[str, TimedSeries],
    fps: float,
    start_offset_s: float,
    duration_s: float | None,
    max_frames: int | None,
) -> ReplayTrajectory:
    topic_starts = [values.times[0] for values in series.values() if values.times]
    topic_ends = [values.times[-1] for values in series.values() if values.times]
    start_ns = max(topic_starts) + int(start_offset_s * 1e9)
    end_ns = min(topic_ends)
    if duration_s is not None:
        end_ns = min(end_ns, start_ns + int(duration_s * 1e9))
    if end_ns <= start_ns:
        raise RuntimeError("no overlapping command time range after applying start/duration")

    step_ns = int(1e9 / fps)
    if step_ns <= 0:
        raise ValueError("--fps is too high")

    n_frames = int((end_ns - start_ns) // step_ns) + 1
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    if n_frames <= 0:
        raise RuntimeError("trajectory has no replay frames")

    actions = [build_action(series, start_ns + idx * step_ns) for idx in range(n_frames)]
    return ReplayTrajectory(start_ns=start_ns, end_ns=start_ns + (n_frames - 1) * step_ns, fps=fps, actions=actions)


def apply_motion_mode(trajectory: ReplayTrajectory, motion_mode: str) -> ReplayTrajectory:
    """Return a trajectory after applying the requested replay-mode constraint."""

    if not trajectory.actions:
        raise ValueError("cannot apply a motion mode to an empty trajectory")

    if motion_mode == MOTION_MODE_FULL:
        # Preserve every recorded dimension: torso, arms, grippers, and base.
        actions = [list(action) for action in trajectory.actions]
    elif motion_mode == MOTION_MODE_BASE_FROZEN:
        # The last three dimensions are vx, vy, and yaw rate. Zeroing them on
        # every frame makes the base remain stationary while the upper body is
        # replayed normally.
        actions = [action[:UPPER_BODY_DIM] + [0.0] * len(BASE_FIELDS) for action in trajectory.actions]
    elif motion_mode == MOTION_MODE_BASE_ONLY:
        # Position commands need a stable target rather than zeros (which could
        # drive the robot to a different pose). Hold the initial recorded
        # upper-body target and only replay the three base velocity components.
        upper_body_hold = trajectory.actions[0][:UPPER_BODY_DIM]
        actions = [upper_body_hold + action[UPPER_BODY_DIM:] for action in trajectory.actions]
    else:
        raise ValueError(f"unsupported motion mode: {motion_mode}")

    return ReplayTrajectory(
        start_ns=trajectory.start_ns,
        end_ns=trajectory.end_ns,
        fps=trajectory.fps,
        actions=actions,
    )


def resolve_first_frame_hold_s(args: argparse.Namespace) -> float:
    """Choose the mode-specific hold default unless the operator set one."""

    if args.first_frame_hold_s is not None:
        return float(args.first_frame_hold_s)
    if args.motion_mode == MOTION_MODE_FULL:
        return DEFAULT_FULL_FIRST_FRAME_HOLD_S
    return DEFAULT_SCOPED_FIRST_FRAME_HOLD_S


def validate_trajectory(
    trajectory: ReplayTrajectory,
    max_action_abs: float | None,
    max_step_delta: float | None,
) -> None:
    previous: list[float] | None = None
    for idx, action in enumerate(trajectory.actions):
        if max_action_abs is not None:
            peak = max(abs(float(value)) for value in action)
            if peak > max_action_abs:
                raise RuntimeError(f"frame {idx} exceeds --max-action-abs: {peak:.6g} > {max_action_abs:.6g}")
        if previous is not None and max_step_delta is not None:
            delta = max(abs(float(value) - float(prev)) for value, prev in zip(action, previous))
            if delta > max_step_delta:
                raise RuntimeError(f"frame {idx} exceeds --max-step-delta: {delta:.6g} > {max_step_delta:.6g}")
        previous = action


def format_action(action: Sequence[float]) -> str:
    upper_body_preview = ", ".join(
        f"{name}={float(value):.4f}" for name, value in zip(ACTION_FIELDS[:8], action[:8])
    )
    base_preview = ", ".join(
        f"{name}={float(value):.4f}"
        for name, value in zip(BASE_FIELDS, action[UPPER_BODY_DIM:])
    )
    return f"{upper_body_preview}, ..., {base_preview}"


def first_frame_position_action(trajectory: ReplayTrajectory) -> list[float]:
    """Return the first upper-body target while keeping the base stationary."""

    if not trajectory.actions:
        raise ValueError("cannot create a first-frame position action for an empty trajectory")
    return trajectory.actions[0][:UPPER_BODY_DIM] + [0.0] * len(BASE_FIELDS)


def hold_action(
    node: CommandPublisher,
    action: Sequence[float],
    control_mode: float,
    duration_s: float,
    period_s: float,
) -> None:
    """Publish an action repeatedly for a bounded duration."""

    deadline = time.perf_counter() + duration_s
    while True:
        start_t = time.perf_counter()
        node.publish_command(action, control_mode)
        rclpy.spin_once(node, timeout_sec=0.0)

        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        elapsed = time.perf_counter() - start_t
        time.sleep(min(max(period_s - elapsed, 0.0), remaining))


def replay(trajectory: ReplayTrajectory, args: argparse.Namespace) -> None:
    period_s = 1.0 / trajectory.fps / args.speed_scale
    realtime = args.publish or args.dry_run_realtime
    first_frame_action = first_frame_position_action(trajectory)

    node: CommandPublisher | None = None
    if args.publish:
        rclpy.init()
        node = CommandPublisher(args.cmd_topic)
        # Let discovery see the publisher before the first command.
        rclpy.spin_once(node, timeout_sec=0.1)

    completed = False
    try:
        if node is not None and args.first_frame_hold_s > 0.0:
            print(
                f"[publish] moving to frame 0 position for {args.first_frame_hold_s:g}s "
                "before replay (base stopped)",
                flush=True,
            )
            hold_action(
                node,
                first_frame_action,
                args.control_mode,
                args.first_frame_hold_s,
                period_s,
            )

        for idx, action in enumerate(trajectory.actions):
            start_t = time.perf_counter()
            if node is not None:
                node.publish_command(action, args.control_mode)
                rclpy.spin_once(node, timeout_sec=0.0)

            if idx % args.log_every_n == 0 or idx == len(trajectory.actions) - 1:
                mode = "publish" if args.publish else "dry-run"
                print(
                    f"[{mode}:{args.motion_mode}] frame={idx}/{len(trajectory.actions) - 1}: "
                    f"{format_action(action)}",
                    flush=True,
                )

            if realtime:
                elapsed = time.perf_counter() - start_t
                time.sleep(max(period_s - elapsed, 0.0))
        completed = True
    finally:
        if node is not None:
            # Do not command a new pose after an interrupted or failed replay.
            if completed and args.first_frame_hold_s > 0.0:
                print(
                    f"[publish] replay complete; moving to frame 0 position for "
                    f"{args.first_frame_hold_s:g}s (base stopped)",
                    flush=True,
                )
                hold_action(
                    node,
                    first_frame_action,
                    args.control_mode,
                    args.first_frame_hold_s,
                    period_s,
                )
            if args.idle_on_exit:
                node.publish_command([0.0] * ACTION_DIM, control_mode=0.0)
                rclpy.spin_once(node, timeout_sec=0.1)
            node.destroy_node()
            rclpy.shutdown()


def main() -> None:
    args = parse_args()
    args.first_frame_hold_s = resolve_first_frame_hold_s(args)
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.first_frame_hold_s < 0:
        raise SystemExit("--first-frame-hold-s must be non-negative")
    if args.speed_scale <= 0:
        raise SystemExit("--speed-scale must be positive")
    if args.start_offset_s < 0:
        raise SystemExit("--start-offset-s must be non-negative")
    if args.duration_s is not None and args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    if args.log_every_n <= 0:
        raise SystemExit("--log-every-n must be positive")

    source = resolve_bag_source(args.bag)
    print(f"[load] bag={source.display_path}", flush=True)
    print(f"[load] uri={source.uri}", flush=True)

    series = read_command_series(source, args.allow_missing_topics)
    for topic in REQUIRED_TOPICS:
        values = series[topic]
        print(f"[load] {topic}: {len(values.times)} messages", flush=True)

    trajectory = build_trajectory(
        series,
        fps=args.fps,
        start_offset_s=args.start_offset_s,
        duration_s=args.duration_s,
        max_frames=args.max_frames,
    )
    trajectory = apply_motion_mode(trajectory, args.motion_mode)
    validate_trajectory(trajectory, args.max_action_abs, args.max_step_delta)

    duration_s = (trajectory.end_ns - trajectory.start_ns) / 1e9
    print(
        f"[ready] frames={len(trajectory.actions)} fps={trajectory.fps:g} "
        f"duration={duration_s:.3f}s motion_mode={args.motion_mode} "
        f"first_frame_hold={args.first_frame_hold_s:g}s "
        f"cmd_topic={args.cmd_topic} publish={args.publish}",
        flush=True,
    )
    if not args.publish:
        print("[ready] dry-run only; pass --publish to send robot commands.", flush=True)

    replay(trajectory, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
