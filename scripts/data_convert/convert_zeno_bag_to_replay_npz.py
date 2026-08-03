#!/usr/bin/env python3
"""Convert one Zeno ROS 2 MCAP bag into the native-rate replay NPZ v2 schema.

The normal training converter intentionally synchronizes everything to a
single camera-rate clock.  That is the right representation for policy
training, but it discards the native control timing needed by replay.  This
standalone tool instead writes two independently timed, causally aligned
streams:

* ``state_timestamp_s[Ns]`` + ``state[Ns, 23]`` at the torso feedback rate
  (normally about 100 Hz).  ``state[:, 20:23]`` is measured odometry velocity.
* ``state_base_twist[Ns, 3]`` is the recorded ``/zeno/h1/twist/cmd`` sampled
  at the same state instants with causal zero-order hold (ZOH).
* ``action_timestamp_s[Na]`` + ``action[Na, 23]`` at the torso command rate
  (normally about 300 Hz).  Its final three entries are the recorded Twist
  command, also causally ZOH-aligned.

The two timestamp arrays share one bag-time origin.  Their first values are
not forced to zero: a primary torso sample is retained only once every field
needed for *that* vector has an earlier-or-equal source sample.  This avoids
inventing a value from the future at the beginning of the bag.

Run with ROS 2's system Python so ``rosbag2_py`` and ROS message types are
available:

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 scripts/data_convert/convert_zeno_bag_to_replay_npz.py \\
      --bag /path/to/rosbag2_dir \\
      --output /path/to/trajectory_replay_v2.npz

No images are decoded or written.  The source bag is never modified.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    import rosbag2_py
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
except ImportError as exc:
    raise SystemExit(
        "ROS 2 Python packages are unavailable. Run, for example:\n"
        "  source /opt/ros/humble/setup.bash && /usr/bin/python3 "
        "scripts/data_convert/convert_zeno_bag_to_replay_npz.py ..."
    ) from exc


SCHEMA_NAME = "zeno.replay.v2"
VECTOR_DIM = 23
UPPER_BODY_DIM = 20

STATE_TORSO = "/zeno/h1/wheelarm/torso/joint_state"
STATE_LEFT_ARM = "/zeno/h1/wheelarm/left_arm/joint_state"
STATE_RIGHT_ARM = "/zeno/h1/wheelarm/right_arm/joint_state"
STATE_LEFT_GRIPPER = "/zeno/h1/left_gripper/joint_state"
STATE_RIGHT_GRIPPER = "/zeno/h1/right_gripper/joint_state"
ODOM = "/zeno/h1/sensor/odom_raw"

ACTION_TORSO = "/zeno/h1/wheelarm/torso/joint_cmd"
ACTION_LEFT_ARM = "/zeno/h1/wheelarm/left_arm/joint_cmd"
ACTION_RIGHT_ARM = "/zeno/h1/wheelarm/right_arm/joint_cmd"
ACTION_LEFT_GRIPPER = "/zeno/h1/left_gripper/joint_cmd"
ACTION_RIGHT_GRIPPER = "/zeno/h1/right_gripper/joint_cmd"
TWIST_CMD = "/zeno/h1/twist/cmd"

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

# These aliases match the deployment bridge and existing training converter.
JOINT_NAME_ALIASES: dict[str, list[str]] = {
    "head_pan": ["torso_head_pan"],
    "head_tilt": ["torso_head_tilt"],
    "left_gripper": ["left_arm_gripper"],
    "right_gripper": ["right_arm_gripper"],
}

EXPECTED_TOPIC_TYPES = {
    STATE_TORSO: "sensor_msgs/msg/JointState",
    STATE_LEFT_ARM: "sensor_msgs/msg/JointState",
    STATE_RIGHT_ARM: "sensor_msgs/msg/JointState",
    STATE_LEFT_GRIPPER: "sensor_msgs/msg/JointState",
    STATE_RIGHT_GRIPPER: "sensor_msgs/msg/JointState",
    ODOM: "nav_msgs/msg/Odometry",
    ACTION_TORSO: "sensor_msgs/msg/JointState",
    ACTION_LEFT_ARM: "sensor_msgs/msg/JointState",
    ACTION_RIGHT_ARM: "sensor_msgs/msg/JointState",
    ACTION_LEFT_GRIPPER: "sensor_msgs/msg/JointState",
    ACTION_RIGHT_GRIPPER: "sensor_msgs/msg/JointState",
    TWIST_CMD: "geometry_msgs/msg/Twist",
}

STATE_VECTOR_TOPICS = (
    STATE_TORSO,
    STATE_LEFT_ARM,
    STATE_RIGHT_ARM,
    STATE_LEFT_GRIPPER,
    STATE_RIGHT_GRIPPER,
    ODOM,
)
ACTION_VECTOR_TOPICS = (
    ACTION_TORSO,
    ACTION_LEFT_ARM,
    ACTION_RIGHT_ARM,
    ACTION_LEFT_GRIPPER,
    ACTION_RIGHT_GRIPPER,
    TWIST_CMD,
)
REQUIRED_TOPICS = tuple(dict.fromkeys((*STATE_VECTOR_TOPICS, *ACTION_VECTOR_TOPICS)))


@dataclass(frozen=True)
class BagSource:
    """Resolved ROS bag source for ``rosbag2_py``."""

    uri: str
    storage_id: str
    display_path: Path


@dataclass(frozen=True)
class TimedSeries:
    """A sorted, duplicate-free source stream used for causal ZOH lookup."""

    topic: str
    times_ns: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.times_ns.ndim != 1 or self.values.ndim != 2:
            raise ValueError(f"{self.topic}: invalid series shapes {self.times_ns.shape}/{self.values.shape}")
        if len(self.times_ns) != len(self.values) or len(self.times_ns) < 2:
            raise ValueError(f"{self.topic}: requires at least two aligned samples")
        if np.any(np.diff(self.times_ns) <= 0):
            raise ValueError(f"{self.topic}: timestamps must be strictly increasing after normalization")
        if not np.isfinite(self.values).all():
            raise ValueError(f"{self.topic}: source values contain NaN/Inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one ROS 2 MCAP bag to native-rate Zeno replay NPZ v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bag",
        type=Path,
        required=True,
        help="ROS bag directory, metadata.yaml file, or standalone .mcap file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz path. It is atomically exposed only after validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output NPZ after successful conversion.",
    )
    return parser.parse_args()


def resolve_bag_source(raw_path: Path) -> BagSource:
    """Accept the common ROS bag directory, metadata file, or MCAP-file forms."""

    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"bag path does not exist: {path}")
    if path.is_file() and path.name == "metadata.yaml":
        return BagSource(uri=str(path.parent), storage_id="mcap", display_path=path.parent)
    if path.is_file() and path.suffix.lower() == ".mcap":
        return BagSource(uri=str(path), storage_id="mcap", display_path=path)
    if path.is_dir():
        return BagSource(uri=str(path), storage_id="mcap", display_path=path)
    raise ValueError(f"--bag must be a bag directory, metadata.yaml, or .mcap file: {path}")


def open_reader(source: BagSource) -> Any:
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=source.uri, storage_id=source.storage_id),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Unable to open {source.display_path} as an MCAP ROS 2 bag. "
            "Confirm ros-humble-rosbag2-storage-mcap is installed and ROS 2 is sourced."
        ) from exc
    return reader


def extract_named_positions(message: JointState, expected_fields: Sequence[str]) -> np.ndarray:
    """Extract a stable field order, accepting the recorded bridge aliases."""

    positions = [float(value) for value in message.position]
    names = [str(name) for name in message.name]
    if names:
        by_name = {name: value for name, value in zip(names, positions, strict=True)}
        values: list[float] = []
        missing: list[str] = []
        for field in expected_fields:
            candidates = (field, *JOINT_NAME_ALIASES.get(field, ()))
            matching_name = next((name for name in candidates if name in by_name), None)
            if matching_name is None:
                missing.append(field)
            else:
                values.append(by_name[matching_name])
        if missing:
            raise ValueError(
                f"JointState is missing fields {missing}; available={names}; "
                f"aliases={JOINT_NAME_ALIASES}"
            )
    else:
        if len(positions) < len(expected_fields):
            raise ValueError(f"JointState has {len(positions)} positions; expected {len(expected_fields)}")
        values = positions[: len(expected_fields)]
    return np.asarray(values, dtype=np.float32)


def decode_odom_velocity(message: Odometry) -> np.ndarray:
    twist = message.twist.twist
    return np.asarray([twist.linear.x, twist.linear.y, twist.angular.z], dtype=np.float32)


def decode_twist_command(message: Twist) -> np.ndarray:
    return np.asarray([message.linear.x, message.linear.y, message.angular.z], dtype=np.float32)


Decoder = Callable[[Any], np.ndarray]


def decoder_for_topic(topic: str) -> tuple[type[Any], Decoder]:
    if topic in {STATE_TORSO, ACTION_TORSO}:
        return JointState, lambda message: extract_named_positions(message, TORSO_FIELDS)
    if topic in {STATE_LEFT_ARM, ACTION_LEFT_ARM}:
        return JointState, lambda message: extract_named_positions(message, LEFT_ARM_FIELDS)
    if topic in {STATE_RIGHT_ARM, ACTION_RIGHT_ARM}:
        return JointState, lambda message: extract_named_positions(message, RIGHT_ARM_FIELDS)
    if topic in {STATE_LEFT_GRIPPER, ACTION_LEFT_GRIPPER}:
        return JointState, lambda message: extract_named_positions(message, LEFT_GRIPPER_FIELDS)
    if topic in {STATE_RIGHT_GRIPPER, ACTION_RIGHT_GRIPPER}:
        return JointState, lambda message: extract_named_positions(message, RIGHT_GRIPPER_FIELDS)
    if topic == ODOM:
        return Odometry, decode_odom_velocity
    if topic == TWIST_CMD:
        return Twist, decode_twist_command
    raise KeyError(f"unsupported topic: {topic}")


def normalize_series(topic: str, timestamps: list[int], values: list[np.ndarray]) -> TimedSeries:
    """Sort a stream and retain the last sample for duplicate bag timestamps."""

    if not timestamps:
        raise RuntimeError(f"required topic has no messages: {topic}")
    times = np.asarray(timestamps, dtype=np.int64)
    matrix = np.stack(values).astype(np.float32, copy=False)
    if len(times) != len(matrix):
        raise AssertionError(f"{topic}: timestamp/value collection length mismatch")

    # SequentialReader is normally time-ordered, but normalizing protects the
    # causal lookup contract when a bag has out-of-order chunks or equal stamps.
    order = np.argsort(times, kind="stable")
    if not np.array_equal(order, np.arange(len(times))):
        times = times[order]
        matrix = matrix[order]
    if len(times) > 1 and np.any(np.diff(times) == 0):
        # For a right-continuous ZOH, the final record at a repeated stamp is
        # the effective command/measurement. Keep that one deterministically.
        keep = np.concatenate((np.diff(times) != 0, np.asarray([True])))
        removed = int(len(times) - int(np.count_nonzero(keep)))
        print(f"[read] {topic}: collapsed {removed} duplicate timestamp sample(s)", flush=True)
        times = times[keep]
        matrix = matrix[keep]
    return TimedSeries(topic=topic, times_ns=times, values=matrix)


def read_required_series(source: BagSource) -> dict[str, TimedSeries]:
    """Deserialize only the 12 state/action topics required by replay."""

    reader = open_reader(source)
    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    missing = [topic for topic in REQUIRED_TOPICS if topic not in topic_types]
    if missing:
        raise RuntimeError("bag is missing required replay topic(s): " + ", ".join(missing))
    wrong_types = [
        f"{topic} ({topic_types[topic]!r}, expected {EXPECTED_TOPIC_TYPES[topic]!r})"
        for topic in REQUIRED_TOPICS
        if topic_types[topic] != EXPECTED_TOPIC_TYPES[topic]
    ]
    if wrong_types:
        raise RuntimeError("unexpected required topic type(s): " + "; ".join(wrong_types))

    try:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(REQUIRED_TOPICS)))
    except Exception:
        # Some older rosbag2_py builds lack filtering. The topic check below
        # still makes the conversion correct, just less I/O efficient.
        pass

    timestamps: dict[str, list[int]] = defaultdict(list)
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    decoded = 0
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic not in EXPECTED_TOPIC_TYPES:
            continue
        message_type, decoder = decoder_for_topic(topic)
        message = deserialize_message(serialized, message_type)
        value = decoder(message)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f"{topic} at {timestamp_ns}: decoded invalid value {value}")
        timestamps[topic].append(int(timestamp_ns))
        values[topic].append(value)
        decoded += 1
        if decoded % 100_000 == 0:
            print(f"[read] decoded {decoded} required messages", flush=True)

    series = {
        topic: normalize_series(topic, timestamps[topic], values[topic])
        for topic in REQUIRED_TOPICS
    }
    print(f"[read] decoded {decoded} required messages total", flush=True)
    return series


def native_reference_times(
    primary: TimedSeries,
    component_topics: Sequence[str],
    series: dict[str, TimedSeries],
) -> np.ndarray:
    """Keep primary samples only over the causal coverage intersection."""

    start_ns = max(int(series[topic].times_ns[0]) for topic in component_topics)
    end_ns = min(int(series[topic].times_ns[-1]) for topic in component_topics)
    reference = primary.times_ns[
        (primary.times_ns >= start_ns) & (primary.times_ns <= end_ns)
    ]
    if len(reference) < 2:
        raise RuntimeError(
            f"{primary.topic}: fewer than two primary samples overlap all required component streams "
            f"({start_ns}..{end_ns})"
        )
    return reference


def zoh_at(series: TimedSeries, reference_times_ns: np.ndarray) -> np.ndarray:
    """Causally sample the last source sample at or before each reference time."""

    indices = np.searchsorted(series.times_ns, reference_times_ns, side="right") - 1
    if np.any(indices < 0):
        first_bad = int(np.flatnonzero(indices < 0)[0])
        raise RuntimeError(
            f"{series.topic}: no historical value for reference time "
            f"{int(reference_times_ns[first_bad])}"
        )
    return series.values[indices]


def concatenate_zoh(
    topics: Sequence[str], reference_times_ns: np.ndarray, series: dict[str, TimedSeries]
) -> np.ndarray:
    parts = [zoh_at(series[topic], reference_times_ns) for topic in topics]
    vector = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if vector.shape != (len(reference_times_ns), VECTOR_DIM):
        raise AssertionError(f"expected [N,{VECTOR_DIM}] vector, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise RuntimeError("constructed replay vector contains NaN/Inf")
    return vector


def relative_seconds(times_ns: np.ndarray, origin_ns: int) -> np.ndarray:
    values = (times_ns.astype(np.int64, copy=False) - np.int64(origin_ns)).astype(np.float64) / 1e9
    if len(values) < 2 or np.any(np.diff(values) <= 0.0):
        raise RuntimeError("output timestamps are not strictly increasing")
    return values


def nominal_rate_hz(timestamps_s: np.ndarray) -> float:
    return 1.0 / float(np.median(np.diff(timestamps_s)))


def validate_payload(
    state_times_s: np.ndarray,
    state: np.ndarray,
    state_base_twist: np.ndarray,
    action_times_s: np.ndarray,
    action: np.ndarray,
) -> None:
    if state.shape != (len(state_times_s), VECTOR_DIM):
        raise AssertionError(f"state shape mismatch: {state.shape}")
    if state_base_twist.shape != (len(state_times_s), len(BASE_FIELDS)):
        raise AssertionError(f"state_base_twist shape mismatch: {state_base_twist.shape}")
    if action.shape != (len(action_times_s), VECTOR_DIM):
        raise AssertionError(f"action shape mismatch: {action.shape}")
    arrays = (state_times_s, state, state_base_twist, action_times_s, action)
    if not all(np.isfinite(array).all() for array in arrays):
        raise AssertionError("payload contains NaN/Inf")
    if np.any(np.diff(state_times_s) <= 0.0) or np.any(np.diff(action_times_s) <= 0.0):
        raise AssertionError("payload timestamps must be strictly increasing")


def write_atomically(
    output_path: Path,
    *,
    source: BagSource,
    origin_ns: int,
    state_times_s: np.ndarray,
    state: np.ndarray,
    state_base_twist: np.ndarray,
    action_times_s: np.ndarray,
    action: np.ndarray,
) -> None:
    """Save, reopen-check, then atomically replace the requested output path."""

    temporary_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.npz")
    if temporary_path.exists():
        raise FileExistsError(f"temporary output path already exists: {temporary_path}")
    try:
        np.savez_compressed(
            temporary_path,
            schema=np.asarray(SCHEMA_NAME),
            schema_version=np.asarray(2, dtype=np.int64),
            source_bag=np.asarray(str(source.display_path)),
            source_storage_id=np.asarray(source.storage_id),
            shared_time_origin_bag_ns=np.asarray(origin_ns, dtype=np.int64),
            shared_time_origin_contract=np.asarray(
                "All timestamp arrays are seconds relative to the earliest required replay-topic "
                "bag timestamp. Each stream begins only after its own causal component coverage starts."
            ),
            state_timestamp_s=state_times_s.astype(np.float64, copy=False),
            state=state.astype(np.float32, copy=False),
            state_base_twist=state_base_twist.astype(np.float32, copy=False),
            action_timestamp_s=action_times_s.astype(np.float64, copy=False),
            action=action.astype(np.float32, copy=False),
            state_fields=np.asarray(ACTION_FIELDS),
            action_fields=np.asarray(ACTION_FIELDS),
            state_primary_topic=np.asarray(STATE_TORSO),
            action_primary_topic=np.asarray(ACTION_TORSO),
            state_component_topics=np.asarray(STATE_VECTOR_TOPICS),
            action_component_topics=np.asarray(ACTION_VECTOR_TOPICS),
            state_base_twist_topic=np.asarray(TWIST_CMD),
            zoh_contract=np.asarray(
                "Every non-primary vector component is the last recorded source sample at or before "
                "that vector timestamp (causal zero-order hold; no future sample is used)."
            ),
        )
        with np.load(temporary_path, allow_pickle=False) as archive:
            required = {
                "schema",
                "schema_version",
                "state_timestamp_s",
                "state",
                "state_base_twist",
                "action_timestamp_s",
                "action",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise RuntimeError(f"atomic NPZ validation missing key(s): {missing}")
            validate_payload(
                np.asarray(archive["state_timestamp_s"], dtype=np.float64),
                np.asarray(archive["state"], dtype=np.float32),
                np.asarray(archive["state_base_twist"], dtype=np.float32),
                np.asarray(archive["action_timestamp_s"], dtype=np.float64),
                np.asarray(archive["action"], dtype=np.float32),
            )
            if str(archive["schema"].item()) != SCHEMA_NAME:
                raise RuntimeError("atomic NPZ validation found an unexpected schema name")
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def main() -> int:
    args = parse_args()
    source = resolve_bag_source(args.bag)
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError(f"--output must end in .npz: {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[input] bag={source.display_path}", flush=True)
    series = read_required_series(source)

    # The origin covers both streams and is deliberately one raw bag-time
    # origin, rather than independently zeroing state/action time arrays.
    origin_ns = min(int(item.times_ns[0]) for item in series.values())
    state_reference_ns = native_reference_times(series[STATE_TORSO], STATE_VECTOR_TOPICS, series)
    action_reference_ns = native_reference_times(series[ACTION_TORSO], ACTION_VECTOR_TOPICS, series)

    state = concatenate_zoh(STATE_VECTOR_TOPICS, state_reference_ns, series)
    state_base_twist = zoh_at(series[TWIST_CMD], state_reference_ns).astype(np.float32, copy=False)
    action = concatenate_zoh(ACTION_VECTOR_TOPICS, action_reference_ns, series)
    state_times_s = relative_seconds(state_reference_ns, origin_ns)
    action_times_s = relative_seconds(action_reference_ns, origin_ns)
    validate_payload(state_times_s, state, state_base_twist, action_times_s, action)

    write_atomically(
        output_path,
        source=source,
        origin_ns=origin_ns,
        state_times_s=state_times_s,
        state=state,
        state_base_twist=state_base_twist,
        action_times_s=action_times_s,
        action=action,
    )

    print(f"[done] output={output_path}")
    print(
        f"[done] state: frames={len(state_times_s)} rate≈{nominal_rate_hz(state_times_s):.6f}Hz "
        f"range={state_times_s[0]:.9f}..{state_times_s[-1]:.9f}s shape={state.shape}"
    )
    print(
        f"[done] action: frames={len(action_times_s)} rate≈{nominal_rate_hz(action_times_s):.6f}Hz "
        f"range={action_times_s[0]:.9f}..{action_times_s[-1]:.9f}s shape={action.shape}"
    )
    print(
        "[done] state[:,20:23]=odom_raw velocity; state_base_twist and action[:,20:23]="
        "twist/cmd with causal ZOH."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
