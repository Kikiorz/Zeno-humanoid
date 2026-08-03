#!/usr/bin/env python3
"""Replay an NPZ through the normal 24-D Robot8 auto whole-body topic.

This is deliberately a single-file ROS2 worker: it does not load a model,
start a socket worker, or need images.  By default it follows the selected
source's original timestamps exactly, and emits the same deploy ABI as the
model bridge:

    /zeno/h1/auto/wholebody/cmd
    std_msgs/msg/Float64MultiArray
    [1.0, upper_body_20, base_twist_3]

The default ``--replay-source state`` retains the original trajectory-replay
contract: the first 20 values are joint state positions and the final three
values come from the NPZ action tail (recorded ``/zeno/h1/twist/cmd``).
Measured ``state[20:23]`` odom velocities are never published as commands.

For model-output NPZs, use ``--replay-source action``. That publishes all 23
command dimensions directly from ``action`` and therefore mirrors an ACT
deployment output rather than replaying the measured upper-body state.

By request, execution publishes immediately by default after live preflight
checks.  Use ``--dry-run`` only when a non-publishing preview is wanted:

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 replay_zeno_npz_state.py

    # Replay a model's full 23-D predicted command.
    /usr/bin/python3 replay_zeno_npz_state.py --npz /path/model_output.npz \
      --replay-source action

Pass ``--rate-hz HZ`` only when an explicit fixed-rate resample is wanted.
On normal exit or Ctrl-C the script sends the deployment idle ``[0.0] * 24``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


VECTOR_DIM = 23
UPPER_BODY_DIM = 20
BASE_DIM = 3
COMMAND_DIM = 24
DEFAULT_CMD_TOPIC = "/zeno/h1/auto/wholebody/cmd"
DEFAULT_NPZ = "/home/zeno-rp/2027icra/Data/replay/8.1DEMO-1_full_merged_smoothed.npz"
DEFAULT_ACTUAL_STATE_OUTPUT_DIR = Path(DEFAULT_NPZ).parent
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
    """Replay input with independent native state/action clocks when present."""

    state_timestamps_s: np.ndarray
    states: np.ndarray
    action_timestamps_s: np.ndarray
    actions: np.ndarray
    state_base_twists: np.ndarray | None
    path: Path
    native_timing_schema: bool


@dataclass(frozen=True)
class ReplayTrajectory:
    """Commands together with the wall-clock schedule used for publication.

    ``times_s`` remains on the original NPZ clock. ``schedule_times_s`` is
    identical unless optional fixed-rate transition frames are inserted. With
    the default ``state``
    source, ``upper_states`` comes from NPZ state[0:20] and ``base_twists``
    from NPZ action[20:23]. With the ``action`` source, both come from the
    23-D model action. If optional transition frames are inserted,
    neighbouring values may share a timestamp because the inserted safety
    ramp lengthens wall-clock replay.
    """

    times_s: np.ndarray
    schedule_times_s: np.ndarray
    upper_states: np.ndarray
    base_twists: np.ndarray
    rate_hz: float
    source_timing: bool


@dataclass
class ActualStateCapture:
    """Measured state samples paired with the exact replay command sent.

    The capture is intentionally owned by the replay process.  It therefore
    does not add a second subscriber to the auto-command topic and cannot
    influence the replay script's command-subscriber preflight.
    """

    source_path: Path
    output_path: Path
    rate_hz: float
    max_state_age_s: float
    replay_source: str
    timestamps_s: list[float]
    source_times_s: list[float]
    frame_indices: list[int]
    states: list[np.ndarray]
    actions: list[np.ndarray]
    cache_ages_s: list[np.ndarray]
    dropped_samples: dict[str, int]

    @classmethod
    def create(
        cls,
        source_path: Path,
        output_path: Path,
        rate_hz: float,
        max_state_age_s: float,
        replay_source: str,
    ) -> "ActualStateCapture":
        return cls(
            source_path=source_path,
            output_path=output_path,
            rate_hz=rate_hz,
            max_state_age_s=max_state_age_s,
            replay_source=replay_source,
            timestamps_s=[],
            source_times_s=[],
            frame_indices=[],
            states=[],
            actions=[],
            cache_ages_s=[],
            dropped_samples={},
        )

    def drop(self, reason: str) -> None:
        self.dropped_samples[reason] = self.dropped_samples.get(reason, 0) + 1

    def append(
        self,
        *,
        replay_elapsed_s: float,
        source_time_s: float,
        frame_index: int,
        state: np.ndarray,
        action: np.ndarray,
        cache_ages_s: np.ndarray,
    ) -> None:
        if state.shape != (VECTOR_DIM,) or action.shape != (VECTOR_DIM,):
            raise ValueError(f"actual-state capture requires 23-D state/action, got {state.shape}/{action.shape}")
        if cache_ages_s.shape != (6,):
            raise ValueError(f"actual-state cache ages must be 6-D, got {cache_ages_s.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(cache_ages_s).all():
            raise ValueError("actual-state capture received NaN/Inf")
        self.timestamps_s.append(float(replay_elapsed_s))
        self.source_times_s.append(float(source_time_s))
        self.frame_indices.append(int(frame_index))
        self.states.append(state.astype(np.float32, copy=True))
        self.actions.append(action.astype(np.float32, copy=True))
        self.cache_ages_s.append(cache_ages_s.astype(np.float32, copy=True))

    def write(self) -> tuple[Path, Path] | None:
        """Atomically save actual state and paired sent command after replay."""

        if not self.states:
            return None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_name(
            f".{self.output_path.stem}.{os.getpid()}.tmp.npz"
        )
        np.savez_compressed(
            temporary_path,
            # ``state`` is measured live feedback; ``action`` is the exact
            # 23-D command built by this replay process for that row.
            timestamp_s=np.asarray(self.timestamps_s, dtype=np.float64),
            source_timestamp_s=np.asarray(self.source_times_s, dtype=np.float64),
            replay_frame_index=np.asarray(self.frame_indices, dtype=np.int64),
            state=np.stack(self.states).astype(np.float32, copy=False),
            action=np.stack(self.actions).astype(np.float32, copy=False),
            state_cache_age_s=np.stack(self.cache_ages_s).astype(np.float32, copy=False),
            source=np.asarray(str(self.source_path)),
            replay_source=np.asarray(self.replay_source),
            state_fields=np.asarray(ACTION_FIELDS),
            state_cache_order=np.asarray(
                ("torso", "left_arm", "right_arm", "left_gripper", "right_gripper", "odom")
            ),
            sampling_contract=np.asarray(
                "Actual 23-D feedback sampled in the replay process immediately after each "
                "publish; action is the exact [upper_state, base_twist] command sent for that replay frame."
            ),
        )
        temporary_path.replace(self.output_path)
        summary_path = self.output_path.with_suffix(".json")
        summary_path.write_text(
            json.dumps(
                {
                    "source_npz": str(self.source_path),
                    "output": str(self.output_path),
                    "sample_count": len(self.states),
                    "nominal_replay_rate_hz": self.rate_hz,
                    "max_state_age_s": self.max_state_age_s,
                    "replay_source": self.replay_source,
                    "dropped_samples": self.dropped_samples,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.output_path, summary_path


def parse_rate_hz(raw_value: str) -> float | None:
    """Parse ``source`` or an explicit positive fixed replay frequency."""

    if raw_value.lower() in {"source", "native", "auto"}:
        return None
    try:
        rate_hz = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be 'source' or a positive number") from exc
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise argparse.ArgumentTypeError("must be 'source' or a finite positive number")
    return rate_hz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay either [state[0:20], action[20:23]] (default) or full action[0:23] "
            "to the normal 24-D /zeno/h1/auto/wholebody/cmd deploy topic. Publishes by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ,
        help=(
            "Legacy NPZ: timestamp_s[N], state[N,23], action[N,23]. Native-clock NPZ: "
            "state_timestamp_s[Ns], state[Ns,23], action_timestamp_s[Na], action[Na,23]."
        ),
    )
    parser.add_argument(
        "--replay-source",
        choices=("state", "action"),
        default="state",
        help=(
            "state (default): publish [state[0:20], action[20:23]] for an edited trajectory; "
            "action: publish action[0:23] for a model-output NPZ."
        ),
    )
    parser.add_argument("--cmd-topic", default=DEFAULT_CMD_TOPIC)
    parser.add_argument(
        "--rate-hz",
        type=parse_rate_hz,
        default="source",
        metavar="{source|HZ}",
        help=(
            "source (default): preserve the selected state/action source timestamps; "
            "a positive HZ value: explicitly resample onto a fixed publish clock."
        ),
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without creating ROS publishers. Default publishes after preflight.",
    )
    # Compatibility with the previous revision: publishing is already the
    # default, so this flag is intentionally a no-op.
    parser.add_argument("--publish", dest="dry_run", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run-realtime",
        action="store_true",
        help="In --dry-run, retain the selected source/fixed wall-clock timing instead of printing quickly.",
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
        help="Permit publishing even if no subscriber matches --cmd-topic.",
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
    parser.add_argument("--odom-topic", default="/zeno/h1/sensor/odom_raw")
    parser.add_argument("--log-every-n", type=int, default=20)
    parser.add_argument(
        "--transition-s",
        type=float,
        default=0.0,
        help=(
            "For an upper-body jump larger than --transition-threshold-rad, insert a "
            "linear upper-body transition of this duration. Zero preserves the selected NPZ source exactly."
        ),
    )
    parser.add_argument(
        "--transition-threshold-rad",
        type=float,
        default=0.30,
        help="Upper-body per-frame jump that triggers optional --transition-s smoothing.",
    )
    parser.add_argument(
        "--record-actual-state",
        action="store_true",
        help=(
            "During replay, record fresh live JointState + odom feedback after each publish and "
            "save it to a separate NPZ with the exact 23-D command sent for every captured frame. "
            "Stale feedback rows are skipped; use replay_frame_index to align the result."
        ),
    )
    parser.add_argument(
        "--actual-state-output",
        type=Path,
        default=None,
        help=(
            "NPZ written by --record-actual-state. Defaults to "
            "Data/replay/<input-npz-stem>_actual_state.npz."
        ),
    )
    parser.add_argument(
        "--actual-state-max-age-s",
        type=float,
        default=0.25,
        help="Skip capture rows if any live JointState/odom cache is older than this.",
    )
    parser.add_argument(
        "--overwrite-actual-state",
        action="store_true",
        help="Allow --record-actual-state to replace an existing output NPZ and summary JSON.",
    )
    return parser.parse_args()


def load_source_state(raw_path: str) -> SourceState:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        native_required = ("state_timestamp_s", "state", "action_timestamp_s", "action")
        if all(name in archive for name in native_required):
            state_timestamps_s = np.asarray(archive["state_timestamp_s"], dtype=np.float64)
            states = np.asarray(archive["state"], dtype=np.float64)
            action_timestamps_s = np.asarray(archive["action_timestamp_s"], dtype=np.float64)
            actions = np.asarray(archive["action"], dtype=np.float64)
            state_base_twists = (
                np.asarray(archive["state_base_twist"], dtype=np.float64)
                if "state_base_twist" in archive
                else None
            )
            native_timing_schema = True
        else:
            missing = [name for name in ("timestamp_s", "state", "action") if name not in archive]
            if missing:
                raise ValueError(
                    "NPZ must contain either native-clock arrays "
                    "(state_timestamp_s/state/action_timestamp_s/action) or legacy arrays "
                    f"(timestamp_s/state/action); missing: {', '.join(missing)}"
                )
            legacy_timestamps_s = np.asarray(archive["timestamp_s"], dtype=np.float64)
            state_timestamps_s = legacy_timestamps_s
            action_timestamps_s = legacy_timestamps_s
            states = np.asarray(archive["state"], dtype=np.float64)
            actions = np.asarray(archive["action"], dtype=np.float64)
            state_base_twists = None
            native_timing_schema = False

    def validate_stream(label: str, timestamps_s: np.ndarray, values: np.ndarray) -> None:
        if timestamps_s.ndim != 1:
            raise ValueError(f"{label}_timestamp_s must have shape [N], got {timestamps_s.shape}")
        if values.ndim != 2 or values.shape[1] != VECTOR_DIM:
            raise ValueError(f"{label} must have shape [N,{VECTOR_DIM}], got {values.shape}")
        if len(timestamps_s) != len(values) or len(values) < 2:
            raise ValueError(
                f"{label}_timestamp_s and {label} must have the same length with at least two frames: "
                f"timestamps={len(timestamps_s)}, values={len(values)}"
            )
        if not np.isfinite(timestamps_s).all() or not np.isfinite(values).all():
            raise ValueError(f"{label}_timestamp_s and {label} must contain only finite values")
        if np.any(np.diff(timestamps_s) <= 0.0):
            raise ValueError(f"{label}_timestamp_s must be strictly increasing")

    validate_stream("state", state_timestamps_s, states)
    validate_stream("action", action_timestamps_s, actions)
    if state_base_twists is not None:
        if state_base_twists.shape != (len(state_timestamps_s), BASE_DIM):
            raise ValueError(
                "state_base_twist must have shape "
                f"[{len(state_timestamps_s)},{BASE_DIM}], got {state_base_twists.shape}"
            )
        if not np.isfinite(state_base_twists).all():
            raise ValueError("state_base_twist must contain only finite values")
    return SourceState(
        state_timestamps_s=state_timestamps_s,
        states=states,
        action_timestamps_s=action_timestamps_s,
        actions=actions,
        state_base_twists=state_base_twists,
        path=path,
        native_timing_schema=native_timing_schema,
    )


def resolve_actual_state_output(source: SourceState, args: argparse.Namespace) -> Path | None:
    """Resolve and protect the optional live-feedback capture path."""

    if not args.record_actual_state:
        return None
    if args.dry_run:
        raise ValueError("--record-actual-state requires publishing; remove --dry-run")
    if not math.isfinite(args.actual_state_max_age_s) or args.actual_state_max_age_s <= 0.0:
        raise ValueError("--actual-state-max-age-s must be finite and positive")
    raw_output = args.actual_state_output
    output = (
        raw_output.expanduser().resolve()
        if raw_output is not None
        else (DEFAULT_ACTUAL_STATE_OUTPUT_DIR / f"{source.path.stem}_actual_state.npz").resolve()
    )
    if output == source.path:
        raise ValueError("--actual-state-output must be a new NPZ, not the replay input NPZ")
    summary = output.with_suffix(".json")
    if not args.overwrite_actual_state and (output.exists() or summary.exists()):
        existing = output if output.exists() else summary
        raise FileExistsError(
            f"actual-state output already exists: {existing}; pass --overwrite-actual-state to replace it"
        )
    return output


def build_trajectory(
    source: SourceState,
    rate_hz: float | None,
    start_offset_s: float,
    duration_s: float | None,
    max_frames: int | None,
    replay_source: str,
) -> ReplayTrajectory:
    if rate_hz is not None and (not math.isfinite(rate_hz) or rate_hz <= 0.0):
        raise ValueError("--rate-hz must be 'source' or a finite positive number")
    if not math.isfinite(start_offset_s) or start_offset_s < 0.0:
        raise ValueError("--start-offset-s must be finite and non-negative")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0.0):
        raise ValueError("--duration-s must be finite and positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if replay_source not in {"state", "action"}:
        raise ValueError(f"unsupported replay source: {replay_source!r}")

    selected_timestamps_s = (
        source.state_timestamps_s if replay_source == "state" else source.action_timestamps_s
    )
    source_start_s = float(selected_timestamps_s[0])
    source_end_s = float(selected_timestamps_s[-1])
    start_s = source_start_s + start_offset_s
    if start_s > source_end_s:
        raise ValueError("--start-offset-s is after the end of the NPZ")
    end_s = source_end_s if duration_s is None else min(source_end_s, start_s + duration_s)
    if end_s < start_s:
        raise ValueError("no source time remains after applying start/duration")

    if rate_hz is None:
        selected_indices = np.flatnonzero(
            (selected_timestamps_s >= start_s) & (selected_timestamps_s <= end_s)
        )
        if not len(selected_indices):
            raise ValueError("no selected source samples remain after applying start/duration")
        if max_frames is not None:
            selected_indices = selected_indices[:max_frames]
        times_s = selected_timestamps_s[selected_indices]
        source_timing = True
        nominal_rate_hz = 1.0 / float(np.median(np.diff(selected_timestamps_s)))
    else:
        # Fixed-rate mode deliberately changes timing. It is retained for
        # legacy deployments that explicitly want a regular command clock.
        frame_count = int(math.floor((end_s - start_s) * rate_hz + 1e-9)) + 1
        times_s = start_s + np.arange(frame_count, dtype=np.float64) / rate_hz
        if times_s[-1] < end_s - 1e-9:
            times_s = np.append(times_s, end_s)
        if max_frames is not None:
            times_s = times_s[:max_frames]
        selected_indices = None
        source_timing = False
        nominal_rate_hz = rate_hz

    if replay_source == "state":
        if source_timing:
            assert selected_indices is not None
            upper_states = source.states[selected_indices, :UPPER_BODY_DIM]
            if source.state_base_twists is not None:
                base_twists = source.state_base_twists[selected_indices]
            else:
                base_indices = np.searchsorted(source.action_timestamps_s, times_s, side="right") - 1
                base_indices = np.clip(base_indices, 0, len(source.actions) - 1)
                base_twists = source.actions[base_indices, UPPER_BODY_DIM:]
        else:
            upper_states = np.empty((len(times_s), UPPER_BODY_DIM), dtype=np.float64)
            for index in range(UPPER_BODY_DIM):
                upper_states[:, index] = np.interp(
                    times_s, source.state_timestamps_s, source.states[:, index]
                )
            if source.state_base_twists is not None:
                base_indices = np.searchsorted(source.state_timestamps_s, times_s, side="right") - 1
                base_indices = np.clip(base_indices, 0, len(source.state_base_twists) - 1)
                base_twists = source.state_base_twists[base_indices]
            else:
                base_indices = np.searchsorted(source.action_timestamps_s, times_s, side="right") - 1
                base_indices = np.clip(base_indices, 0, len(source.actions) - 1)
                base_twists = source.actions[base_indices, UPPER_BODY_DIM:]
    else:
        action_indices = np.searchsorted(source.action_timestamps_s, times_s, side="right") - 1
        action_indices = np.clip(action_indices, 0, len(source.actions) - 1)
        upper_states = source.actions[action_indices, :UPPER_BODY_DIM]
        base_twists = source.actions[action_indices, UPPER_BODY_DIM:]
    return ReplayTrajectory(
        times_s=times_s,
        schedule_times_s=times_s.copy(),
        upper_states=upper_states,
        base_twists=base_twists,
        rate_hz=nominal_rate_hz,
        source_timing=source_timing,
    )


def add_large_jump_transitions(
    trajectory: ReplayTrajectory,
    transition_s: float,
    threshold_rad: float,
) -> ReplayTrajectory:
    """Optionally insert ramps at large *upper-body* discontinuities.

    The default transition duration is zero, so normal use preserves the
    selected source. When requested in explicit fixed-rate mode, inserted
    frames lengthen wall-clock replay rather than hiding a jump inside one
    command period. Native source-timing mode refuses this deliberate timing
    modification.
    """

    if not math.isfinite(transition_s) or transition_s < 0.0:
        raise ValueError("--transition-s must be finite and non-negative")
    if not math.isfinite(threshold_rad) or threshold_rad <= 0.0:
        raise ValueError("--transition-threshold-rad must be finite and positive")
    if transition_s == 0.0 or len(trajectory.upper_states) < 2:
        return trajectory
    if trajectory.source_timing:
        raise ValueError(
            "--transition-s changes source timing; pass an explicit --rate-hz HZ to use it"
        )

    transition_steps = max(1, int(math.ceil(transition_s * trajectory.rate_hz)))
    output_upper_states: list[np.ndarray] = [trajectory.upper_states[0]]
    output_base_twists: list[np.ndarray] = [trajectory.base_twists[0]]
    output_times: list[float] = [float(trajectory.times_s[0])]
    output_schedule_times: list[float] = [float(trajectory.schedule_times_s[0])]
    schedule_offset_s = 0.0
    for index in range(1, len(trajectory.upper_states)):
        previous_upper = output_upper_states[-1]
        previous_base = output_base_twists[-1]
        target_upper = trajectory.upper_states[index]
        target_base = trajectory.base_twists[index]
        max_upper_body_delta = float(np.max(np.abs(target_upper - previous_upper)))
        if max_upper_body_delta > threshold_rad:
            for fraction in np.linspace(1.0 / transition_steps, 1.0, transition_steps):
                ramped_upper = previous_upper + fraction * (target_upper - previous_upper)
                # The base trajectory is not altered by an upper-body safety
                # ramp.  Keep its previous Twist until the target frame.
                output_upper_states.append(ramped_upper)
                output_base_twists.append(target_base if fraction == 1.0 else previous_base)
                output_times.append(float(trajectory.times_s[index]))
                output_schedule_times.append(output_schedule_times[-1] + 1.0 / trajectory.rate_hz)
            schedule_offset_s += (transition_steps - 1) / trajectory.rate_hz
        else:
            output_upper_states.append(target_upper)
            output_base_twists.append(target_base)
            output_times.append(float(trajectory.times_s[index]))
            output_schedule_times.append(float(trajectory.schedule_times_s[index]) + schedule_offset_s)
    return ReplayTrajectory(
        times_s=np.asarray(output_times, dtype=np.float64),
        schedule_times_s=np.asarray(output_schedule_times, dtype=np.float64),
        upper_states=np.asarray(output_upper_states, dtype=np.float64),
        base_twists=np.asarray(output_base_twists, dtype=np.float64),
        rate_hz=trajectory.rate_hz,
        source_timing=False,
    )


def zero_arm_plateaus(
    source: SourceState, replay_source: str, minimum_frames: int = 20
) -> list[tuple[int, int]]:
    """Find long exact-zero arm/gripper stretches in the published source."""

    values = source.states if replay_source == "state" else source.actions
    all_zero = np.all(values[:, 4:UPPER_BODY_DIM] == 0.0, axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([False], all_zero, [False])).astype(np.int8)))
    return [
        (int(start), int(end))
        for start, end in edges.reshape(-1, 2)
        if end - start >= minimum_frames
    ]


def source_values_and_timestamps(
    source: SourceState, replay_source: str
) -> tuple[np.ndarray, np.ndarray]:
    if replay_source == "state":
        return source.states, source.state_timestamps_s
    return source.actions, source.action_timestamps_s


def max_upper_body_step(trajectory: ReplayTrajectory) -> tuple[int, float]:
    if len(trajectory.upper_states) < 2:
        return 0, 0.0
    deltas = np.max(np.abs(np.diff(trajectory.upper_states, axis=0)), axis=1)
    index = int(np.argmax(deltas)) + 1
    return index, float(deltas[index - 1])


def format_command(upper_state: Sequence[float], base_twist: Sequence[float]) -> str:
    head = ", ".join(
        f"{name}={float(value):.4f}"
        for name, value in zip(ACTION_FIELDS[:6], upper_state[:6], strict=True)
    )
    base = ", ".join(
        f"{name}={float(value):.4f}"
        for name, value in zip(ACTION_FIELDS[-3:], base_twist, strict=True)
    )
    return f"{head}, ..., {base}"


def print_source_summary(source: SourceState, trajectory: ReplayTrajectory, args: argparse.Namespace) -> None:
    state_dt_s = np.diff(source.state_timestamps_s)
    action_dt_s = np.diff(source.action_timestamps_s)
    state_rate_hz = 1.0 / float(np.median(state_dt_s))
    action_rate_hz = 1.0 / float(np.median(action_dt_s))
    if args.replay_source == "state":
        base_label = "state_base_twist" if source.state_base_twists is not None else "action[20:23]"
        command_layout = f"[1.0, state[0:20], {base_label}]"
    else:
        command_layout = "[1.0, action[0:23]]"
    print(
        f"[load] npz={source.path}\n"
        f"[load] state={source.states.shape}, duration="
        f"{source.state_timestamps_s[-1] - source.state_timestamps_s[0]:.3f}s, "
        f"native_rate≈{state_rate_hz:.3f}Hz; action={source.actions.shape}, duration="
        f"{source.action_timestamps_s[-1] - source.action_timestamps_s[0]:.3f}s, "
        f"native_rate≈{action_rate_hz:.3f}Hz\n"
        f"[ready] timing={'source timestamps' if trajectory.source_timing else 'fixed-rate resample'}, "
        f"output_frames={len(trajectory.upper_states)}, nominal_rate≈{trajectory.rate_hz:g}Hz, "
        f"output_duration={trajectory.schedule_times_s[-1] - trajectory.schedule_times_s[0]:.3f}s\n"
        f"[ready] replay_source={args.replay_source}; Float64MultiArray={command_layout}\n"
        f"[ready] cmd_topic={args.cmd_topic}",
        flush=True,
    )
    source_label = "state" if args.replay_source == "state" else "action"
    _, source_times_s = source_values_and_timestamps(source, args.replay_source)
    for start, end in zero_arm_plateaus(source, args.replay_source):
        print(
            f"[warning] {source_label} itself has an all-zero arm/gripper stretch: "
            f"frames {start}:{end - 1}, source_time={source_times_s[start]:.3f}.."
            f"{source_times_s[end - 1]:.3f}s. It will be replayed as supplied.",
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
    if args.replay_source == "state":
        base_source = "state_base_twist" if source.state_base_twists is not None else "action[20:23]"
        print(
            "[note] upper body uses state[0:20]; the final base command slots use "
            f"{base_source} from recorded Twist. state[20:23] odometry velocities are not "
            "published as commands.",
            flush=True,
        )
    else:
        print(
            "[note] all 23 active command slots use action. This is the correct mode for a model-output NPZ.",
            flush=True,
        )


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


def extract_odom_velocity(message: Any) -> list[float] | None:
    """Read the measured base-twist tail of the normal 23-D state."""

    try:
        twist = message.twist.twist
        values = [float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)]
    except (AttributeError, TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def create_ros_publisher(args: argparse.Namespace) -> tuple[Any, Any]:
    """Create the same auto whole-body publisher used by the deployment bridge."""

    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python packages are unavailable. Source ROS first, for example: "
            "source /opt/ros/humble/setup.bash && /usr/bin/python3 replay_zeno_npz_state.py"
        ) from exc

    class CommandPublisher(Node):
        def __init__(self) -> None:
            super().__init__("zeno_npz_state_replay")
            # Same QoS shortcut as the deployment bridge: KEEP_LAST depth 10,
            # reliable and volatile in ROS2 Humble's default QoS profile.
            self.publisher = self.create_publisher(Float64MultiArray, args.cmd_topic, 10)
            self.current_joint_positions: dict[str, list[float]] = {}
            self.current_joint_received_s: dict[str, float] = {}
            self.current_odom_velocity: list[float] | None = None
            self.current_odom_received_s: float | None = None
            self.subscriptions_keepalive = []
            for group, fields in REQUIRED_JOINTS.items():
                topic = getattr(args, JOINT_TOPIC_PARAMS[group])
                self.subscriptions_keepalive.append(
                    self.create_subscription(JointState, topic, self.joint_callback(group, fields), 10)
                )
            if args.record_actual_state:
                self.subscriptions_keepalive.append(
                    self.create_subscription(Odometry, args.odom_topic, self.odom_callback, 10)
                )

        def joint_callback(self, group: str, fields: Sequence[str]) -> Any:
            def callback(message: Any) -> None:
                values = extract_joint_positions(message, fields)
                if values is not None and all(math.isfinite(value) for value in values):
                    self.current_joint_positions[group] = values
                    self.current_joint_received_s[group] = time.monotonic()

            return callback

        def odom_callback(self, message: Any) -> None:
            values = extract_odom_velocity(message)
            if values is not None:
                self.current_odom_velocity = values
                self.current_odom_received_s = time.monotonic()

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

        def command_subscriber_counts(self) -> dict[str, int]:
            return {args.cmd_topic: int(self.publisher.get_subscription_count())}

        def current_actual_state(
            self, max_age_s: float
        ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
            """Return a fresh whole-body feedback snapshot for live capture."""

            now_s = time.monotonic()
            values: list[float] = []
            ages_s: list[float] = []
            for group in REQUIRED_JOINTS:
                joint_values = self.current_joint_positions.get(group)
                received_s = self.current_joint_received_s.get(group)
                if joint_values is None or received_s is None:
                    return None, None, f"missing:{group}"
                age_s = now_s - received_s
                if age_s < 0.0 or age_s > max_age_s:
                    return None, None, f"stale:{group}"
                values.extend(joint_values)
                ages_s.append(age_s)
            if self.current_odom_velocity is None or self.current_odom_received_s is None:
                return None, None, "missing:odom"
            odom_age_s = now_s - self.current_odom_received_s
            if odom_age_s < 0.0 or odom_age_s > max_age_s:
                return None, None, "stale:odom"
            values.extend(self.current_odom_velocity)
            ages_s.append(odom_age_s)
            state = np.asarray(values, dtype=np.float32)
            ages = np.asarray(ages_s, dtype=np.float32)
            if state.shape != (VECTOR_DIM,):
                return None, None, f"shape:{state.shape}"
            if not np.isfinite(state).all():
                return None, None, "nonfinite"
            return state, ages, None

        def publish_frame(self, upper_state: Sequence[float], base_twist: Sequence[float]) -> None:
            if len(upper_state) != UPPER_BODY_DIM:
                raise ValueError(
                    f"upper state command must contain {UPPER_BODY_DIM} values, got {len(upper_state)}"
                )
            if len(base_twist) != BASE_DIM:
                raise ValueError(f"base action tail must contain {BASE_DIM} values, got {len(base_twist)}")
            command = [1.0, *[float(value) for value in upper_state], *[float(value) for value in base_twist]]
            if len(command) != COMMAND_DIM:
                raise RuntimeError(f"whole-body command must contain {COMMAND_DIM} values, got {len(command)}")
            message = Float64MultiArray()
            message.data = command
            self.publisher.publish(message)

        def publish_idle(self) -> None:
            message = Float64MultiArray()
            message.data = [0.0] * COMMAND_DIM
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


def drain_callbacks(rclpy_module: Any, node: Any, max_callbacks: int = 32) -> None:
    """Consume pending feedback callbacks before a live-state snapshot.

    The normal replay path deliberately does not need a continuously spinning
    executor.  When capture is requested, this bounded non-blocking drain
    prevents the five fast JointState streams from sitting behind the replay
    publish loop while leaving the control schedule unchanged.  The remaining
    wait time is then spent in ``spin_for`` below.
    """

    for _ in range(max_callbacks):
        rclpy_module.spin_once(node, timeout_sec=0.0)


def preflight_publish(rclpy_module: Any, node: Any, first_state: Sequence[float], args: argparse.Namespace) -> None:
    """Wait for DDS/state input, then reject a mismatched absolute-pose start."""

    spin_for(rclpy_module, node, args.discovery_wait_s)
    subscriber_counts = node.command_subscriber_counts()
    missing_command_topics = [topic for topic, count in subscriber_counts.items() if count == 0]
    if missing_command_topics and not args.allow_no_command_subscriber:
        raise RuntimeError(
            "no subscriber matched whole-body command topic after "
            f"{args.discovery_wait_s:g}s: {', '.join(missing_command_topics)}. Refusing to start. "
            "Use --allow-no-command-subscriber only for a non-robot transport test."
        )
    count_text = ", ".join(f"{topic}={count}" for topic, count in subscriber_counts.items())
    print(f"[preflight] matched command subscribers: {count_text}", flush=True)

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
    """Repeat deployment idle so a short DDS loss does not leave control active."""

    if not rclpy_module.ok():
        print("[publish] ROS context already stopped; could not send deployment idle.", flush=True)
        return
    for repeat in range(IDLE_PUBLISH_REPEATS):
        node.publish_idle()
        rclpy_module.spin_once(node, timeout_sec=0.0)
        if repeat + 1 < IDLE_PUBLISH_REPEATS:
            time.sleep(0.05)


def replay(
    trajectory: ReplayTrajectory,
    args: argparse.Namespace,
    actual_state_capture: ActualStateCapture | None = None,
) -> None:
    realtime = not args.dry_run or args.dry_run_realtime
    rclpy_module: Any | None = None
    node: Any | None = None
    completed = False

    if not args.dry_run:
        rclpy_module, node = create_ros_publisher(args)

    try:
        if node is not None:
            preflight_publish(rclpy_module, node, trajectory.upper_states[0], args)
        start_wall_s = time.perf_counter()
        capture_start_monotonic_s = time.monotonic()
        for index, (upper_state, base_twist) in enumerate(
            zip(trajectory.upper_states, trajectory.base_twists, strict=True)
        ):
            if node is not None:
                node.publish_frame(upper_state, base_twist)
                if actual_state_capture is None:
                    rclpy_module.spin_once(node, timeout_sec=0.0)
                else:
                    # Keep the state queues current before snapshotting. This
                    # is capture-only and is compensated by the same absolute
                    # source/fixed timing deadline below.
                    drain_callbacks(rclpy_module, node)
                    measured_state, cache_ages_s, reason = node.current_actual_state(
                        actual_state_capture.max_state_age_s
                    )
                    if reason is not None:
                        actual_state_capture.drop(reason)
                    else:
                        assert measured_state is not None and cache_ages_s is not None
                        sent_action = np.concatenate(
                            (
                                np.asarray(upper_state, dtype=np.float32),
                                np.asarray(base_twist, dtype=np.float32),
                            )
                        )
                        actual_state_capture.append(
                            replay_elapsed_s=time.monotonic() - capture_start_monotonic_s,
                            source_time_s=float(trajectory.times_s[index]),
                            frame_index=index,
                            state=measured_state,
                            action=sent_action,
                            cache_ages_s=cache_ages_s,
                        )

            if index % args.log_every_n == 0 or index == len(trajectory.upper_states) - 1:
                mode = "publish" if not args.dry_run else "dry-run"
                print(
                    f"[{mode}] frame={index}/{len(trajectory.upper_states) - 1} "
                    f"source_time={trajectory.times_s[index]:.3f}s: "
                    f"{format_command(upper_state, base_twist)}",
                    flush=True,
                )

            if realtime and index + 1 < len(trajectory.upper_states):
                deadline_s = start_wall_s + (
                    trajectory.schedule_times_s[index + 1] - trajectory.schedule_times_s[0]
                )
                remaining_s = max(0.0, deadline_s - time.perf_counter())
                if actual_state_capture is not None and node is not None:
                    # Unlike plain replay, consume feedback throughout the
                    # idle portion of the period so the next sample remains
                    # fresh even when commands publish faster than feedback.
                    spin_for(rclpy_module, node, remaining_s)
                else:
                    time.sleep(remaining_s)
        completed = True
    finally:
        try:
            if node is not None:
                status = "after completion" if completed else "after interruption/failure"
                publish_idle(rclpy_module, node)
                print(f"[publish] sent {IDLE_PUBLISH_REPEATS} deployment idle command(s) {status}.", flush=True)
        finally:
            try:
                if actual_state_capture is not None:
                    result = actual_state_capture.write()
                    if result is None:
                        print(
                            "[capture] no fresh actual-state rows were recorded; no output NPZ written.",
                            file=sys.stderr,
                        )
                    else:
                        output, summary = result
                        print(
                            f"[capture] wrote {output} ({len(actual_state_capture.states)} actual-state rows)",
                            flush=True,
                        )
                        print(f"[capture] summary {summary}", flush=True)
            finally:
                if node is not None:
                    node.destroy_node()
                    if rclpy_module.ok():
                        rclpy_module.shutdown()


def main() -> None:
    args = parse_args()
    if args.log_every_n <= 0:
        raise SystemExit("--log-every-n must be positive")
    if not math.isfinite(args.discovery_wait_s) or args.discovery_wait_s < 0.0:
        raise SystemExit("--discovery-wait-s must be finite and non-negative")
    if not math.isfinite(args.preflight_timeout_s) or args.preflight_timeout_s <= 0.0:
        raise SystemExit("--preflight-timeout-s must be finite and positive")
    if not math.isfinite(args.start_max_position_error) or args.start_max_position_error <= 0.0:
        raise SystemExit("--start-max-position-error must be finite and positive")
    source = load_source_state(args.npz)
    actual_state_output = resolve_actual_state_output(source, args)
    trajectory = build_trajectory(
        source,
        rate_hz=args.rate_hz,
        start_offset_s=args.start_offset_s,
        duration_s=args.duration_s,
        max_frames=args.max_frames,
        replay_source=args.replay_source,
    )
    unsmoothed_frame_count = len(trajectory.upper_states)
    trajectory = add_large_jump_transitions(
        trajectory,
        transition_s=args.transition_s,
        threshold_rad=args.transition_threshold_rad,
    )
    print_source_summary(source, trajectory, args)
    inserted_frame_count = len(trajectory.upper_states) - unsmoothed_frame_count
    if inserted_frame_count:
        print(
            f"[ready] --transition-s inserted {inserted_frame_count} linear ramp frame(s); "
            f"wall-clock replay is {trajectory.schedule_times_s[-1] - trajectory.times_s[-1]:.3f}s longer.",
            flush=True,
        )
    actual_state_capture = None
    if actual_state_output is not None:
        actual_state_capture = ActualStateCapture.create(
            source_path=source.path,
            output_path=actual_state_output,
            rate_hz=trajectory.rate_hz,
            max_state_age_s=args.actual_state_max_age_s,
            replay_source=args.replay_source,
        )
        print(
            f"[capture] enabled: real JointState + odom will be written to {actual_state_output}; "
            f"freshness={args.actual_state_max_age_s:g}s",
            flush=True,
        )
    if args.dry_run:
        print("[ready] dry-run only; omit --dry-run to publish auto whole-body commands.", flush=True)
    else:
        print("[ready] publishing auto whole-body commands after preflight.", flush=True)
    # Match the deployment bridge: turn SIGTERM into a normal interruption so
    # the replay finally block has one chance to publish the idle command.
    def stop_handler(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_handler)
    try:
        replay(trajectory, args, actual_state_capture)
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
