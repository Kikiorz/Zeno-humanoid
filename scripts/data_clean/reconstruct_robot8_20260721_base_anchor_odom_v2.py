#!/usr/bin/env python3
"""Create a physically anchored, noise-cleaned Robot8 LeRobot dataset (v2).

This is intentionally a *new* dataset, not an overwrite of the original or
of ``base_clean_cmdseg_v1``.  It replays the original ROS bags at exactly the
same 20 Hz timestamps used by the converter, reads the real
``/zeno/h1/sensor/odom_raw`` pose, and uses long stationary plateaus as
high-confidence SE(2) anchors.

V2 base-label semantics
=======================
``action[20:23]`` is changed from the recorded low-level ``/twist/cmd`` into
the desired physical body velocity ``[vx, vy, wz]`` of an endpoint-constrained
odom trajectory.  This is the only label definition for which image motion,
the target trajectory, and its final physical pose live in the same
coordinate system.  The original command remains losslessly available in the
provenance sidecar.

Consequently, a V2 policy must be deployed with an odom-feedback inverse
mapper before publishing to ``/zeno/h1/auto/wholebody/cmd``.  The script fits
and stores that mapper, but it does *not* pretend an offline open-loop model
can guarantee a future real-robot endpoint.

Trajectory policy
=================
For every pair of long stationary odom plateaus, V2 makes the plateau poses
hard endpoints.  It then tries candidates in this order:

1. a smooth, endpoint-constrained version of the observed physical path;
2. a rotate-translate-rotate path only when it is feasible *and* remains
   tightly synchronized with the observed camera path.

Candidates must exactly meet the anchors, lower velocity variation, obey
physical speed limits, and stay close to the recorded path.  Otherwise the
observed physical path is retained.  Thus rotate/translate separation is a
strictly gated preference rather than a source of label/image mismatch.
Stable plateaus are explicitly assigned zero desired base velocity.

The result is auditable under ``meta/base_anchor_odom_v2/`` and reuses the
existing MP4 files through hard links by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from rosbags.highlevel import AnyReader


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = (
    REPO_ROOT
    / "Data"
    / "lerobot"
    / "robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23"
)
DEFAULT_BAG_DATA = REPO_ROOT / "Data" / "2026_07_21"
DEFAULT_OUTPUT_DATASET = (
    REPO_ROOT
    / "Data"
    / "lerobot"
    / "robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23_base_anchor_odom_v2"
)
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis" / "robot8_20260721_base_anchor_odom_v2"

ACTION_DIM = 23
BASE_SLICE = slice(20, 23)
FPS = 20.0
DT = 1.0 / FPS


@dataclass(frozen=True)
class V2Config:
    fps: float = FPS
    # A plateau is deliberately stricter than V1's command-only stop: it must
    # be stationary in both command and measured physical velocity for 0.3 s.
    # The pose-spread gate makes these short but genuine stops reliable and
    # preserves data03's rotation/translation checkpoint.
    anchor_min_frames: int = 6
    command_stop_translation_mps: float = 0.005
    command_stop_rotation_rps: float = 0.020
    odom_stop_translation_mps: float = 0.010
    odom_stop_rotation_rps: float = 0.030
    anchor_max_pose_spread_m: float = 0.002
    anchor_max_yaw_spread_rad: float = 0.003
    smooth_window: int = 5
    smooth_max_path_deviation_m: float = 0.040
    smooth_max_yaw_deviation_rad: float = 0.070
    rtr_max_path_deviation_m: float = 0.030
    rtr_max_yaw_deviation_rad: float = math.radians(3.0)
    min_tv_reduction_fraction: float = 0.03
    physical_speed_limit_mps: float = 0.160
    physical_rotation_limit_rps: float = 0.220
    candidate_peak_tolerance: float = 1.05
    endpoint_tolerance_m: float = 1e-5
    endpoint_tolerance_rad: float = 1e-5
    alignment_tolerance: float = 2e-5
    # Metadata defaults keep the established V2 output byte-for-byte
    # compatible.  The V3 launcher supplies distinct values so it can never
    # overwrite V2 provenance by accident.
    dataset_version: str = "base_anchor_odom_v2"
    metadata_dir_name: str = "base_anchor_odom_v2"
    # V3 optional command-mode reconstruction. It uses the recorded command
    # only as a timing prior, preserves exact local SE(2) endpoints, and falls
    # back to the V2 physical path whenever frame-synchronous gates reject it.
    enable_command_mode_separation: bool = False
    command_mode_translation_mps: float = 0.008
    command_mode_rotation_rps: float = 0.025
    command_mode_min_run_frames: int = 4
    separation_checkpoint_min_frames: int = 4
    separation_checkpoint_translation_mps: float = 0.025
    separation_checkpoint_rotation_rps: float = 0.050
    separation_min_block_frames: int = 8
    separation_relative_max_path_deviation_m: float = 0.015
    separation_relative_p95_path_deviation_m: float = 0.008
    separation_relative_max_yaw_deviation_rad: float = math.radians(1.5)
    separation_relative_p95_yaw_deviation_rad: float = math.radians(0.5)
    separation_max_tv_increase_fraction: float = 0.05
    # V3's preferred safe path: take the already visual-aligned V2 physical
    # trajectory and smooth XY and yaw independently a second time between
    # stationary anchors.  This removes short reversal/command-jitter without
    # pretending that genuine simultaneous holonomic translation+yaw should be
    # time-separated.  Zero disables this optional second pass.
    secondary_smooth_window: int = 0
    secondary_relative_max_path_deviation_m: float = 0.015
    secondary_relative_p95_path_deviation_m: float = 0.008
    secondary_relative_max_yaw_deviation_rad: float = math.radians(1.5)
    secondary_relative_p95_yaw_deviation_rad: float = math.radians(0.5)
    secondary_min_tv_reduction_fraction: float = 0.03

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


@dataclass
class RawEpisode:
    episode_index: int
    bag_path: str
    sample_times_ns: np.ndarray
    raw_command: np.ndarray
    odom_twist: np.ndarray
    odom_pose: np.ndarray  # world x, y, unwrapped yaw; shape (N, 3)
    odom_offset_ms: np.ndarray


@dataclass
class Anchor:
    episode_index: int
    anchor_index: int
    start_frame: int
    end_frame: int  # exclusive
    reference_pose_x_m: float
    reference_pose_y_m: float
    reference_pose_yaw_rad: float
    max_position_spread_m: float
    max_yaw_spread_rad: float


@dataclass
class SegmentAudit:
    episode_index: int
    segment_index: int
    start_frame: int
    end_frame: int  # pose/action boundary, inclusive target pose at end
    num_action_frames: int
    strategy: str
    accepted: bool
    raw_end_x_m: float
    raw_end_y_m: float
    raw_end_yaw_rad: float
    target_end_x_m: float
    target_end_y_m: float
    target_end_yaw_rad: float
    endpoint_position_error_m: float
    endpoint_yaw_error_rad: float
    max_path_deviation_m: float
    max_yaw_deviation_rad: float
    raw_path_length_m: float
    target_path_length_m: float
    raw_velocity_total_variation: float
    reference_velocity_total_variation: float
    target_velocity_total_variation: float
    raw_mode_overlap_frames: int
    target_mode_overlap_frames: int
    raw_peak_speed_mps: float
    target_peak_speed_mps: float
    raw_peak_rotation_rps: float
    target_peak_rotation_rps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--bag-data-dir", type=Path, default=DEFAULT_BAG_DATA)
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--preview-episodes",
        type=int,
        nargs="+",
        default=[3],
        help="Episode indexes for raw-vs-target physical trajectory previews.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset to reconstruct; omitted reconstructs all episodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read/reconstruct/audit but do not create the derived dataset.",
    )
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Reuse source videos through hard links (default) or copies.",
    )
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--anchor-min-frames", type=int, default=6)
    parser.add_argument("--smooth-max-path-deviation-m", type=float, default=0.040)
    parser.add_argument("--rtr-max-path-deviation-m", type=float, default=0.030)
    parser.add_argument("--dataset-version", default="base_anchor_odom_v2")
    parser.add_argument("--metadata-dir-name", default="base_anchor_odom_v2")
    parser.add_argument(
        "--enable-command-mode-separation",
        action="store_true",
        help=(
            "Enable the V3 safe command-mode candidate: translation/rotation are "
            "separated only in blocks that pass visual/path and endpoint gates."
        ),
    )
    parser.add_argument("--command-mode-min-run-frames", type=int, default=4)
    parser.add_argument("--separation-checkpoint-min-frames", type=int, default=4)
    parser.add_argument("--separation-min-block-frames", type=int, default=8)
    parser.add_argument("--separation-relative-max-path-deviation-m", type=float, default=0.015)
    parser.add_argument("--separation-relative-p95-path-deviation-m", type=float, default=0.008)
    parser.add_argument("--separation-relative-max-yaw-deviation-deg", type=float, default=1.5)
    parser.add_argument("--separation-relative-p95-yaw-deviation-deg", type=float, default=0.5)
    parser.add_argument(
        "--secondary-smooth-window",
        type=int,
        default=0,
        help=(
            "Optional odd endpoint-constrained smoothing width applied to the V2 physical path. "
            "Use 0 (default) to retain exact V2 behavior; V3 uses 11."
        ),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> V2Config:
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise SystemExit("--smooth-window must be a positive odd integer")
    if args.anchor_min_frames < 6:
        raise SystemExit("--anchor-min-frames must be at least 6")
    if args.smooth_max_path_deviation_m <= 0 or args.rtr_max_path_deviation_m <= 0:
        raise SystemExit("path-deviation limits must be positive")
    if not args.dataset_version or not args.metadata_dir_name:
        raise SystemExit("--dataset-version and --metadata-dir-name must be non-empty")
    if "/" in args.metadata_dir_name or args.metadata_dir_name in {".", ".."}:
        raise SystemExit("--metadata-dir-name must be one safe directory component")
    if args.command_mode_min_run_frames < 1 or args.separation_checkpoint_min_frames < 1:
        raise SystemExit("command/checkpoint run lengths must be positive")
    if args.separation_min_block_frames < 2:
        raise SystemExit("--separation-min-block-frames must be at least 2")
    if args.separation_relative_max_path_deviation_m <= 0 or args.separation_relative_p95_path_deviation_m <= 0:
        raise SystemExit("separation path-deviation limits must be positive")
    if args.separation_relative_max_yaw_deviation_deg <= 0 or args.separation_relative_p95_yaw_deviation_deg <= 0:
        raise SystemExit("separation yaw-deviation limits must be positive")
    if args.separation_relative_p95_path_deviation_m > args.separation_relative_max_path_deviation_m:
        raise SystemExit("separation p95 path-deviation limit must be <= its max limit")
    if args.separation_relative_p95_yaw_deviation_deg > args.separation_relative_max_yaw_deviation_deg:
        raise SystemExit("separation p95 yaw-deviation limit must be <= its max limit")
    if args.secondary_smooth_window < 0 or (
        args.secondary_smooth_window > 0
        and (args.secondary_smooth_window < 3 or args.secondary_smooth_window % 2 == 0)
    ):
        raise SystemExit("--secondary-smooth-window must be 0 or an odd integer >= 3")
    return V2Config(
        smooth_window=args.smooth_window,
        anchor_min_frames=args.anchor_min_frames,
        smooth_max_path_deviation_m=args.smooth_max_path_deviation_m,
        rtr_max_path_deviation_m=args.rtr_max_path_deviation_m,
        dataset_version=args.dataset_version,
        metadata_dir_name=args.metadata_dir_name,
        enable_command_mode_separation=args.enable_command_mode_separation,
        command_mode_min_run_frames=args.command_mode_min_run_frames,
        separation_checkpoint_min_frames=args.separation_checkpoint_min_frames,
        separation_min_block_frames=args.separation_min_block_frames,
        separation_relative_max_path_deviation_m=args.separation_relative_max_path_deviation_m,
        separation_relative_p95_path_deviation_m=args.separation_relative_p95_path_deviation_m,
        separation_relative_max_yaw_deviation_rad=math.radians(args.separation_relative_max_yaw_deviation_deg),
        separation_relative_p95_yaw_deviation_rad=math.radians(args.separation_relative_p95_yaw_deviation_deg),
        secondary_smooth_window=args.secondary_smooth_window,
    )


def load_converter_module() -> ModuleType:
    """Load the original converter so bag discovery/sampling cannot drift."""
    path = REPO_ROOT / "scripts" / "data_convert" / "convert_zeno_h1_v30.py"
    # The converter imports the reusable head-stereo preprocessor beside it.
    # Dynamic ``spec_from_file_location`` loading does not add ``path.parent``
    # to ``sys.path`` the way a normal ``python path/to/script.py`` launch
    # does, so make that import contract explicit before executing the module.
    converter_dir = str(path.parent)
    if converter_dir not in sys.path:
        sys.path.insert(0, converter_dir)
    spec = importlib.util.spec_from_file_location("robot8_v2_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixed_list_to_numpy(table: pa.Table, column_name: str, width: int) -> np.ndarray:
    column = table[column_name].combine_chunks()
    if getattr(column.type, "list_size", None) != width:
        raise ValueError(f"{column_name} must be fixed-size list[{width}], got {column.type}")
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(column), width)


def fixed_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected 2D values, got {values.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def contiguous_episode_bounds(episode_indices: np.ndarray) -> list[tuple[int, int, int]]:
    boundaries = np.flatnonzero(np.diff(episode_indices) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_indices)]))
    bounds: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for start, end in zip(starts, ends, strict=True):
        episode = int(episode_indices[start])
        if episode in seen:
            raise ValueError(f"episode {episode} is not contiguous")
        seen.add(episode)
        bounds.append((episode, int(start), int(end)))
    return bounds


def nearest_indices(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Vectorized copy of the converter's strict-nearest lookup rule."""
    right = np.searchsorted(times, query)
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    choose_right = np.abs(times[right] - query) < np.abs(query - times[left])
    return np.where(right == 0, 0, np.where(right == len(times) - 1, right, np.where(choose_right, right, left)))


def yaw_from_quaternion(quaternion: Any) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def read_raw_episode(
    *,
    converter: ModuleType,
    bag_path: Path,
    episode_index: int,
) -> RawEpisode:
    """Replay the converter's 20 Hz sampling and return physical odom data."""
    camera_names = ["head_cam", "left_arm_cam", "right_arm_cam"]
    topics = converter.enabled_topics(camera_names)
    topic_set = set(topics)
    topic_times: dict[str, list[int]] = {topic: [] for topic in topics}
    odom_messages: list[tuple[int, Any]] = []
    command_messages: list[tuple[int, Any]] = []

    with AnyReader([bag_path]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topic_set]
        if not connections:
            raise RuntimeError(f"No converter topics found in {bag_path}")
        for connection, timestamp, raw in reader.messages(connections=connections):
            topic_times[connection.topic].append(timestamp)
            if connection.topic == converter.ODOM:
                odom_messages.append((timestamp, reader.deserialize(raw, connection.msgtype)))
            elif connection.topic == converter.TWIST_CMD:
                command_messages.append((timestamp, reader.deserialize(raw, connection.msgtype)))

    missing = [topic for topic, values in topic_times.items() if not values]
    if missing:
        raise RuntimeError(f"{bag_path} is missing converter topic(s): {missing}")
    if not odom_messages or not command_messages:
        raise RuntimeError(f"{bag_path} has no odom or twist/cmd messages")

    arrays = {topic: np.asarray(values, dtype=np.int64) for topic, values in topic_times.items()}
    t_start = max(values[0] for values in arrays.values())
    t_end = min(values[-1] for values in arrays.values())
    step_ns = int(1e9 / FPS)
    if t_end <= t_start:
        raise RuntimeError(f"{bag_path} has no common topic time range")
    sample_times = t_start + np.arange((t_end - t_start) // step_ns + 1, dtype=np.int64) * step_ns

    odom_times = np.asarray([item[0] for item in odom_messages], dtype=np.int64)
    command_times = np.asarray([item[0] for item in command_messages], dtype=np.int64)
    odom_indices = nearest_indices(odom_times, sample_times)
    command_indices = nearest_indices(command_times, sample_times)

    pose = np.empty((len(sample_times), 3), dtype=np.float64)
    twist = np.empty((len(sample_times), 3), dtype=np.float64)
    command = np.empty((len(sample_times), 3), dtype=np.float64)
    for output_index, message_index in enumerate(odom_indices):
        message = odom_messages[int(message_index)][1]
        position = message.pose.pose.position
        pose[output_index] = (float(position.x), float(position.y), yaw_from_quaternion(message.pose.pose.orientation))
        measured = message.twist.twist
        twist[output_index] = (float(measured.linear.x), float(measured.linear.y), float(measured.angular.z))
    for output_index, message_index in enumerate(command_indices):
        message = command_messages[int(message_index)][1]
        command[output_index] = (float(message.linear.x), float(message.linear.y), float(message.angular.z))
    pose[:, 2] = np.unwrap(pose[:, 2])

    return RawEpisode(
        episode_index=episode_index,
        bag_path=str(bag_path),
        sample_times_ns=sample_times,
        raw_command=command,
        odom_twist=twist,
        odom_pose=pose,
        odom_offset_ms=(odom_times[odom_indices] - sample_times).astype(np.float64) / 1e6,
    )


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous true runs as half-open ``[start, end)`` pairs."""
    values = np.asarray(mask, dtype=bool)
    if len(values) == 0:
        return []
    padded = np.concatenate(([False], values, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[index]), int(edges[index + 1])) for index in range(0, len(edges), 2)]


def box_smooth(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if width <= 1 or len(values) <= 2:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    return np.stack(
        [np.convolve(padded[:, dimension], kernel, mode="valid") for dimension in range(values.shape[1])],
        axis=1,
    )


def se2_body_twist_from_path(path: np.ndarray, dt: float) -> np.ndarray:
    """Differentiate world ``x,y,yaw`` poses into body-frame physical twist."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return np.empty((0, 3), dtype=np.float64)
    delta_world = np.diff(path[:, :2], axis=0)
    yaw = path[:-1, 2]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    output = np.empty((len(path) - 1, 3), dtype=np.float64)
    output[:, 0] = (cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]) / dt
    output[:, 1] = (-sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]) / dt
    output[:, 2] = np.diff(path[:, 2]) / dt
    return output


def integrate_body_twist(initial_pose: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    """Integrate physical body twist with the same SE(2) convention as deployment."""
    twist = np.asarray(twist, dtype=np.float64)
    path = np.empty((len(twist) + 1, 3), dtype=np.float64)
    path[0] = np.asarray(initial_pose, dtype=np.float64)
    for index, (vx, vy, wz) in enumerate(twist):
        x, y, yaw = path[index]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        path[index + 1] = (
            x + dt * (vx * cos_yaw - vy * sin_yaw),
            y + dt * (vx * sin_yaw + vy * cos_yaw),
            yaw + dt * wz,
        )
    return path


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())


def physical_velocity_total_variation(twist: np.ndarray) -> float:
    if len(twist) < 2:
        return 0.0
    # Rotation is expressed in rad/s; its natural scale is kept separate but
    # weighted so a sharp yaw pulse is not ignored by the noise gate.
    delta_translation = np.linalg.norm(np.diff(twist[:, :2], axis=0), axis=1)
    delta_rotation = np.abs(np.diff(twist[:, 2])) * 0.20
    return float((delta_translation + delta_rotation).sum())


def mode_overlap_frames(twist: np.ndarray) -> int:
    translation = np.linalg.norm(twist[:, :2], axis=1) > 0.010
    rotation = np.abs(twist[:, 2]) > 0.030
    return int((translation & rotation).sum())


def peak_translation(twist: np.ndarray) -> float:
    return float(np.linalg.norm(twist[:, :2], axis=1).max(initial=0.0))


def peak_rotation(twist: np.ndarray) -> float:
    return float(np.abs(twist[:, 2]).max(initial=0.0))


def anchor_reference_pose(pose: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Robust central-pose estimate plus plateau spread diagnostics."""
    reference = np.median(pose, axis=0)
    position_spread = float(np.linalg.norm(pose[:, :2] - reference[:2], axis=1).max(initial=0.0))
    yaw_spread = float(np.abs(pose[:, 2] - reference[2]).max(initial=0.0))
    return reference, position_spread, yaw_spread


def detect_anchors(raw: RawEpisode, config: V2Config) -> list[Anchor]:
    command_speed = np.linalg.norm(raw.raw_command[:, :2], axis=1)
    physical_speed = np.linalg.norm(raw.odom_twist[:, :2], axis=1)
    stationary = (
        (command_speed <= config.command_stop_translation_mps)
        & (np.abs(raw.raw_command[:, 2]) <= config.command_stop_rotation_rps)
        & (physical_speed <= config.odom_stop_translation_mps)
        & (np.abs(raw.odom_twist[:, 2]) <= config.odom_stop_rotation_rps)
    )
    anchors: list[Anchor] = []
    for start, end in find_runs(stationary):
        if end - start < config.anchor_min_frames:
            continue
        reference, position_spread, yaw_spread = anchor_reference_pose(raw.odom_pose[start:end])
        if position_spread > config.anchor_max_pose_spread_m or yaw_spread > config.anchor_max_yaw_spread_rad:
            continue
        anchors.append(
            Anchor(
                episode_index=raw.episode_index,
                anchor_index=len(anchors),
                start_frame=start,
                end_frame=end,
                reference_pose_x_m=float(reference[0]),
                reference_pose_y_m=float(reference[1]),
                reference_pose_yaw_rad=float(reference[2]),
                max_position_spread_m=position_spread,
                max_yaw_spread_rad=yaw_spread,
            )
        )
    return anchors


def anchor_pose(anchor: Anchor) -> np.ndarray:
    return np.array(
        [anchor.reference_pose_x_m, anchor.reference_pose_y_m, anchor.reference_pose_yaw_rad],
        dtype=np.float64,
    )


def retarget_path_endpoints(raw_path: np.ndarray, start_pose: np.ndarray, end_pose: np.ndarray) -> np.ndarray:
    """Preserve the observed path shape while making both endpoints exact."""
    raw_path = np.asarray(raw_path, dtype=np.float64)
    fraction = np.linspace(0.0, 1.0, len(raw_path), dtype=np.float64)[:, None]
    start_error = np.asarray(start_pose, dtype=np.float64) - raw_path[0]
    end_error = np.asarray(end_pose, dtype=np.float64) - raw_path[-1]
    output = raw_path + (1.0 - fraction) * start_error + fraction * end_error
    output[0] = start_pose
    output[-1] = end_pose
    return output


def endpoint_smoothed_path(raw_path: np.ndarray, start_pose: np.ndarray, end_pose: np.ndarray, width: int) -> np.ndarray:
    """Smooth residual motion around the endpoint chord without moving anchors."""
    retargeted = retarget_path_endpoints(raw_path, start_pose, end_pose)
    fraction = np.linspace(0.0, 1.0, len(retargeted), dtype=np.float64)[:, None]
    chord = (1.0 - fraction) * start_pose + fraction * end_pose
    residual = retargeted - chord
    output = chord + box_smooth(residual, width)
    # The edge-padded box filter can alter residual endpoints.  Correct them
    # linearly, then set exact values to avoid numerical ambiguity.
    start_error = start_pose - output[0]
    end_error = end_pose - output[-1]
    output += (1.0 - fraction) * start_error + fraction * end_error
    output[0] = start_pose
    output[-1] = end_pose
    return output


def minjerk_fraction(count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    values = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
    return 10.0 * values**3 - 15.0 * values**4 + 6.0 * values**5


def required_minjerk_frames(distance: float, limit: float, config: V2Config) -> int:
    if distance <= 1e-8:
        return 0
    # A unit min-jerk displacement has a peak derivative of 1.875.
    return int(math.ceil(1.875 * distance / max(limit, 1e-8) / config.dt))


def distribute_spare_frames(required: list[int], total: int) -> list[int] | None:
    active = [index for index, count in enumerate(required) if count > 0]
    minimum = sum(required)
    if minimum > total:
        return None
    output = required.copy()
    if not active:
        return output
    spare = total - minimum
    weights = np.asarray([max(required[index], 1) for index in active], dtype=np.float64)
    allocation = np.floor(spare * weights / weights.sum()).astype(int)
    for index, extra in zip(active, allocation, strict=True):
        output[index] += int(extra)
    remainder = spare - int(allocation.sum())
    for index in active[:remainder]:
        output[index] += 1
    return output


def append_minjerk_phase(
    result: list[np.ndarray],
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    frames: int,
) -> None:
    if frames <= 0:
        return
    fraction = minjerk_fraction(frames)[1:, None]
    values = (1.0 - fraction) * start_pose + fraction * end_pose
    result.extend(values)


def rtr_candidate_path(start_pose: np.ndarray, end_pose: np.ndarray, frames: int, config: V2Config) -> np.ndarray | None:
    """Build a feasible physical rotate-translate-rotate candidate path.

    This candidate is intentionally optional.  It is accepted later only when
    it also stays very close to the recorded physical path.
    """
    if frames <= 0:
        return None
    displacement = end_pose[:2] - start_pose[:2]
    distance = float(np.linalg.norm(displacement))
    if distance > 1e-8:
        world_heading = math.atan2(float(displacement[1]), float(displacement[0]))
        heading = float(start_pose[2] + wrap_angle(world_heading - start_pose[2]))
    else:
        heading = float(start_pose[2])
    final_yaw = float(start_pose[2] + wrap_angle(end_pose[2] - start_pose[2]))
    first_rotation = abs(float(heading - start_pose[2]))
    final_rotation = abs(float(final_yaw - heading))
    required = [
        required_minjerk_frames(first_rotation, config.physical_rotation_limit_rps, config),
        required_minjerk_frames(distance, config.physical_speed_limit_mps, config),
        required_minjerk_frames(final_rotation, config.physical_rotation_limit_rps, config),
    ]
    allocation = distribute_spare_frames(required, frames)
    if allocation is None:
        return None
    poses: list[np.ndarray] = [np.asarray(start_pose, dtype=np.float64).copy()]
    turn_pose = np.array([start_pose[0], start_pose[1], heading], dtype=np.float64)
    translate_pose = np.array([end_pose[0], end_pose[1], heading], dtype=np.float64)
    final_pose = np.array([end_pose[0], end_pose[1], final_yaw], dtype=np.float64)
    append_minjerk_phase(poses, poses[-1], turn_pose, allocation[0])
    append_minjerk_phase(poses, poses[-1], translate_pose, allocation[1])
    append_minjerk_phase(poses, poses[-1], final_pose, allocation[2])
    if len(poses) != frames + 1:
        return None
    output = np.asarray(poses, dtype=np.float64)
    output[0] = start_pose
    output[-1] = end_pose
    return output


def command_mode_runs(modes: np.ndarray) -> list[tuple[int, int, int]]:
    """Return ``(mode, start, end)`` runs for an integer per-frame mode array."""
    modes = np.asarray(modes, dtype=np.int8)
    if len(modes) == 0:
        return []
    changes = np.flatnonzero(modes[1:] != modes[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(modes)]))
    return [
        (int(modes[start]), int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def merge_short_command_mode_runs(modes: np.ndarray, minimum_frames: int) -> np.ndarray:
    """Remove accidental one-to-three-frame mode blips without inventing motion.

    A short run is merged only when both neighbours agree. This deliberately
    leaves an ambiguous short transition untouched instead of alternating
    translation and rotation labels to force a prettier statistic.
    """
    result = np.asarray(modes, dtype=np.int8).copy()
    for _ in range(max(1, len(result))):
        changed = False
        runs = command_mode_runs(result)
        for index, (mode, start, end) in enumerate(runs):
            if end - start >= minimum_frames or index == 0 or index == len(runs) - 1:
                continue
            previous_mode = runs[index - 1][0]
            next_mode = runs[index + 1][0]
            if previous_mode == next_mode and previous_mode != mode:
                result[start:end] = previous_mode
                changed = True
                break
        if not changed:
            break
    return result


def command_mode_labels(raw_command: np.ndarray, config: V2Config) -> np.ndarray:
    """Classify recorded command timing into zero, translation, or rotation."""
    command = np.asarray(raw_command, dtype=np.float64)
    translation = np.linalg.norm(command[:, :2], axis=1)
    rotation = np.abs(command[:, 2])
    has_translation = translation > config.command_mode_translation_mps
    has_rotation = rotation > config.command_mode_rotation_rps
    modes = np.zeros(len(command), dtype=np.int8)
    modes[has_translation & ~has_rotation] = 1
    modes[has_rotation & ~has_translation] = 2
    both = has_translation & has_rotation
    # True simultaneous operator commands are rare. Choose the stronger
    # normalized intent rather than retaining an overlap in a strict candidate.
    translation_strength = translation / max(config.command_mode_translation_mps, 1e-8)
    rotation_strength = rotation / max(config.command_mode_rotation_rps, 1e-8)
    modes[both & (translation_strength >= rotation_strength)] = 1
    modes[both & (translation_strength < rotation_strength)] = 2
    return merge_short_command_mode_runs(modes, config.command_mode_min_run_frames)


def modewise_smoothed_twist(reference_path: np.ndarray, modes: np.ndarray, config: V2Config) -> np.ndarray:
    """Keep only the physical component selected by each command-mode run."""
    reference_twist = se2_body_twist_from_path(reference_path, config.dt)
    output = np.zeros_like(reference_twist)
    for mode, start, end in command_mode_runs(modes):
        if mode == 1:
            output[start:end, :2] = box_smooth(reference_twist[start:end, :2], config.smooth_window)
        elif mode == 2:
            output[start:end, 2:3] = box_smooth(reference_twist[start:end, 2:3], config.smooth_window)
    return output


def mode_correction_weights(modes: np.ndarray, desired_mode: int) -> np.ndarray:
    """Raised-cosine weights that keep endpoint corrections smooth within runs."""
    weights = np.zeros(len(modes), dtype=np.float64)
    for mode, start, end in command_mode_runs(modes):
        if mode != desired_mode:
            continue
        count = end - start
        phase = np.arange(1, count + 1, dtype=np.float64) / float(count + 1)
        # Nonzero edges retain enough authority for exact SE(2) endpoints,
        # while the central peak avoids adding a sharp correction at switches.
        weights[start:end] = 0.10 + np.sin(math.pi * phase) ** 2
    return weights


def command_mode_separated_candidate(
    reference_path: np.ndarray,
    raw_command: np.ndarray,
    config: V2Config,
) -> tuple[np.ndarray | None, np.ndarray, str]:
    """Build a locally endpoint-exact strict T/R candidate from command timing.

    Translation is allocated only to recorded translation-mode frames and yaw
    only to rotation-mode frames. Endpoint yaw is corrected first; world-XY
    residual is then distributed over translation frames using the corrected
    yaw. This is valid for the H1's holonomic base and avoids V2's unsuitable
    non-holonomic ``rotate-to-heading`` assumption.
    """
    reference_path = np.asarray(reference_path, dtype=np.float64)
    frames = len(reference_path) - 1
    if frames < 1 or len(raw_command) != frames:
        return None, np.zeros(max(frames, 0), dtype=np.int8), "invalid_block_shape"
    modes = command_mode_labels(raw_command, config)
    if not np.any(modes == 1) or not np.any(modes == 2):
        return None, modes, "missing_translation_or_rotation_mode"

    target_twist = modewise_smoothed_twist(reference_path, modes, config)
    rotation_weights = mode_correction_weights(modes, 2)
    translation_weights = mode_correction_weights(modes, 1)
    target_yaw_delta = float(reference_path[-1, 2] - reference_path[0, 2])
    current_yaw_delta = float(config.dt * target_twist[:, 2].sum())
    if rotation_weights.sum() <= 1e-10:
        return None, modes, "missing_rotation_authority"
    target_twist[:, 2] += (target_yaw_delta - current_yaw_delta) * rotation_weights / (
        config.dt * rotation_weights.sum()
    )

    yaw = np.empty(frames, dtype=np.float64)
    yaw[0] = reference_path[0, 2]
    if frames > 1:
        yaw[1:] = yaw[0] + np.cumsum(target_twist[:-1, 2]) * config.dt
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    world_steps = np.column_stack(
        (
            cos_yaw * target_twist[:, 0] - sin_yaw * target_twist[:, 1],
            sin_yaw * target_twist[:, 0] + cos_yaw * target_twist[:, 1],
        )
    ) * config.dt
    residual_world = reference_path[-1, :2] - reference_path[0, :2] - world_steps.sum(axis=0)
    if translation_weights.sum() <= 1e-10:
        return None, modes, "missing_translation_authority"
    world_correction = residual_world[None, :] * (translation_weights[:, None] / translation_weights.sum())
    target_twist[:, 0] += (cos_yaw * world_correction[:, 0] + sin_yaw * world_correction[:, 1]) / config.dt
    target_twist[:, 1] += (-sin_yaw * world_correction[:, 0] + cos_yaw * world_correction[:, 1]) / config.dt
    candidate = integrate_body_twist(reference_path[0], target_twist, config.dt)
    endpoint_position_error = float(np.linalg.norm(candidate[-1, :2] - reference_path[-1, :2]))
    endpoint_yaw_error = abs(float(candidate[-1, 2] - reference_path[-1, 2]))
    if endpoint_position_error > config.endpoint_tolerance_m or endpoint_yaw_error > config.endpoint_tolerance_rad:
        return None, modes, "endpoint_numerical_error"
    return candidate, modes, "candidate"


def separation_checkpoints(raw: RawEpisode, start: int, end: int, config: V2Config) -> list[int]:
    """Split an anchor interval at genuine low-motion checkpoints only."""
    command_speed = np.linalg.norm(raw.raw_command[:, :2], axis=1)
    physical_speed = np.linalg.norm(raw.odom_twist[:, :2], axis=1)
    stationary = (
        (command_speed <= config.command_stop_translation_mps)
        & (np.abs(raw.raw_command[:, 2]) <= config.command_stop_rotation_rps)
        & (physical_speed <= config.separation_checkpoint_translation_mps)
        & (np.abs(raw.odom_twist[:, 2]) <= config.separation_checkpoint_rotation_rps)
    )
    candidates: list[int] = []
    for run_start, run_end in find_runs(stationary):
        if run_end - run_start < config.separation_checkpoint_min_frames:
            continue
        midpoint = (run_start + run_end - 1) // 2
        if start + config.separation_min_block_frames <= midpoint <= end - config.separation_min_block_frames:
            candidates.append(midpoint)
    points = [start]
    for point in sorted(set(candidates)):
        if point - points[-1] >= config.separation_min_block_frames:
            points.append(point)
    if end - points[-1] < config.separation_min_block_frames and len(points) > 1:
        points.pop()
    points.append(end)
    return points


def separation_candidate_is_valid(
    raw_path: np.ndarray,
    reference_path: np.ndarray,
    candidate: np.ndarray,
    config: V2Config,
) -> tuple[bool, dict[str, float | int], str]:
    """Apply endpoint, physical, smoothness, and visual-timing gates to V3."""
    aligned_raw = retarget_path_endpoints(raw_path, reference_path[0], reference_path[-1])
    metrics = path_metrics(aligned_raw, candidate, config)
    reference_metrics = path_metrics(aligned_raw, reference_path, config)
    relative_xy = np.linalg.norm(candidate[:, :2] - reference_path[:, :2], axis=1)
    relative_yaw = np.abs(candidate[:, 2] - reference_path[:, 2])
    if float(np.linalg.norm(candidate[-1, :2] - reference_path[-1, :2])) > config.endpoint_tolerance_m:
        return False, metrics, "endpoint_position"
    if abs(float(candidate[-1, 2] - reference_path[-1, 2])) > config.endpoint_tolerance_rad:
        return False, metrics, "endpoint_yaw"
    if metrics["max_path_deviation_m"] > config.smooth_max_path_deviation_m:
        return False, metrics, "visual_path_alignment"
    if metrics["max_yaw_deviation_rad"] > config.smooth_max_yaw_deviation_rad:
        return False, metrics, "visual_yaw_alignment"
    if float(relative_xy.max(initial=0.0)) > config.separation_relative_max_path_deviation_m:
        return False, metrics, "relative_path_alignment"
    if float(np.quantile(relative_xy, 0.95)) > config.separation_relative_p95_path_deviation_m:
        return False, metrics, "relative_path_p95"
    if float(relative_yaw.max(initial=0.0)) > config.separation_relative_max_yaw_deviation_rad:
        return False, metrics, "relative_yaw_alignment"
    if float(np.quantile(relative_yaw, 0.95)) > config.separation_relative_p95_yaw_deviation_rad:
        return False, metrics, "relative_yaw_p95"
    if metrics["target_mode_overlap_frames"] != 0:
        return False, metrics, "target_mode_overlap"
    if metrics["target_peak_speed_mps"] > config.physical_speed_limit_mps:
        return False, metrics, "speed_limit"
    if metrics["target_peak_rotation_rps"] > config.physical_rotation_limit_rps:
        return False, metrics, "rotation_limit"
    if metrics["target_peak_speed_mps"] > max(0.010, float(metrics["raw_peak_speed_mps"]) * config.candidate_peak_tolerance):
        return False, metrics, "speed_peak"
    if metrics["target_peak_rotation_rps"] > max(0.030, float(metrics["raw_peak_rotation_rps"]) * config.candidate_peak_tolerance):
        return False, metrics, "rotation_peak"
    if metrics["target_path_length_m"] > max(0.005, float(metrics["raw_path_length_m"]) * 1.02):
        return False, metrics, "path_length_excess"
    if float(metrics["target_velocity_total_variation"]) > float(reference_metrics["target_velocity_total_variation"]) * (
        1.0 + config.separation_max_tv_increase_fraction
    ):
        return False, metrics, "velocity_variation"
    if int(reference_metrics["target_mode_overlap_frames"]) == 0:
        return False, metrics, "no_reference_overlap"
    return True, metrics, "accepted"


def secondary_smooth_candidate_is_valid(
    raw_path: np.ndarray,
    reference_path: np.ndarray,
    candidate: np.ndarray,
    config: V2Config,
) -> tuple[bool, dict[str, float | int], str]:
    """Validate a conservative second smoothing pass over an accepted V2 path.

    ``raw_path`` is the anchor-retargeted measured odom path and provides the
    original frame-synchronous visual guard. ``reference_path`` is the V2
    output.  The candidate may only make a very small additional change to V2
    while demonstrably reducing velocity total variation.  Translation and yaw
    are filtered independently in pose space; their real simultaneous timing
    is retained rather than invented from a low-level command stream.
    """
    visual_metrics = path_metrics(raw_path, candidate, config)
    reference_metrics = path_metrics(reference_path, candidate, config)
    relative_xy = np.linalg.norm(candidate[:, :2] - reference_path[:, :2], axis=1)
    relative_yaw = np.abs(candidate[:, 2] - reference_path[:, 2])
    if float(np.linalg.norm(candidate[-1, :2] - reference_path[-1, :2])) > config.endpoint_tolerance_m:
        return False, visual_metrics, "endpoint_position"
    if abs(float(candidate[-1, 2] - reference_path[-1, 2])) > config.endpoint_tolerance_rad:
        return False, visual_metrics, "endpoint_yaw"
    # The original V2 visual contract remains a hard outer guard.
    if visual_metrics["max_path_deviation_m"] > config.smooth_max_path_deviation_m:
        return False, visual_metrics, "visual_path_alignment"
    if visual_metrics["max_yaw_deviation_rad"] > config.smooth_max_yaw_deviation_rad:
        return False, visual_metrics, "visual_yaw_alignment"
    # This tighter inner guard means the second pass cannot drift visibly far
    # from the previously accepted V2 path even when the original raw odom had
    # a small endpoint-retargeting offset.
    if float(relative_xy.max(initial=0.0)) > config.secondary_relative_max_path_deviation_m:
        return False, visual_metrics, "relative_path_alignment"
    if float(np.quantile(relative_xy, 0.95)) > config.secondary_relative_p95_path_deviation_m:
        return False, visual_metrics, "relative_path_p95"
    if float(relative_yaw.max(initial=0.0)) > config.secondary_relative_max_yaw_deviation_rad:
        return False, visual_metrics, "relative_yaw_alignment"
    if float(np.quantile(relative_yaw, 0.95)) > config.secondary_relative_p95_yaw_deviation_rad:
        return False, visual_metrics, "relative_yaw_p95"
    if reference_metrics["target_peak_speed_mps"] > config.physical_speed_limit_mps:
        return False, visual_metrics, "speed_limit"
    if reference_metrics["target_peak_rotation_rps"] > config.physical_rotation_limit_rps:
        return False, visual_metrics, "rotation_limit"
    if reference_metrics["target_peak_speed_mps"] > max(
        0.010, float(reference_metrics["raw_peak_speed_mps"]) * config.candidate_peak_tolerance
    ):
        return False, visual_metrics, "speed_peak"
    if reference_metrics["target_peak_rotation_rps"] > max(
        0.030, float(reference_metrics["raw_peak_rotation_rps"]) * config.candidate_peak_tolerance
    ):
        return False, visual_metrics, "rotation_peak"
    if reference_metrics["target_path_length_m"] > float(reference_metrics["raw_path_length_m"]) + 1e-8:
        return False, visual_metrics, "path_length_increase"
    reference_tv = float(reference_metrics["raw_velocity_total_variation"])
    candidate_tv = float(reference_metrics["target_velocity_total_variation"])
    if reference_tv > 1e-8 and candidate_tv > reference_tv * (
        1.0 - config.secondary_min_tv_reduction_fraction
    ):
        return False, visual_metrics, "insufficient_noise_reduction"
    return True, visual_metrics, "accepted"


def path_metrics(raw_path: np.ndarray, target_path: np.ndarray, config: V2Config) -> dict[str, float | int]:
    raw_twist = se2_body_twist_from_path(raw_path, config.dt)
    target_twist = se2_body_twist_from_path(target_path, config.dt)
    endpoint_position_error = float(np.linalg.norm(target_path[-1, :2] - raw_path[-1, :2]))
    endpoint_yaw_error = abs(float(target_path[-1, 2] - raw_path[-1, 2]))
    return {
        "endpoint_position_error_m": endpoint_position_error,
        "endpoint_yaw_error_rad": endpoint_yaw_error,
        "max_path_deviation_m": float(np.linalg.norm(target_path[:, :2] - raw_path[:, :2], axis=1).max()),
        "max_yaw_deviation_rad": float(np.abs(target_path[:, 2] - raw_path[:, 2]).max()),
        "raw_path_length_m": path_length(raw_path),
        "target_path_length_m": path_length(target_path),
        "raw_velocity_total_variation": physical_velocity_total_variation(raw_twist),
        "target_velocity_total_variation": physical_velocity_total_variation(target_twist),
        "raw_mode_overlap_frames": mode_overlap_frames(raw_twist),
        "target_mode_overlap_frames": mode_overlap_frames(target_twist),
        "raw_peak_speed_mps": peak_translation(raw_twist),
        "target_peak_speed_mps": peak_translation(target_twist),
        "raw_peak_rotation_rps": peak_rotation(raw_twist),
        "target_peak_rotation_rps": peak_rotation(target_twist),
    }


def candidate_is_valid(
    raw_path: np.ndarray,
    candidate: np.ndarray,
    config: V2Config,
    *,
    max_path_deviation_m: float,
    max_yaw_deviation_rad: float,
    require_tv_reduction: bool,
) -> tuple[bool, dict[str, float | int], str]:
    metrics = path_metrics(raw_path, candidate, config)
    if metrics["endpoint_position_error_m"] > config.endpoint_tolerance_m:
        return False, metrics, "endpoint_position"
    if metrics["endpoint_yaw_error_rad"] > config.endpoint_tolerance_rad:
        return False, metrics, "endpoint_yaw"
    if metrics["max_path_deviation_m"] > max_path_deviation_m:
        return False, metrics, "path_alignment"
    if metrics["max_yaw_deviation_rad"] > max_yaw_deviation_rad:
        return False, metrics, "yaw_alignment"
    if metrics["target_peak_speed_mps"] > config.physical_speed_limit_mps:
        return False, metrics, "speed_limit"
    if metrics["target_peak_rotation_rps"] > config.physical_rotation_limit_rps:
        return False, metrics, "rotation_limit"
    # A smoothing candidate should not create a faster spike than the observed
    # trajectory, even when it is below the global safety limit.
    if metrics["target_peak_speed_mps"] > max(
        0.010, float(metrics["raw_peak_speed_mps"]) * config.candidate_peak_tolerance
    ):
        return False, metrics, "speed_peak"
    if metrics["target_peak_rotation_rps"] > max(
        0.030, float(metrics["raw_peak_rotation_rps"]) * config.candidate_peak_tolerance
    ):
        return False, metrics, "rotation_peak"
    if require_tv_reduction:
        raw_tv = float(metrics["raw_velocity_total_variation"])
        target_tv = float(metrics["target_velocity_total_variation"])
        if raw_tv > 1e-8 and target_tv > raw_tv * (1.0 - config.min_tv_reduction_fraction):
            return False, metrics, "insufficient_noise_reduction"
    return True, metrics, "accepted"


def audit_from_metrics(
    *,
    episode_index: int,
    segment_index: int,
    start_frame: int,
    end_frame: int,
    strategy: str,
    accepted: bool,
    raw_path: np.ndarray,
    target_path: np.ndarray,
    metrics: dict[str, float | int],
    reference_velocity_total_variation: float | None = None,
) -> SegmentAudit:
    return SegmentAudit(
        episode_index=episode_index,
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        num_action_frames=end_frame - start_frame,
        strategy=strategy,
        accepted=accepted,
        raw_end_x_m=float(raw_path[-1, 0]),
        raw_end_y_m=float(raw_path[-1, 1]),
        raw_end_yaw_rad=float(raw_path[-1, 2]),
        target_end_x_m=float(target_path[-1, 0]),
        target_end_y_m=float(target_path[-1, 1]),
        target_end_yaw_rad=float(target_path[-1, 2]),
        endpoint_position_error_m=float(metrics["endpoint_position_error_m"]),
        endpoint_yaw_error_rad=float(metrics["endpoint_yaw_error_rad"]),
        max_path_deviation_m=float(metrics["max_path_deviation_m"]),
        max_yaw_deviation_rad=float(metrics["max_yaw_deviation_rad"]),
        raw_path_length_m=float(metrics["raw_path_length_m"]),
        target_path_length_m=float(metrics["target_path_length_m"]),
        raw_velocity_total_variation=float(metrics["raw_velocity_total_variation"]),
        reference_velocity_total_variation=(
            float(metrics["target_velocity_total_variation"])
            if reference_velocity_total_variation is None
            else float(reference_velocity_total_variation)
        ),
        target_velocity_total_variation=float(metrics["target_velocity_total_variation"]),
        raw_mode_overlap_frames=int(metrics["raw_mode_overlap_frames"]),
        target_mode_overlap_frames=int(metrics["target_mode_overlap_frames"]),
        raw_peak_speed_mps=float(metrics["raw_peak_speed_mps"]),
        target_peak_speed_mps=float(metrics["target_peak_speed_mps"]),
        raw_peak_rotation_rps=float(metrics["raw_peak_rotation_rps"]),
        target_peak_rotation_rps=float(metrics["target_peak_rotation_rps"]),
    )


def reconstruct_target_episode(
    raw: RawEpisode,
    anchors: list[Anchor],
    config: V2Config,
) -> tuple[np.ndarray, np.ndarray, list[SegmentAudit]]:
    """Build a globally continuous, anchor-exact target physical trajectory."""
    raw_pose = raw.odom_pose
    frame_count = len(raw_pose)
    target_pose = raw_pose.copy()
    audits: list[SegmentAudit] = []
    if len(anchors) < 2:
        # No trustworthy pair means do not invent a physical path.  The raw
        # physical pose is still used below, so V2 label semantics remain
        # coherent and a later audit can explain the lack of reconstruction.
        target_twist = se2_body_twist_from_path(target_pose, config.dt)
        return target_pose, np.vstack((target_twist, np.zeros((1, 3)))), audits

    # Set every confirmed plateau to its robust physical reference.  Since
    # plateau pose spread is sub-millimetric, this removes stop jitter without
    # changing meaningful visual motion.
    for anchor in anchors:
        target_pose[anchor.start_frame : anchor.end_frame] = anchor_pose(anchor)

    for segment_index, (left, right) in enumerate(zip(anchors[:-1], anchors[1:], strict=True)):
        # Start at the final stationary frame so the first target transition is
        # temporally aligned with the command that exits the plateau.  The end
        # is the first stationary frame of the following plateau.
        start = left.end_frame - 1
        end = right.start_frame
        if end <= start:
            continue
        start_pose = anchor_pose(left)
        end_pose = anchor_pose(right)
        raw_path = raw_pose[start : end + 1]
        retained = retarget_path_endpoints(raw_path, start_pose, end_pose)
        chosen = retained
        strategy = "retain_observed_physical_path"
        accepted = False

        smooth_candidate = endpoint_smoothed_path(raw_path, start_pose, end_pose, config.smooth_window)
        smooth_ok, smooth_metrics, smooth_reason = candidate_is_valid(
            retained,
            smooth_candidate,
            config,
            max_path_deviation_m=config.smooth_max_path_deviation_m,
            max_yaw_deviation_rad=config.smooth_max_yaw_deviation_rad,
            require_tv_reduction=True,
        )
        rtr_candidate = rtr_candidate_path(start_pose, end_pose, end - start, config)
        rtr_ok = False
        rtr_metrics: dict[str, float | int] | None = None
        rtr_reason = "infeasible_time_or_speed"
        if rtr_candidate is not None:
            rtr_ok, rtr_metrics, rtr_reason = candidate_is_valid(
                retained,
                rtr_candidate,
                config,
                max_path_deviation_m=config.rtr_max_path_deviation_m,
                max_yaw_deviation_rad=config.rtr_max_yaw_deviation_rad,
                require_tv_reduction=False,
            )

        # R-T-R gets priority only when it preserves frame-wise visual
        # alignment under its much tighter geometric gate.
        if rtr_ok and rtr_candidate is not None:
            chosen = rtr_candidate
            strategy = "accepted_rotate_translate_rotate"
            accepted = True
            metrics = rtr_metrics
        elif smooth_ok:
            chosen = smooth_candidate
            strategy = "accepted_endpoint_smooth"
            accepted = True
            metrics = smooth_metrics
        else:
            metrics = path_metrics(retained, retained, config)
            # Keep rejection reason in the strategy so the audit explains why
            # a block did not receive a virtual reconstruction.
            if rtr_candidate is None:
                strategy = f"retain_observed_physical_path_rtr_{rtr_reason}_smooth_{smooth_reason}"
            else:
                strategy = f"retain_observed_physical_path_rtr_{rtr_reason}_smooth_{smooth_reason}"

        secondary_reference_tv: float | None = None
        if config.secondary_smooth_window:
            v2_reference = chosen
            secondary_reference_tv = physical_velocity_total_variation(
                se2_body_twist_from_path(v2_reference, config.dt)
            )
            secondary_candidate = endpoint_smoothed_path(
                v2_reference,
                start_pose,
                end_pose,
                config.secondary_smooth_window,
            )
            secondary_ok, secondary_metrics, secondary_reason = secondary_smooth_candidate_is_valid(
                retained,
                v2_reference,
                secondary_candidate,
                config,
            )
            if secondary_ok:
                chosen = secondary_candidate
                strategy = f"accepted_secondary_endpoint_smooth_from_{strategy}"
                metrics = secondary_metrics
                # In V3 the accepted flag means the *additional* de-jitter
                # pass was accepted; V2's own status remains in the strategy.
                accepted = True
            else:
                metrics = path_metrics(retained, chosen, config)
                strategy = f"retain_v2_{strategy}_secondary_{secondary_reason}"
                accepted = False

        if not config.enable_command_mode_separation:
            # Keep the established V2 output exactly unchanged unless V3 has
            # been explicitly requested.  This is important because these
            # labels have a different deployment bridge requirement from the
            # raw command data.
            target_pose[start : end + 1] = chosen
            audits.append(
                audit_from_metrics(
                    episode_index=raw.episode_index,
                    segment_index=segment_index,
                    start_frame=start,
                    end_frame=end,
                    strategy=strategy,
                    accepted=accepted,
                    raw_path=retained,
                    target_path=chosen,
                    metrics=metrics,
                    reference_velocity_total_variation=secondary_reference_tv,
                )
            )
            continue

        # V3 is deliberately local: checkpoint boundaries are genuine
        # stationary intervals, and each strict T/R candidate must remain
        # visually close to this V2 physical reference.  A rejected block is
        # simply left as V2; we never force command timing onto a visually
        # inconsistent part of the recording just to increase the separation
        # rate.
        checkpoints = separation_checkpoints(raw, start, end, config)
        for block_index, (block_start, block_end) in enumerate(
            zip(checkpoints[:-1], checkpoints[1:], strict=True)
        ):
            local_start = block_start - start
            local_end = block_end - start
            reference_block = chosen[local_start : local_end + 1].copy()
            raw_block = raw_pose[block_start : block_end + 1]
            aligned_raw_block = retarget_path_endpoints(
                raw_block, reference_block[0], reference_block[-1]
            )
            audit_index = segment_index * 10000 + block_index

            if block_end - block_start < config.separation_min_block_frames:
                block_metrics = path_metrics(aligned_raw_block, reference_block, config)
                audits.append(
                    audit_from_metrics(
                        episode_index=raw.episode_index,
                        segment_index=audit_index,
                        start_frame=block_start,
                        end_frame=block_end,
                        strategy=f"fallback_v2_{strategy}_block_too_short",
                        accepted=False,
                        raw_path=aligned_raw_block,
                        target_path=reference_block,
                        metrics=block_metrics,
                    )
                )
                continue

            candidate, _modes, candidate_reason = command_mode_separated_candidate(
                reference_block,
                raw.raw_command[block_start:block_end],
                config,
            )
            if candidate is None:
                block_metrics = path_metrics(aligned_raw_block, reference_block, config)
                block_strategy = f"fallback_v2_{strategy}_mode_{candidate_reason}"
                block_target = reference_block
                block_accepted = False
            else:
                block_accepted, block_metrics, validation_reason = separation_candidate_is_valid(
                    raw_block,
                    reference_block,
                    candidate,
                    config,
                )
                if block_accepted:
                    block_target = candidate
                    block_strategy = "accepted_command_mode_separated"
                    chosen[local_start : local_end + 1] = candidate
                else:
                    block_target = reference_block
                    block_strategy = f"fallback_v2_{strategy}_mode_{validation_reason}"

            audits.append(
                audit_from_metrics(
                    episode_index=raw.episode_index,
                    segment_index=audit_index,
                    start_frame=block_start,
                    end_frame=block_end,
                    strategy=block_strategy,
                    accepted=block_accepted,
                    raw_path=aligned_raw_block,
                    target_path=block_target,
                    metrics=block_metrics,
                )
            )

        target_pose[start : end + 1] = chosen

    target_twist = se2_body_twist_from_path(target_pose, config.dt)
    target_twist = np.vstack((target_twist, np.zeros((1, 3), dtype=np.float64)))
    # Anchor plateau labels must be exactly zero, including their final frame;
    # the planned segment begins at ``anchor.end - 1`` and overwrites the
    # boundary transition only when a genuine movement follows.
    for anchor in anchors:
        target_twist[anchor.start_frame : anchor.end_frame] = 0.0
    # A planned segment can begin at ``left.end - 1``. Restore that boundary
    # velocity after zeroing its plateau, so the global path stays consistent.
    for left, right in zip(anchors[:-1], anchors[1:], strict=True):
        boundary = left.end_frame - 1
        if boundary < right.start_frame:
            target_twist[boundary] = se2_body_twist_from_path(
                target_pose[boundary : boundary + 2], config.dt
            )[0]
    return target_pose, target_twist, audits


def vector_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"expected a 2D array with >=2 rows, got {values.shape}")
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.asarray([len(values)], dtype=np.int64),
        "q01": quantiles[0],
        "q10": quantiles[1],
        "q50": quantiles[2],
        "q90": quantiles[3],
        "q99": quantiles[4],
    }


def json_stats(stats: dict[str, np.ndarray]) -> dict[str, list[float] | list[int]]:
    return {key: value.tolist() for key, value in stats.items()}


def replace_action_in_table(table: pa.Table, actions: np.ndarray) -> pa.Table:
    if actions.shape != (len(table), ACTION_DIM):
        raise ValueError(f"action shape {actions.shape} does not match source rows={len(table)}")
    index = table.schema.get_field_index("action")
    if index < 0:
        raise KeyError("source parquet has no action column")
    # Preserve exact Arrow field metadata/child name for LeRobot compatibility.
    return table.set_column(index, table.schema.field(index), fixed_list_array(actions))


def write_episode_row_groups(table: pa.Table, destination: Path, bounds: list[tuple[int, int, int]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(destination, table.schema, compression="snappy") as writer:
        for _, start, end in bounds:
            writer.write_table(table.slice(start, end - start))


def update_episode_action_stats(episodes_path: Path, episode_stats: dict[int, dict[str, np.ndarray]]) -> None:
    table = pq.read_table(episodes_path)
    episode_ids = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    arrays: list[pa.Array | pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for field in table.schema:
        if not field.name.startswith("stats/action/"):
            arrays.append(table[field.name])
            fields.append(field)
            continue
        stat_key = field.name.rsplit("/", maxsplit=1)[-1]
        values = table[field.name].to_pylist()
        for row, episode in enumerate(episode_ids):
            values[row] = episode_stats[int(episode)][stat_key].tolist()
        arrays.append(pa.array(values, type=field.type))
        fields.append(field)
    updated = pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=table.schema.metadata))
    pq.write_table(updated, episodes_path, compression="snappy")


def validate_episode_action_stats(episodes_path: Path, episode_stats: dict[int, dict[str, np.ndarray]]) -> None:
    table = pq.read_table(episodes_path)
    episode_ids = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    for field in table.schema:
        if not field.name.startswith("stats/action/"):
            continue
        stat_key = field.name.rsplit("/", maxsplit=1)[-1]
        values = table[field.name].to_pylist()
        for row, episode in enumerate(episode_ids):
            actual = np.asarray(values[row], dtype=np.float64)
            expected = np.asarray(episode_stats[int(episode)][stat_key], dtype=np.float64)
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=1e-7):
                raise AssertionError(f"per-episode action stats mismatch episode={episode} stat={stat_key}")


def copy_tree_with_video_mode(source: Path, destination: Path, video_mode: str) -> None:
    """Copy mutable metadata/data while linking only immutable source MP4s."""
    for child in source.iterdir():
        target = destination / child.name
        if child.name == "videos":
            for file in sorted(child.rglob("*")):
                relative = file.relative_to(child)
                target_file = target / relative
                if file.is_dir():
                    target_file.mkdir(parents=True, exist_ok=True)
                    continue
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if video_mode == "hardlink":
                    os.link(file, target_file)
                else:
                    shutil.copy2(file, target_file)
        elif child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def fit_feedback_arx(raw_episodes: list[RawEpisode]) -> dict[str, Any]:
    """Fit a one-step physical velocity model for the future deployment mapper.

    The model is ``v[t+1] = bias + state_matrix @ v[t] + command_matrix @ u[t]``.
    It is not used to assert endpoint correctness during conversion; it is only
    exported so a deployment bridge can turn a physical desired velocity into
    a feedback-corrected low-level command.
    """
    def rows(episodes: Iterable[RawEpisode]) -> tuple[np.ndarray, np.ndarray]:
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for episode in episodes:
            if len(episode.odom_twist) < 2:
                continue
            features.append(
                np.column_stack(
                    (
                        np.ones(len(episode.odom_twist) - 1),
                        episode.odom_twist[:-1],
                        episode.raw_command[:-1],
                    )
                )
            )
            targets.append(episode.odom_twist[1:])
        return np.concatenate(features), np.concatenate(targets)

    train_episodes = [episode for episode in raw_episodes if episode.episode_index % 5 != 4]
    heldout_episodes = [episode for episode in raw_episodes if episode.episode_index % 5 == 4]
    x_train, y_train = rows(train_episodes)
    coefficient, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    x_test, y_test = rows(heldout_episodes)
    prediction = x_test @ coefficient
    residual = y_test - prediction
    denominator = ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - (residual**2).sum(axis=0) / np.maximum(denominator, 1e-12)
    full_x, full_y = rows(raw_episodes)
    full_coefficient, *_ = np.linalg.lstsq(full_x, full_y, rcond=None)
    bias = full_coefficient[0]
    state_matrix = full_coefficient[1:4].T
    command_matrix = full_coefficient[4:7].T
    # Some recordings never independently excite every base direction (for
    # example, a teleoperator may never command lateral velocity).  The ARX
    # least-squares fit remains valid, but its 3x3 command block is then rank
    # deficient and has no ordinary inverse.  This mapper is deployment
    # metadata only: use a Moore--Penrose inverse in that case so observed
    # directions retain their least-squares correction while unobservable
    # directions are not fabricated as arbitrarily large commands.
    singular_values = np.linalg.svd(command_matrix, compute_uv=False)
    condition_number = float(np.linalg.cond(command_matrix))
    effective_rank = int(np.linalg.matrix_rank(command_matrix))
    if effective_rank == command_matrix.shape[0] and np.isfinite(condition_number) and condition_number < 1e8:
        command_inverse = np.linalg.inv(command_matrix)
        inverse_method = "ordinary_inverse"
    else:
        command_inverse = np.linalg.pinv(command_matrix, rcond=1e-8)
        inverse_method = "moore_penrose_pseudoinverse"
    return {
        "model": "v_next = bias + state_matrix @ v_current + command_matrix @ u_command",
        "bias": bias.tolist(),
        "state_matrix": state_matrix.tolist(),
        "command_matrix": command_matrix.tolist(),
        "command_matrix_inverse": command_inverse.tolist(),
        "command_matrix_condition_number": condition_number,
        "command_matrix_effective_rank": effective_rank,
        "command_matrix_singular_values": singular_values.tolist(),
        "command_matrix_inverse_method": inverse_method,
        "heldout_episode_rule": "episode_index % 5 == 4",
        "heldout_one_step_mae": np.abs(residual).mean(axis=0).tolist(),
        "heldout_one_step_r2": r2.tolist(),
        "fitted_samples": int(len(full_y)),
        "heldout_samples": int(len(y_test)),
        "recommended_feedback_gain": 0.50,
    }


def validate_derived_arrays(
    source_actions: np.ndarray,
    target_actions: np.ndarray,
    source_table: pa.Table,
    output_table: pa.Table,
) -> None:
    if not source_table.schema.equals(output_table.schema, check_metadata=True):
        raise AssertionError("derived parquet schema or metadata changed")
    if not np.array_equal(source_actions[:, :20], target_actions[:, :20]):
        raise AssertionError("non-base action labels changed")
    for name in source_table.column_names:
        if name != "action" and not source_table[name].equals(output_table[name]):
            raise AssertionError(f"non-action column changed: {name}")
    written_actions = fixed_list_to_numpy(output_table, "action", ACTION_DIM)
    if not np.array_equal(written_actions, target_actions):
        raise AssertionError("written action column differs from target action")
    if not np.isfinite(written_actions).all():
        raise AssertionError("derived action contains non-finite values")


def validate_target_trajectory(
    target_pose: np.ndarray,
    target_twist: np.ndarray,
    anchors: list[Anchor],
    audits: list[SegmentAudit],
    config: V2Config,
) -> None:
    """Assert the derived label integrates back to every target anchor exactly."""
    reconstructed = integrate_body_twist(target_pose[0], target_twist[:-1], config.dt)
    position_error = np.linalg.norm(reconstructed[:, :2] - target_pose[:, :2], axis=1)
    yaw_error = np.abs(reconstructed[:, 2] - target_pose[:, 2])
    if float(position_error.max()) > config.endpoint_tolerance_m or float(yaw_error.max()) > config.endpoint_tolerance_rad:
        raise AssertionError(
            "target physical twist does not reconstruct its target pose: "
            f"max_position_error={position_error.max():.8g}, max_yaw_error={yaw_error.max():.8g}"
        )
    for anchor in anchors:
        reference = anchor_pose(anchor)
        observed = target_pose[anchor.start_frame : anchor.end_frame]
        if not np.allclose(observed, reference, rtol=0.0, atol=config.endpoint_tolerance_m):
            raise AssertionError(f"target anchor plateau moved episode={anchor.episode_index} anchor={anchor.anchor_index}")
    for audit in audits:
        start = audit.start_frame
        end = audit.end_frame
        path = integrate_body_twist(target_pose[start], target_twist[start:end], config.dt)
        position_delta = float(np.linalg.norm(path[-1, :2] - target_pose[end, :2]))
        yaw_delta = abs(float(path[-1, 2] - target_pose[end, 2]))
        if position_delta > config.endpoint_tolerance_m or yaw_delta > config.endpoint_tolerance_rad:
            raise AssertionError(
                f"segment endpoint failed episode={audit.episode_index} segment={audit.segment_index}: "
                f"{position_delta:.8g}m {yaw_delta:.8g}rad"
            )


def write_provenance(
    *,
    destination: Path,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    global_indices: np.ndarray,
    raw_command: np.ndarray,
    raw_odom_twist: np.ndarray,
    target_physical_twist: np.ndarray,
    raw_odom_pose: np.ndarray,
    target_pose: np.ndarray,
    anchors: list[Anchor],
    audits: list[SegmentAudit],
    dynamics: dict[str, Any],
    config: V2Config,
    source: Path,
) -> None:
    meta = destination / "meta" / config.metadata_dir_name
    meta.mkdir(parents=True, exist_ok=True)
    provenance = pa.table(
        {
            "episode_index": pa.array(episode_indices.astype(np.int64)),
            "frame_index": pa.array(frame_indices.astype(np.int64)),
            "index": pa.array(global_indices.astype(np.int64)),
            "raw_base_command": fixed_list_array(raw_command),
            "raw_odom_twist": fixed_list_array(raw_odom_twist),
            "target_base_physical_twist": fixed_list_array(target_physical_twist),
            "raw_odom_pose_xyyaw": fixed_list_array(raw_odom_pose),
            "target_odom_pose_xyyaw": fixed_list_array(target_pose),
        }
    )
    pq.write_table(provenance, meta / "action_provenance.parquet", compression="snappy")
    if anchors:
        pq.write_table(pa.Table.from_pylist([asdict(anchor) for anchor in anchors]), meta / "pose_anchors.parquet", compression="snappy")
    if audits:
        pq.write_table(pa.Table.from_pylist([asdict(audit) for audit in audits]), meta / "segment_audit.parquet", compression="snappy")
    (meta / "dynamics_feedback_mapper.json").write_text(
        json.dumps(dynamics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": config.dataset_version,
        "source_dataset": str(source.resolve()),
        "action_semantics": {
            "training_action": "action[0:20] unchanged; action[20:23] is desired physical odom body twist, not raw twist/cmd",
            "raw_command": "preserved in action_provenance.parquet/raw_base_command",
            "raw_physical_motion": "preserved in action_provenance.parquet/raw_odom_twist and raw_odom_pose_xyyaw",
            "target_pose": "stored as target_odom_pose_xyyaw and exactly reconstructed by target_base_physical_twist",
            "deployment_requirement": "use dynamics_feedback_mapper.json with an odom-feedback physical_desired bridge mode; do not send this action tail directly as twist/cmd",
        },
        "guarantees": {
            "target_anchor_endpoints": "numerically checked by physical SE(2) integration",
            "visual_alignment": (
                "the accepted V2 physical path receives a second endpoint-constrained XY/yaw smoothing pass "
                "only when it remains inside both raw-path and V2-relative visual gates; rejected segments "
                "preserve V2 exactly"
                if config.secondary_smooth_window
                else (
                    "only endpoint-smooth/R-T-R candidates that pass frame-synchronous path/yaw gates are accepted; "
                    "other blocks preserve observed physical path"
                    if not config.enable_command_mode_separation
                    else "only local command-mode translation/rotation candidates that pass strict frame-synchronous "
                    "path/yaw gates are accepted; every rejected block preserves the V2 physical path"
                )
            ),
            "real_robot_endpoint": "requires the exported odom-feedback mapper; an offline open-loop model is not represented as a guarantee",
        },
        "config": asdict(config),
        "anchor_count": len(anchors),
        "segment_count": len(audits),
        "accepted_replan_segment_count": int(sum(audit.accepted for audit in audits)),
        "video_reuse": "hardlink or copy; no re-encode",
    }
    (meta / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def line_plot(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    box: tuple[int, int, int, int],
    color: str,
    scale: tuple[float, float],
) -> None:
    x0, y0, x1, y1 = box
    low, high = scale
    if high <= low:
        high = low + 1.0
    points = []
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        x = x0 + (x1 - x0) * index / max(1, len(values) - 1)
        y = y1 - (y1 - y0) * (value - low) / (high - low)
        points.append((int(round(x)), int(round(y))))
    if len(points) >= 2:
        draw.line(points, fill=color, width=2)


def write_preview(
    analysis_dir: Path,
    raw: RawEpisode,
    target_pose: np.ndarray,
    target_twist: np.ndarray,
    anchors: list[Anchor],
    audits: list[SegmentAudit],
    *,
    dataset_version: str,
    artifact_label: str,
) -> None:
    """Write a compact physical-path inspection artifact without matplotlib."""
    analysis_dir.mkdir(parents=True, exist_ok=True)
    episode = raw.episode_index
    np.savez_compressed(
        analysis_dir / f"episode_{episode:02d}_{artifact_label}_preview.npz",
        raw_odom_pose=raw.odom_pose,
        raw_odom_twist=raw.odom_twist,
        raw_command=raw.raw_command,
        target_odom_pose=target_pose,
        target_physical_twist=target_twist,
    )
    (analysis_dir / f"episode_{episode:02d}_{artifact_label}_audit.json").write_text(
        json.dumps(
            {
                "bag_path": raw.bag_path,
                "anchors": [asdict(anchor) for anchor in anchors],
                "segments": [asdict(audit) for audit in audits],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    width, height = 1800, 1420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (30, 20),
        f"Robot8 {dataset_version} | episode {episode:02d} | raw=gray target=blue",
        fill="#17202a",
        font=font,
    )
    chart_left, chart_right = 125, 1740
    chart_height = 190
    names = ("physical vx (m/s)", "physical vy (m/s)", "physical wz (rad/s)")
    for row, name in enumerate(names):
        top = 70 + row * 220
        bottom = top + chart_height
        raw_values = raw.odom_twist[:, row]
        target_values = target_twist[:, row]
        maximum = float(max(np.abs(raw_values).max(initial=0.0), np.abs(target_values).max(initial=0.0)))
        margin = max(0.005, maximum * 0.08)
        low, high = -maximum - margin, maximum + margin
        zero = bottom - (bottom - top) * (0.0 - low) / (high - low)
        draw.rectangle((chart_left, top, chart_right, bottom), outline="#b8c2cc", width=1)
        draw.line((chart_left, int(zero), chart_right, int(zero)), fill="#d5dce3", width=1)
        draw.text((30, top + 8), name, fill="#17202a", font=font)
        line_plot(draw, raw_values, (chart_left, top, chart_right, bottom), "#40464d", (low, high))
        line_plot(draw, target_values, (chart_left, top, chart_right, bottom), "#1769e0", (low, high))

    path_top, path_bottom = 780, 1370
    path_left, path_right = 125, 1740
    combined = np.concatenate((raw.odom_pose[:, :2], target_pose[:, :2]), axis=0)
    low_xy = combined.min(axis=0)
    high_xy = combined.max(axis=0)
    span = np.maximum(high_xy - low_xy, 0.05)
    low_xy -= 0.08 * span
    high_xy += 0.08 * span
    draw.rectangle((path_left, path_top, path_right, path_bottom), outline="#b8c2cc", width=1)
    draw.text((30, path_top + 8), "physical odom XY path", fill="#17202a", font=font)

    def points(path: np.ndarray) -> list[tuple[int, int]]:
        return [
            (
                int(round(path_left + (path_right - path_left) * (x - low_xy[0]) / (high_xy[0] - low_xy[0]))),
                int(round(path_bottom - (path_bottom - path_top) * (y - low_xy[1]) / (high_xy[1] - low_xy[1]))),
            )
            for x, y in path[:, :2]
        ]

    raw_points = points(raw.odom_pose)
    target_points = points(target_pose)
    draw.line(raw_points, fill="#40464d", width=3)
    draw.line(target_points, fill="#1769e0", width=3)
    for anchor in anchors:
        point = target_points[anchor.start_frame]
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="#d62728")
    accepted = sum(audit.accepted for audit in audits)
    draw.text((125, 1384), f"anchors={len(anchors)} | accepted path replans={accepted}/{len(audits)} | target endpoints are SE(2)-integrated", fill="#17202a", font=font)
    image.save(analysis_dir / f"episode_{episode:02d}_{artifact_label}_preview.png")


def validate_written_target_trajectory(
    written_target_twist: np.ndarray,
    target_pose: np.ndarray,
    config: V2Config,
) -> None:
    """Validate physical endpoints after float32 parquet serialization."""
    reconstructed = integrate_body_twist(target_pose[0], written_target_twist[:-1], config.dt)
    position_error = float(np.linalg.norm(reconstructed[:, :2] - target_pose[:, :2], axis=1).max())
    yaw_error = float(np.abs(reconstructed[:, 2] - target_pose[:, 2]).max())
    if position_error > 5e-5 or yaw_error > 5e-5:
        raise AssertionError(
            f"float32 target integration error too large: {position_error:.8g}m {yaw_error:.8g}rad"
        )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    source = args.source_dataset.resolve()
    destination = args.output_dataset.resolve()
    bag_data = args.bag_data_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    if not (source / "data").is_dir() or not (source / "meta").is_dir() or not (source / "videos").is_dir():
        raise FileNotFoundError(f"not a complete LeRobot source dataset: {source}")
    if not bag_data.is_dir():
        raise FileNotFoundError(f"raw bag directory not found: {bag_data}")
    if destination == source:
        raise SystemExit("--output-dataset must differ from --source-dataset")
    if destination.exists() and not args.dry_run:
        raise SystemExit(f"refusing to overwrite existing output dataset: {destination}")
    if args.episodes is not None and not args.dry_run:
        raise SystemExit("V2 has a single physical action semantic; subset output is unsafe. Use --dry-run for subsets.")

    data_paths = sorted((source / "data").rglob("*.parquet"))
    if len(data_paths) != 1:
        raise RuntimeError(f"expected exactly one source data parquet, found {len(data_paths)}")
    source_data_path = data_paths[0]
    source_table = pq.read_table(source_data_path)
    source_actions = fixed_list_to_numpy(source_table, "action", ACTION_DIM)
    source_states = fixed_list_to_numpy(source_table, "observation.state", ACTION_DIM)
    episode_indices = np.asarray(source_table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    frame_indices = np.asarray(source_table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    global_indices = np.asarray(source_table["index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    bounds = contiguous_episode_bounds(episode_indices)
    available_episodes = {episode for episode, _, _ in bounds}
    selected = set(args.episodes) if args.episodes is not None else available_episodes
    unknown = selected - available_episodes
    if unknown:
        raise SystemExit(f"unknown --episodes values: {sorted(unknown)}")

    converter = load_converter_module()
    excluded = {
        "rosbag2_2026_07_21_15_09_46",
        "rosbag2_2026_07_21_16_11_11",
    }
    bag_paths = converter.collect_bag_paths(bag_data, exclude_bags=excluded)
    if len(bag_paths) != len(bounds):
        raise RuntimeError(
            f"converter bag count ({len(bag_paths)}) does not match source episodes ({len(bounds)})"
        )

    print(f"Source: {source}", flush=True)
    print(f"Bags:   {bag_data}", flush=True)
    print(
        f"Mode:   {'dry-run' if args.dry_run else f'create derived {config.dataset_version} dataset'}",
        flush=True,
    )
    print(f"Frames: {len(source_table)}, episodes: {len(bounds)}", flush=True)
    started = time.monotonic()

    raw_episodes: list[RawEpisode] = []
    for bag_path, (episode, start, end) in zip(bag_paths, bounds, strict=True):
        print(f"[odom {episode:02d}/{len(bounds) - 1:02d}] {bag_path}", flush=True)
        raw = read_raw_episode(converter=converter, bag_path=bag_path, episode_index=episode)
        expected_frames = end - start
        if len(raw.odom_pose) != expected_frames:
            raise AssertionError(
                f"episode={episode} bag samples={len(raw.odom_pose)} but source frames={expected_frames}"
            )
        command_error = float(np.abs(raw.raw_command - source_actions[start:end, BASE_SLICE]).max())
        state_error = float(np.abs(raw.odom_twist - source_states[start:end, BASE_SLICE]).max())
        if command_error > config.alignment_tolerance or state_error > config.alignment_tolerance:
            raise AssertionError(
                f"raw/source alignment failed episode={episode}: command={command_error:.8g}, state={state_error:.8g}"
            )
        raw_episodes.append(raw)

    target_actions = source_actions.copy()
    all_raw_command: list[np.ndarray] = []
    all_raw_twist: list[np.ndarray] = []
    all_raw_pose: list[np.ndarray] = []
    all_target_twist: list[np.ndarray] = []
    all_target_pose: list[np.ndarray] = []
    all_anchors: list[Anchor] = []
    all_audits: list[SegmentAudit] = []
    target_by_episode: dict[int, tuple[np.ndarray, np.ndarray, list[Anchor], list[SegmentAudit]]] = {}
    episode_action_stats: dict[int, dict[str, np.ndarray]] = {}

    for raw, (episode, start, end) in zip(raw_episodes, bounds, strict=True):
        anchors = detect_anchors(raw, config)
        if episode in selected:
            target_pose, target_twist, audits = reconstruct_target_episode(raw, anchors, config)
        else:
            target_pose = raw.odom_pose.copy()
            target_twist = np.vstack(
                (se2_body_twist_from_path(target_pose, config.dt), np.zeros((1, 3), dtype=np.float64))
            )
            audits = []
        validate_target_trajectory(target_pose, target_twist, anchors, audits, config)
        target_actions[start:end, BASE_SLICE] = target_twist.astype(np.float32)
        episode_action_stats[episode] = vector_stats(target_actions[start:end])
        all_raw_command.append(raw.raw_command)
        all_raw_twist.append(raw.odom_twist)
        all_raw_pose.append(raw.odom_pose)
        all_target_twist.append(target_twist)
        all_target_pose.append(target_pose)
        all_anchors.extend(anchors)
        all_audits.extend(audits)
        target_by_episode[episode] = (target_pose, target_twist, anchors, audits)

    raw_command = np.concatenate(all_raw_command)
    raw_odom_twist = np.concatenate(all_raw_twist)
    raw_odom_pose = np.concatenate(all_raw_pose)
    target_twist = np.concatenate(all_target_twist)
    target_pose = np.concatenate(all_target_pose)
    if len(target_twist) != len(source_table) or len(target_pose) != len(source_table):
        raise AssertionError("raw episode concatenation does not match source rows")
    if not np.array_equal(source_actions[:, :20], target_actions[:, :20]):
        raise AssertionError("base conversion changed upper-body labels")
    if not np.isfinite(target_actions).all():
        raise AssertionError("target actions are non-finite")

    dynamics = fit_feedback_arx(raw_episodes)
    changed_vs_raw_command = np.any(
        np.abs(target_actions[:, BASE_SLICE].astype(np.float64) - source_actions[:, BASE_SLICE].astype(np.float64)) > 1e-7,
        axis=1,
    )
    physical_replans = sum(audit.accepted for audit in all_audits)
    strategies = {
        strategy: sum(audit.strategy == strategy for audit in all_audits)
        for strategy in sorted({audit.strategy for audit in all_audits})
    }
    summary: dict[str, Any] = {
        "version": config.dataset_version,
        "source_dataset": str(source),
        "output_dataset": str(destination),
        "dry_run": bool(args.dry_run),
        "frames": len(source_table),
        "episodes": len(bounds),
        "base_action_semantics": "desired physical odom body twist",
        "anchor_count": len(all_anchors),
        "anchor_motion_segment_count": len(all_audits),
        "accepted_replan_segments": int(physical_replans),
        "strategies": strategies,
        "base_frames_changed_from_raw_command": int(changed_vs_raw_command.sum()),
        "base_frame_change_fraction_from_raw_command": float(changed_vs_raw_command.mean()),
        "odom_alignment_offset_ms_abs_p99": float(np.quantile(np.abs(np.concatenate([raw.odom_offset_ms for raw in raw_episodes])), 0.99)),
        "feedback_mapper_heldout_r2": dynamics["heldout_one_step_r2"],
        "config": asdict(config),
    }
    if config.enable_command_mode_separation:
        audited_frames = sum(audit.num_action_frames for audit in all_audits)
        accepted_audits = [
            audit for audit in all_audits if audit.strategy == "accepted_command_mode_separated"
        ]
        raw_overlap = sum(audit.raw_mode_overlap_frames for audit in all_audits)
        target_overlap = sum(audit.target_mode_overlap_frames for audit in all_audits)
        summary["command_mode_separation"] = {
            "audited_block_count": len(all_audits),
            "accepted_block_count": len(accepted_audits),
            "accepted_action_frames": int(sum(audit.num_action_frames for audit in accepted_audits)),
            "accepted_action_frame_fraction": (
                float(sum(audit.num_action_frames for audit in accepted_audits) / audited_frames)
                if audited_frames
                else 0.0
            ),
            "audited_raw_mode_overlap_frames": int(raw_overlap),
            "audited_target_mode_overlap_frames": int(target_overlap),
            "audited_mode_overlap_reduction_fraction": (
                float((raw_overlap - target_overlap) / raw_overlap) if raw_overlap else 0.0
            ),
            "acceptance_contract": (
                "candidate is endpoint-exact, zero-overlap, physical-limit bounded, no longer than "
                "the aligned raw path, no noisier than the V2 reference, and visually close to both "
                "the raw recording and V2 reference"
            ),
        }
    if config.secondary_smooth_window:
        secondary_audits = [
            audit
            for audit in all_audits
            if audit.strategy.startswith("accepted_secondary_endpoint_smooth_from_")
        ]
        secondary_frames = sum(audit.num_action_frames for audit in secondary_audits)
        audited_frames = sum(audit.num_action_frames for audit in all_audits)
        reference_tv = sum(audit.reference_velocity_total_variation for audit in all_audits)
        final_tv = sum(audit.target_velocity_total_variation for audit in all_audits)
        summary["secondary_decoupled_smoothing"] = {
            "window_frames": config.secondary_smooth_window,
            "accepted_segment_count": len(secondary_audits),
            "accepted_action_frames": int(secondary_frames),
            "accepted_action_frame_fraction": float(secondary_frames / audited_frames) if audited_frames else 0.0,
            "reference_v2_velocity_total_variation": float(reference_tv),
            "final_velocity_total_variation": float(final_tv),
            "velocity_total_variation_reduction_vs_v2": (
                float((reference_tv - final_tv) / reference_tv) if reference_tv else 0.0
            ),
            "method": (
                "endpoint-constrained independent XY/yaw pose smoothing between stationary anchors; "
                "genuine simultaneous translation+yaw timing is retained rather than force-separated"
            ),
            "visual_guard": (
                "outer V2 raw-path gate plus inner V2-relative max/p95 XY and yaw gates; "
                "candidate must reduce velocity total variation by at least 3%"
            ),
        }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    for episode in dict.fromkeys(args.preview_episodes):
        if episode not in target_by_episode:
            print(f"Skipping preview episode={episode}: not present", flush=True)
            continue
        raw = raw_episodes[episode]
        pose, twist, anchors, audits = target_by_episode[episode]
        artifact_label = (
            "anchor_v2"
            if config.dataset_version == "base_anchor_odom_v2"
            else config.dataset_version.replace("/", "_")
        )
        write_preview(
            analysis_dir,
            raw,
            pose,
            twist,
            anchors,
            audits,
            dataset_version=config.dataset_version,
            artifact_label=artifact_label,
        )
        print(f"Wrote {config.dataset_version} preview episode={episode:02d}", flush=True)
    (analysis_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    stage = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if stage.exists():
        raise RuntimeError(f"staging directory already exists: {stage}")
    try:
        copy_tree_with_video_mode(source, stage, args.video_mode)
        target_table = replace_action_in_table(source_table, target_actions)
        output_data_path = stage / source_data_path.relative_to(source)
        write_episode_row_groups(target_table, output_data_path, bounds)
        written_table = pq.read_table(output_data_path)
        validate_derived_arrays(source_actions, target_actions, source_table, written_table)
        written_actions = fixed_list_to_numpy(written_table, "action", ACTION_DIM)
        for episode, start, end in bounds:
            validate_written_target_trajectory(
                written_actions[start:end, BASE_SLICE].astype(np.float64),
                target_by_episode[episode][0],
                config,
            )

        stats_path = stage / "meta" / "stats.json"
        stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
        stats_payload["action"] = json_stats(vector_stats(target_actions))
        stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        episode_stats_path = stage / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        update_episode_action_stats(episode_stats_path, episode_action_stats)
        validate_episode_action_stats(episode_stats_path, episode_action_stats)
        write_provenance(
            destination=stage,
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            global_indices=global_indices,
            raw_command=raw_command,
            raw_odom_twist=raw_odom_twist,
            target_physical_twist=target_twist,
            raw_odom_pose=raw_odom_pose,
            target_pose=target_pose,
            anchors=all_anchors,
            audits=all_audits,
            dynamics=dynamics,
            config=config,
            source=source,
        )
        for source_video in source.joinpath("videos").rglob("*.mp4"):
            output_video = stage / source_video.relative_to(source)
            if not output_video.is_file():
                raise AssertionError(f"missing output video: {output_video}")
            if args.video_mode == "hardlink" and not os.path.samefile(source_video, output_video):
                raise AssertionError(f"video is not a hard link: {output_video}")
        stage.rename(destination)
    except Exception:
        print(f"V2 creation failed; staging preserved for inspection: {stage}", file=sys.stderr, flush=True)
        raise

    summary["elapsed_s"] = time.monotonic() - started
    (destination / "meta" / config.metadata_dir_name / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created {config.dataset_version} dataset: {destination}", flush=True)
    print(
        f"Anchors={len(all_anchors)}, accepted_replans={physical_replans}/{len(all_audits)}, "
        f"elapsed={summary['elapsed_s']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
