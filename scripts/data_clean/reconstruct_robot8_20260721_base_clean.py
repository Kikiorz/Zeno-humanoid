#!/usr/bin/env python3
"""Create a non-destructive, training-safe base-label-cleaned LeRobot dataset.

The source Robot8 dataset stores the human ``/twist/cmd`` target in the last
three action dimensions ``[base_vx, base_vy, base_rotation]``.  This tool
creates a *new* LeRobot-v3 root and only replaces those three training labels.
Images, state, all upper-body labels, timestamps, episode indexing, and video
files stay unchanged.

Why this is command-space (rather than odometry-space) reconstruction
---------------------------------------------------------------------
The recorded odometry pose is physically self-consistent, but the robot has a
stable command-to-motion coupling: a commanded translation also changes its
measured yaw.  Thus it is impossible to impose both of these constraints with
the current log format:

* command labels are purely ``translation`` or purely ``rotation``; and
* physical odometry follows the same yaw-free translation path.

For imitation learning, the safe first version is to preserve the original
*command-space* SE(2) endpoint of every cleaned segment.  It never invents a
large virtual camera path.  A separate future physical-clean version would
need a calibrated inverse dynamics model.

Cleaning policy
---------------
Stable command stops (0.3 s by default) partition each episode.  The default
``noise_only`` profile only changes segments with a concrete noise signature;
the optional ``light_smooth`` profile additionally applies a short centred
filter to otherwise-clean pure motions:

* pure translation: remove small closed lateral correction loops; optionally
  smooth ``vx/vy``;
* pure rotation: retain the recorded velocity profile; optionally smooth
  ``wz``;
* true composite motion: keep the original label exactly.

Every accepted candidate must preserve the raw segment endpoint (within
floating-point tolerance), stay close to its original command-integrated path,
and not exceed the original peak command by more than a small tolerance.
Segments that fail any gate are retained unchanged.  This conservative policy
is deliberate: label/image consistency matters more for training than forcing
every movement into an aesthetically ideal rotate-then-translate motion.

The generated root includes ``meta/base_clean_v1/`` with the original and
cleaned base labels plus a per-segment audit.  The three videos are hard-linked
by default, so no video re-encoding or source overwrite occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = (
    REPO_ROOT
    / "Data"
    / "lerobot"
    / "robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23"
)
DEFAULT_OUTPUT_DATASET = (
    REPO_ROOT
    / "Data"
    / "lerobot"
    / "robot8_20260721_zeno_h1_auto_cmd_v30_3cam_640x480_nocrop_all23_base_clean_cmdseg_v1"
)
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis" / "robot8_20260721_base_clean_cmdseg_v1"

ACTION_DIM = 23
BASE_SLICE = slice(20, 23)
FPS = 20.0
DT = 1.0 / FPS


@dataclass(frozen=True)
class CleanConfig:
    """All thresholds are expressed in the original command coordinate system."""

    fps: float = FPS
    stop_translation_speed: float = 0.005
    stop_rotation_speed: float = 0.020
    min_stop_frames: int = 6
    short_pulse_frames: int = 5
    short_pulse_distance_m: float = 0.012
    short_pulse_yaw_rad: float = 0.030
    translation_min_distance_m: float = 0.050
    rotation_min_yaw_rad: float = 0.050
    pure_endpoint_tolerance_m: float = 1e-5
    pure_endpoint_tolerance_rad: float = 1e-5
    stop_endpoint_tolerance_m: float = 1e-5
    stop_endpoint_tolerance_rad: float = 1e-5
    # A 5-frame (250 ms) centred filter is available for the optional
    # light_smooth profile.  It limits visual/action timing adjustment to
    # ±100 ms, rather than the ±150 ms 7-frame exploratory setting.
    smooth_window: int = 5
    smooth_pure_motion: bool = False
    weak_component_net_distance_m: float = 0.010
    weak_component_max_excursion_m: float = 0.080
    weak_component_min_excursion_m: float = 0.005
    max_path_deviation_m: float = 0.060
    path_deviation_fraction: float = 0.050
    min_path_deviation_m: float = 0.006
    max_yaw_path_deviation_rad: float = 0.080
    peak_increase_fraction: float = 0.050
    endpoint_tolerance_m: float = 2e-5
    endpoint_tolerance_rad: float = 2e-5

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


@dataclass
class SegmentAudit:
    episode_index: int
    segment_index: int
    kind: str
    strategy: str
    accepted: bool
    start_frame: int
    end_frame: int  # exclusive
    num_frames: int
    start_time_s: float
    end_time_s: float
    raw_end_x_m: float
    raw_end_y_m: float
    raw_end_yaw_rad: float
    clean_end_x_m: float
    clean_end_y_m: float
    clean_end_yaw_rad: float
    endpoint_position_error_m: float
    endpoint_yaw_error_rad: float
    raw_path_length_m: float
    max_path_deviation_m: float
    max_yaw_deviation_rad: float
    raw_peak_translation_mps: float
    clean_peak_translation_mps: float
    raw_peak_rotation_rps: float
    clean_peak_rotation_rps: float
    changed_frame_count: int
    removed_axes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--preview-episodes",
        type=int,
        nargs="+",
        default=[3],
        help="Episodes for before/after preview artifacts; data03 is episode 3.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset to clean. Omitted means all episodes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write previews/audit only; do not create a dataset.")
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How the new root reuses source MP4s. Hardlink is default and uses no extra video disk space.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Odd centred moving-average width at 20 Hz (default: 5 = 250 ms).",
    )
    parser.add_argument(
        "--profile",
        choices=("noise_only", "light_smooth"),
        default="noise_only",
        help=(
            "Cleaning strength. noise_only (default) removes only explicit "
            "closed correction loops; light_smooth also filters every accepted pure motion."
        ),
    )
    parser.add_argument("--max-path-deviation-m", type=float, default=0.060)
    parser.add_argument("--max-yaw-path-deviation-rad", type=float, default=0.080)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CleanConfig:
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise SystemExit("--smooth-window must be a positive odd number")
    if args.max_path_deviation_m <= 0 or args.max_yaw_path_deviation_rad <= 0:
        raise SystemExit("path-deviation limits must be positive")
    return CleanConfig(
        smooth_window=args.smooth_window,
        smooth_pure_motion=args.profile == "light_smooth",
        max_path_deviation_m=args.max_path_deviation_m,
        max_yaw_path_deviation_rad=args.max_yaw_path_deviation_rad,
    )


def fixed_list_to_numpy(table: pa.Table, column_name: str, width: int) -> np.ndarray:
    column = table[column_name].combine_chunks()
    if getattr(column.type, "list_size", None) != width:
        raise ValueError(f"{column_name} must be fixed-size list[{width}], got {column.type}")
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(column), width)


def fixed_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D array, got {values.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def contiguous_episode_bounds(episode_indices: np.ndarray) -> list[tuple[int, int, int]]:
    """Return ``(episode_index, start, end)`` for contiguous table episodes."""
    if episode_indices.ndim != 1 or len(episode_indices) == 0:
        raise ValueError("episode_indices must be a non-empty 1D array")
    boundaries = np.flatnonzero(np.diff(episode_indices) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_indices)]))
    output: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for start, end in zip(starts, ends, strict=True):
        episode = int(episode_indices[start])
        if episode in seen:
            raise ValueError(f"episode {episode} is not contiguous in the source parquet")
        seen.add(episode)
        output.append((episode, int(start), int(end)))
    return output


def integrate_command(base_actions: np.ndarray, dt: float) -> np.ndarray:
    """Integrate the deployment's body-frame holonomic command convention."""
    actions = np.asarray(base_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"base_actions must have shape (N, 3), got {actions.shape}")
    path = np.zeros((len(actions) + 1, 3), dtype=np.float64)
    for index, (vx, vy, wz) in enumerate(actions):
        x, y, yaw = path[index]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        path[index + 1, 0] = x + dt * (vx * cos_yaw - vy * sin_yaw)
        path[index + 1, 1] = y + dt * (vx * sin_yaw + vy * cos_yaw)
        path[index + 1, 2] = yaw + dt * wz
    return path


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous true-runs as ``[start, end)``."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def debounced_stationary_mask(base: np.ndarray, config: CleanConfig) -> np.ndarray:
    """Mark reliable stops and absorb only tiny enclosed command blips."""
    speed = np.linalg.norm(base[:, :2], axis=1)
    stationary = (speed <= config.stop_translation_speed) & (
        np.abs(base[:, 2]) <= config.stop_rotation_speed
    )
    # A short command burst fully enclosed by stationary samples is usually an
    # accidental key tap. It becomes part of the stop only if its own endpoint
    # movement is tiny; a deliberate short motion is left alone.
    for start, end in find_runs(~stationary):
        if start == 0 or end == len(stationary) or end - start > config.short_pulse_frames:
            continue
        if not stationary[start - 1] or not stationary[end]:
            continue
        path = integrate_command(base[start:end], config.dt)
        distance = float(np.linalg.norm(path[-1, :2]))
        yaw = abs(float(path[-1, 2]))
        if distance <= config.short_pulse_distance_m and yaw <= config.short_pulse_yaw_rad:
            stationary[start:end] = True
    return stationary


def long_stop_runs(base: np.ndarray, config: CleanConfig) -> list[tuple[int, int]]:
    stationary = debounced_stationary_mask(base, config)
    return [(start, end) for start, end in find_runs(stationary) if end - start >= config.min_stop_frames]


def box_smooth(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if width <= 1 or len(values) <= 2:
        return values.copy()
    if width % 2 == 0:
        raise ValueError("smoothing width must be odd")
    pad = width // 2
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(values.shape[1])], axis=1)


def correction_profile(length: int) -> np.ndarray:
    """Discrete minimum-jerk velocity profile normalized to sum to one."""
    if length <= 0:
        raise ValueError("profile length must be positive")
    if length < 3:
        return np.full(length, 1.0 / length, dtype=np.float64)
    time_axis = np.linspace(0.0, 1.0, length, dtype=np.float64)
    profile = 30.0 * time_axis**2 * (1.0 - time_axis) ** 2
    total = float(profile.sum())
    if total <= 1e-12:
        return np.full(length, 1.0 / length, dtype=np.float64)
    return profile / total


def peak_translation(base: np.ndarray) -> float:
    return float(np.linalg.norm(base[:, :2], axis=1).max(initial=0.0))


def peak_rotation(base: np.ndarray) -> float:
    return float(np.abs(base[:, 2]).max(initial=0.0))


def classify_motion(raw: np.ndarray, raw_path: np.ndarray, config: CleanConfig) -> str:
    """Classify only genuinely one-mode command segments.

    Endpoint cancellation alone is not enough: a segment can return to its
    initial yaw after a real rotate-and-correct manoeuvre.  Inspecting the
    orthogonal command channel prevents that manoeuvre from being mislabelled
    as a pure translation (or vice versa).
    """
    distance = float(np.linalg.norm(raw_path[-1, :2]))
    yaw = abs(float(raw_path[-1, 2]))
    if distance <= config.stop_endpoint_tolerance_m and yaw <= config.stop_endpoint_tolerance_rad:
        return "closed_loop"
    if (
        distance >= config.translation_min_distance_m
        and yaw <= config.pure_endpoint_tolerance_rad
        and peak_rotation(raw) <= config.stop_rotation_speed
    ):
        return "pure_translation"
    if (
        distance <= config.pure_endpoint_tolerance_m
        and yaw >= config.rotation_min_yaw_rad
        and peak_translation(raw) <= config.stop_translation_speed
    ):
        return "pure_rotation"
    return "composite"


def make_translation_candidate(raw: np.ndarray, raw_path: np.ndarray, config: CleanConfig) -> tuple[np.ndarray, list[str]]:
    """Remove a weak correction loop while preserving the command endpoint.

    In the training-default ``noise_only`` profile the dominant translation
    trace is byte-for-byte retained.  This avoids moving the label onset or
    offset relative to its recorded camera frame.  ``light_smooth`` is kept as
    a deliberately explicit ablation for cases where broader filtering is
    desired.
    """
    target_xy = raw_path[-1, :2].copy()
    candidate_xy = (
        box_smooth(raw[:, :2], config.smooth_window)
        if config.smooth_pure_motion
        else raw[:, :2].copy()
    )
    removed_axes: list[str] = []
    for axis, name in enumerate(("vx", "vy")):
        cumulative = np.cumsum(raw[:, axis], dtype=np.float64) * config.dt
        excursion = float(cumulative.max(initial=0.0) - cumulative.min(initial=0.0))
        net = abs(float(target_xy[axis]))
        if (
            net <= config.weak_component_net_distance_m
            and config.weak_component_min_excursion_m <= excursion <= config.weak_component_max_excursion_m
        ):
            # This is the target signature of an accidental tap followed by a
            # correction: appreciable temporary excursion, almost no net move.
            candidate_xy[:, axis] = 0.0
            removed_axes.append(name)
    if not config.smooth_pure_motion and not removed_axes:
        # Exact no-op is important: motion that already has no diagnosed
        # correction loop must retain its original timing and floating-point
        # values, not receive a numerically invisible endpoint correction.
        return raw.copy(), removed_axes
    residual = target_xy - candidate_xy.sum(axis=0) * config.dt
    candidate_xy += correction_profile(len(raw))[:, None] * (residual / config.dt)
    # Preserve all untouched dimensions, including harmless floating-point
    # residues in an already-pure segment.  Their removal is not a useful
    # training target and would inflate the audit's changed-frame count.
    candidate = raw.copy()
    candidate[:, :2] = candidate_xy
    return candidate, removed_axes


def make_rotation_candidate(raw: np.ndarray, raw_path: np.ndarray, config: CleanConfig) -> np.ndarray:
    """Optionally smooth a translation-free rotation with exact yaw integral."""
    target_yaw = float(raw_path[-1, 2])
    candidate_wz = box_smooth(raw[:, 2:3], config.smooth_window)[:, 0]
    residual = target_yaw - float(candidate_wz.sum() * config.dt)
    candidate_wz += correction_profile(len(raw)) * (residual / config.dt)
    candidate = raw.copy()
    candidate[:, 2] = candidate_wz
    return candidate


def allowed_position_deviation(raw_path: np.ndarray, config: CleanConfig) -> float:
    length = path_length(raw_path)
    return min(
        config.max_path_deviation_m,
        max(config.min_path_deviation_m, config.path_deviation_fraction * length),
    )


def candidate_metrics(raw: np.ndarray, candidate: np.ndarray, config: CleanConfig) -> dict[str, float]:
    raw_path = integrate_command(raw, config.dt)
    clean_path = integrate_command(candidate, config.dt)
    position_error = float(np.linalg.norm(clean_path[-1, :2] - raw_path[-1, :2]))
    yaw_error = abs(float(clean_path[-1, 2] - raw_path[-1, 2]))
    max_position_deviation = float(np.linalg.norm(clean_path[:, :2] - raw_path[:, :2], axis=1).max())
    max_yaw_deviation = float(np.abs(clean_path[:, 2] - raw_path[:, 2]).max())
    return {
        "endpoint_position_error_m": position_error,
        "endpoint_yaw_error_rad": yaw_error,
        "max_path_deviation_m": max_position_deviation,
        "max_yaw_deviation_rad": max_yaw_deviation,
        "raw_path_length_m": path_length(raw_path),
        "raw_peak_translation_mps": peak_translation(raw),
        "clean_peak_translation_mps": peak_translation(candidate),
        "raw_peak_rotation_rps": peak_rotation(raw),
        "clean_peak_rotation_rps": peak_rotation(candidate),
    }


def accept_candidate(raw: np.ndarray, candidate: np.ndarray, metrics: dict[str, float], config: CleanConfig) -> tuple[bool, str]:
    if metrics["endpoint_position_error_m"] > config.endpoint_tolerance_m:
        return False, "endpoint_position"
    if metrics["endpoint_yaw_error_rad"] > config.endpoint_tolerance_rad:
        return False, "endpoint_yaw"
    raw_path = integrate_command(raw, config.dt)
    if metrics["max_path_deviation_m"] > allowed_position_deviation(raw_path, config):
        return False, "path_deviation"
    if metrics["max_yaw_deviation_rad"] > config.max_yaw_path_deviation_rad:
        return False, "yaw_deviation"
    translation_limit = max(1e-5, metrics["raw_peak_translation_mps"] * (1.0 + config.peak_increase_fraction))
    rotation_limit = max(1e-5, metrics["raw_peak_rotation_rps"] * (1.0 + config.peak_increase_fraction))
    if metrics["clean_peak_translation_mps"] > translation_limit:
        return False, "translation_peak"
    if metrics["clean_peak_rotation_rps"] > rotation_limit:
        return False, "rotation_peak"
    return True, "accepted"


def audit_from_arrays(
    *,
    episode_index: int,
    segment_index: int,
    kind: str,
    strategy: str,
    accepted: bool,
    start: int,
    end: int,
    raw: np.ndarray,
    clean: np.ndarray,
    config: CleanConfig,
    removed_axes: Iterable[str] = (),
) -> SegmentAudit:
    raw_path = integrate_command(raw, config.dt)
    clean_path = integrate_command(clean, config.dt)
    metrics = candidate_metrics(raw, clean, config)
    changed = np.any(np.abs(raw - clean) > 1e-7, axis=1)
    return SegmentAudit(
        episode_index=episode_index,
        segment_index=segment_index,
        kind=kind,
        strategy=strategy,
        accepted=accepted,
        start_frame=start,
        end_frame=end,
        num_frames=end - start,
        start_time_s=start / config.fps,
        end_time_s=end / config.fps,
        raw_end_x_m=float(raw_path[-1, 0]),
        raw_end_y_m=float(raw_path[-1, 1]),
        raw_end_yaw_rad=float(raw_path[-1, 2]),
        clean_end_x_m=float(clean_path[-1, 0]),
        clean_end_y_m=float(clean_path[-1, 1]),
        clean_end_yaw_rad=float(clean_path[-1, 2]),
        endpoint_position_error_m=metrics["endpoint_position_error_m"],
        endpoint_yaw_error_rad=metrics["endpoint_yaw_error_rad"],
        raw_path_length_m=metrics["raw_path_length_m"],
        max_path_deviation_m=metrics["max_path_deviation_m"],
        max_yaw_deviation_rad=metrics["max_yaw_deviation_rad"],
        raw_peak_translation_mps=metrics["raw_peak_translation_mps"],
        clean_peak_translation_mps=metrics["clean_peak_translation_mps"],
        raw_peak_rotation_rps=metrics["raw_peak_rotation_rps"],
        clean_peak_rotation_rps=metrics["clean_peak_rotation_rps"],
        changed_frame_count=int(changed.sum()),
        removed_axes=",".join(removed_axes),
    )


def clean_episode(base: np.ndarray, episode_index: int, config: CleanConfig) -> tuple[np.ndarray, list[SegmentAudit]]:
    """Conservatively clean one episode without changing its command endpoint."""
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError(f"Expected base action shape (N, 3), got {base.shape}")
    clean = base.copy()
    stops = long_stop_runs(base, config)
    audits: list[SegmentAudit] = []
    segment_index = 0

    def process(start: int, end: int, is_stop: bool) -> None:
        nonlocal segment_index
        if end <= start:
            return
        raw = base[start:end]
        raw_path = integrate_command(raw, config.dt)
        kind = "stable_stop" if is_stop else classify_motion(raw, raw_path, config)
        candidate = raw.copy()
        strategy = "kept_raw"
        accepted = False
        removed_axes: list[str] = []

        if is_stop:
            if (
                np.linalg.norm(raw_path[-1, :2]) <= config.stop_endpoint_tolerance_m
                and abs(raw_path[-1, 2]) <= config.stop_endpoint_tolerance_rad
            ):
                candidate = np.zeros_like(raw)
                strategy = "zero_stable_stop"
                metrics = candidate_metrics(raw, candidate, config)
                accepted, reason = accept_candidate(raw, candidate, metrics, config)
                if not accepted:
                    candidate = raw.copy()
                    strategy = f"kept_stop_{reason}"
            else:
                strategy = "kept_stop_nonzero_endpoint"
        elif kind == "closed_loop":
            # Small enclosed blips between stable anchors are label noise. The
            # strict endpoint gate makes this removal safe for the action path.
            candidate = np.zeros_like(raw)
            strategy = "zero_closed_loop"
            metrics = candidate_metrics(raw, candidate, config)
            accepted, reason = accept_candidate(raw, candidate, metrics, config)
            if not accepted:
                candidate = raw.copy()
                strategy = f"kept_closed_loop_{reason}"
        elif kind == "pure_translation":
            candidate, removed_axes = make_translation_candidate(raw, raw_path, config)
            if not config.smooth_pure_motion and not removed_axes:
                strategy = "kept_pure_translation_no_noise"
            else:
                strategy = (
                    "smooth_pure_translation"
                    if config.smooth_pure_motion
                    else "remove_translation_correction_loop"
                )
                metrics = candidate_metrics(raw, candidate, config)
                accepted, reason = accept_candidate(raw, candidate, metrics, config)
                if not accepted:
                    candidate = raw.copy()
                    removed_axes = []
                    strategy = f"kept_translation_{reason}"
        elif kind == "pure_rotation":
            if not config.smooth_pure_motion:
                strategy = "kept_pure_rotation_no_noise"
            else:
                candidate = make_rotation_candidate(raw, raw_path, config)
                strategy = "smooth_pure_rotation"
                metrics = candidate_metrics(raw, candidate, config)
                accepted, reason = accept_candidate(raw, candidate, metrics, config)
                if not accepted:
                    candidate = raw.copy()
                    strategy = f"kept_rotation_{reason}"
        else:
            strategy = "kept_composite"

        clean[start:end] = candidate
        audits.append(
            audit_from_arrays(
                episode_index=episode_index,
                segment_index=segment_index,
                kind=kind,
                strategy=strategy,
                accepted=accepted,
                start=start,
                end=end,
                raw=raw,
                clean=candidate,
                config=config,
                removed_axes=removed_axes,
            )
        )
        segment_index += 1

    cursor = 0
    for stop_start, stop_end in stops:
        process(cursor, stop_start, is_stop=False)
        process(stop_start, stop_end, is_stop=True)
        cursor = stop_end
    process(cursor, len(base), is_stop=False)

    raw_episode_path = integrate_command(base, config.dt)
    clean_episode_path = integrate_command(clean, config.dt)
    endpoint_pos = float(np.linalg.norm(raw_episode_path[-1, :2] - clean_episode_path[-1, :2]))
    endpoint_yaw = abs(float(raw_episode_path[-1, 2] - clean_episode_path[-1, 2]))
    if endpoint_pos > config.endpoint_tolerance_m or endpoint_yaw > config.endpoint_tolerance_rad:
        # Never emit a derived episode whose command destination moved. Returning
        # the raw labels is safer than a global correction that could pollute a
        # visually aligned segment.
        rejected: list[SegmentAudit] = []
        for audit in audits:
            if audit.accepted:
                audit.strategy = "reverted_episode_endpoint"
                audit.accepted = False
            rejected.append(audit)
        return base.copy(), rejected
    return clean, audits


def vector_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"Expected at least two vector samples, got {values.shape}")
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


def copy_tree_with_video_mode(source: Path, destination: Path, video_mode: str) -> None:
    """Copy mutable files and link/copy immutable MP4s without touching source."""
    for child in source.iterdir():
        target = destination / child.name
        if child.name == "videos":
            for source_file in sorted(child.rglob("*")):
                relative = source_file.relative_to(child)
                target_file = target / relative
                if source_file.is_dir():
                    target_file.mkdir(parents=True, exist_ok=True)
                    continue
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if video_mode == "hardlink":
                    try:
                        os.link(source_file, target_file)
                    except OSError as exc:
                        raise RuntimeError(
                            f"Could not hard-link {source_file}; rerun with --video-mode copy if needed."
                        ) from exc
                else:
                    shutil.copy2(source_file, target_file)
        elif child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def replace_action_in_table(table: pa.Table, actions: np.ndarray) -> pa.Table:
    if len(table) != len(actions) or actions.shape != (len(table), ACTION_DIM):
        raise ValueError(f"action shape {actions.shape} does not match table rows {len(table)}")
    column_index = table.schema.get_field_index("action")
    if column_index < 0:
        raise KeyError("source table has no action column")
    # Preserve the original Arrow child-field name and any field metadata.
    # Passing just the string "action" makes Arrow synthesize a new
    # ``item`` child field on some versions, which needlessly changes the
    # LeRobot schema despite identical values.
    return table.set_column(column_index, table.schema.field(column_index), fixed_list_array(actions))


def write_episode_row_groups(table: pa.Table, destination: Path, episode_bounds: list[tuple[int, int, int]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    compression = "snappy"
    with pq.ParquetWriter(destination, table.schema, compression=compression) as writer:
        for _, start, end in episode_bounds:
            writer.write_table(table.slice(start, end - start))


def update_episode_action_stats(episodes_path: Path, episode_stats: dict[int, dict[str, np.ndarray]]) -> None:
    table = pq.read_table(episodes_path)
    episode_ids = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    arrays: list[pa.Array | pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for field in table.schema:
        name = field.name
        if not name.startswith("stats/action/"):
            arrays.append(table[name])
            fields.append(field)
            continue
        stat_key = name.rsplit("/", maxsplit=1)[-1]
        values = table[name].to_pylist()
        for row_index, episode_index in enumerate(episode_ids):
            values[row_index] = episode_stats[int(episode_index)][stat_key].tolist()
        arrays.append(pa.array(values, type=field.type))
        fields.append(field)
    updated = pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=table.schema.metadata))
    pq.write_table(updated, episodes_path, compression="snappy")


def validate_episode_action_stats(episodes_path: Path, episode_stats: dict[int, dict[str, np.ndarray]]) -> None:
    """Ensure rewritten per-episode action normalization metadata is exact."""
    table = pq.read_table(episodes_path)
    episode_ids = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    for field in table.schema:
        name = field.name
        if not name.startswith("stats/action/"):
            continue
        stat_key = name.rsplit("/", maxsplit=1)[-1]
        values = table[name].to_pylist()
        for row_index, episode_index in enumerate(episode_ids):
            actual = np.asarray(values[row_index], dtype=np.float64)
            expected = np.asarray(episode_stats[int(episode_index)][stat_key], dtype=np.float64)
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=1e-7):
                raise AssertionError(
                    f"episode action stats mismatch for episode={episode_index}, stat={stat_key}"
                )


def write_provenance(
    destination: Path,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    global_indices: np.ndarray,
    raw_base: np.ndarray,
    clean_base: np.ndarray,
    audits: list[SegmentAudit],
    config: CleanConfig,
    source: Path,
) -> None:
    meta_dir = destination / "meta" / "base_clean_v1"
    meta_dir.mkdir(parents=True, exist_ok=True)
    changed = np.any(np.abs(raw_base - clean_base) > 1e-7, axis=1)
    provenance = pa.table(
        {
            "episode_index": pa.array(episode_indices.astype(np.int64)),
            "frame_index": pa.array(frame_indices.astype(np.int64)),
            "index": pa.array(global_indices.astype(np.int64)),
            "raw_base_action": fixed_list_array(raw_base),
            "clean_base_action": fixed_list_array(clean_base),
            "changed": pa.array(changed),
        }
    )
    pq.write_table(provenance, meta_dir / "action_provenance.parquet", compression="snappy")
    audit_rows = [asdict(audit) for audit in audits]
    if audit_rows:
        pq.write_table(pa.Table.from_pylist(audit_rows), meta_dir / "segment_audit.parquet", compression="snappy")
    manifest = {
        "version": "base_clean_cmdseg_v1",
        "source_dataset": str(source.resolve()),
        "action_semantics": {
            "training_action": "action[0:20] unchanged; action[20:23] is the clean command-space target",
            "raw_action": "preserved in meta/base_clean_v1/action_provenance.parquet",
            "state_images": "unchanged raw logged observations",
            "endpoint_constraint": "per accepted segment, original twist/cmd SE(2) endpoint is preserved",
            "not_physical_odom_constraint": "odom pose is intentionally not used as a hard endpoint in v1 because platform dynamics couple translation to yaw",
        },
        "config": asdict(config),
        "segment_count": len(audits),
        "accepted_segment_count": int(sum(audit.accepted for audit in audits)),
        "changed_segment_count": int(sum(audit.changed_frame_count > 0 for audit in audits)),
        "changed_frame_count": int(changed.sum()),
        "video_reuse": "hardlink or copy; no re-encode",
    }
    (meta_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def line_plot(draw: ImageDraw.ImageDraw, values: np.ndarray, box: tuple[int, int, int, int], color: str, scale: tuple[float, float]) -> None:
    x0, y0, x1, y1 = box
    low, high = scale
    if high <= low:
        high = low + 1.0
    values = np.asarray(values, dtype=np.float64)
    points = []
    for index, value in enumerate(values):
        x = x0 + (x1 - x0) * index / max(1, len(values) - 1)
        y = y1 - (y1 - y0) * (value - low) / (high - low)
        points.append((int(round(x)), int(round(y))))
    if len(points) >= 2:
        draw.line(points, fill=color, width=2)


def write_preview(
    analysis_dir: Path,
    episode_index: int,
    raw_base: np.ndarray,
    clean_base: np.ndarray,
    audits: list[SegmentAudit],
    config: CleanConfig,
) -> None:
    """Write an inspectable PNG/NPZ/JSON preview without matplotlib."""
    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_path = integrate_command(raw_base, config.dt)
    clean_path = integrate_command(clean_base, config.dt)
    np.savez_compressed(
        analysis_dir / f"episode_{episode_index:02d}_base_clean_preview.npz",
        raw_base=raw_base.astype(np.float32),
        clean_base=clean_base.astype(np.float32),
        raw_path=raw_path,
        clean_path=clean_path,
    )
    (analysis_dir / f"episode_{episode_index:02d}_segment_audit.json").write_text(
        json.dumps([asdict(audit) for audit in audits], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image_width, image_height = 1800, 1420
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    draw.text((30, 22), f"Robot8 base-clean cmdseg-v1 | episode {episode_index:02d} | raw=gray clean=blue", fill="#17202a", font=title_font)
    names = ("base_vx (m/s)", "base_vy (m/s)", "base_rotation (rad/s)")
    colors = ("#2f3640", "#2f3640", "#2f3640")
    chart_left, chart_right = 120, 1740
    chart_height = 220
    for row, (name, color) in enumerate(zip(names, colors, strict=True)):
        top = 80 + row * 250
        bottom = top + chart_height
        raw = raw_base[:, row]
        clean = clean_base[:, row]
        margin = max(0.005, 0.08 * float(max(np.abs(raw).max(initial=0.0), np.abs(clean).max(initial=0.0))))
        low = float(min(raw.min(initial=0.0), clean.min(initial=0.0)) - margin)
        high = float(max(raw.max(initial=0.0), clean.max(initial=0.0)) + margin)
        zero_y = bottom - (bottom - top) * (0.0 - low) / (high - low)
        draw.rectangle((chart_left, top, chart_right, bottom), outline="#b8c2cc", width=1)
        draw.line((chart_left, int(zero_y), chart_right, int(zero_y)), fill="#d5dce3", width=1)
        draw.text((30, top + 8), name, fill="#17202a", font=title_font)
        line_plot(draw, raw, (chart_left, top, chart_right, bottom), color, (low, high))
        line_plot(draw, clean, (chart_left, top, chart_right, bottom), "#1769e0", (low, high))

    # Command-integrated path panel.
    path_top, path_bottom = 900, 1370
    path_left, path_right = 120, 1740
    all_xy = np.concatenate((raw_path[:, :2], clean_path[:, :2]), axis=0)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 0.05)
    pad = 0.08 * span
    min_xy -= pad
    max_xy += pad
    draw.rectangle((path_left, path_top, path_right, path_bottom), outline="#b8c2cc", width=1)
    draw.text((30, path_top + 8), "command-integrated XY path", fill="#17202a", font=title_font)

    def xy_points(path: np.ndarray) -> list[tuple[int, int]]:
        points = []
        for x, y in path[:, :2]:
            px = path_left + (path_right - path_left) * (x - min_xy[0]) / (max_xy[0] - min_xy[0])
            py = path_bottom - (path_bottom - path_top) * (y - min_xy[1]) / (max_xy[1] - min_xy[1])
            points.append((int(round(px)), int(round(py))))
        return points

    draw.line(xy_points(raw_path), fill="#2f3640", width=3)
    draw.line(xy_points(clean_path), fill="#1769e0", width=3)
    draw.ellipse((xy_points(raw_path)[0][0] - 4, xy_points(raw_path)[0][1] - 4, xy_points(raw_path)[0][0] + 4, xy_points(raw_path)[0][1] + 4), fill="#2ca25f")
    draw.ellipse((xy_points(raw_path)[-1][0] - 4, xy_points(raw_path)[-1][1] - 4, xy_points(raw_path)[-1][0] + 4, xy_points(raw_path)[-1][1] + 4), fill="#d62728")
    changed_count = int(np.any(np.abs(raw_base - clean_base) > 1e-7, axis=1).sum())
    draw.text((125, 1382), f"changed frames: {changed_count}/{len(raw_base)} | accepted segments: {sum(a.accepted for a in audits)}/{len(audits)}", fill="#17202a", font=title_font)
    image.save(analysis_dir / f"episode_{episode_index:02d}_base_clean_preview.png")


def validate_derived_arrays(
    source_actions: np.ndarray,
    clean_actions: np.ndarray,
    source_table: pa.Table,
    output_table: pa.Table,
) -> None:
    if not source_table.schema.equals(output_table.schema, check_metadata=True):
        raise AssertionError("derived parquet schema or schema metadata changed")
    if not np.array_equal(source_actions[:, :20], clean_actions[:, :20]):
        raise AssertionError("non-base action dimensions changed")
    for name in source_table.column_names:
        if name == "action":
            continue
        if not source_table[name].equals(output_table[name]):
            raise AssertionError(f"non-action column changed: {name}")
    output_actions = fixed_list_to_numpy(output_table, "action", ACTION_DIM)
    if not np.array_equal(output_actions, clean_actions):
        raise AssertionError("written action column does not match intended cleaned actions")
    if not np.isfinite(output_actions).all():
        raise AssertionError("cleaned action contains non-finite values")


def validate_written_segment_endpoints(
    raw_base: np.ndarray,
    written_base: np.ndarray,
    episode_indices: np.ndarray,
    audits: list[SegmentAudit],
    config: CleanConfig,
) -> None:
    """Recheck endpoint constraints after the final float32 parquet round trip."""
    bounds = {
        episode: (start, end)
        for episode, start, end in contiguous_episode_bounds(episode_indices)
    }
    # Float32 serialization is expected to be far below this allowance; the
    # bound simply separates harmless representation noise from a real write
    # or indexing error.
    position_tolerance = max(5e-5, 2.0 * config.endpoint_tolerance_m)
    yaw_tolerance = max(5e-5, 2.0 * config.endpoint_tolerance_rad)
    for audit in audits:
        if audit.changed_frame_count == 0:
            continue
        episode_start, _ = bounds[audit.episode_index]
        start = episode_start + audit.start_frame
        end = episode_start + audit.end_frame
        raw_path = integrate_command(raw_base[start:end], config.dt)
        written_path = integrate_command(written_base[start:end], config.dt)
        position_error = float(np.linalg.norm(written_path[-1, :2] - raw_path[-1, :2]))
        yaw_error = abs(float(written_path[-1, 2] - raw_path[-1, 2]))
        if position_error > position_tolerance or yaw_error > yaw_tolerance:
            raise AssertionError(
                "written segment endpoint moved beyond tolerance: "
                f"episode={audit.episode_index} segment={audit.segment_index} "
                f"position_error={position_error:.8f} yaw_error={yaw_error:.8f}"
            )


def prepare_clean_actions(
    source_actions: np.ndarray,
    episode_indices: np.ndarray,
    selected_episodes: set[int] | None,
    config: CleanConfig,
) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]], list[SegmentAudit]]:
    cleaned = source_actions.copy()
    audits: list[SegmentAudit] = []
    per_episode_stats: dict[int, dict[str, np.ndarray]] = {}
    for episode_index, start, end in contiguous_episode_bounds(episode_indices):
        raw_base = source_actions[start:end, BASE_SLICE]
        if selected_episodes is None or episode_index in selected_episodes:
            clean_base, episode_audits = clean_episode(raw_base, episode_index, config)
            cleaned[start:end, BASE_SLICE] = clean_base.astype(np.float32)
            audits.extend(episode_audits)
        else:
            # Record why this episode is an intentional untouched subset.
            episode_audits = []
        per_episode_stats[episode_index] = vector_stats(cleaned[start:end])
    return cleaned, per_episode_stats, audits


def main() -> None:
    args = parse_args()
    config = build_config(args)
    source = args.source_dataset.resolve()
    destination = args.output_dataset.resolve()
    analysis_dir = args.analysis_dir.resolve()
    if not (source / "data").is_dir() or not (source / "meta").is_dir() or not (source / "videos").is_dir():
        raise FileNotFoundError(f"Not a complete LeRobot dataset root: {source}")
    if destination == source:
        raise SystemExit("--output-dataset must be different from --source-dataset")
    if destination.exists() and not args.dry_run:
        raise SystemExit(f"Refusing to overwrite existing output dataset: {destination}")

    data_paths = sorted((source / "data").rglob("*.parquet"))
    if len(data_paths) != 1:
        raise RuntimeError(f"Expected exactly one data parquet for this dataset, found {len(data_paths)}")
    source_data_path = data_paths[0]
    table = pq.read_table(source_data_path)
    source_actions = fixed_list_to_numpy(table, "action", ACTION_DIM)
    episode_indices = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    global_indices = np.asarray(table["index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    available_episodes = {episode for episode, _, _ in contiguous_episode_bounds(episode_indices)}
    selected_episodes = set(args.episodes) if args.episodes is not None else None
    if selected_episodes is not None:
        unknown = sorted(selected_episodes - available_episodes)
        if unknown:
            raise SystemExit(f"Unknown --episodes: {unknown}; available={sorted(available_episodes)}")

    print(f"Source: {source}", flush=True)
    print(f"Mode:   {'dry-run' if args.dry_run else 'create derived dataset'}", flush=True)
    print(f"Frames: {len(table)}, episodes: {len(available_episodes)}, clean target: {sorted(selected_episodes) if selected_episodes is not None else 'all'}", flush=True)
    started = time.monotonic()
    clean_actions, episode_action_stats, audits = prepare_clean_actions(
        source_actions,
        episode_indices,
        selected_episodes,
        config,
    )
    changed = np.any(np.abs(source_actions[:, BASE_SLICE] - clean_actions[:, BASE_SLICE]) > 1e-7, axis=1)
    global_action_stats = vector_stats(clean_actions)

    # Preview artifacts are always generated. They make the data change
    # inspectable before anyone launches a training job.
    bounds_by_episode = {episode: (start, end) for episode, start, end in contiguous_episode_bounds(episode_indices)}
    for episode in dict.fromkeys(args.preview_episodes):
        if episode not in bounds_by_episode:
            print(f"Skipping preview episode {episode}: not present", flush=True)
            continue
        start, end = bounds_by_episode[episode]
        episode_audits = [audit for audit in audits if audit.episode_index == episode]
        write_preview(
            analysis_dir,
            episode,
            source_actions[start:end, BASE_SLICE],
            clean_actions[start:end, BASE_SLICE],
            episode_audits,
            config,
        )
        print(f"Wrote preview for episode {episode:02d}", flush=True)

    preview_summary = {
        "source_dataset": str(source),
        "output_dataset": str(destination),
        "dry_run": bool(args.dry_run),
        "frames": len(table),
        "episodes": len(available_episodes),
        "selected_episodes": sorted(selected_episodes) if selected_episodes is not None else "all",
        "changed_frames": int(changed.sum()),
        "changed_frame_fraction": float(changed.mean()),
        "accepted_segments": int(sum(audit.accepted for audit in audits)),
        "changed_segments": int(sum(audit.changed_frame_count > 0 for audit in audits)),
        "total_audited_segments": len(audits),
        "strategies": {
            strategy: sum(audit.strategy == strategy for audit in audits)
            for strategy in sorted({audit.strategy for audit in audits})
        },
        "config": asdict(config),
        "command_space_note": "Endpoint preservation uses the original action command SE(2) convention, not odom pose.",
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "cleaning_summary.json").write_text(
        json.dumps(preview_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print(json.dumps(preview_summary, ensure_ascii=False, indent=2), flush=True)
        return

    stage = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if stage.exists():
        raise RuntimeError(f"Staging path already exists; refusing to touch it: {stage}")
    stage.parent.mkdir(parents=True, exist_ok=True)
    try:
        copy_tree_with_video_mode(source, stage, args.video_mode)
        clean_table = replace_action_in_table(table, clean_actions)
        output_data_path = stage / source_data_path.relative_to(source)
        write_episode_row_groups(clean_table, output_data_path, contiguous_episode_bounds(episode_indices))
        written_table = pq.read_table(output_data_path)
        validate_derived_arrays(source_actions, clean_actions, table, written_table)
        written_actions = fixed_list_to_numpy(written_table, "action", ACTION_DIM)
        validate_written_segment_endpoints(
            source_actions[:, BASE_SLICE],
            written_actions[:, BASE_SLICE],
            episode_indices,
            audits,
            config,
        )

        stats_path = stage / "meta" / "stats.json"
        stats_json = json.loads(stats_path.read_text(encoding="utf-8"))
        stats_json["action"] = json_stats(global_action_stats)
        stats_path.write_text(json.dumps(stats_json, ensure_ascii=False, indent=2), encoding="utf-8")
        episodes_path = stage / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        update_episode_action_stats(episodes_path, episode_action_stats)
        validate_episode_action_stats(episodes_path, episode_action_stats)
        write_provenance(
            stage,
            episode_indices,
            frame_indices,
            global_indices,
            source_actions[:, BASE_SLICE],
            clean_actions[:, BASE_SLICE],
            audits,
            config,
            source,
        )
        # Validate video reuse explicitly: every source file has a matching
        # destination file and hardlink mode must point to the same inode.
        for source_video in source.joinpath("videos").rglob("*.mp4"):
            output_video = stage / source_video.relative_to(source)
            if not output_video.is_file():
                raise AssertionError(f"Missing copied video: {output_video}")
            if args.video_mode == "hardlink" and not os.path.samefile(source_video, output_video):
                raise AssertionError(f"Expected hard-linked video: {output_video}")
        stage.rename(destination)
    except Exception:
        print(f"Creation failed. Staging directory retained for inspection: {stage}", file=sys.stderr, flush=True)
        raise

    preview_summary["elapsed_s"] = time.monotonic() - started
    (destination / "meta" / "base_clean_v1" / "generation_summary.json").write_text(
        json.dumps(preview_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created derived dataset: {destination}", flush=True)
    print(f"Changed {int(changed.sum())}/{len(changed)} frames; elapsed {preview_summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
