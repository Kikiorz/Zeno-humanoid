#!/usr/bin/env python3
"""Validate structural alignment of a locally converted LeRobot v3 dataset.

The checker is deliberately read-only.  It verifies the aspects that can
silently make a robot dataset unsafe to train on: data/episode accounting,
contiguous frame and global indexes, fixed-rate timestamps, finite state/action
vectors, and an exact decoded-frame match between each episode and each camera
video.  It emits progress lines followed by one machine-readable
``VALIDATION_RESULT=...`` line for the terminal UI.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"[error] {message}", flush=True)


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(message)
    print(f"[warning] {message}", flush=True)


def fixed_list_values(table, name: str, expected_dim: int, errors: list[str]) -> np.ndarray | None:
    """Return a fixed-size Arrow list column as a two-dimensional NumPy array."""

    column = table[name].combine_chunks()
    list_size = getattr(column.type, "list_size", None)
    if list_size != expected_dim:
        add_error(errors, f"{name} has vector width {list_size}; expected {expected_dim}")
        return None
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(column), expected_dim)


def read_episode_rows(root: Path, errors: list[str]) -> list[dict[str, Any]]:
    paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not paths:
        add_error(errors, "missing meta/episodes parquet files")
        return []
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def check_episode_metadata(
    rows: list[dict[str, Any]],
    info: dict[str, Any],
    errors: list[str],
) -> dict[int, int]:
    """Check episode rows and return expected frame counts keyed by episode."""

    expected_total = int(info.get("total_frames", -1))
    expected_episodes = int(info.get("total_episodes", -1))
    if len(rows) != expected_episodes:
        add_error(errors, f"episode metadata has {len(rows)} rows; info.json says {expected_episodes}")

    lengths: dict[int, int] = {}
    next_data_index = 0
    for expected_episode, row in enumerate(rows):
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        if episode_index != expected_episode:
            add_error(errors, f"episode index {episode_index} is out of order; expected {expected_episode}")
        if length <= 0:
            add_error(errors, f"episode {episode_index} has non-positive length {length}")
        if start != next_data_index or end != start + length:
            add_error(
                errors,
                f"episode {episode_index} data range [{start}, {end}) does not match expected "
                f"[{next_data_index}, {next_data_index + length})",
            )
        next_data_index = end
        lengths[episode_index] = length
    if next_data_index != expected_total:
        add_error(errors, f"episode lengths total {next_data_index}; info.json says {expected_total} frames")
    return lengths


def check_data_files(
    root: Path,
    fps: float,
    expected_dims: int,
    expected_lengths: dict[int, int],
    expected_total: int,
    errors: list[str],
) -> Counter[int]:
    """Validate all tabular frames without loading the whole dataset at once."""

    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        add_error(errors, "missing data parquet files")
        return Counter()

    counts: Counter[int] = Counter()
    last_frame: dict[int, int] = {}
    last_timestamp: dict[int, float] = {}
    next_index = 0
    expected_delta = 1.0 / fps
    timestamp_tolerance = max(2e-4, expected_delta * 0.01)

    columns = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index"]
    for path in paths:
        print(f"[check] data {path.relative_to(root)}", flush=True)
        table = pq.read_table(path, columns=columns)
        missing = [name for name in columns if name not in table.column_names]
        if missing:
            add_error(errors, f"{path.relative_to(root)} is missing columns {missing}")
            continue

        state = fixed_list_values(table, "observation.state", expected_dims, errors)
        action = fixed_list_values(table, "action", expected_dims, errors)
        if state is not None and not np.isfinite(state).all():
            add_error(errors, f"non-finite observation.state values in {path.relative_to(root)}")
        if action is not None and not np.isfinite(action).all():
            add_error(errors, f"non-finite action values in {path.relative_to(root)}")

        indexes = np.asarray(table["index"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
        episode_indexes = np.asarray(
            table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64
        )
        frame_indexes = np.asarray(
            table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64
        )
        timestamps = np.asarray(table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64)
        if not np.isfinite(timestamps).all():
            add_error(errors, f"non-finite timestamps in {path.relative_to(root)}")
        if not np.array_equal(indexes, np.arange(next_index, next_index + len(indexes), dtype=np.int64)):
            add_error(errors, f"global index is not contiguous in {path.relative_to(root)}")
        next_index += len(indexes)

        for episode_index in np.unique(episode_indexes):
            positions = np.flatnonzero(episode_indexes == episode_index)
            frames = frame_indexes[positions]
            times = timestamps[positions]
            previous_frame = last_frame.get(int(episode_index), -1)
            expected_frames = np.arange(previous_frame + 1, previous_frame + 1 + len(frames), dtype=np.int64)
            if not np.array_equal(frames, expected_frames):
                add_error(errors, f"episode {episode_index} frame_index is not contiguous")
            if len(times) > 1 and not np.allclose(np.diff(times), expected_delta, rtol=0, atol=timestamp_tolerance):
                add_error(errors, f"episode {episode_index} timestamps are not spaced at {fps:g} Hz")
            if int(episode_index) in last_timestamp and len(times) and not math.isclose(
                times[0] - last_timestamp[int(episode_index)], expected_delta, rel_tol=0, abs_tol=timestamp_tolerance
            ):
                add_error(errors, f"episode {episode_index} timestamp gap across parquet files")
            if len(frames):
                last_frame[int(episode_index)] = int(frames[-1])
                last_timestamp[int(episode_index)] = float(times[-1])
            counts[int(episode_index)] += len(frames)

    if next_index != expected_total:
        add_error(errors, f"data parquet rows total {next_index}; expected {expected_total}")
    for episode_index, expected_length in expected_lengths.items():
        if counts[episode_index] != expected_length:
            add_error(
                errors,
                f"episode {episode_index} has {counts[episode_index]} data frames; expected {expected_length}",
            )
    unexpected = sorted(set(counts) - set(expected_lengths))
    if unexpected:
        add_error(errors, f"data references unknown episode indexes {unexpected}")
    return counts


def decoded_video_frames(video_path: Path) -> tuple[int, float | None]:
    """Decode a video to count the actual frames that training will consume."""

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate is not None else None
        frame_count = sum(1 for _ in container.decode(stream))
    return frame_count, rate


def check_videos(
    root: Path,
    rows: list[dict[str, Any]],
    video_keys: list[str],
    fps: float,
    errors: list[str],
) -> None:
    """Confirm every camera video has exactly one decoded frame per data frame."""

    for row in rows:
        episode_index = int(row["episode_index"])
        expected_frames = int(row["length"])
        for key in video_keys:
            prefix = f"videos/{key}"
            required = [f"{prefix}/chunk_index", f"{prefix}/file_index", f"{prefix}/from_timestamp", f"{prefix}/to_timestamp"]
            missing = [name for name in required if name not in row]
            if missing:
                add_error(errors, f"episode {episode_index} missing video metadata for {key}: {missing}")
                continue
            chunk_index = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            video_path = root / "videos" / key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            if not video_path.is_file():
                add_error(errors, f"episode {episode_index} missing video {video_path.relative_to(root)}")
                continue
            print(f"[check] video episode={episode_index} camera={key}", flush=True)
            try:
                actual_frames, actual_fps = decoded_video_frames(video_path)
            except Exception as exc:
                add_error(errors, f"cannot decode {video_path.relative_to(root)}: {exc}")
                continue
            if actual_frames != expected_frames:
                add_error(
                    errors,
                    f"episode {episode_index} {key} video has {actual_frames} frames; expected {expected_frames}",
                )
            if actual_fps is None or not math.isclose(actual_fps, fps, rel_tol=0, abs_tol=0.01):
                add_error(errors, f"episode {episode_index} {key} video fps {actual_fps}; expected {fps:g}")
            metadata_duration = float(row[f"{prefix}/to_timestamp"]) - float(row[f"{prefix}/from_timestamp"])
            expected_duration = expected_frames / fps
            if not math.isclose(metadata_duration, expected_duration, rel_tol=0, abs_tol=0.01):
                add_error(
                    errors,
                    f"episode {episode_index} {key} metadata duration {metadata_duration:.4f}s; "
                    f"expected {expected_duration:.4f}s",
                )


def validate(root: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    """Run all checks and return errors, warnings, and a compact summary."""

    errors: list[str] = []
    warnings: list[str] = []
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        add_error(errors, "missing meta/info.json")
        return errors, warnings, {}

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 0))
    if fps <= 0:
        add_error(errors, f"invalid fps in info.json: {fps}")
        return errors, warnings, {}
    features = info.get("features", {})
    for feature_name in ("observation.state", "action"):
        shape = features.get(feature_name, {}).get("shape")
        if shape != [23]:
            add_error(errors, f"{feature_name} shape is {shape}; expected [23]")
    video_keys = [name for name, value in features.items() if value.get("dtype") == "video"]
    if not video_keys:
        add_error(errors, "dataset has no video features")

    rows = read_episode_rows(root, errors)
    expected_lengths = check_episode_metadata(rows, info, errors)
    check_data_files(root, fps, 23, expected_lengths, int(info.get("total_frames", 0)), errors)
    check_videos(root, rows, video_keys, fps, errors)

    summary = {
        "episodes": int(info.get("total_episodes", 0)),
        "frames": int(info.get("total_frames", 0)),
        "cameras": video_keys,
        "fps": fps,
    }
    return errors, warnings, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="Local LeRobot dataset root to inspect.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_path).expanduser().resolve()
    print(f"[check] dataset {root}", flush=True)
    try:
        errors, warnings, summary = validate(root)
    except Exception as exc:
        errors = [f"validator crashed: {exc}"]
        warnings = []
        summary = {}
        print(f"[error] {errors[0]}", flush=True)
    result = {"ok": not errors, "errors": errors, "warnings": warnings, "summary": summary}
    print("VALIDATION_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
