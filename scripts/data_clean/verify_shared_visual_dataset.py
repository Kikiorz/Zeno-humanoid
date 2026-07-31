#!/usr/bin/env python3
"""Fail closed before two label variants share a frozen-DINO visual cache.

V3 changes actions and action statistics but intentionally preserves every
visual frame from its rectified raw source through hard links.  A memfd cache
is addressed by absolute LeRobot index, so sharing one cache is correct only
when the two datasets have exactly the same index/episode/frame mapping and
the same camera-video inodes.  This tool checks that contract explicitly instead
of relying on matching dataset names or a permissive cache loader.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


CAMERA_KEYS = (
    "observation.images.head_cam",
    "observation.images.left_arm_cam",
    "observation.images.right_arm_cam",
)
INDEX_COLUMNS = ("index", "episode_index", "frame_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--derived-dataset", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing required metadata: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def dataset_info(root: Path) -> dict:
    info = read_json(root / "meta" / "info.json")
    fps = info.get("fps")
    episodes = info.get("total_episodes")
    frames = info.get("total_frames")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or fps <= 0
        or not isinstance(episodes, int)
        or episodes <= 0
        or not isinstance(frames, int)
        or frames <= 0
    ):
        raise RuntimeError(
            f"Dataset metadata must be a non-empty dataset with a positive fps for {root}: "
            f"fps={fps!r}, total_episodes={episodes!r}, total_frames={frames!r}"
        )
    for key in CAMERA_KEYS:
        feature = info.get("features", {}).get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
            raise RuntimeError(f"{root} has invalid visual feature {key}: {feature!r}")
    return info


def sorted_parquet_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise RuntimeError(f"No parquet data files under {root / 'data'}")
    return paths


def _batch_column_values(batch: pa.RecordBatch, name: str) -> list[int]:
    return [int(value.as_py()) for value in batch.column(name)]


def verify_index_mapping(source: Path, derived: Path) -> tuple[int, str]:
    source_paths = sorted_parquet_paths(source)
    derived_paths = sorted_parquet_paths(derived)
    source_rel = [path.relative_to(source / "data") for path in source_paths]
    derived_rel = [path.relative_to(derived / "data") for path in derived_paths]
    if source_rel != derived_rel:
        raise RuntimeError(
            "Parquet layout differs; cannot share cache. "
            f"source={source_rel}, derived={derived_rel}"
        )

    digest = hashlib.sha256()
    total_rows = 0
    for relative in source_rel:
        source_path = source / "data" / relative
        derived_path = derived / "data" / relative
        source_file = pq.ParquetFile(source_path)
        derived_file = pq.ParquetFile(derived_path)
        source_batches = source_file.iter_batches(columns=list(INDEX_COLUMNS), batch_size=65_536)
        derived_batches = derived_file.iter_batches(columns=list(INDEX_COLUMNS), batch_size=65_536)
        missing = object()
        for batch_index, pair in enumerate(itertools.zip_longest(source_batches, derived_batches, fillvalue=missing)):
            source_batch, derived_batch = pair
            if source_batch is missing or derived_batch is missing:
                raise RuntimeError(f"Parquet batch count differs in {relative}")
            if source_batch.num_rows != derived_batch.num_rows:
                raise RuntimeError(
                    f"Index mapping row count differs in {relative} batch={batch_index}: "
                    f"source={source_batch.num_rows}, derived={derived_batch.num_rows}"
                )
            for column in INDEX_COLUMNS:
                left = _batch_column_values(source_batch, column)
                right = _batch_column_values(derived_batch, column)
                if left != right:
                    first = next(index for index, (a, b) in enumerate(zip(left, right, strict=True)) if a != b)
                    raise RuntimeError(
                        f"Index mapping differs in {relative} batch={batch_index}, column={column}, "
                        f"row={first}: source={left[first]}, derived={right[first]}"
                    )
                for value in left:
                    digest.update(int(value).to_bytes(8, byteorder="little", signed=True))
            total_rows += source_batch.num_rows

    return total_rows, digest.hexdigest()


def iter_videos(root: Path) -> Iterable[Path]:
    videos_root = root / "videos"
    paths = sorted(videos_root.glob("**/*.mp4"))
    if not paths:
        raise RuntimeError(f"No MP4 files under {videos_root}")
    return paths


def verify_hardlinked_videos(source: Path, derived: Path) -> int:
    source_paths = list(iter_videos(source))
    derived_paths = list(iter_videos(derived))
    source_rel = [path.relative_to(source / "videos") for path in source_paths]
    derived_rel = [path.relative_to(derived / "videos") for path in derived_paths]
    if source_rel != derived_rel:
        raise RuntimeError("Video layouts differ; cannot share cache")
    for relative in source_rel:
        source_video = source / "videos" / relative
        derived_video = derived / "videos" / relative
        if not os.path.samefile(source_video, derived_video):
            raise RuntimeError(
                f"V3 video is not a hard link to source; cannot share cache: {relative}"
            )
    return len(source_rel)


def verify_topcam_provenance(source: Path, derived: Path) -> str:
    source_meta = read_json(source / "meta" / "topcam_rectification.json")
    derived_meta = read_json(derived / "meta" / "topcam_rectification.json")
    if source_meta != derived_meta:
        raise RuntimeError("Topcam rectification provenance differs between source and V3 dataset")
    common_required = {
        "head_stereo_model_resize_mode": "letterbox",
        "generic_center_crop_applied_to_head_stereo": False,
    }
    actual = {key: source_meta.get(key) for key in common_required}
    if actual != common_required:
        raise RuntimeError(f"Unexpected common topcam processing contract: {actual}")
    expected_feature_to_eye = {"head_cam": "left"}
    if source_meta.get("head_camera_feature_to_eye") != expected_feature_to_eye:
        raise RuntimeError(
            "Topcam feature mapping is not the required independent left/right contract: "
            f"{source_meta.get('head_camera_feature_to_eye')!r}"
        )
    if source_meta.get("selected_model_topcam_eye") != "left":
        raise RuntimeError("The shared visual dataset must expose only the aligned left topcam eye")
    if source_meta.get("output_eyes") != ["left"] or source_meta.get("emitted_head_eyes") != ["left"]:
        raise RuntimeError("The shared visual dataset must emit only the aligned left topcam eye")
    profile = source_meta.get("profile")
    if profile in {None, "legacy_data_process"}:
        required_pipeline = (
            "split_left_right_then_opencv_fisheye_rectify_then_crop_then_"
            "independent_left_right_rgb_resize"
        )
        if source_meta.get("pipeline") != required_pipeline:
            raise RuntimeError(f"Unexpected legacy topcam pipeline: {source_meta.get('pipeline')!r}")
        crop = source_meta.get("crop")
        if crop != {"x": 20, "y": 0, "width": 1240, "height": 620}:
            raise RuntimeError(f"Unexpected legacy topcam crop: {crop!r}")
    elif profile == "cam_20260729":
        required_pipeline = (
            "split_left_right_then_cam_20260729_opencv_fisheye_rectify_then_"
            "independent_left_right_rgb_resize"
        )
        if source_meta.get("pipeline") != required_pipeline:
            raise RuntimeError(f"Unexpected cam_20260729 topcam pipeline: {source_meta.get('pipeline')!r}")
        if source_meta.get("spatial_crop") != {"x": 20, "y": 0, "width": 1240, "height": 620}:
            raise RuntimeError("cam_20260729 must apply the fixed x=20,y=0,1240x620 crop")
        if source_meta.get("rectified_size_per_eye") != {"width": 1280, "height": 720}:
            raise RuntimeError(
                f"Unexpected cam_20260729 rectified size: {source_meta.get('rectified_size_per_eye')!r}"
            )
        if source_meta.get("cropped_size_per_eye") != {"width": 1240, "height": 620}:
            raise RuntimeError(
                f"Unexpected cam_20260729 cropped size: {source_meta.get('cropped_size_per_eye')!r}"
            )
        quality = source_meta.get("calibration_quality")
        if not isinstance(quality, dict) or float(quality.get("vertical_error_p95_px", float("inf"))) > 1.0:
            raise RuntimeError(f"cam_20260729 calibration quality is not acceptable: {quality!r}")
    elif profile == "data_process_20260729":
        # The active 2026-07-29 contract must use the exact two JSON files
        # supplied under Data/process, not the similarly shaped NPZ profile.
        required_pipeline = (
            "split_left_right_then_opencv_fisheye_rectify_then_crop_then_"
            "independent_left_right_rgb_resize"
        )
        if source_meta.get("pipeline") != required_pipeline:
            raise RuntimeError(
                f"Unexpected data_process_20260729 topcam pipeline: {source_meta.get('pipeline')!r}"
            )
        if source_meta.get("contract_source") != "Data/process/rectify_topcam_stereo.py":
            raise RuntimeError("data_process_20260729 must identify Data/process as its source of truth")
        crop = {"x": 20, "y": 0, "width": 1240, "height": 620}
        if source_meta.get("crop") != crop or source_meta.get("spatial_crop") != crop:
            raise RuntimeError("data_process_20260729 must apply the fixed x=20,y=0,1240x620 crop")
        if source_meta.get("rectified_size_per_eye") != {"width": 1280, "height": 620}:
            raise RuntimeError(
                "data_process_20260729 must rectify to 1280x620 before the fixed crop"
            )
        if source_meta.get("cropped_size_per_eye") != {"width": 1240, "height": 620}:
            raise RuntimeError("data_process_20260729 has an invalid cropped size")
        required_hashes = {
            "calibration_sha256": "53be6cfdc7a82bb3acb119ddcee9d4c504a1e6a89e9c035c54106807844043cc",
            "processing_sha256": "71633cda55868d1a2af8fadd8035de9cb7ebc180a89ac340461ac4dc5e3fc46e",
        }
        actual_hashes = {key: source_meta.get(key) for key in required_hashes}
        if actual_hashes != required_hashes:
            raise RuntimeError(f"data_process_20260729 calibration provenance mismatch: {actual_hashes!r}")
    else:
        raise RuntimeError(f"Unknown topcam rectification profile: {profile!r}")
    return hashlib.sha256(
        json.dumps(source_meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    args = parse_args()
    source = args.source_dataset.expanduser().resolve()
    derived = args.derived_dataset.expanduser().resolve()
    if source == derived:
        raise SystemExit("--source-dataset and --derived-dataset must be distinct")
    source_info = dataset_info(source)
    derived_info = dataset_info(derived)
    for key in ("fps", "total_episodes", "total_frames"):
        if source_info.get(key) != derived_info.get(key):
            raise RuntimeError(
                f"Source/V3 metadata differs for {key}: "
                f"source={source_info.get(key)!r}, derived={derived_info.get(key)!r}"
            )
    rows, index_hash = verify_index_mapping(source, derived)
    videos = verify_hardlinked_videos(source, derived)
    topcam_hash = verify_topcam_provenance(source, derived)
    print(
        "shared visual-cache compatibility verified: "
        f"rows={rows}, videos={videos}, index_mapping_sha256={index_hash}, "
        f"topcam_provenance_sha256={topcam_hash}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
