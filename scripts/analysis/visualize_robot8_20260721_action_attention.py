#!/usr/bin/env python3
"""Render action-conditioned visual attribution for data03 and base3x-100k.

This is deliberately *not* a generic ViT self-attention visualization.  ACT's
decoder uses the same action query for all 23 output coordinates, so raw
decoder attention cannot say which pixels matter for ``base_vx`` versus (for
example) the left arm.  Instead this script uses gradient-times-activation on
the frozen DINOv3 feature map for a chosen action output:

    | d(action) / d(DINO feature) * DINO feature |

The feature-map gradient is still valid even though DINOv3 is frozen.  For the
two arm rows, the scalar target is the output projected along the arm's
currently predicted joint-motion direction.  This lets one heatmap represent
the seven joints of one arm without conflating it with the other arm.

The rendered videos retain data03's 20 Hz imagery.  Attribution is recomputed
at a configurable lower rate (1 Hz by default) and held until the next update;
doing a separate backward pass for every 20 Hz frame would be needlessly slow
and visually unstable.  The static key-frame PNGs use freshly computed maps.

The action target is the newest ACT chunk token (t+0).  It explains the visual
evidence used by the model at that observation.  It is not an attribution of a
temporally ensembled command, whose deployed value mixes action chunks from
multiple earlier observations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "Data" / "2026_07_21"
REPLAY_SOURCE = REPO_ROOT / "scripts" / "analysis" / "replay_robot8_act_base_compare.py"
CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_base3x_decoder7_ddp128_100k"
    / "checkpoints"
    / "100000"
    / "pretrained_model"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "analysis" / "robot8_20260721_data03_base3x_100k_action_attention"
)

CAMERA_NAMES = ("head_cam", "left_arm_cam", "right_arm_cam")
BASE_TARGETS = (
    ("base_vx", 20, "base vx"),
    ("base_vy", 21, "base vy"),
    ("base_rotation", 22, "base yaw"),
)
ARM_TARGETS = (
    ("left_arm", tuple(range(4, 11)), "left arm"),
    ("right_arm", tuple(range(11, 18)), "right arm"),
)
DEFAULT_BASE_FRAMES = (388, 543, 806)
DEFAULT_ARM_FRAMES = (43, 1081, 1554)


@dataclass(frozen=True)
class Target:
    """One action-conditioned visual attribution target."""

    key: str
    indices: tuple[int, ...]
    label: str
    kind: str  # "single" or "arm_direction"


@dataclass
class Attribution:
    """A visual map and scalar metadata for one target at one input frame."""

    key: str
    label: str
    kind: str
    maps: np.ndarray  # (3 cameras, 30, 40), non-negative gradient*activation
    camera_shares: list[float]
    raw_action: float | None
    motion_norm: float | None
    dominant_action_field: str | None
    target_score: float


def import_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--episode-index", type=int, default=3, help="data03 is zero-based episode index 3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional smoke-test frame limit")
    parser.add_argument(
        "--base-frames",
        type=int,
        nargs="+",
        default=list(DEFAULT_BASE_FRAMES),
        help="data03 key frames for the base static PNGs",
    )
    parser.add_argument(
        "--arm-frames",
        type=int,
        nargs="+",
        default=list(DEFAULT_ARM_FRAMES),
        help="data03 key frames for the arm static PNGs",
    )
    parser.add_argument(
        "--attribution-stride",
        type=int,
        default=20,
        help="Recompute visual attribution every N original 20 Hz frames (default: 20 = 1 Hz)",
    )
    parser.add_argument("--video-fps", type=float, default=20.0, help="Rendered video rate")
    parser.add_argument("--skip-static", action="store_true", help="Do not write key-frame PNGs")
    parser.add_argument("--skip-video", action="store_true", help="Do not write the two 20 Hz videos")
    parser.add_argument(
        "--keep-intermediate-video",
        action="store_true",
        help="Keep the temporary mp4v files used before H.264 compression",
    )
    return parser.parse_args()


def make_worker(replay: Any, checkpoint: Path, device: str | None) -> Any:
    """Use exactly the all-23D deployment preprocessing and checkpoint loader."""
    return replay.DEPLOY_WORKER.ActWorker(
        checkpoint=checkpoint,
        device=device,
        image_size=None,  # the checkpoint stores 640 x 480, no crop
        center_crop_fraction=1.0,
        use_amp=False,  # attribution is FP32 for stable gradient maps
        clamp_actions=True,
        action_clip_margin=0.05,
        n_action_steps=None,
        temporal_ensemble_coeff=None,
        frozen_action_indices=(),
        fixed_action_values={},
    )


def action_stats(worker: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the action mean/std used by the deployment unnormalizer."""
    for step in getattr(worker.postprocessor, "steps", []):
        stats = getattr(step, "stats", None)
        if not stats or "action" not in stats:
            continue
        action = stats["action"]
        mean = np.asarray(action["mean"], dtype=np.float32)
        std = np.asarray(action["std"], dtype=np.float32)
        if mean.shape == (23,) and std.shape == (23,):
            return mean, std
    raise RuntimeError("Could not recover action mean/std from the checkpoint postprocessor")


def make_targets() -> tuple[Target, ...]:
    base = tuple(Target(key, (index,), label, "single") for key, index, label in BASE_TARGETS)
    arms = tuple(Target(key, indices, label, "arm_direction") for key, indices, label in ARM_TARGETS)
    return base + arms


def decoded_images(worker: Any, episode: Any, frame_index: int) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    """Decode the no-crop RGB inputs and prepare the raw deployment observation."""
    image_bytes = episode.image_bytes(frame_index)
    display_images: dict[str, np.ndarray] = {}
    observation: dict[str, torch.Tensor] = {
        "observation.state": torch.from_numpy(episode.states[frame_index].astype(np.float32, copy=True))
    }
    for camera_name, feature_key in worker.image_keys.items():
        image = worker_module_decode(worker, image_bytes[camera_name])
        if image is None:
            raise RuntimeError(f"Could not decode {camera_name} at frame {frame_index}")
        observation[feature_key] = torch.from_numpy(image)
        display_images[camera_name] = np.ascontiguousarray(
            np.clip(np.transpose(image, (1, 2, 0)) * 255.0, 0.0, 255.0).astype(np.uint8)
        )
    return display_images, observation


def worker_module_decode(worker: Any, image_bytes: bytes) -> np.ndarray | None:
    """Resolve decode_image from the dynamically imported deployment module."""
    # The module is carried by the worker's class module.  Using it avoids
    # importing a second copy of the LeRobot deployment stack.
    module = sys.modules[worker.__class__.__module__]
    return module.decode_image(image_bytes, worker.image_size, worker.center_crop_fraction)


def cached_dino_features(worker: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute frozen DINO features once, then make them a differentiable leaf.

    Gradient stops at the DINO feature map, not at image pixels.  This is both
    faithful to the frozen backbone used during training and much cheaper than
    back-propagating through every DINOv3 block.
    """
    config_image_keys = list(worker.policy.config.image_features)
    worker_image_keys = list(worker.image_keys.values())
    if config_image_keys != worker_image_keys:
        raise RuntimeError(
            "Camera order mismatch between ACT config and deployment worker: "
            f"{config_image_keys} vs {worker_image_keys}"
        )
    with torch.no_grad():
        feature_maps = [worker.policy.model.backbone(batch[key])["feature_map"] for key in config_image_keys]
    return torch.stack(feature_maps, dim=1).detach().float().requires_grad_(True)


def raw_actions(normalized_actions: torch.Tensor, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    """Inverse the checkpoint's mean/std action normalization without a device hop."""
    normalized = normalized_actions.detach()[0, 0].float().cpu().numpy()
    return normalized * action_std + action_mean


def target_score(
    target: Target,
    normalized_actions: torch.Tensor,
    raw_action: np.ndarray,
    state: np.ndarray,
    action_std: np.ndarray,
    action_fields: tuple[str, ...],
) -> tuple[torch.Tensor, float | None, float | None, str | None]:
    """Build a scalar action target in physical output units.

    A single coordinate uses that raw coordinate.  An arm target is the raw
    output vector projected along its predicted joint displacement from the
    current observed pose; it therefore selects the arm's *current movement*
    rather than merely its absolute pose.
    """
    indices = np.asarray(target.indices, dtype=np.int64)
    coeff = np.zeros((23,), dtype=np.float32)
    if target.kind == "single":
        index = int(indices[0])
        coeff[index] = action_std[index]
        return (
            normalized_actions[0, 0] @ torch.as_tensor(coeff, device=normalized_actions.device),
            float(raw_action[index]),
            None,
            action_fields[index],
        )

    delta = raw_action[indices] - state[indices]
    motion_norm = float(np.linalg.norm(delta))
    if motion_norm > 1e-6:
        direction = delta / motion_norm
    else:
        # A stationary arm has no well-defined movement direction.  Use the
        # largest normalized joint response so every frame remains renderable,
        # and mark its motion magnitude as zero in the title.
        local = normalized_actions.detach()[0, 0, indices].abs().float().cpu().numpy()
        direction = np.zeros_like(delta)
        direction[int(np.argmax(local))] = 1.0
    coeff[indices] = direction * action_std[indices]
    dominant_local = int(np.argmax(np.abs(delta)))
    dominant_index = int(indices[dominant_local])
    return (
        normalized_actions[0, 0] @ torch.as_tensor(coeff, device=normalized_actions.device),
        None,
        motion_norm,
        action_fields[dominant_index],
    )


def compute_attributions(
    worker: Any,
    episode: Any,
    frame_index: int,
    targets: Iterable[Target],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_fields: tuple[str, ...],
) -> tuple[dict[str, Attribution], dict[str, np.ndarray], np.ndarray]:
    """Run DINO once and one ACT backward pass per requested action target."""
    display_images, observation = decoded_images(worker, episode, frame_index)
    batch = worker.preprocessor(observation)
    features = cached_dino_features(worker, batch)
    model_batch = {
        "observation.state": batch["observation.state"],
        "observation.dino_features": features,
    }
    state = episode.states[frame_index]
    results: dict[str, Attribution] = {}

    for target in targets:
        worker.policy.model.zero_grad(set_to_none=True)
        features.grad = None
        actions, _ = worker.policy.model(model_batch)
        raw_action = raw_actions(actions, action_mean, action_std)
        score, scalar_action, motion_norm, dominant_field = target_score(
            target,
            actions,
            raw_action,
            state,
            action_std,
            action_fields,
        )
        score.backward()
        if features.grad is None:
            raise RuntimeError(f"No gradient reached frozen DINO features for target {target.key}")

        # Sum the channel contribution at each 16 x 16 DINO token.  Absolute
        # value lets positive and negative evidence both remain visible.
        maps = (features.detach() * features.grad.detach()).abs().sum(dim=2)[0]
        maps_np = maps.float().cpu().numpy().astype(np.float32, copy=False)
        total = float(maps_np.sum())
        shares = (maps_np.sum(axis=(1, 2)) / total).tolist() if total > 0 else [0.0, 0.0, 0.0]
        results[target.key] = Attribution(
            key=target.key,
            label=target.label,
            kind=target.kind,
            maps=maps_np,
            camera_shares=[float(value) for value in shares],
            raw_action=scalar_action,
            motion_norm=motion_norm,
            dominant_action_field=dominant_field,
            target_score=float(score.detach().cpu()),
        )

    del features, model_batch, batch
    return results, display_images, raw_action


def normalize_maps(maps: np.ndarray) -> np.ndarray:
    """Use one robust scale across three cameras, preserving their relative mass."""
    ceiling = float(np.quantile(maps, 0.995))
    if not np.isfinite(ceiling) or ceiling <= 1e-12:
        return np.zeros_like(maps, dtype=np.float32)
    return np.clip(maps / ceiling, 0.0, 1.0).astype(np.float32, copy=False)


def overlay_attribution(image_rgb: np.ndarray, token_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Overlay a 30 x 40 token map on the full-resolution RGB observation."""
    width, height = size
    image = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    heat = cv2.resize(token_map, (width, height), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0.0, 1.0)
    heat_u8 = np.clip(heat * 255.0, 0.0, 255.0).astype(np.uint8)
    # applyColorMap returns BGR; convert the underlying RGB image to BGR so
    # the returned panel can be written directly by cv2.VideoWriter.
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    alpha = (0.62 * heat)[..., None]
    return np.clip(image_bgr * (1.0 - alpha) + color * alpha, 0.0, 255.0).astype(np.uint8)


def target_label(attribution: Attribution) -> str:
    if attribution.kind == "single":
        if attribution.key == "base_rotation":
            unit = "rad/s"
        else:
            unit = "m/s"
        return f"{attribution.label}: {attribution.raw_action:+.3f} {unit}"
    return (
        f"{attribution.label}: |dq|={attribution.motion_norm:.3f} rad "
        f"({attribution.dominant_action_field})"
    )


def compose_panel(
    display_images: dict[str, np.ndarray],
    attributions: list[Attribution],
    *,
    title: str,
    tile_size: tuple[int, int] = (320, 240),
) -> np.ndarray:
    """Make a labelled rows=targets, columns=cameras BGR visualisation."""
    tile_w, tile_h = tile_size
    left = 180
    header = 70
    footer = 10
    width = left + tile_w * len(CAMERA_NAMES)
    height = header + tile_h * len(attributions) + footer
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (238, 238, 238), 1, cv2.LINE_AA)

    for column, camera_name in enumerate(CAMERA_NAMES):
        x = left + column * tile_w
        cv2.putText(
            canvas,
            camera_name,
            (x + 8, header - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    for row, attribution in enumerate(attributions):
        y = header + row * tile_h
        if attribution.kind == "arm_direction":
            label_lines = [
                attribution.label,
                f"|dq|={attribution.motion_norm:.3f} rad",
                f"({attribution.dominant_action_field})",
            ]
            label_ys = (93, 116, 139)
            label_scale = 0.47
        else:
            label_lines = [target_label(attribution)]
            label_ys = (112,)
            label_scale = 0.48
        for label, label_y in zip(label_lines, label_ys, strict=True):
            cv2.putText(
                canvas,
                label,
                (8, y + label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

        normalized = normalize_maps(attribution.maps)
        for column, camera_name in enumerate(CAMERA_NAMES):
            x = left + column * tile_w
            panel = overlay_attribution(display_images[camera_name], normalized[column], tile_size)
            canvas[y : y + tile_h, x : x + tile_w] = panel
            text = f"{attribution.camera_shares[column] * 100.0:.1f}%"
            cv2.putText(panel, text, (8, tile_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, text, (8, tile_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 250), 1, cv2.LINE_AA)
            canvas[y : y + tile_h, x : x + tile_w] = panel
    return canvas


def h264_writer(path: Path, frame_size: tuple[int, int], fps: float) -> tuple[cv2.VideoWriter, Path]:
    """Write a portable H.264 mp4 via a temporary broadly-supported mp4v file."""
    intermediate = path.with_name(f"{path.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {intermediate}")
    return writer, intermediate


def compress_h264(intermediate: Path, output: Path, *, keep_intermediate: bool) -> None:
    if shutil.which("ffmpeg") is None:
        intermediate.replace(output)
        return
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(intermediate),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    if not keep_intermediate:
        intermediate.unlink(missing_ok=True)


def attribution_record(
    frame_index: int,
    timestamp_s: float,
    raw_action: np.ndarray,
    attributions: dict[str, Attribution],
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "model_action_raw": raw_action.tolist(),
        "targets": {
            key: {
                "camera_shares": value.camera_shares,
                "raw_action": value.raw_action,
                "motion_norm": value.motion_norm,
                "dominant_action_field": value.dominant_action_field,
                "target_score": value.target_score,
            }
            for key, value in attributions.items()
        },
    }


def render_videos(
    worker: Any,
    episode: Any,
    targets: tuple[Target, ...],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_fields: tuple[str, ...],
    output_dir: Path,
    attribution_stride: int,
    fps: float,
    keep_intermediate: bool,
) -> list[dict[str, Any]]:
    """Render full data03 20 Hz imagery while holding 1 Hz attribution maps."""
    if attribution_stride < 1:
        raise ValueError("--attribution-stride must be at least 1")

    base_targets = [target.key for target in targets if target.kind == "single"]
    arm_targets = [target.key for target in targets if target.kind == "arm_direction"]
    frame_count = len(episode.timestamps_s)
    # left label + three 320-wide camera views; dimensions are even for H.264.
    base_size = (180 + 3 * 320, 70 + 3 * 240 + 10)
    arm_size = (180 + 3 * 320, 70 + 2 * 240 + 10)
    base_path = output_dir / "data03_base_action_attention.mp4"
    arm_path = output_dir / "data03_arm_action_attention.mp4"
    base_writer, base_tmp = h264_writer(base_path, base_size, fps)
    arm_writer, arm_tmp = h264_writer(arm_path, arm_size, fps)

    records: list[dict[str, Any]] = []
    current_attributions: dict[str, Attribution] | None = None
    current_update_frame = -1
    try:
        for frame_index in range(frame_count):
            if frame_index % attribution_stride == 0:
                current_attributions, display_images, raw_action = compute_attributions(
                    worker,
                    episode,
                    frame_index,
                    targets,
                    action_mean,
                    action_std,
                    action_fields,
                )
                current_update_frame = frame_index
                records.append(
                    attribution_record(
                        frame_index,
                        float(episode.timestamps_s[frame_index]),
                        raw_action,
                        current_attributions,
                    )
                )
                print(
                    f"attribution {frame_index + 1:4d}/{frame_count} "
                    f"t={episode.timestamps_s[frame_index]:6.2f}s",
                    flush=True,
                )
            else:
                display_images, _ = decoded_images(worker, episode, frame_index)

            assert current_attributions is not None
            time_label = (
                f"data03 t={episode.timestamps_s[frame_index]:.2f}s | "
                f"maps at t={episode.timestamps_s[current_update_frame]:.2f}s | "
                "base3x 100k | grad x DINO feature"
            )
            base_panel = compose_panel(
                display_images,
                [current_attributions[key] for key in base_targets],
                title=time_label,
            )
            arm_panel = compose_panel(
                display_images,
                [current_attributions[key] for key in arm_targets],
                title=time_label,
            )
            base_writer.write(base_panel)
            arm_writer.write(arm_panel)
    finally:
        base_writer.release()
        arm_writer.release()

    compress_h264(base_tmp, base_path, keep_intermediate=keep_intermediate)
    compress_h264(arm_tmp, arm_path, keep_intermediate=keep_intermediate)
    return records


def render_static_keyframes(
    worker: Any,
    episode: Any,
    targets: tuple[Target, ...],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_fields: tuple[str, ...],
    output_dir: Path,
    base_frames: Iterable[int],
    arm_frames: Iterable[int],
) -> list[dict[str, Any]]:
    """Write high-information static panes at base and arm motion moments."""
    records: list[dict[str, Any]] = []
    base_targets = [target.key for target in targets if target.kind == "single"]
    arm_targets = [target.key for target in targets if target.kind == "arm_direction"]
    requests = (("base", tuple(base_frames), base_targets), ("arm", tuple(arm_frames), arm_targets))
    for group, frames, target_keys in requests:
        for frame_index in frames:
            if not 0 <= frame_index < len(episode.timestamps_s):
                print(f"Skipping {group} static frame {frame_index}: outside episode", flush=True)
                continue
            values, display_images, raw_action = compute_attributions(
                worker,
                episode,
                frame_index,
                targets,
                action_mean,
                action_std,
                action_fields,
            )
            title = (
                f"data03 t={episode.timestamps_s[frame_index]:.2f}s | "
                "base3x 100k | grad x DINO feature"
            )
            panel = compose_panel(display_images, [values[key] for key in target_keys], title=title)
            png_path = output_dir / f"data03_{group}_attention_frame_{frame_index:04d}.png"
            if not cv2.imwrite(str(png_path), panel):
                raise RuntimeError(f"Could not write {png_path}")
            records.append(
                attribution_record(
                    frame_index,
                    float(episode.timestamps_s[frame_index]),
                    raw_action,
                    values,
                )
            )
            print(f"wrote {png_path.name}", flush=True)
    return records


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")
    if args.attribution_stride < 1:
        raise SystemExit("--attribution-stride must be at least 1")
    if args.skip_static and args.skip_video:
        raise SystemExit("Nothing requested: remove --skip-static or --skip-video")

    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    replay = import_module("robot8_attention_replay", REPLAY_SOURCE)
    episode = replay.build_episode(args.data_dir, args.episode_index, args.max_frames)
    worker = make_worker(replay, checkpoint, args.device)
    action_mean, action_std = action_stats(worker)
    targets = make_targets()
    action_fields = tuple(replay.DEPLOY_WORKER.ACTION_FIELDS)
    started = time.monotonic()
    static_records: list[dict[str, Any]] = []
    video_records: list[dict[str, Any]] = []
    try:
        if not args.skip_static:
            static_records = render_static_keyframes(
                worker,
                episode,
                targets,
                action_mean,
                action_std,
                action_fields,
                args.output_dir,
                args.base_frames,
                args.arm_frames,
            )
        if not args.skip_video:
            video_records = render_videos(
                worker,
                episode,
                targets,
                action_mean,
                action_std,
                action_fields,
                args.output_dir,
                args.attribution_stride,
                args.video_fps,
                args.keep_intermediate_video,
            )
    finally:
        del worker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "checkpoint": str(checkpoint),
        "episode_index": args.episode_index,
        "bag_path": str(episode.bag_path),
        "frame_count": len(episode.timestamps_s),
        "duration_s": float(episode.timestamps_s[-1]),
        "input_sampling": "20 Hz bridge-style latest causal state/image cache",
        "visualization": {
            "method": "absolute gradient-times-activation at frozen DINOv3 feature map",
            "feature_shape": [3, 768, 30, 40],
            "patch_pixels": [16, 16],
            "image_size": [640, 480],
            "crop": "none",
            "action_target": "ACT newest action-chunk token t+0",
            "base_targets": [target.key for target in targets if target.kind == "single"],
            "arm_targets": "per-arm raw output projected onto current predicted 7-joint motion direction",
            "video_attribution_stride_frames": args.attribution_stride,
            "video_attribution_rate_hz": 20.0 / args.attribution_stride,
            "video_frame_rate_hz": args.video_fps,
        },
        "static_records": static_records,
        "video_attribution_records": video_records,
        "elapsed_s": time.monotonic() - started,
    }
    summary_path = args.output_dir / "attention_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {summary['elapsed_s']:.1f}s; summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
