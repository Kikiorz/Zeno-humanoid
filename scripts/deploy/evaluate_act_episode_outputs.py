#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

from human_new_pick_act_dinov2_worker import (  # noqa: E402
    ACTION_DIM,
    NORM_EPS,
    resolve_checkpoint_path,
    set_norm_eps,
)


ACTION_NAMES = [
    "torso_lift",
    "torso_waist",
    "head_pan",
    "head_tilt",
    *[f"left_arm_j{i}" for i in range(7)],
    *[f"right_arm_j{i}" for i in range(7)],
    "left_gripper",
    "right_gripper",
    "base_vx",
    "base_vy",
    "base_rotation",
]
ARM_SLICE = slice(4, 18)
UPPER_BODY_SLICE = slice(0, 20)
IMAGE_KEYS = [
    "observation.images.head_cam",
    "observation.images.left_arm_cam",
    "observation.images.right_arm_cam",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ACT checkpoint outputs on one LeRobot episode.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "Data" / "lerobot" / "robot4_20260623_zeno_h1_auto_cmd_v30",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "train"
        / "robot4_20260623_act_dinov3_base_fullft"
        / "checkpoints"
        / "080000",
    )
    parser.add_argument("--episode-index", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--summary-out", type=Path, default=Path("/tmp/robot4_episode_output_summary.json"))
    parser.add_argument("--npz-out", type=Path, default=Path("/tmp/robot4_episode_outputs.npz"))
    return parser.parse_args()


def load_policy(checkpoint_path: Path, device: str, n_action_steps: int | None) -> tuple[Any, Any, Any, Any]:
    config = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True)
    config.device = device
    if hasattr(config, "dinov2_pretrained"):
        config.dinov2_pretrained = False
    if hasattr(config, "dinov2_pretrained_weights"):
        config.dinov2_pretrained_weights = None
    if n_action_steps is not None:
        config.n_action_steps = n_action_steps
    if getattr(config, "temporal_ensemble_coeff", None) is not None:
        config.temporal_ensemble_coeff = None

    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(checkpoint_path, config=config, local_files_only=True)
    policy.to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    set_norm_eps(preprocessor, NORM_EPS)
    set_norm_eps(postprocessor, NORM_EPS)
    return config, policy, preprocessor, postprocessor


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def metric_block(values: np.ndarray) -> dict[str, float]:
    flat = np.abs(values).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "p90": float(np.quantile(flat, 0.90)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
    }


def error_block(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(np.square(err)))),
        "max_abs": float(np.max(np.abs(err))),
    }


def per_dim_stats(pred: np.ndarray, gt: np.ndarray, state: np.ndarray, indices: range) -> list[dict[str, float | str]]:
    rows = []
    for idx in indices:
        pred_dim = pred[:, idx]
        gt_dim = gt[:, idx]
        state_dim = state[:, idx]
        rows.append(
            {
                "index": idx,
                "name": ACTION_NAMES[idx],
                "pred_mean": float(pred_dim.mean()),
                "pred_std": float(pred_dim.std()),
                "pred_min": float(pred_dim.min()),
                "pred_max": float(pred_dim.max()),
                "gt_mean": float(gt_dim.mean()),
                "gt_std": float(gt_dim.std()),
                "gt_min": float(gt_dim.min()),
                "gt_max": float(gt_dim.max()),
                "pred_gt_mae": float(np.mean(np.abs(pred_dim - gt_dim))),
                "pred_state_abs_delta_mean": float(np.mean(np.abs(pred_dim - state_dim))),
                "gt_state_abs_delta_mean": float(np.mean(np.abs(gt_dim - state_dim))),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dataset = LeRobotDataset(
        args.dataset_root.name,
        root=args.dataset_root,
        episodes=[args.episode_index],
        download_videos=False,
        video_backend=args.video_backend,
        return_uint8=False,
    )
    config, policy, preprocessor, postprocessor = load_policy(
        checkpoint_path,
        args.device,
        args.n_action_steps,
    )

    total = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
    predictions: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    states: list[np.ndarray] = []
    frame_indices: list[int] = []

    device_type = args.device.split(":", maxsplit=1)[0]
    autocast_enabled = args.use_amp and device_type == "cuda"
    started = time.time()
    print(
        f"Evaluating episode={args.episode_index}, frames={total}, "
        f"checkpoint={checkpoint_path}, n_action_steps={config.n_action_steps}",
        flush=True,
    )

    with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=autocast_enabled):
        for i in range(total):
            item = dataset[i]
            observation = {"observation.state": item["observation.state"]}
            for image_key in IMAGE_KEYS:
                observation[image_key] = item[image_key]

            batch = preprocessor(observation)
            action = policy.select_action(batch)
            action = postprocessor(action)
            pred = np.squeeze(tensor_to_numpy(action)).astype(np.float32)
            gt = tensor_to_numpy(item["action"]).astype(np.float32).reshape(-1)
            state = tensor_to_numpy(item["observation.state"]).astype(np.float32).reshape(-1)
            if pred.shape != (ACTION_DIM,) or gt.shape != (ACTION_DIM,) or state.shape != (ACTION_DIM,):
                raise ValueError(f"bad shape at local frame {i}: pred={pred.shape}, gt={gt.shape}, state={state.shape}")
            predictions.append(pred)
            ground_truth.append(gt)
            states.append(state)
            frame_indices.append(int(tensor_to_numpy(item["frame_index"]).item()))

            if args.progress_every > 0 and (i + 1 == total or (i + 1) % args.progress_every == 0):
                fps = (i + 1) / max(time.time() - started, 1e-6)
                print(f"  {i + 1}/{total} frames, {fps:.2f} fps", flush=True)

    pred_arr = np.stack(predictions)
    gt_arr = np.stack(ground_truth)
    state_arr = np.stack(states)
    frame_arr = np.asarray(frame_indices, dtype=np.int64)

    arm_pred = pred_arr[:, ARM_SLICE]
    arm_gt = gt_arr[:, ARM_SLICE]
    arm_state = state_arr[:, ARM_SLICE]
    upper_pred = pred_arr[:, UPPER_BODY_SLICE]
    upper_gt = gt_arr[:, UPPER_BODY_SLICE]
    upper_state = state_arr[:, UPPER_BODY_SLICE]

    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "episode_index": args.episode_index,
        "frames": int(total),
        "duration_s_at_20hz": float(total / 20.0),
        "device": args.device,
        "use_amp": args.use_amp,
        "n_action_steps": int(config.n_action_steps),
        "full_action_pred_vs_gt": error_block(pred_arr, gt_arr),
        "arm_pred_vs_gt": error_block(arm_pred, arm_gt),
        "upper_body_pred_vs_gt": error_block(upper_pred, upper_gt),
        "arm_abs_pred_minus_state": metric_block(arm_pred - arm_state),
        "arm_abs_gt_minus_state": metric_block(arm_gt - arm_state),
        "upper_abs_pred_minus_state": metric_block(upper_pred - upper_state),
        "upper_abs_gt_minus_state": metric_block(upper_gt - upper_state),
        "arm_pred_std_mean": float(np.mean(np.std(arm_pred, axis=0))),
        "arm_gt_std_mean": float(np.mean(np.std(arm_gt, axis=0))),
        "arm_pred_range_mean": float(np.mean(np.max(arm_pred, axis=0) - np.min(arm_pred, axis=0))),
        "arm_gt_range_mean": float(np.mean(np.max(arm_gt, axis=0) - np.min(arm_gt, axis=0))),
        "arm_per_dim": per_dim_stats(pred_arr, gt_arr, state_arr, range(4, 18)),
    }

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.npz_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.npz_out,
        prediction=pred_arr,
        ground_truth=gt_arr,
        state=state_arr,
        frame_index=frame_arr,
        action_names=np.asarray(ACTION_NAMES),
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"summary_out={args.summary_out}", flush=True)
    print(f"npz_out={args.npz_out}", flush=True)


if __name__ == "__main__":
    main()
