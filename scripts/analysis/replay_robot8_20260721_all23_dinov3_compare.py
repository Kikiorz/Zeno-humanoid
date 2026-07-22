#!/usr/bin/env python3
"""Replay data03 through the two all-23D DINOv3 ACT deployment snapshots.

The raw ROS bag is sampled exactly as the 20 Hz ROS bridge would use it: each
tick receives the latest causally available state and compressed image from
every topic.  Each model is replayed in the deployed ACT FIFO-100 mode and in
the per-tick temporal-ensemble comparison mode.  The latter is ACT-specific;
it is not LeRobot's generic RTCInferenceEngine.

Outputs per model:
  * ``*_base_compare.png``: vx/vy/wz and dead-reckoned command path;
  * ``*_base_signals.csv``: the three base signals and their paths;
  * ``*_actions.npz``: all recorded/model 23D actions and input states;
  * ``summary.json``: latency, base errors, and path metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "scripts" / "analysis" / "replay_robot8_act_base_compare.py"
RAW_DATA_DIR = REPO_ROOT / "Data" / "2026_07_21"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "robot8_20260721_data03_all23_normal_vs_base3x"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "normal_035000": {
        "checkpoint": (
            REPO_ROOT
            / "outputs"
            / "train"
            / "robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_decoder7_ddp128_100k"
            / "checkpoints"
            / "035000"
            / "pretrained_model"
        ),
        "step": 35000,
        "description": "ordinary all-23D loss weights",
    },
    "base3x_030000": {
        "checkpoint": (
            REPO_ROOT
            / "outputs"
            / "train"
            / "robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_base3x_decoder7_ddp128_100k"
            / "checkpoints"
            / "030000"
            / "pretrained_model"
        ),
        "step": 30000,
        "description": "all-23D with 3x base_vx/base_vy/base_rotation L1 weights",
    },
}


def load_shared_module() -> Any:
    spec = importlib.util.spec_from_file_location("robot8_shared_act_replay", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import shared replay implementation: {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_worker(replay: Any, checkpoint: Path, *, temporal_ensemble: bool, device: str | None) -> Any:
    """Construct the exact all-23D deployment worker settings for one mode."""
    return replay.DEPLOY_WORKER.ActWorker(
        checkpoint=checkpoint,
        device=device,
        image_size=None,  # use 640x480 stored in the checkpoint config
        center_crop_fraction=1.0,
        use_amp=True,
        clamp_actions=True,
        action_clip_margin=0.05,
        n_action_steps=1 if temporal_ensemble else None,
        temporal_ensemble_coeff=0.01 if temporal_ensemble else None,
        frozen_action_indices=(),
        fixed_action_values={},
    )


def action_error_metrics(actions: np.ndarray, demo_actions: np.ndarray) -> dict[str, float]:
    error = actions - demo_actions
    base_error = error[:, -3:]
    return {
        "full_action_mae": float(np.mean(np.abs(error))),
        "full_action_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "base_mae": float(np.mean(np.abs(base_error))),
        "base_rmse": float(np.sqrt(np.mean(np.square(base_error)))),
        "base_max_abs": float(np.max(np.abs(base_error))),
        "base_vx_mae": float(np.mean(np.abs(base_error[:, 0]))),
        "base_vy_mae": float(np.mean(np.abs(base_error[:, 1]))),
        "base_wz_mae": float(np.mean(np.abs(base_error[:, 2]))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--episode-index", type=int, default=3, help="data03 is zero-based episode index 3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available")
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test limit; omit for full data03")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="Model snapshots to replay",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")

    np.random.seed(0)
    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay = load_shared_module()
    episode = replay.build_episode(args.data_dir, args.episode_index, args.max_frames)
    demo_path = replay.integrate_holonomic_base(episode.demo_actions)
    summary: dict[str, Any] = {
        "episode_index": args.episode_index,
        "bag_path": str(episode.bag_path),
        "num_frames": len(episode.timestamps_s),
        "duration_s": float(episode.timestamps_s[-1]),
        "fps": replay.FPS,
        "input_sampling": "20 Hz bridge-style latest causal message cache for every topic",
        "deployment_settings": {
            "cameras": ["head_cam", "left_arm_cam", "right_arm_cam"],
            "image_size": [640, 480],
            "center_crop_fraction": 1.0,
            "state_action_dimensions": 23,
            "frozen_fields": [],
            "use_amp": True,
            "action_clip_margin": 0.05,
        },
        "mode_definition": {
            "rtc_off": "ACT deployed default: 100-step FIFO action chunks at 20 Hz.",
            "rtc_on": "ACT per-tick temporal ensemble: n_action_steps=1, coefficient=0.01.",
            "generic_rtc": "Not used: ACT does not support LeRobot RTCInferenceEngine.",
        },
        "models": {},
    }

    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        checkpoint = Path(spec["checkpoint"])
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint}")
        print(f"Replaying {model_name}: {checkpoint}", flush=True)
        worker_off = make_worker(replay, checkpoint, temporal_ensemble=False, device=args.device)
        worker_on = make_worker(replay, checkpoint, temporal_ensemble=True, device=args.device)
        try:
            actions_off, metrics_off = replay.replay_mode(worker_off, episode)
            actions_on, metrics_on = replay.replay_mode(worker_on, episode)
        finally:
            del worker_off
            del worker_on
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        path_off = replay.integrate_holonomic_base(actions_off)
        path_on = replay.integrate_holonomic_base(actions_on)
        csv_path = args.output_dir / f"{model_name}_base_signals.csv"
        replay.write_csv(
            csv_path,
            episode.timestamps_s,
            episode.demo_actions,
            actions_off,
            actions_on,
            demo_path,
            path_off,
            path_on,
        )
        npz_path = args.output_dir / f"{model_name}_actions.npz"
        np.savez_compressed(
            npz_path,
            timestamps_s=episode.timestamps_s,
            states=episode.states,
            demo_actions=episode.demo_actions,
            rtc_off_actions=actions_off,
            rtc_on_actions=actions_on,
            demo_path=demo_path,
            rtc_off_path=path_off,
            rtc_on_path=path_on,
        )
        plot_path = args.output_dir / f"{model_name}_base_compare.png"
        path_metrics = replay.make_plot(
            plot_path,
            int(spec["step"]),
            episode,
            actions_off,
            actions_on,
            metrics_off,
            metrics_on,
        )
        summary["models"][model_name] = {
            "description": spec["description"],
            "checkpoint": str(checkpoint),
            "csv": str(csv_path),
            "all_actions_npz": str(npz_path),
            "plot": str(plot_path),
            "rtc_off": {**metrics_off, **action_error_metrics(actions_off, episode.demo_actions)},
            "rtc_on": {**metrics_on, **action_error_metrics(actions_on, episode.demo_actions)},
            **path_metrics,
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
