#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "robot8_20260716_act_dinov3_base_frozen_100k_640x480_crop2of3_curated30"
REPO_ID = f"QRP123/{RUN_ID}"
DATASET_REPO_ID = "robot8_20260716_zeno_h1_auto_cmd_v30_center_crop_2of3_640x480_curated30"
TRAINING_SOURCE_COMMIT = "b4ce7106c3441fc2380e109691c28ff15a227dc5"
CHECKPOINT = REPO_ROOT / "outputs" / "train" / RUN_ID / "checkpoints" / "100000" / "pretrained_model"
TRAIN_LOG = REPO_ROOT / "scripts" / "train_log" / RUN_ID / "train.log"
SHARED_DEPLOY_DIR = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k"
DEPLOY_DIR = REPO_ROOT / "scripts" / "deploy" / "robot8_20260716_3cam_act_dinov3_base_frozen_100k"
REQUIRED_CHECKPOINT_FILES = (
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def current_source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def validate_inputs() -> None:
    missing = [
        str(CHECKPOINT / name)
        for name in REQUIRED_CHECKPOINT_FILES
        if not (CHECKPOINT / name).is_file()
    ]
    for path in (TRAIN_LOG, SHARED_DEPLOY_DIR, DEPLOY_DIR):
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing upload input(s):\n" + "\n".join(missing))


def upload_model_card(api: HfApi, source_commit: str) -> None:
    card = f"""---
library_name: lerobot
tags:
- act
- dinov3
- robot8
---

# {RUN_ID}

- Source data: `Data/2026_07_16`
- Curated training dataset: `{DATASET_REPO_ID}` (30 episodes)
- Cameras: `head_cam,left_arm_cam,right_arm_cam`
- Checkpoint step: `100000`
- Training source commit: `{TRAINING_SOURCE_COMMIT}`
- Deployment package commit: `{source_commit}`
- Image preprocessing: center crop width and height to `2/3`, then resize to `640x480`
- State/action: 23D whole-body vector
- Policy: ACT, `chunk_size=100`, `n_action_steps=100`, no temporal ensemble
- Vision: DINOv3 ViT-B/16 LVD backbone, frozen during training

The seven checkpoint and processor files are stored at the repository root. The model includes
the complete DINOv3 backbone weights and can be loaded offline.

Deployment code is provided in both directories below:

- `deploy/robot8_20260716_3cam_act_dinov3_base_frozen_100k/`: model-specific wrapper and instructions
- `deploy/robot8_act_dinov3_base_frozen_100k/`: shared worker and ROS2 bridge implementation

Copy both directories into `<Zeno-humanoid>/scripts/deploy/`. See the model-specific deployment
README for dry-run and command-publishing instructions.
"""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md") as file:
        file.write(card)
        file.flush()
        api.upload_file(
            path_or_fileobj=file.name,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Add model card for {RUN_ID}",
        )


def upload(source_commit: str) -> None:
    validate_inputs()
    api = HfApi()
    print(f"[upload] creating public model repository {REPO_ID}", flush=True)
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)

    print(f"[upload] checkpoint {CHECKPOINT} -> {REPO_ID}:/", flush=True)
    api.upload_folder(
        folder_path=str(CHECKPOINT),
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo=".",
        commit_message=f"Upload checkpoint 100000 for {RUN_ID}",
    )

    for deploy_dir in (SHARED_DEPLOY_DIR, DEPLOY_DIR):
        print(f"[upload] deploy {deploy_dir.name}", flush=True)
        api.upload_folder(
            folder_path=str(deploy_dir),
            repo_id=REPO_ID,
            repo_type="model",
            path_in_repo=f"deploy/{deploy_dir.name}",
            ignore_patterns=["__pycache__/*", "*.pyc"],
            commit_message=f"Upload {deploy_dir.name} deploy code",
        )

    print(f"[upload] training log {TRAIN_LOG}", flush=True)
    api.upload_file(
        path_or_fileobj=str(TRAIN_LOG),
        path_in_repo="logs/train.log",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Upload training log for {RUN_ID}",
    )
    upload_model_card(api, source_commit)
    print(f"[done] https://huggingface.co/{REPO_ID}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the robot8 2026-07-16 ACT+DINOv3 checkpoint to Hugging Face."
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help="Git commit containing the deployment package; defaults to the current HEAD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload(args.source_commit or current_source_commit())


if __name__ == "__main__":
    main()
