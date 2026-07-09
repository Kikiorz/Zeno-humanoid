#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "train"
LOG_ROOT = REPO_ROOT / "scripts" / "train_log"
DEPLOY_DIR = REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k"

RUNS = [
    {
        "run_id": "robot8_20260708_3cam_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709",
        "repo_id": "QRP123/robot8_20260708_3cam_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709",
        "dataset": "Data/2026_07_08",
        "cameras": "head_cam,left_arm_cam,right_arm_cam",
    },
    {
        "run_id": "robot8_20260709_head_right_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709",
        "repo_id": "QRP123/robot8_20260709_head_right_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709",
        "dataset": "Data/2026_07_09",
        "cameras": "head_cam,right_arm_cam",
    },
]


def checkpoint_for_step(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"


def is_checkpoint_ready(checkpoint: Path) -> bool:
    required = ["model.safetensors", "config.json"]
    return all((checkpoint / name).is_file() for name in required)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for step_dir in checkpoints_dir.iterdir():
        checkpoint = step_dir / "pretrained_model"
        if not is_checkpoint_ready(checkpoint):
            continue
        try:
            step = int(step_dir.name)
        except ValueError:
            continue
        candidates.append((step, checkpoint))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def process_running(run_id: str) -> bool:
    proc = Path("/proc")
    for cmdline_path in proc.glob("[0-9]*/cmdline"):
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        except OSError:
            continue
        if run_id in cmdline and "lerobot_train" in cmdline:
            return True
    return False


def wait_for_checkpoint(run: dict[str, str], final_step: int, interval_s: int) -> Path:
    run_id = run["run_id"]
    run_dir = OUTPUT_ROOT / run_id
    final_checkpoint = checkpoint_for_step(run_dir, final_step)
    while True:
        if is_checkpoint_ready(final_checkpoint):
            return final_checkpoint
        latest = latest_checkpoint(run_dir)
        if latest is not None and not process_running(run_id):
            return latest
        print(
            f"[wait] {run_id}: final checkpoint not ready; "
            f"latest={latest if latest else 'none'}; sleeping {interval_s}s",
            flush=True,
        )
        time.sleep(interval_s)


def checkpoint_step(checkpoint: Path) -> str:
    return checkpoint.parent.parent.name


def upload_model_card(api: HfApi, run: dict[str, str], checkpoint: Path, source_commit: str | None) -> None:
    step = checkpoint_step(checkpoint)
    source = source_commit or "unknown"
    card = f"""---
library_name: lerobot
tags:
- act
- dinov3
- robot8
---

# {run["run_id"]}

- Dataset: `{run["dataset"]}`
- Cameras: `{run["cameras"]}`
- Checkpoint step: `{step}`
- Source commit: `{source}`
- Image preprocessing: 640x480, center crop 2/3 before resize
- Policy: ACT with DINOv3 ViT-B/16 LVD backbone, frozen vision backbone

The checkpoint files are stored at the repository root. Deployment helper code is under
`deploy/robot8_act_dinov3_base_frozen_100k/`.
"""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md") as f:
        f.write(card)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo="README.md",
            repo_id=run["repo_id"],
            repo_type="model",
            commit_message=f"Add model card for {run['run_id']}",
        )


def upload_run(api: HfApi, run: dict[str, str], checkpoint: Path, source_commit: str | None) -> None:
    repo_id = run["repo_id"]
    run_id = run["run_id"]
    print(f"[upload] creating {repo_id}", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"[upload] checkpoint {checkpoint} -> {repo_id}:/", flush=True)
    api.upload_folder(
        folder_path=str(checkpoint),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=".",
        commit_message=f"Upload checkpoint {checkpoint_step(checkpoint)} for {run_id}",
    )

    print(f"[upload] deploy code -> {repo_id}:/deploy/{DEPLOY_DIR.name}", flush=True)
    api.upload_folder(
        folder_path=str(DEPLOY_DIR),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"deploy/{DEPLOY_DIR.name}",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        commit_message=f"Upload deploy code for {run_id}",
    )

    train_log = LOG_ROOT / run_id / "train.log"
    if train_log.is_file():
        api.upload_file(
            path_or_fileobj=str(train_log),
            path_in_repo="logs/train.log",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload train log for {run_id}",
        )

    upload_model_card(api, run, checkpoint, source_commit)
    print(f"[done] {repo_id}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload robot8 July 2026 ACT checkpoints to Hugging Face.")
    parser.add_argument("--final-step", type=int, default=100000)
    parser.add_argument("--interval-s", type=int, default=600)
    parser.add_argument("--no-wait", action="store_true", help="Upload only if a checkpoint is already present.")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--only", choices=[run["run_id"] for run in RUNS], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    runs = [run for run in RUNS if args.only is None or run["run_id"] == args.only]
    for run in runs:
        run_dir = OUTPUT_ROOT / run["run_id"]
        checkpoint = checkpoint_for_step(run_dir, args.final_step)
        if not is_checkpoint_ready(checkpoint):
            if args.no_wait:
                checkpoint = latest_checkpoint(run_dir)
            else:
                checkpoint = wait_for_checkpoint(run, args.final_step, args.interval_s)
        if checkpoint is None:
            if args.no_wait:
                raise FileNotFoundError(f"no checkpoint available for {run['run_id']}")
            raise RuntimeError(f"wait ended without checkpoint for {run['run_id']}")
        upload_run(api, run, checkpoint, args.source_commit)


if __name__ == "__main__":
    main()
