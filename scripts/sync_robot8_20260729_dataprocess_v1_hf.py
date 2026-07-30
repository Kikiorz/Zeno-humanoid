#!/usr/bin/env python3
"""Synchronize final 2026-07-29 Data/process ACT checkpoints through HF Hub.

Run ``upload`` on the training machine after both jobs have reached 100k, then
run ``download`` on the target workstation.  Only inference files are moved to
the workstation checkpoint location; optimizer/RNG training state stays on the
training machine.  Repositories are private by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_STEP = 100_000
OWNER_DEFAULT = "QRP123"
MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


@dataclass(frozen=True)
class Run:
    key: str
    run_id: str
    hub_name: str
    label: str

    def repo_id(self, owner: str) -> str:
        # Hub model names are limited to 96 characters.  ``run_id`` is kept
        # lossless for local/remote artifact paths, while this concise alias
        # remains stable and readable on the Hub.
        return f"{owner}/{self.hub_name}"


RUNS = (
    Run(
        key="normal",
        run_id=(
            "robot8_20260729_act_dinov3_3cam_640x480_"
            "topcam_left_dataprocess_v1_all23_decoder7_b32_100k"
        ),
        hub_name="robot8-20260729-act-dinov3-topcam-left-all23-100k",
        label="ordinary all-23D objective",
    ),
    Run(
        key="v3",
        run_id=(
            "robot8_20260729_act_dinov3_3cam_640x480_"
            "topcam_left_dataprocess_v1_all23_base_anchor_odom_v3_"
            "decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k"
        ),
        hub_name="robot8-20260729-act-dinov3-topcam-left-v3-smooth-100k",
        label="V3 smooth base-label curriculum",
    ),
)


def selected_runs(which: str) -> Iterable[Run]:
    return RUNS if which == "both" else tuple(run for run in RUNS if run.key == which)


def checkpoint(root: Path, run: Run) -> Path:
    return root / "outputs" / "train" / run.run_id / "checkpoints" / f"{FINAL_STEP:06d}" / "pretrained_model"


def run_root(root: Path, run: Run) -> Path:
    return root / "outputs" / "train" / run.run_id


def require_final_checkpoint(root: Path, run: Run) -> Path:
    run_dir = run_root(root, run)
    ckpt = checkpoint(root, run)
    missing = [name for name in MODEL_FILES if not (ckpt / name).is_file()]
    step_file = run_dir / "checkpoints" / f"{FINAL_STEP:06d}" / "training_state" / "training_step.json"
    success = run_dir / "TRAINING_SUCCEEDED"
    if missing or not step_file.is_file() or not success.is_file():
        details = [str(ckpt / name) for name in missing]
        if not step_file.is_file():
            details.append(str(step_file))
        if not success.is_file():
            details.append(str(success))
        raise FileNotFoundError("final training artifact is incomplete:\n" + "\n".join(details))
    payload = json.loads(step_file.read_text(encoding="utf-8"))
    if int(payload.get("step", -1)) != FINAL_STEP:
        raise ValueError(f"{step_file} does not record step {FINAL_STEP}")
    return ckpt


def source_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model_card(run: Run, repo_id: str, commit: str) -> str:
    return f"""---
library_name: lerobot
tags:
- act
- dinov3
- robot8
- topcam-rectified
---

# {run.run_id}

- Variant: {run.label}
- Final checkpoint: `{FINAL_STEP}`
- Cameras: `head_cam,left_arm_cam,right_arm_cam`
- State/action: 23D whole-body vector
- Policy: ACT, decoder layers 7, chunk/action horizon 100
- Vision: frozen DINOv3 ViT-B/16 LVD backbone
- Image contract: raw 2560×720 top stereo JPEG is split, rectified/aligned with
  `Data/process` calibration, cropped x=20/y=0/1240×620, then only the left
  RGB eye is letterboxed to 640×480.  No generic crop or stretch is used.
- Deployment source commit: `{commit}`

The repository root contains the complete inference checkpoint only (model,
configuration and pre/post-processors).  It deliberately excludes optimizer
and RNG state.  The corresponding deployment wrapper is under
`deploy/robot8_20260729_3cam_act_dinov3_topcam_left_dataprocess_v1/`.

Repository: https://huggingface.co/{repo_id}
"""


def upload(
    root: Path,
    artifact_root: Path,
    run: Run,
    owner: str,
    public: bool,
) -> None:
    """Upload a staged final checkpoint plus deployment assets from ``root``.

    ``artifact_root`` is normally the repository root.  Keeping it separate
    also lets a workstation upload a final checkpoint staged from the training
    server without copying the whole source tree or any optimizer state.
    """
    ckpt = require_final_checkpoint(artifact_root, run)
    repo_id = run.repo_id(owner)
    api = HfApi()
    api.whoami()  # Fail before any write when no valid token is configured.
    api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=".",
        commit_message=f"Upload final {FINAL_STEP} inference checkpoint",
    )

    deploy_root = root / "scripts" / "deploy"
    for directory in (
        deploy_root / "robot8_20260729_3cam_act_dinov3_topcam_left_dataprocess_v1",
        deploy_root / "robot8_act_dinov3_base_frozen_100k",
    ):
        api.upload_folder(
            folder_path=str(directory),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"deploy/{directory.name}",
            ignore_patterns=["__pycache__/*", "*.pyc"],
            commit_message=f"Upload {directory.name} deployment code",
        )

    for calibration in (
        root / "Data" / "process" / "top_stereo_calibration_basalt_kb4_compat.json",
        root / "Data" / "process" / "processing_metadata_centered_crop_1240x620.json",
    ):
        api.upload_file(
            path_or_fileobj=str(calibration),
            path_in_repo=f"deploy/calibration/{calibration.name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload Data/process calibration {calibration.name}",
        )

    log = artifact_root / "scripts" / "train_log" / run.run_id / "train.log"
    if log.is_file():
        api.upload_file(
            path_or_fileobj=str(log),
            path_in_repo="logs/train.log",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload training log",
        )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md") as handle:
        handle.write(model_card(run, repo_id, source_commit(root)))
        handle.flush()
        api.upload_file(
            path_or_fileobj=handle.name,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Write model card",
        )
    print(f"uploaded https://huggingface.co/{repo_id}")


def download(root: Path, run: Run, owner: str, overwrite: bool) -> None:
    target = checkpoint(root, run)
    existing = [name for name in MODEL_FILES if (target / name).is_file()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to replace existing checkpoint at {target}; pass --overwrite after checking it"
        )
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=run.repo_id(owner),
        repo_type="model",
        allow_patterns=[*MODEL_FILES, "README.md"],
        local_dir=str(target),
    )
    missing = [name for name in MODEL_FILES if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"HF download is incomplete for {run.repo_id(owner)}: {missing}")
    print(f"downloaded {run.repo_id(owner)} -> {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("upload", "download"))
    parser.add_argument("--only", choices=("normal", "v3", "both"), default="both")
    parser.add_argument("--owner", default=OWNER_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Root containing outputs/train artifacts for upload; defaults to --repo-root",
    )
    parser.add_argument("--public", action="store_true", help="Make newly-created model repositories public")
    parser.add_argument("--overwrite", action="store_true", help="Allow download to replace existing model files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    artifact_root = (args.artifact_root or root).resolve()
    for run in selected_runs(args.only):
        if args.mode == "upload":
            upload(root, artifact_root, run, args.owner, args.public)
        else:
            download(root, run, args.owner, args.overwrite)


if __name__ == "__main__":
    main()
