#!/usr/bin/env python3
"""Upload/download a complete TeaRoom normal ACT checkpoint through HF Hub.

The Hub model repository contains both deployable inference files and the
optimizer/scheduler/RNG state needed to resume the exact saved step.  Upload is
intentionally permitted only after the trainer has atomically published a
numbered checkpoint with a matching ``training_step.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "robot8_20260729_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_ddp32_b16_seqcache_60k"
OWNER_DEFAULT = "QRP123"
DEPLOY_DIR_NAME = "robot8_20260729_tearoom_3cam_act_dinov3_topcam_left_cam20260729"
SHARED_DEPLOY_DIR_NAME = "robot8_act_dinov3_base_frozen_100k"
MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def normalize_socks_proxy() -> None:
    """Make a common local ``socks://`` setting valid for httpx/HF Hub."""
    for name in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(name)
        if value and value.startswith("socks://"):
            os.environ[name] = "socks5://" + value.removeprefix("socks://")


def checkpoint_dir(artifact_root: Path, step: int) -> Path:
    return artifact_root / "outputs" / "train" / RUN_ID / "checkpoints" / f"{step:06d}"


def checkpoint_paths(artifact_root: Path, step: int) -> tuple[Path, Path, Path]:
    checkpoint = checkpoint_dir(artifact_root, step)
    return checkpoint, checkpoint / "pretrained_model", checkpoint / "training_state"


def require_strict_download_target(
    artifact_root: Path, step: int, checkpoint: Path, pretrained: Path
) -> None:
    """Reject a target that is not exactly ``checkpoints/{step:06d}``.

    Keeping this check close to the Hub download protects against accidentally
    materializing e.g. a requested step 4,000 below ``checkpoints/040000``.
    The artifact root is resolved by ``main`` so these comparisons are lexical
    and unambiguous.
    """
    expected_checkpoint = checkpoint_dir(artifact_root, step)
    expected_pretrained = expected_checkpoint / "pretrained_model"
    if checkpoint != expected_checkpoint or pretrained != expected_pretrained:
        raise RuntimeError(
            "unsafe checkpoint download target: "
            f"expected {expected_checkpoint}, got {checkpoint}"
        )
    if checkpoint.name != f"{step:06d}" or checkpoint.parent.name != "checkpoints":
        raise RuntimeError(
            "checkpoint target must be exactly "
            f"checkpoints/{step:06d}, got {checkpoint}"
        )
    if checkpoint.is_symlink() or pretrained.is_symlink():
        raise RuntimeError(
            "refusing to download through a checkpoint/pretrained_model symlink: "
            f"{checkpoint}"
        )


def require_deployable_pretrained(artifact_root: Path, step: int) -> tuple[Path, Path, Path]:
    """Check only the inference payload, without requiring resume state."""
    checkpoint, pretrained, training_state = checkpoint_paths(artifact_root, step)
    require_strict_download_target(artifact_root, step, checkpoint, pretrained)
    missing = [pretrained / name for name in MODEL_FILES if not (pretrained / name).is_file()]
    if missing:
        details = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"deployable checkpoint {step} is incomplete:\n{details}")
    return checkpoint, pretrained, training_state


def require_complete_checkpoint(artifact_root: Path, step: int) -> tuple[Path, Path, Path]:
    checkpoint, pretrained, training_state = require_deployable_pretrained(artifact_root, step)
    missing: list[Path] = []
    step_file = training_state / "training_step.json"
    if not step_file.is_file():
        missing.append(step_file)
    if missing:
        details = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"checkpoint {step} is incomplete:\n{details}")
    payload = json.loads(step_file.read_text(encoding="utf-8"))
    if int(payload.get("step", -1)) != step:
        raise ValueError(f"{step_file} does not record step {step}")
    return checkpoint, pretrained, training_state


def require_remote_recorded_step(repo_id: str, step: int) -> int:
    """Verify the Hub model card records the exact requested checkpoint step."""
    readme = hf_hub_download(repo_id=repo_id, repo_type="model", filename="README.md")
    text = Path(readme).read_text(encoding="utf-8")
    match = re.search(r"(?m)^- Saved step:\s*`(\d+)`\s*$", text)
    if not match:
        raise ValueError(
            f"{repo_id} README.md does not contain the expected '- Saved step: `N`' record"
        )
    recorded_step = int(match.group(1))
    if recorded_step != step:
        raise ValueError(
            f"Hub checkpoint step mismatch: requested {step}, but {repo_id} README.md records "
            f"step {recorded_step}"
        )
    return recorded_step


def source_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def default_repo_id(owner: str, step: int) -> str:
    return f"{owner}/robot8-20260729-tearoom-act-dinov3-normal-{step // 1000}k"


def model_card(repo_id: str, step: int, commit: str) -> str:
    return f"""---
library_name: lerobot
tags:
- act
- dinov3
- robot8
- tearoom
- topcam-rectified
---

# TeaRoom normal ACT+DINOv3 checkpoint — step {step}

- Training run: `{RUN_ID}`
- Saved step: `{step}`
- Policy: ACT, decoder layers 7, frozen DINOv3 ViT-B/16 LVD backbone
- Cameras: `head_cam,left_arm_cam,right_arm_cam`
- State/action: normal raw 23D command vectors
- Visual contract: raw 2560×720 top stereo JPEG → split eyes → calibrated
  `cam_20260729` NPZ fisheye rectification/alignment → fixed left crop
  `(20, 0, 1240, 620)` → left RGB → 640×480 letterbox
- Deployment source commit: `{commit}`

Files at the repository root are the deployable `pretrained_model` files.
`training_state/` retains the optimizer, scheduler, RNG, and recorded training
step for exact resume.  Deployment wrappers and the calibration asset are under
`deploy/`.

Repository: https://huggingface.co/{repo_id}
"""


def upload(root: Path, artifact_root: Path, step: int, repo_id: str, public: bool) -> None:
    _, pretrained, training_state = require_complete_checkpoint(artifact_root, step)
    normalize_socks_proxy()
    api = HfApi()
    api.whoami()  # Verify credentials before creating or mutating a repository.
    api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
    api.upload_folder(
        folder_path=str(pretrained),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=".",
        commit_message=f"Upload deployable checkpoint step {step}",
    )
    api.upload_folder(
        folder_path=str(training_state),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="training_state",
        commit_message=f"Upload resumable training state step {step}",
    )

    deploy_root = root / "scripts" / "deploy"
    for directory in (deploy_root / DEPLOY_DIR_NAME, deploy_root / SHARED_DEPLOY_DIR_NAME):
        if not directory.is_dir():
            raise FileNotFoundError(f"deployment directory missing: {directory}")
        api.upload_folder(
            folder_path=str(directory),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"deploy/{directory.name}",
            ignore_patterns=["__pycache__/*", "*.pyc"],
            commit_message=f"Upload deployment code: {directory.name}",
        )
    for source, destination in (
        (
            root / "scripts" / "data_convert" / "topcam_stereo_rectify_cam_20260729.py",
            "deploy/data_convert/topcam_stereo_rectify_cam_20260729.py",
        ),
        (
            root / "scripts" / "data_convert" / "cam" / "stereo_params_20260729_172611.npz",
            "deploy/calibration/stereo_params_20260729_172611.npz",
        ),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"calibration deployment asset missing: {source}")
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=destination,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload deployment asset: {source.name}",
        )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md") as file:
        file.write(model_card(repo_id, step, source_commit(root)))
        file.flush()
        api.upload_file(
            path_or_fileobj=file.name,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Document checkpoint step {step}",
        )
    print(f"uploaded https://huggingface.co/{repo_id}")


def download(
    root: Path,
    artifact_root: Path,
    step: int,
    repo_id: str,
    overwrite: bool,
    deploy_only: bool,
) -> None:
    checkpoint, pretrained, training_state = checkpoint_paths(artifact_root, step)
    require_strict_download_target(artifact_root, step, checkpoint, pretrained)
    files_to_check = [pretrained / name for name in MODEL_FILES]
    if not deploy_only:
        files_to_check.append(training_state / "training_step.json")
    existing = [path for path in files_to_check if path.exists()]
    if existing and not overwrite:
        details = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to replace existing checkpoint files:\n{details}\npass --overwrite after checking them")
    normalize_socks_proxy()
    recorded_step = require_remote_recorded_step(repo_id, step)
    print(
        "download request: "
        f"repo={repo_id}, requested_step={step}, recorded_step={recorded_step}, "
        f"deploy_only={deploy_only}"
    )
    print(f"download target (strict): {checkpoint} [checkpoints/{step:06d}]")
    pretrained.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=list(MODEL_FILES),
        local_dir=str(pretrained),
    )
    if deploy_only:
        # This intentionally avoids ``training_state/**``: optimizer, scheduler,
        # and RNG files are not deployment dependencies.
        require_deployable_pretrained(artifact_root, step)
        print(f"downloaded deployable pretrained_model only: {repo_id} -> {pretrained}")
    else:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["training_state/**"],
            local_dir=str(checkpoint),
        )
        require_complete_checkpoint(artifact_root, step)
        print(f"downloaded complete resumable checkpoint: {repo_id} -> {checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("upload", "download"))
    parser.add_argument("--step", type=int, default=2_000, help="Numbered checkpoint step to synchronize")
    parser.add_argument("--owner", default=OWNER_DEFAULT)
    parser.add_argument("--repo-id", help="HF repository; defaults to a private TeaRoom step-specific repository")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Source checkout containing deploy code")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Root containing outputs/train artifacts; defaults to --repo-root",
    )
    parser.add_argument("--public", action="store_true", help="Make a newly-created model repository public")
    parser.add_argument("--overwrite", action="store_true", help="Allow download to overwrite existing checkpoint files")
    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help=(
            "Download only deployable pretrained_model files (model, config, and processors); "
            "do not download optimizer/RNG training_state"
        ),
    )
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("--step must be positive")
    if args.deploy_only and args.mode != "download":
        parser.error("--deploy-only is valid only with download")
    return args


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    artifact_root = (args.artifact_root or root).expanduser().resolve()
    repo_id = args.repo_id or default_repo_id(args.owner, args.step)
    if args.mode == "upload":
        upload(root, artifact_root, args.step, repo_id, args.public)
    else:
        download(root, artifact_root, args.step, repo_id, args.overwrite, args.deploy_only)


if __name__ == "__main__":
    main()
