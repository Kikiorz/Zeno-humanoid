#!/usr/bin/env python3
"""Transfer selected 2026-07-31 ACT+DINOv3 inference checkpoints via HF Hub.

Default artifacts deliberately contain two final models plus two representative
intermediate snapshots: RAW-40k, RAW-100k, V3-70k, and V3-100k. They are placed
under outputs/train/<run>/checkpoints/<step>/pretrained_model, so either
deployment wrapper can load them without copying optimizer/RNG training state.

Run upload on the training server, then download in this repository on the
target workstation. Model repositories are private unless --public is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    default_steps: tuple[int, ...]
    description: str

    def repo_id(self, owner: str) -> str:
        return f"{owner}/{self.hub_name}"


RUNS = (
    Run(
        key="raw",
        run_id=(
            "robot8_20260731_act_dinov3_3cam_640x480_topcam_left_cam20260729_all15_"
            "decoder7_b32_bf16_nogc_resume_100k_onthefly"
        ),
        hub_name="robot8-20260731-act-dinov3-raw",
        default_steps=(40_000, 100_000),
        description="normal 23D raw-command labels",
    ),
    Run(
        key="v3",
        run_id=(
            "robot8_20260731_act_dinov3_3cam_640x480_topcam_left_cam20260729_all15_"
            "base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_bf16_"
            "nogc_resume_100k_onthefly"
        ),
        hub_name="robot8-20260731-act-dinov3-v3",
        default_steps=(70_000, 100_000),
        description="V3 endpoint-anchored, decoupled-smoothed physical-base labels",
    ),
)


def selected_runs(which: str) -> Iterable[Run]:
    return RUNS if which == "both" else tuple(run for run in RUNS if run.key == which)


def configure_hf_transport() -> None:
    """Make Hub downloads reliable on workstations using a local SOCKS proxy.

    Some client stacks accept only the standard socks5:// spelling, while the
    local environment exports socks://. In that situation Xet can create many
    stalled streams for large safetensors, so use ordinary resumable LFS unless
    the caller explicitly chose another setting.
    """

    normalized_socks_proxy = False
    for name in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(name)
        if value and value.startswith("socks://"):
            os.environ[name] = "socks5://" + value.removeprefix("socks://")
            normalized_socks_proxy = True
    if normalized_socks_proxy:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def parse_steps(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    try:
        steps = tuple(sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--steps must be comma-separated integers") from exc
    if not steps or any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("--steps must contain positive integers")
    return steps


def checkpoint(root: Path, run: Run, step: int) -> Path:
    return (
        root
        / "outputs"
        / "train"
        / run.run_id
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )


def validate_source_checkpoint(root: Path, run: Run, step: int) -> Path:
    model_dir = checkpoint(root, run, step)
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    step_file = model_dir.parent / "training_state" / "training_step.json"
    if not step_file.is_file():
        missing.append(str(step_file))
    if missing:
        raise FileNotFoundError(
            f"Incomplete checkpoint for {run.key} step {step}:\\n"
            + "\\n".join(str(model_dir / name) if name in MODEL_FILES else name for name in missing)
        )
    payload = json.loads(step_file.read_text(encoding="utf-8"))
    if int(payload.get("step", -1)) != step:
        raise ValueError(f"{step_file} records {payload.get('step')!r}, expected {step}")
    return model_dir


def model_card(run: Run, owner: str) -> str:
    snapshot_lines = "\\n".join(f"- {step:06d}" for step in run.default_steps)
    return f"""---
library_name: lerobot
tags:
- act
- dinov3
- robot8
- topcam-rectified
---

# {run.run_id}

- Variant: {run.description}
- Cameras: head_cam,left_arm_cam,right_arm_cam
- State/action: 23D whole-body vector
- Policy: ACT with a seven-layer decoder and frozen DINOv3 ViT-B/16 LVD vision
- Image contract: raw 2560x720 top stereo JPEG -> fisheye rectify/align ->
  crop x=20/y=0/1240x620 -> left RGB -> 640x480 letterbox.
- Available inference checkpoints:

{snapshot_lines}

The repository contains inference files only. Local deployment code is tracked
in Kikiorz/Zeno-humanoid under scripts/deploy/.
"""


def upload(root: Path, run: Run, owner: str, steps: tuple[int, ...], public: bool) -> None:
    api = HfApi()
    api.whoami()
    repo_id = run.repo_id(owner)
    api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
    for step in steps:
        model_dir = validate_source_checkpoint(root, run, step)
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"checkpoints/{step:06d}/pretrained_model",
            commit_message=f"Upload {run.key} inference checkpoint {step:06d}",
        )
        print(f"uploaded {repo_id} checkpoint {step:06d}", flush=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as manifest:
        json.dump(
            {
                "run_id": run.run_id,
                "variant": run.key,
                "selected_steps": list(steps),
                "inference_files": list(MODEL_FILES),
            },
            manifest,
            indent=2,
        )
        manifest.flush()
        api.upload_file(
            path_or_fileobj=manifest.name,
            path_in_repo="snapshots.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Record selected inference snapshots",
        )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md") as card:
        card.write(model_card(run, owner))
        card.flush()
        api.upload_file(
            path_or_fileobj=card.name,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Write model card",
        )


def download(root: Path, run: Run, owner: str, steps: tuple[int, ...], overwrite: bool) -> None:
    target_root = root / "outputs" / "train" / run.run_id
    existing = [
        checkpoint(root, run, step) / name
        for step in steps
        for name in MODEL_FILES
        if (checkpoint(root, run, step) / name).is_file()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing model files; inspect them or pass --overwrite:\\n"
            + "\\n".join(str(path) for path in existing)
        )
    patterns = [
        f"checkpoints/{step:06d}/pretrained_model/{name}"
        for step in steps
        for name in MODEL_FILES
    ]
    snapshot_download(
        repo_id=run.repo_id(owner),
        repo_type="model",
        allow_patterns=[*patterns, "README.md", "snapshots.json"],
        local_dir=str(target_root),
    )
    for step in steps:
        missing = [name for name in MODEL_FILES if not (checkpoint(root, run, step) / name).is_file()]
        if missing:
            raise RuntimeError(
                f"HF download incomplete for {run.repo_id(owner)} step {step}: {missing}"
            )
    print(f"downloaded {run.repo_id(owner)} -> {target_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("upload", "download"))
    parser.add_argument("--only", choices=("raw", "v3", "both"), default="both")
    parser.add_argument("--owner", default=OWNER_DEFAULT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT,
        help="Root containing outputs/train for upload or receiving them for download.",
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
        help="Optional comma-separated steps applied to every selected variant.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--public", action="store_true", help="Create new HF model repositories as public.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_transport()
    root = args.artifact_root.resolve()
    for run in selected_runs(args.only):
        steps = args.steps or run.default_steps
        if args.mode == "upload":
            upload(root, run, args.owner, steps, args.public)
        else:
            download(root, run, args.owner, steps, args.overwrite)


if __name__ == "__main__":
    main()
