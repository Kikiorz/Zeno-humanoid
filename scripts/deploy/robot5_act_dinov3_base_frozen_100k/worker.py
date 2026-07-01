#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch


ROBOT = "robot5"
ACTION_DIM = 23
NORM_EPS = 1e-6
IMAGE_KEYS = {
    "head_cam": "observation.images.head_cam",
    "left_arm_cam": "observation.images.left_arm_cam",
    "right_arm_cam": "observation.images.right_arm_cam",
}

REPO_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot5_20260623_act_dinov3_base_fullft"
    / "checkpoints"
    / "080000"
    / "pretrained_model"
)


def send_message(sock: socket.socket, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    size = struct.unpack("!I", recv_exact(sock, 4))[0]
    return pickle.loads(recv_exact(sock, size))


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    candidates: list[tuple[int, str, Path]] = []
    for step_dir in checkpoints_dir.iterdir():
        pretrained = step_dir / "pretrained_model"
        if not (pretrained / "model.safetensors").is_file():
            continue
        try:
            step = int(step_dir.name)
        except ValueError:
            step = -1
        candidates.append((step, step_dir.name, pretrained))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def resolve_checkpoint(path: str | None) -> Path:
    candidate = Path(path).expanduser() if path else DEFAULT_CHECKPOINT
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if (candidate / "model.safetensors").is_file():
        return candidate
    if (candidate / "pretrained_model" / "model.safetensors").is_file():
        return candidate / "pretrained_model"
    latest = latest_checkpoint(candidate)
    if latest is not None:
        return latest
    raise FileNotFoundError(f"model.safetensors not found under: {candidate}")


def decode_image(image_bytes: bytes, image_size: int) -> np.ndarray | None:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))


def set_norm_eps(pipeline: Any) -> None:
    for step in getattr(pipeline, "steps", []):
        if hasattr(step, "eps"):
            step.eps = NORM_EPS


def load_action_bounds(checkpoint: Path, margin: float) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None
    state_files: list[Path] = []
    for manifest_name in ("policy_postprocessor.json", "policy_preprocessor.json"):
        manifest_path = checkpoint / manifest_name
        if not manifest_path.is_file():
            continue
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        for step in manifest.get("steps", []):
            state_file = step.get("state_file")
            if state_file:
                state_files.append(checkpoint / state_file)
    for state_file in state_files:
        if not state_file.is_file():
            continue
        tensors = load_file(str(state_file), device="cpu")
        if "action.min" not in tensors or "action.max" not in tensors:
            continue
        action_min = tensors["action.min"].numpy().astype(np.float32).reshape(-1)
        action_max = tensors["action.max"].numpy().astype(np.float32).reshape(-1)
        if action_min.shape != (ACTION_DIM,) or action_max.shape != (ACTION_DIM,):
            continue
        action_range = np.maximum(action_max - action_min, NORM_EPS)
        return action_min - margin * action_range, action_max + margin * action_range
    return None


class ActWorker:
    def __init__(
        self,
        checkpoint: Path,
        device: str | None,
        image_size: int,
        use_amp: bool,
        clamp_actions: bool,
        action_clip_margin: float,
        n_action_steps: int | None,
        temporal_ensemble_coeff: float | None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.use_amp = use_amp
        self.clamp_actions = clamp_actions
        self.action_bounds = load_action_bounds(checkpoint, action_clip_margin)

        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        config.device = self.device
        if hasattr(config, "dinov2_pretrained"):
            config.dinov2_pretrained = False
        if hasattr(config, "dinov2_pretrained_weights"):
            config.dinov2_pretrained_weights = None
        if n_action_steps is not None:
            config.n_action_steps = n_action_steps
        if temporal_ensemble_coeff is not None:
            config.temporal_ensemble_coeff = temporal_ensemble_coeff
            if n_action_steps is None:
                config.n_action_steps = 1
        if getattr(config, "temporal_ensemble_coeff", None) is not None and getattr(config, "n_action_steps", 1) != 1:
            raise ValueError("temporal ensemble requires n_action_steps=1")

        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(checkpoint, config=config, local_files_only=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        set_norm_eps(self.preprocessor)
        set_norm_eps(self.postprocessor)

    def select_action(self, state: Sequence[float], images: dict[str, bytes]) -> list[float]:
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.shape != (ACTION_DIM,):
            raise ValueError(f"state shape {state_np.shape}, expected {(ACTION_DIM,)}")
        if not np.isfinite(state_np).all():
            raise ValueError("state contains NaN or Inf")

        observation: dict[str, torch.Tensor] = {"observation.state": torch.from_numpy(state_np)}
        for image_name, feature_key in IMAGE_KEYS.items():
            image_bytes = images.get(image_name)
            if not image_bytes:
                raise ValueError(f"missing image {image_name}")
            image = decode_image(image_bytes, self.image_size)
            if image is None:
                raise ValueError(f"failed to decode image {image_name}")
            observation[feature_key] = torch.from_numpy(image)

        batch = self.preprocessor(observation)
        device_type = self.device.split(":", maxsplit=1)[0]
        with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=self.use_amp and device_type == "cuda"):
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        action_np = np.squeeze(action_np).astype(np.float32)
        if action_np.shape != (ACTION_DIM,):
            raise ValueError(f"action shape {action_np.shape}, expected {(ACTION_DIM,)}")
        if not np.isfinite(action_np).all():
            raise ValueError("action contains NaN or Inf")
        if self.clamp_actions and self.action_bounds is not None:
            low, high = self.action_bounds
            action_np = np.clip(action_np, low, high)
        return action_np.astype(float).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{ROBOT} ACT+DINOv3 inference worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--clamp-actions", dest="clamp_actions", action="store_true", default=True)
    parser.add_argument("--no-clamp-actions", dest="clamp_actions", action="store_false")
    parser.add_argument("--action-clip-margin", type=float, default=0.05)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    return parser.parse_args()


def serve_client(client: socket.socket, worker: ActWorker) -> None:
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        request = recv_message(client)
        start = time.perf_counter()
        try:
            action = worker.select_action(request["state"], request["images"])
            send_message(client, {"ok": True, "action": action, "latency_s": time.perf_counter() - start})
        except Exception as exc:
            send_message(client, {"ok": False, "error": str(exc)})


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint_path)
    worker = ActWorker(
        checkpoint,
        args.device,
        args.image_size,
        args.use_amp,
        args.clamp_actions,
        args.action_clip_margin,
        args.n_action_steps,
        args.temporal_ensemble_coeff,
    )
    clamp_status = "enabled" if worker.clamp_actions and worker.action_bounds is not None else "disabled"
    print(
        f"[{ROBOT} worker] checkpoint={checkpoint}; device={worker.device}; "
        f"n_action_steps={worker.policy.config.n_action_steps}; "
        f"temporal_ensemble_coeff={worker.policy.config.temporal_ensemble_coeff}; "
        f"action_clamp={clamp_status}; listening={args.host}:{args.port}",
        flush=True,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        while True:
            client, address = server.accept()
            print(f"[{ROBOT} worker] bridge connected: {address}", flush=True)
            with client:
                try:
                    serve_client(client, worker)
                except ConnectionError:
                    print(f"[{ROBOT} worker] bridge disconnected", flush=True)


if __name__ == "__main__":
    main()
