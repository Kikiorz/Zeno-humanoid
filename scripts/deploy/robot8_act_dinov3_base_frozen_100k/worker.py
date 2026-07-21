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


ROBOT = "robot8"
ACTION_DIM = 23
NORM_EPS = 1e-6
ACTION_FIELDS = (
    "torso_lift",
    "torso_waist",
    "head_pan",
    "head_tilt",
    *(f"left_arm_j{i}" for i in range(7)),
    *(f"right_arm_j{i}" for i in range(7)),
    "left_gripper",
    "right_gripper",
    "base_vx",
    "base_vy",
    "base_rotation",
)
ACTION_FIELD_TO_INDEX = {name: index for index, name in enumerate(ACTION_FIELDS)}
KNOWN_IMAGE_KEYS = {
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
    / "robot8_20260708_3cam_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709"
    / "checkpoints"
    / "100000"
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


def checkpoint_image_size(config: PreTrainedConfig) -> tuple[int, int]:
    for feature in config.input_features.values():
        feature_type = getattr(feature, "type", None)
        feature_type_name = getattr(feature_type, "value", feature_type)
        if feature_type_name != "VISUAL":
            continue
        shape = tuple(int(v) for v in feature.shape)
        if len(shape) != 3:
            continue
        _, height, width = shape
        return width, height
    return 224, 224


def checkpoint_image_keys(config: PreTrainedConfig) -> dict[str, str]:
    image_keys: dict[str, str] = {}
    for feature_key, feature in config.input_features.items():
        feature_type = getattr(feature, "type", None)
        feature_type_name = getattr(feature_type, "value", feature_type)
        if feature_type_name != "VISUAL":
            continue
        camera_name = feature_key.rsplit(".", maxsplit=1)[-1]
        if camera_name not in KNOWN_IMAGE_KEYS:
            raise ValueError(f"unsupported visual feature in checkpoint: {feature_key}")
        image_keys[camera_name] = feature_key
    if not image_keys:
        raise ValueError("checkpoint has no visual input features")
    return image_keys


def center_crop_image(image: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 1.0:
        return image
    height, width = image.shape[:2]
    crop_width = max(1, min(width, int(round(width * fraction))))
    crop_height = max(1, min(height, int(round(height * fraction))))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width]


def decode_image(image_bytes: bytes, image_size: tuple[int, int], center_crop_fraction: float) -> np.ndarray | None:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_bgr = center_crop_image(image_bgr, center_crop_fraction)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, image_size, interpolation=cv2.INTER_LINEAR)
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


def optional_float(value: str) -> float | None:
    if value.lower() in {"none", "off", "false", "null"}:
        return None
    return float(value)


def parse_frozen_fields(raw: str) -> tuple[str, ...]:
    """Parse 23D command fields that must be zeroed at the worker boundary."""
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in ACTION_FIELD_TO_INDEX]
    if unknown:
        known = ", ".join(ACTION_FIELDS)
        raise ValueError(f"Unknown --frozen-fields value(s): {unknown}. Known fields: {known}")
    return tuple(dict.fromkeys(names))


class ActWorker:
    def __init__(
        self,
        checkpoint: Path,
        device: str | None,
        image_size: tuple[int, int] | None,
        center_crop_fraction: float,
        use_amp: bool,
        clamp_actions: bool,
        action_clip_margin: float,
        n_action_steps: int | None,
        temporal_ensemble_coeff: float | None,
        frozen_action_indices: Sequence[int] = (),
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.clamp_actions = clamp_actions
        self.frozen_action_indices = tuple(sorted(set(int(index) for index in frozen_action_indices)))
        if any(index < 0 or index >= ACTION_DIM for index in self.frozen_action_indices):
            raise ValueError(f"frozen action indices must be within [0, {ACTION_DIM - 1}]")
        self.action_bounds = load_action_bounds(checkpoint, action_clip_margin)

        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        resolved_image_size = image_size or checkpoint_image_size(config)
        self.image_size = (int(resolved_image_size[0]), int(resolved_image_size[1]))
        self.center_crop_fraction = center_crop_fraction
        self.image_keys = checkpoint_image_keys(config)
        config.device = self.device
        # The checkpoint contains the complete vision backbone.  Disable
        # initializer-only pretrained weights so deployment never needs a
        # torchvision download (notably for ResNet checkpoints).
        if hasattr(config, "pretrained_backbone_weights"):
            config.pretrained_backbone_weights = None
        if hasattr(config, "dinov2_pretrained"):
            config.dinov2_pretrained = False
        if hasattr(config, "dinov2_pretrained_weights"):
            config.dinov2_pretrained_weights = None
        if n_action_steps is not None:
            if not 1 <= n_action_steps <= int(config.chunk_size):
                raise ValueError(
                    f"n_action_steps must be in [1, {config.chunk_size}], got {n_action_steps}"
                )
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
        if self.frozen_action_indices:
            # Dataset conversion uses the same raw zero convention.  Applying
            # it before the checkpoint normalizer prevents frozen joint sensor
            # jitter (or a bad frozen sensor sample) from becoming a model
            # input.
            state_np = state_np.copy()
            state_np[list(self.frozen_action_indices)] = 0.0
        if not np.isfinite(state_np).all():
            raise ValueError("state contains NaN or Inf")

        observation: dict[str, torch.Tensor] = {"observation.state": torch.from_numpy(state_np)}
        for image_name, feature_key in self.image_keys.items():
            image_bytes = images.get(image_name)
            if not image_bytes:
                raise ValueError(f"missing image {image_name}")
            image = decode_image(image_bytes, self.image_size, self.center_crop_fraction)
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
        if self.frozen_action_indices:
            # Last model-side safety boundary.  The ROS bridge repeats this
            # mask immediately before publication as independent protection.
            action_np[list(self.frozen_action_indices)] = 0.0
        if not np.isfinite(action_np).all():
            raise ValueError("action contains NaN or Inf")
        if self.clamp_actions and self.action_bounds is not None:
            low, high = self.action_bounds
            action_np = np.clip(action_np, low, high)
        return action_np.astype(float).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{ROBOT} ACT inference worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional override for image resize size; defaults to checkpoint input shape",
    )
    parser.add_argument(
        "--center-crop-fraction",
        type=float,
        default=1.0,
        help="Center crop fraction before resize. Use 0.6666667 for crop2of3 checkpoints.",
    )
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--clamp-actions", dest="clamp_actions", action="store_true", default=True)
    parser.add_argument("--no-clamp-actions", dest="clamp_actions", action="store_false")
    parser.add_argument("--action-clip-margin", type=float, default=0.05)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=optional_float, default=None)
    parser.add_argument(
        "--frozen-fields",
        default="",
        help=(
            "Comma-separated 23D state/action fields forced to zero before model input "
            "and after model output; e.g. torso_lift,torso_waist"
        ),
    )
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
    if not (0 < args.center_crop_fraction <= 1.0):
        raise SystemExit("--center-crop-fraction must be in the range (0, 1]")
    try:
        frozen_fields = parse_frozen_fields(args.frozen_fields)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    checkpoint = resolve_checkpoint(args.checkpoint_path)
    args.temporal_ensemble_coeff=None
    worker = ActWorker(
        checkpoint,
        args.device,
        args.image_size,
        args.center_crop_fraction,
        args.use_amp,
        args.clamp_actions,
        args.action_clip_margin,
        args.n_action_steps,
        args.temporal_ensemble_coeff,
        [ACTION_FIELD_TO_INDEX[field] for field in frozen_fields],
    )
    clamp_status = "enabled" if worker.clamp_actions and worker.action_bounds is not None else "disabled"
    print(
        f"[{ROBOT} worker] checkpoint={checkpoint}; device={worker.device}; "
        f"cameras={','.join(worker.image_keys)}; image_size={worker.image_size}; "
        f"center_crop_fraction={worker.center_crop_fraction}; "
        f"n_action_steps={worker.policy.config.n_action_steps}; "
        f"temporal_ensemble_coeff={worker.policy.config.temporal_ensemble_coeff}; "
        f"frozen_fields={','.join(frozen_fields) if frozen_fields else 'none'}; "
        f"action_clamp={clamp_status}; listening={args.host}:{args.port}",
        flush=True,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        try:
            while True:
                client, address = server.accept()
                print(f"[{ROBOT} worker] bridge connected: {address}", flush=True)
                with client:
                    try:
                        # Do not reuse queued ACT actions after a bridge restart.
                        worker.policy.reset()
                        serve_client(client, worker)
                    except ConnectionError:
                        print(f"[{ROBOT} worker] bridge disconnected", flush=True)
        except KeyboardInterrupt:
            print(f"[{ROBOT} worker] stopped", flush=True)


if __name__ == "__main__":
    main()
