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

REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402


ACTION_DIM = 23
NORM_EPS = 1e-6
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "human_new_pick_act_dinov2_dinoft_20260609_093740"
)
DEFAULT_DATASET_STATS = REPO_ROOT / "Data" / "lerobot" / "human_new_pick_zeno_h1_v30" / "meta" / "stats.json"

# Embedded from Data/lerobot/human_new_pick_zeno_h1_v30/meta/stats.json so deployment
# does not require the training dataset on the robot computer.
EMBEDDED_ACTION_MIN = np.asarray(
    [
        -0.04453912004828453,
        -0.08728089183568954,
        -0.8109204173088074,
        -0.4053068161010742,
        -0.23436853289604187,
        -0.0791284441947937,
        -0.09168343245983124,
        1.1333708763122559,
        -0.09339592605829239,
        -0.1672649383544922,
        -0.2150437831878662,
        -1.9192146062850952,
        -0.5235917568206787,
        -0.1744491159915924,
        0.28437376022338867,
        -1.524909496307373,
        -0.7853744626045227,
        -0.8922497630119324,
        -4.0201587718502463e-13,
        -0.003583333222195506,
        -0.05930357053875923,
        -0.03990600258111954,
        -5.826073029232843e-15,
    ],
    dtype=np.float32,
)
EMBEDDED_ACTION_MAX = np.asarray(
    [
        6.9454602659446696e-12,
        0.14305707812309265,
        0.13418078422546387,
        0.1760386973619461,
        0.3229830861091614,
        0.10675428807735443,
        0.4779173731803894,
        2.0329668521881104,
        0.20694151520729065,
        0.14939837157726288,
        0.3766157329082489,
        0.2568458020687103,
        0.4499939978122711,
        1.2510985136032104,
        2.172759532928467,
        0.2769336998462677,
        0.478831946849823,
        1.046875,
        0.7357653379440308,
        3.221508502960205,
        0.19143040478229523,
        0.16519059240818024,
        0.2768456041812897,
    ],
    dtype=np.float32,
)
EMBEDDED_ACTION_STATS_SOURCE = "embedded human_new_pick_zeno_h1_v30 action min/max"

IMAGE_KEYS = {
    "head_cam": "observation.images.head_cam",
    "left_arm_cam": "observation.images.left_arm_cam",
    "right_arm_cam": "observation.images.right_arm_cam",
}


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

    checkpoints: list[tuple[int, str, Path]] = []
    for step_dir in checkpoints_dir.iterdir():
        pretrained = step_dir / "pretrained_model"
        if not (step_dir.is_dir() and (pretrained / "model.safetensors").is_file()):
            continue
        try:
            step = int(step_dir.name)
        except ValueError:
            step = -1
        checkpoints.append((step, step_dir.name, pretrained))

    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return checkpoints[-1][2]


def resolve_checkpoint_path(path: str | Path | None) -> Path:
    candidate = Path(path).expanduser() if path else DEFAULT_RUN_DIR
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()

    if (candidate / "model.safetensors").is_file():
        return candidate
    if (candidate / "pretrained_model" / "model.safetensors").is_file():
        return candidate / "pretrained_model"

    checkpoint = latest_checkpoint(candidate)
    if checkpoint is not None:
        return checkpoint

    raise FileNotFoundError(
        "Could not find model.safetensors. Pass a run dir, checkpoint dir, "
        f"or pretrained_model dir. Got: {candidate}"
    )


def resolve_stats_path(stats_path: str | Path | None, checkpoint_path: Path) -> Path | None:
    if stats_path:
        candidate = Path(stats_path).expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate

    train_config = checkpoint_path / "train_config.json"
    if train_config.is_file():
        with train_config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        dataset_root = cfg.get("dataset", {}).get("root")
        if dataset_root:
            candidate = Path(dataset_root) / "meta" / "stats.json"
            if candidate.is_file():
                return candidate

    return DEFAULT_DATASET_STATS if DEFAULT_DATASET_STATS.is_file() else None


def build_action_bounds(
    action_min: np.ndarray,
    action_max: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if action_min.shape != (ACTION_DIM,) or action_max.shape != (ACTION_DIM,):
        return None

    # Keep a 1e-6 floor so nearly-constant dimensions still get a valid clamp range.
    action_range = np.maximum(action_max - action_min, NORM_EPS)
    low = action_min - margin * action_range
    high = action_max + margin * action_range
    return low, high


def load_action_bounds(stats_path: Path | None, margin: float) -> tuple[tuple[np.ndarray, np.ndarray] | None, str]:
    if stats_path is not None and stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        action_stats = stats.get("action")
        if action_stats:
            action_min = np.asarray(action_stats["min"], dtype=np.float32).reshape(-1)
            action_max = np.asarray(action_stats["max"], dtype=np.float32).reshape(-1)
            bounds = build_action_bounds(action_min, action_max, margin)
            if bounds is not None:
                return bounds, str(stats_path)

    bounds = build_action_bounds(EMBEDDED_ACTION_MIN, EMBEDDED_ACTION_MAX, margin)
    return bounds, EMBEDDED_ACTION_STATS_SOURCE


def decode_image(image_bytes: bytes, image_size: int) -> np.ndarray | None:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))


def set_norm_eps(pipeline: Any, eps: float) -> None:
    for step in getattr(pipeline, "steps", []):
        if hasattr(step, "eps"):
            step.eps = eps


class ActDinoV2Runner:
    def __init__(
        self,
        checkpoint_path: str | None,
        stats_path: str | None,
        device: str | None,
        use_amp: bool,
        image_size: int,
        clamp_actions: bool,
        action_clip_margin: float,
    ) -> None:
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.stats_path = resolve_stats_path(stats_path, self.checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.image_size = image_size
        self.clamp_actions = clamp_actions
        self.action_bounds, self.action_bounds_source = load_action_bounds(self.stats_path, action_clip_margin)

        self.policy, self.preprocessor, self.postprocessor = self.load_policy()

    def load_policy(self) -> tuple[Any, Any, Any]:
        config = PreTrainedConfig.from_pretrained(self.checkpoint_path, local_files_only=True)
        config.device = self.device
        if hasattr(config, "dinov2_pretrained"):
            config.dinov2_pretrained = False

        policy_class = get_policy_class(config.type)
        policy = policy_class.from_pretrained(
            self.checkpoint_path,
            config=config,
            local_files_only=True,
        )
        policy.to(self.device)
        policy.eval()
        policy.reset()

        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(self.checkpoint_path),
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        set_norm_eps(preprocessor, NORM_EPS)
        set_norm_eps(postprocessor, NORM_EPS)
        return policy, preprocessor, postprocessor

    def select_action(self, state: Sequence[float], images: dict[str, bytes]) -> list[float]:
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.shape != (ACTION_DIM,):
            raise ValueError(f"state shape is {state_np.shape}, expected {(ACTION_DIM,)}")
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
        autocast_enabled = self.use_amp and device_type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=autocast_enabled):
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)

        action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        action_np = np.squeeze(action_np).astype(np.float32)
        if action_np.shape != (ACTION_DIM,):
            raise ValueError(f"action shape is {action_np.shape}, expected {(ACTION_DIM,)}")
        if not np.isfinite(action_np).all():
            raise ValueError("action contains NaN or Inf")

        if self.clamp_actions and self.action_bounds is not None:
            low, high = self.action_bounds
            action_np = np.clip(action_np, low, high)
        return action_np.astype(float).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="human_new_pick ACT+DINOv2 socket worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--clamp-actions", dest="clamp_actions", action="store_true", default=True)
    parser.add_argument("--no-clamp-actions", dest="clamp_actions", action="store_false")
    parser.add_argument("--action-clip-margin", type=float, default=0.05)
    return parser.parse_args()


def serve_client(client: socket.socket, runner: ActDinoV2Runner) -> None:
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        request = recv_message(client)
        start = time.perf_counter()
        try:
            action = runner.select_action(request["state"], request["images"])
            send_message(client, {"ok": True, "action": action, "latency_s": time.perf_counter() - start})
        except Exception as exc:
            send_message(client, {"ok": False, "error": str(exc)})


def main() -> None:
    args = parse_args()
    runner = ActDinoV2Runner(
        args.checkpoint_path,
        args.stats_path,
        args.device,
        args.use_amp,
        args.image_size,
        args.clamp_actions,
        args.action_clip_margin,
    )
    bounds_status = "enabled" if runner.action_bounds is not None and runner.clamp_actions else "disabled"
    print(
        f"[worker] loaded {runner.checkpoint_path} on {runner.device}; "
        f"action_stats={runner.action_bounds_source}; action_clamp={bounds_status}; "
        f"listening on {args.host}:{args.port}",
        flush=True,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        while True:
            client, address = server.accept()
            print(f"[worker] bridge connected: {address}", flush=True)
            with client:
                try:
                    serve_client(client, runner)
                except ConnectionError:
                    print("[worker] bridge disconnected", flush=True)


if __name__ == "__main__":
    main()
