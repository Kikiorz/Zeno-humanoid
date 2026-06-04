#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


NORM_EPS = 1e-6
ACTION_DIM = 23
DEFAULT_RUN_PREFIX = "humanmoid_pick_act_dinov2_auto_cmd23"

IMAGE_KEYS = {
    "head_cam": "observation.images.head_cam",
    "left_arm_cam": "observation.images.left_arm_cam",
    "right_arm_cam": "observation.images.right_arm_cam",
}

STATE_MEAN = [
    0.06629907019526637, -0.05590814082025135, 0.11646821813029283,
    -0.6266132368715587, 0.057981648629609396, 0.045238616382455014,
    -0.04236767465822407, 0.08916511262711597, -0.04190582294810735,
    0.007063568155413458, 0.002217752039731857, -0.06180571561676185,
    -0.06819940049314288, 0.0540851632584586, 1.0941904276011145,
    0.02510407164758101, 0.09202907995133942, -0.003169012435936132,
    0.08175789400053053, 1.213155426491627, 0.012292727681802864,
    0.018929421771833183, 0.0030569169527993587,
]
STATE_STD = [
    0.15409339708025388, 0.06973948412041121, 0.1567311718276111,
    4.5253110222064015, 0.06598179155410612, 0.04575642793076321,
    0.05023995200247108, 0.13098848817304856, 0.060494029800189794,
    0.06458363415167175, 0.08812887392272124, 0.32200123709527795,
    0.04918729363212187, 0.07198849763681336, 0.7306975594472097,
    0.07365033358428291, 0.11591193587736273, 0.22251401856895184,
    0.019694628447070185, 0.9797960658418878, 0.03605304428207425,
    0.044003668821573624, 0.04177032635031798,
]
STATE_MIN = [
    -0.46292057633399963, -0.325970858335495, 0.050164032727479935, -12.5,
    -0.1348516047000885, -0.14553292095661163, -0.2010619342327118,
    0.013194689527153969, -0.2447165697813034, -0.18940261006355286,
    -0.612840473651886, -1.2941558361053467, -0.14286258816719055,
    -0.2079734355211258, -0.03769911080598831, -0.24090181291103363,
    -0.2782863974571228, -0.7230868935585022, 0.06618601083755493,
    0.07724879682064056, -0.08328412473201752, -0.019552122801542282,
    -0.3921944499015808,
]
STATE_MAX = [
    0.5548561811447144, 0.19054703414440155, 1.2407492399215698,
    12.49542236328125, 0.5125123858451843, 0.23899443447589874,
    0.10430087894201279, 1.8767874240875244, 0.11387044936418533,
    0.21648737788200378, 0.6689173579216003, 1.0660333633422852,
    0.134470134973526, 0.24190263450145721, 2.425309419631958,
    0.30117493867874146, 0.3583962619304657, 1.0400930643081665,
    1.725986123085022, 2.1730754375457764, 0.17503295838832855,
    0.2400037944316864, 0.5401923656463623,
]

ACTION_MEAN = [
    -0.10365233838721921, 0.005942048270958422, -0.05605904515871872,
    0.06571494646222371, 0.0639829378394914, 0.04378487446630782,
    -0.04039570103033145, 0.0579352438273034, -0.04256340922866348,
    0.009985926210418945, -0.000880431673688093, -0.08291973692791084,
    -0.11328279076740563, 0.044875480293795914, 1.1287692557671676,
    0.028022327834596628, 0.09329303081958389, 0.06336851246769414,
    0.0003146091873079551, 1.6830681658332192, 0.012292727681802864,
    0.018929421771833183, 0.0030569169527993587,
]
ACTION_STD = [
    0.18232188102310073, 0.05465406304543733, 0.07004855443912195,
    0.15955475716193016, 0.09811029229217647, 0.04872495984705921,
    0.06100737530421967, 0.1415972983946157, 0.06144951709354732,
    0.06482474507664222, 0.08863812066931381, 0.33707291526284777,
    0.09921649491013156, 0.07615506879940818, 0.7328309614754072,
    0.07947317622954973, 0.12042855074696196, 0.2125301371944121,
    0.02708957437238535, 1.4588431248692368, 0.03605304428207425,
    0.044003668821573624, 0.04177032635031798,
]
ACTION_MIN = [
    -0.599999189376831, -0.08729226887226105, -0.3251272737979889,
    -0.6976150870323181, -0.14012131094932556, -0.14061851799488068,
    -0.23400533199310303, -4.1348270662243294e-16, -0.24915514886379242,
    -0.19362764060497284, -0.6066493988037109, -1.3754724264144897,
    -0.34757131338119507, -0.25229406356811523, -6.38047221330656e-17,
    -0.2511748969554901, -0.2860746681690216, -0.7242950201034546,
    0.0, 0.0, -0.08328412473201752, -0.019552122801542282,
    -0.3921944499015808,
]
ACTION_MAX = [
    0.0, 0.3146280348300934, 0.19017180800437927, 0.6016148924827576,
    1.1035407781600952, 0.2538188099861145, 0.12104932963848114,
    1.9046839475631714, 0.11672401428222656, 0.22020936012268066,
    0.6801795959472656, 1.0807044506072998, 0.14945131540298462,
    0.25256162881851196, 2.4273970127105713, 0.3349626362323761,
    0.36604490876197815, 1.0446527004241943, 2.950000047683716,
    2.950000047683716, 0.17503295838832855, 0.2400037944316864,
    0.5401923656463623,
]

HEAD_CAM_MEAN = [0.49867124790837025, 0.5207219249645205, 0.4707800516668092]
HEAD_CAM_STD = [0.004661754117081969, 0.0030363431985702665, 0.0014945382795570614]
LEFT_ARM_CAM_MEAN = [0.5278932394937336, 0.5470216758385095, 0.5114780027802325]
LEFT_ARM_CAM_STD = [0.0048412498545484005, 0.004683366412521938, 0.005658771356362135]
RIGHT_ARM_CAM_MEAN = [0.21465043091970087, 0.2286127294677534, 0.22356414038153136]
RIGHT_ARM_CAM_STD = [0.09429959684155328, 0.10075341821000729, 0.088536258845907]


def safe_std(values: Sequence[float]) -> list[float]:
    return [max(abs(float(value)), NORM_EPS) for value in values]


DEPLOY_STATS = {
    "observation.state": {
        "mean": STATE_MEAN,
        "std": safe_std(STATE_STD),
        "min": STATE_MIN,
        "max": STATE_MAX,
    },
    "action": {
        "mean": ACTION_MEAN,
        "std": safe_std(ACTION_STD),
        "min": ACTION_MIN,
        "max": ACTION_MAX,
    },
    "observation.images.head_cam": {
        "mean": HEAD_CAM_MEAN,
        "std": safe_std(HEAD_CAM_STD),
    },
    "observation.images.left_arm_cam": {
        "mean": LEFT_ARM_CAM_MEAN,
        "std": safe_std(LEFT_ARM_CAM_STD),
    },
    "observation.images.right_arm_cam": {
        "mean": RIGHT_ARM_CAM_MEAN,
        "std": safe_std(RIGHT_ARM_CAM_STD),
    },
}


def send_message(sock: socket.socket, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
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


def latest_default_run_dir() -> Path:
    output_dir = REPO_ROOT / "outputs" / "train"
    if not output_dir.is_dir():
        return output_dir / DEFAULT_RUN_PREFIX
    runs = sorted(path for path in output_dir.glob(f"{DEFAULT_RUN_PREFIX}_*") if path.is_dir())
    return runs[-1] if runs else output_dir / DEFAULT_RUN_PREFIX


def resolve_checkpoint_path(path: str | Path | None) -> Path:
    candidate = Path(path).expanduser() if path else latest_default_run_dir()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()

    if (candidate / "model.safetensors").is_file():
        return candidate
    if (candidate / "pretrained_model" / "model.safetensors").is_file():
        return candidate / "pretrained_model"

    checkpoints_dir = candidate / "checkpoints"
    if checkpoints_dir.is_dir():
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
        if checkpoints:
            checkpoints.sort(key=lambda item: (item[0], item[1]))
            return checkpoints[-1][2]

    raise FileNotFoundError(
        "Could not find model.safetensors. Pass a run dir, checkpoint dir, "
        f"or pretrained_model dir. Got: {candidate}"
    )


def decode_image(image_bytes: bytes, image_size: int) -> np.ndarray | None:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))


def set_norm_eps(pipeline, eps: float) -> None:
    for step in getattr(pipeline, "steps", []):
        if hasattr(step, "eps"):
            step.eps = eps


class ActDinoV2Runner:
    def __init__(
        self,
        checkpoint_path: str | None,
        device: str | None,
        use_amp: bool,
        image_size: int,
        clamp_actions: bool,
        action_clip_margin: float,
    ) -> None:
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.image_size = image_size
        self.clamp_actions = clamp_actions

        action_min = np.asarray(ACTION_MIN, dtype=np.float32)
        action_max = np.asarray(ACTION_MAX, dtype=np.float32)
        action_range = np.maximum(action_max - action_min, NORM_EPS)
        self.action_low = action_min - action_clip_margin * action_range
        self.action_high = action_max + action_clip_margin * action_range

        self.policy, self.preprocessor, self.postprocessor = self.load_policy()

    def load_policy(self):
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
        policy.reset()

        preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=DEPLOY_STATS)
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
        if self.clamp_actions:
            action_np = np.clip(action_np, self.action_low, self.action_high)
        return action_np.astype(float).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT DINOv2 model worker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-path", default=None)
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
        args.device,
        args.use_amp,
        args.image_size,
        args.clamp_actions,
        args.action_clip_margin,
    )
    print(
        f"[worker] loaded {runner.checkpoint_path} on {runner.device}; "
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
