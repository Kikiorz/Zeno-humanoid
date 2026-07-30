#!/usr/bin/env python3
"""Local TCP worker template for an external Robot8 algorithm.

This server deliberately keeps the existing Robot8 ROS bridge unchanged.  The
only project-specific responsibilities here are: validate the bridge ABI,
apply the exact 2026-07-29 top-camera geometry, and return one safe 23D action
per request.  Replace only ``ExternalPolicy`` with the external algorithm.

The wire protocol is Python pickle and is therefore intentionally restricted
to a localhost socket.  Never bind this example to a LAN/public interface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import pickle
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


ACTION_DIM = 23
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
assert len(ACTION_FIELDS) == ACTION_DIM
MAX_REQUEST_BYTES = 128 * 1024 * 1024
EXPECTED_CALIBRATION_SHA256 = "6d08b6a01a1431476c2c3c77bee43e3f8f20888f33940af772cf1963e9f6b342"

# Everything needed for the camera contract ships next to this file.  The
# directory can therefore be copied to a machine that does not have this repo.
PACKAGE_DIR = Path(__file__).resolve().parent
CAMERA_DIR = PACKAGE_DIR / "camera"
if str(CAMERA_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_DIR))

from topcam_stereo_rectify_cam_20260729 import (  # noqa: E402
    TopStereoRectificationError,
    TopStereoRectifier,
)


DEFAULT_CALIBRATION = (
    CAMERA_DIR / "stereo_params_20260729_172611.npz"
)


@dataclass(frozen=True)
class RobotObservation:
    """One 20 Hz Robot8 observation in raw state + calibrated image geometry.

    All images are RGB ``uint8`` arrays in HWC layout, shape ``(480, 640, 3)``
    with the default CLI resolution.  ``state`` is raw, unnormalized float32
    in the exact 23D order documented in README.md.
    """

    state: np.ndarray
    head_cam_rgb: np.ndarray
    left_arm_cam_rgb: np.ndarray
    right_arm_cam_rgb: np.ndarray


class ExternalPolicy:
    """Replace this class with the external model adapter.

    Keep ``predict`` synchronous: the ROS bridge sends one request then waits
    for one action.  Return exactly one *raw*, unnormalized 23D command, not a
    normalized vector or an action chunk.  Add your own joint/base limits and
    smoothing before returning the action.
    """

    def __init__(self) -> None:
        # TODO: Load the external model and any immutable configuration here.
        # Do not load a new model or recreate camera maps inside predict().
        pass

    def reset(self) -> None:
        """Clear temporal state when the bridge reconnects, if the model has any."""

        # TODO: Reset recurrent state / action queue if applicable.
        pass

    def predict(self, observation: RobotObservation) -> np.ndarray:
        """Return one finite raw 23D action for ``observation``.

        This template fails closed until it is implemented.  Do not replace the
        exception with a constant zero action: an active zero command is not a
        safe substitute for a failed policy.
        """

        del observation
        raise NotImplementedError(
            "Implement ExternalPolicy.predict() with the external algorithm before deployment"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_calibration(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required cam_20260729 calibration not found: {path}")
    digest = _sha256(path)
    if digest != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError(
            "Unexpected top-camera calibration SHA-256. This deployment requires "
            f"{EXPECTED_CALIBRATION_SHA256}, got {digest}: {path}"
        )


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("worker client disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Mapping[str, Any]:
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if not 0 < size <= MAX_REQUEST_BYTES:
        raise ValueError(f"invalid request size {size}; max is {MAX_REQUEST_BYTES}")
    message = pickle.loads(_recv_exact(sock, size))
    if not isinstance(message, Mapping):
        raise ValueError("worker request must be a mapping")
    return message


def send_message(sock: socket.socket, message: Mapping[str, Any]) -> None:
    payload = pickle.dumps(dict(message), protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _as_compressed_bytes(images: Mapping[str, Any], name: str) -> bytes:
    value = images.get(name)
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"images[{name!r}] must be compressed-image bytes")
    if len(value) == 0:
        raise ValueError(f"images[{name!r}] is empty")
    return bytes(value)


def _decode_regular_camera(compressed: bytes, image_size: tuple[int, int], name: str) -> np.ndarray:
    """Current left/right-arm convention: raw JPEG -> RGB -> direct resize."""

    image_bgr = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"could not decode {name} compressed image")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(image_rgb, image_size, interpolation=cv2.INTER_LINEAR)


def build_observation(
    request: Mapping[str, Any],
    *,
    rectifier: TopStereoRectifier,
    image_size: tuple[int, int],
) -> RobotObservation:
    raw_state = request.get("state")
    state = np.asarray(raw_state, dtype=np.float32)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"state must have shape ({ACTION_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or Inf")

    images = request.get("images")
    if not isinstance(images, Mapping):
        raise ValueError("request['images'] must be a mapping")
    head_compressed = _as_compressed_bytes(images, "head_cam")
    left_compressed = _as_compressed_bytes(images, "left_arm_cam")
    right_compressed = _as_compressed_bytes(images, "right_arm_cam")

    # This call enforces raw 2560x720 left|right input and applies the exact
    # fisheye rectify -> fixed crop -> RGB -> letterbox training geometry.
    head_rgb = rectifier.decode_and_rectify_left_rgb(
        head_compressed,
        image_size,
        resize_mode="letterbox",
    )
    if head_rgb is None:
        raise ValueError("could not decode head_cam compressed image")
    expected_shape = (image_size[1], image_size[0], 3)
    if head_rgb.shape != expected_shape or head_rgb.dtype != np.uint8:
        raise RuntimeError(
            f"unexpected rectified head_cam output {head_rgb.shape} {head_rgb.dtype}; "
            f"expected {expected_shape} uint8"
        )

    return RobotObservation(
        state=state,
        head_cam_rgb=head_rgb,
        left_arm_cam_rgb=_decode_regular_camera(left_compressed, image_size, "left_arm_cam"),
        right_arm_cam_rgb=_decode_regular_camera(right_compressed, image_size, "right_arm_cam"),
    )


def validate_action(action: Any) -> list[float]:
    action_np = np.asarray(action, dtype=np.float32)
    if action_np.shape != (ACTION_DIM,):
        raise ValueError(f"action must have shape ({ACTION_DIM},), got {action_np.shape}")
    if not np.isfinite(action_np).all():
        raise ValueError("action contains NaN or Inf")
    return action_np.astype(float).tolist()


def serve_client(
    client: socket.socket,
    *,
    policy: ExternalPolicy,
    rectifier: TopStereoRectifier,
    image_size: tuple[int, int],
) -> None:
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        request = recv_message(client)
        started_at = time.perf_counter()
        try:
            observation = build_observation(request, rectifier=rectifier, image_size=image_size)
            action = validate_action(policy.predict(observation))
            response: dict[str, Any] = {
                "ok": True,
                "action": action,
                "latency_s": time.perf_counter() - started_at,
            }
        except Exception as exc:
            # The stock bridge treats ok=False as a failed cycle and publishes
            # idle; do not fabricate a command after preprocessing/model errors.
            response = {"ok": False, "error": str(exc)}
        send_message(client, response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot8 external-algorithm localhost worker template")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 480),
        help="Policy image size. Keep 640 480 for the current cam_20260729 deployment contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("For safety this pickle worker may bind only to 127.0.0.1")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise SystemExit("--image-size values must be positive")

    _validate_calibration(DEFAULT_CALIBRATION)
    try:
        rectifier = TopStereoRectifier(DEFAULT_CALIBRATION)
    except TopStereoRectificationError as exc:
        raise SystemExit(f"Invalid cam_20260729 calibration: {exc}") from exc
    policy = ExternalPolicy()

    print(
        "[robot8 external worker] "
        f"listening={args.host}:{args.port}; image_size={image_size}; "
        "head_cam=raw2560x720->cam_20260729->leftRGB-letterbox; "
        "implement ExternalPolicy.predict() before publishing commands",
        flush=True,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        try:
            while True:
                client, address = server.accept()
                print(f"[robot8 external worker] bridge connected: {address}", flush=True)
                with client:
                    policy.reset()
                    try:
                        serve_client(
                            client,
                            policy=policy,
                            rectifier=rectifier,
                            image_size=image_size,
                        )
                    except ConnectionError:
                        print("[robot8 external worker] bridge disconnected", flush=True)
                    except Exception as exc:
                        # A malformed trusted-local request should close only
                        # this connection, not bring down the listening server.
                        print(f"[robot8 external worker] client protocol error: {exc}", flush=True)
        except KeyboardInterrupt:
            print("[robot8 external worker] stopped", flush=True)


if __name__ == "__main__":
    main()
