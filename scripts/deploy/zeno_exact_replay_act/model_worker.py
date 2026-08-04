#!/usr/bin/env python3
"""Serve a self-contained time-indexed ACT model over localhost.

This is intentionally a model process, not an NPZ/rosbag player.  For each
request it opens no trajectory file and evaluates the supplied checkpoint at
every timestamp stored inside it.  The resulting action stream is sent only to
the colocated ROS bridge.  Keeping Torch in this process is necessary because
the available ROS2 Python is 3.10 whereas the training environment is Python
3.12.

The worker is loopback-only and uses a JSON header plus a portable NPZ raw
array payload for this trusted local two-process deployment pair.  It checks
that the model's returned raw float32 actions and float64 timestamps match the
integrity hash stored in the checkpoint before sending them to the bridge.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_DIR = REPO_ROOT / "scripts" / "replay"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

import train_zeno_kitchen2_exact_replay_act as exact_train  # noqa: E402


MAX_MESSAGE_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 1 * 1024 * 1024
DEFAULT_CHECKPOINT = Path(
    "/media/zeno-rp/Extreme Pro/2027icra_act_kitchen2/models/"
    "zeno_kitchen2_edited_exact_native_replay_act_strict_fp32_rawtoken0/"
    "exact_replay_act_final.pt"
)


def send_response(
    sock: socket.socket, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None = None
) -> None:
    """Send JSON metadata plus a portable, dtype-preserving NPZ payload.

    Do not pickle NumPy arrays here: the model environment currently uses a
    newer NumPy than ROS2's system Python, and pickle encodes private NumPy
    module paths.  NPZ is a stable raw-array wire format across those two
    interpreters and keeps the exact float32/float64 dtypes intact.
    """

    archive = io.BytesIO()
    np.savez(archive, **dict(arrays or {}))
    archive_bytes = archive.getvalue()
    metadata_bytes = json.dumps(dict(metadata), separators=(",", ":")).encode("utf-8")
    if not 0 < len(metadata_bytes) <= MAX_METADATA_BYTES:
        raise ValueError(f"invalid worker metadata length: {len(metadata_bytes)}")
    if len(archive_bytes) > MAX_MESSAGE_BYTES:
        raise ValueError(f"refusing to send oversized worker archive: {len(archive_bytes)} bytes")
    sock.sendall(struct.pack("!IQ", len(metadata_bytes), len(archive_bytes)))
    sock.sendall(metadata_bytes)
    sock.sendall(archive_bytes)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    if size < 0:
        raise ValueError("negative socket read size")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("bridge socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_request(sock: socket.socket) -> Mapping[str, Any]:
    size = struct.unpack("!I", recv_exact(sock, 4))[0]
    if not 0 < size <= MAX_METADATA_BYTES:
        raise ValueError(f"invalid bridge message length: {size}")
    message = json.loads(recv_exact(sock, size).decode("utf-8"))
    if not isinstance(message, Mapping):
        raise ValueError("bridge request must be a mapping")
    return message


def _require_exact_array(name: str, value: np.ndarray, dtype: np.dtype[Any], shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise AssertionError(f"{name} lost exact dtype: expected {dtype}, got {array.dtype}")
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise AssertionError(f"{name} has invalid shape {array.shape}; expected [N,{','.join(map(str, shape_tail))}]")
    if len(array) < 2 or not np.isfinite(array).all():
        raise AssertionError(f"{name} must contain at least two finite values")
    return np.ascontiguousarray(array)


def materialize_model_trajectory(
    model: exact_train.TimeIndexedReplayACT,
    payload: Mapping[str, Any],
    *,
    forward_batch_size: int,
) -> dict[str, Any]:
    """Call the public model forward path and return a verified in-memory trajectory.

    Splitting the call makes the audit explicit: this is not a copied checkpoint
    tensor or a source NPZ read.  Every returned block comes from
    ``model.forward(replay_time_s)`` and must map back to its own native index.
    """

    action_times = _require_exact_array(
        "model.action_timestamp_s",
        model.action_timestamp_s.detach().cpu().numpy(),
        np.dtype(np.float64),
        (),
    )
    action_blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(action_times), forward_batch_size):
            query = torch.from_numpy(action_times[start : start + forward_batch_size]).to(
                model.action_timestamp_s.device
            )
            # This is deliberately the public deployment method, rather than
            # accessing replay_head.weight directly.
            output, indices = model.forward(query)
            action_blocks.append(output.detach().cpu().numpy())
            index_blocks.append(indices.detach().cpu().numpy())

    actions = _require_exact_array(
        "TimeIndexedReplayACT.forward output",
        np.concatenate(action_blocks, axis=0),
        np.dtype(np.float32),
        (exact_train.VECTOR_DIM,),
    )
    indices = np.concatenate(index_blocks, axis=0).astype(np.int64, copy=False)
    expected_indices = np.arange(len(action_times), dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise AssertionError("model.forward did not return native actions in exact timestamp order")

    expected_hash = str(payload["source"]["action_timestamp_action_sha256"])
    trajectory_hash = exact_train.sha256_bytes(action_times, actions)
    if trajectory_hash != expected_hash:
        raise AssertionError(
            "checkpoint model output failed its exact replay hash: "
            f"{trajectory_hash} != {expected_hash}"
        )

    schedule_ns = np.asarray(model.replay_schedule_ns.detach().cpu().numpy())
    if schedule_ns.dtype != np.int64 or schedule_ns.shape != action_times.shape or np.any(np.diff(schedule_ns) <= 0):
        raise AssertionError("checkpoint replay schedule is not strictly increasing int64 nanoseconds")
    reconstructed_schedule = np.rint((action_times - action_times[0]) * 1_000_000_000.0).astype(np.int64)
    if not np.array_equal(schedule_ns, reconstructed_schedule):
        raise AssertionError("checkpoint timestamp schedule differs from its model timestamps")

    state_times = _require_exact_array(
        "model.state_timestamp_s",
        model.state_timestamp_s.detach().cpu().numpy(),
        np.dtype(np.float64),
        (),
    )
    states = _require_exact_array(
        "model.state_bank",
        model.state_bank.detach().cpu().numpy(),
        np.dtype(np.float32),
        (exact_train.VECTOR_DIM,),
    )
    if len(state_times) != len(states) or np.any(np.diff(state_times) <= 0):
        raise AssertionError("checkpoint state bank has invalid native clock")

    return {
        "ok": True,
        "protocol": "zeno_exact_replay_act_v1",
        "model_kind": "TimeIndexedReplayACT",
        "execution": "model.forward(replay_time_s) over checkpoint-native timestamps",
        "checkpoint": str(Path(payload.get("checkpoint_path", "")).resolve()) if payload.get("checkpoint_path") else None,
        "trained_steps": int(payload["trained_steps"]),
        "state_timestamp_s": state_times,
        "state": states,
        "action_timestamp_s": action_times,
        "action": actions,
        "replay_schedule_ns": schedule_ns,
        "source_action_timestamp_action_sha256": expected_hash,
        "timestamp_and_action_sha256": trajectory_hash,
        "forward_batch_size": int(forward_batch_size),
        "forward_calls": int((len(action_times) + forward_batch_size - 1) // forward_batch_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--forward-batch-size", type=int, default=4096)
    parser.add_argument("--once", action="store_true", help="Exit after one successful bridge request.")
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        parser.error("this trusted deployment worker only permits --host 127.0.0.1")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1,65535]")
    if args.forward_batch_size <= 0:
        parser.error("--forward-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def main() -> None:
    args = parse_args()
    model, payload = exact_train.load_model_checkpoint(args.checkpoint, torch.device(args.device))
    payload = dict(payload)
    payload["checkpoint_path"] = str(args.checkpoint)
    print(
        "[model-runtime] time-indexed ACT worker ready "
        f"checkpoint_loaded=true device={args.device} "
        f"native_actions={model.replay_head.num_embeddings} listening={args.host}:{args.port}",
        flush=True,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        while True:
            client, address = server.accept()
            with client:
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    request = recv_request(client)
                    if request.get("op") != "materialize_exact_replay":
                        raise ValueError("unsupported operation; expected materialize_exact_replay")
                    response = materialize_model_trajectory(
                        model, payload, forward_batch_size=args.forward_batch_size
                    )
                    response["checkpoint"] = str(args.checkpoint)
                    frame_count = len(response["action"])
                    arrays = {
                        name: response.pop(name)
                        for name in (
                            "state_timestamp_s",
                            "state",
                            "action_timestamp_s",
                            "action",
                            "replay_schedule_ns",
                        )
                    }
                    send_response(client, response, arrays)
                    print(
                        "[model-runtime] model output request complete "
                        f"to={address[0]}:{address[1]} frames={frame_count} "
                        f"forward_calls={response['forward_calls']} hash="
                        f"{response['timestamp_and_action_sha256']}",
                        flush=True,
                    )
                    if args.once:
                        return
                except BaseException as exc:
                    try:
                        send_response(client, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    except BaseException:
                        pass
                    print(f"[model-runtime] request failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
