#!/usr/bin/env python3
"""Run a ROS bag through this deployment model and save its predicted actions.

This is deliberately an *offline bridge replay*, rather than dataset
conversion.  It uses the same raw compressed camera bytes, 23-D measured
state ordering, calibrated top-camera preprocessing, action clipping, and ACT
100-step FIFO used by ``worker.py`` + ``bridge.py`` in this directory.

At each virtual 20 Hz bridge tick, every input is the latest ROS message whose
bag timestamp is no later than the tick.  Consequently this never leaks a
future image/state sample into the policy (unlike nearest-timestamp conversion).
The output NPZ is compatible with ``scripts/replay/replay_zeno_npz_state.py``:
``state`` is the model input and ``action`` is the predicted 23-D model action.
It also stores ``demo_action`` from the recorded command topics for comparison.

Example:
    conda run --no-capture-output -n lerobot-qrp312 python offline_infer_bag.py \
      --bag /home/zeno-rp/2027icra/Data/2026_07_31/rosbag2_2026_07_31_14_19_39 \
      --output /home/zeno-rp/2027icra/Data/replay/2026_07_31_14_19_39_robot8_20260731_raw100k_deploy_inference.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rosbags.highlevel import AnyReader


REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_DIR = Path(__file__).resolve().parent
BASE_WORKER_PATH = (
    REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py"
)
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "robot8_20260731_act_dinov3_3cam_640x480_topcam_left_cam20260729_all15_"
    "decoder7_b32_bf16_nogc_resume_100k_onthefly"
)
DEFAULT_CALIBRATION = (
    REPO_ROOT / "scripts" / "data_convert" / "cam" / "stereo_params_20260729_172611.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data" / "replay"

FPS = 20.0
STEP_NS = int(1e9 / FPS)
ACTION_DIM = 23

CAMERA_TOPICS = {
    "head_cam": "/zeno/h1/sensor/head_cam/image/compressed",
    "left_arm_cam": "/zeno/h1/sensor/left_arm_cam/image/compressed",
    "right_arm_cam": "/zeno/h1/sensor/right_arm_cam/image/compressed",
}
STATE_TORSO = "/zeno/h1/wheelarm/torso/joint_state"
STATE_LEFT_ARM = "/zeno/h1/wheelarm/left_arm/joint_state"
STATE_RIGHT_ARM = "/zeno/h1/wheelarm/right_arm/joint_state"
STATE_LEFT_GRIPPER = "/zeno/h1/left_gripper/joint_state"
STATE_RIGHT_GRIPPER = "/zeno/h1/right_gripper/joint_state"
ODOM = "/zeno/h1/sensor/odom_raw"

ACTION_TORSO = "/zeno/h1/wheelarm/torso/joint_cmd"
ACTION_LEFT_ARM = "/zeno/h1/wheelarm/left_arm/joint_cmd"
ACTION_RIGHT_ARM = "/zeno/h1/wheelarm/right_arm/joint_cmd"
ACTION_LEFT_GRIPPER = "/zeno/h1/left_gripper/joint_cmd"
ACTION_RIGHT_GRIPPER = "/zeno/h1/right_gripper/joint_cmd"
TWIST_CMD = "/zeno/h1/twist/cmd"

TORSO_FIELDS = ("torso_lift", "torso_waist", "head_pan", "head_tilt")
LEFT_ARM_FIELDS = tuple(f"left_arm_j{index}" for index in range(7))
RIGHT_ARM_FIELDS = tuple(f"right_arm_j{index}" for index in range(7))
LEFT_GRIPPER_FIELDS = ("left_gripper",)
RIGHT_GRIPPER_FIELDS = ("right_gripper",)
JOINT_NAME_ALIASES = {
    "head_pan": ("torso_head_pan",),
    "head_tilt": ("torso_head_tilt",),
    "left_gripper": ("left_arm_gripper",),
    "right_gripper": ("right_arm_gripper",),
}

STATE_STREAMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (STATE_TORSO, TORSO_FIELDS),
    (STATE_LEFT_ARM, LEFT_ARM_FIELDS),
    (STATE_RIGHT_ARM, RIGHT_ARM_FIELDS),
    (STATE_LEFT_GRIPPER, LEFT_GRIPPER_FIELDS),
    (STATE_RIGHT_GRIPPER, RIGHT_GRIPPER_FIELDS),
)
ACTION_STREAMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (ACTION_TORSO, TORSO_FIELDS),
    (ACTION_LEFT_ARM, LEFT_ARM_FIELDS),
    (ACTION_RIGHT_ARM, RIGHT_ARM_FIELDS),
    (ACTION_LEFT_GRIPPER, LEFT_GRIPPER_FIELDS),
    (ACTION_RIGHT_GRIPPER, RIGHT_GRIPPER_FIELDS),
)
INPUT_TOPICS = tuple(CAMERA_TOPICS.values()) + tuple(topic for topic, _ in STATE_STREAMS) + (ODOM,)
DEMO_ACTION_TOPICS = tuple(topic for topic, _ in ACTION_STREAMS) + (TWIST_CMD,)


@dataclass(frozen=True)
class StreamBounds:
    """First/last bag timestamps and count for one required topic."""

    first_ns: int
    last_ns: int
    count: int


def import_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_bag_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"bag path does not exist: {path}")
    if not path.is_dir() and path.suffix.lower() != ".mcap":
        raise argparse.ArgumentTypeError("--bag must be a rosbag directory or an .mcap file")
    return path


def default_output_path(bag_path: Path) -> Path:
    name = bag_path.stem if bag_path.is_file() else bag_path.name
    return DEFAULT_OUTPUT_DIR / f"{name}_robot8_20260731_raw100k_deploy_inference.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=parse_bag_path, required=True, help="ROS bag directory or MCAP file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ. Defaults to Data/replay/<bag>_robot8_20260731_raw100k_deploy_inference.npz.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Run root or pretrained_model directory; default is this wrapper's 100k run root.",
    )
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only process the first N virtual 20 Hz ticks (for a smoke test).",
    )
    parser.add_argument(
        "--max-obs-age-s",
        type=float,
        default=0.5,
        help="Same bridge cache freshness guard; stale ticks are not sent to the model.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N virtual ticks; set 0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output NPZ.")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.max_obs_age_s <= 0.0:
        parser.error("--max-obs-age-s must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    args.output = (args.output or default_output_path(args.bag)).expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    return args


def selected_connections(reader: Any, wanted_topics: Iterable[str]) -> list[Any]:
    wanted = set(wanted_topics)
    found = {connection.topic for connection in reader.connections if connection.topic in wanted}
    missing = sorted(wanted - found)
    if missing:
        raise RuntimeError("Bag is missing required topic(s): " + ", ".join(missing))
    return [connection for connection in reader.connections if connection.topic in wanted]


def scan_bounds(bag_path: Path, topics: Sequence[str]) -> dict[str, StreamBounds]:
    """First pass: determine the shared interval without retaining messages."""

    first: dict[str, int] = {}
    last: dict[str, int] = {}
    counts: dict[str, int] = {topic: 0 for topic in topics}
    with AnyReader([bag_path]) as reader:
        connections = selected_connections(reader, topics)
        for connection, timestamp_ns, _raw in reader.messages(connections=connections):
            topic = connection.topic
            counts[topic] += 1
            first.setdefault(topic, int(timestamp_ns))
            last[topic] = int(timestamp_ns)
    empty = [topic for topic in topics if counts[topic] == 0]
    if empty:
        raise RuntimeError("Required topic(s) have no messages: " + ", ".join(empty))
    return {
        topic: StreamBounds(first_ns=first[topic], last_ns=last[topic], count=counts[topic])
        for topic in topics
    }


def extract_joint_positions(msg: Any, fields: Sequence[str]) -> list[float]:
    """Match the shared deployment bridge's JointState decoding exactly."""

    positions = list(msg.position)
    if not positions:
        raise ValueError("JointState has no position values")
    names = [str(name) for name in msg.name]
    if not names:
        if len(positions) < len(fields):
            raise ValueError(f"JointState has {len(positions)} positions, need {len(fields)}")
        return [float(value) for value in positions[: len(fields)]]

    by_name = {name: float(position) for name, position in zip(names, positions, strict=False)}
    values: list[float] = []
    missing: list[str] = []
    for field in fields:
        aliases = (field, *JOINT_NAME_ALIASES.get(field, ()))
        matched = next((name for name in aliases if name in by_name), None)
        if matched is None:
            missing.append(field)
        else:
            values.append(by_name[matched])
    if missing:
        raise ValueError(f"JointState is missing fields {missing}; available={names}")
    return values


def odom_velocity(msg: Any) -> list[float]:
    twist = msg.twist.twist
    return [float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)]


def twist_velocity(msg: Any) -> list[float]:
    return [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]


def make_state(cache: dict[str, tuple[int, Any]]) -> np.ndarray:
    values: list[float] = []
    for topic, fields in STATE_STREAMS:
        values.extend(extract_joint_positions(cache[topic][1], fields))
    values.extend(odom_velocity(cache[ODOM][1]))
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"state shape {state.shape}, expected ({ACTION_DIM},)")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN/Inf")
    return state


def make_demo_action(cache: dict[str, tuple[int, Any]]) -> np.ndarray:
    values: list[float] = []
    for topic, fields in ACTION_STREAMS:
        values.extend(extract_joint_positions(cache[topic][1], fields))
    values.extend(twist_velocity(cache[TWIST_CMD][1]))
    action = np.asarray(values, dtype=np.float32)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"demo action shape {action.shape}, expected ({ACTION_DIM},)")
    if not np.isfinite(action).all():
        raise ValueError("demo action contains NaN/Inf")
    return action


def cache_is_fresh(
    cache: dict[str, tuple[int, Any]], required_topics: Sequence[str], tick_ns: int, max_age_ns: int
) -> tuple[bool, str]:
    for topic in required_topics:
        item = cache.get(topic)
        if item is None:
            return False, f"missing:{topic}"
        age_ns = tick_ns - item[0]
        if age_ns < 0:
            return False, f"future_cache:{topic}"
        if age_ns > max_age_ns:
            return False, f"stale:{topic}:{age_ns / 1e9:.3f}s"
    return True, ""


def reset_policy_for_reconnect(worker: Any) -> None:
    """Mirror the worker server's reset after the bridge reconnects."""

    worker.policy.reset()


def build_worker(checkpoint_path: Path, device: str | None) -> tuple[Any, Path]:
    if not BASE_WORKER_PATH.is_file():
        raise FileNotFoundError(f"Shared deployment worker not found: {BASE_WORKER_PATH}")
    if not DEFAULT_CALIBRATION.is_file():
        raise FileNotFoundError(f"Top-camera calibration not found: {DEFAULT_CALIBRATION}")
    deploy_worker = import_module("robot8_20260731_offline_shared_worker", BASE_WORKER_PATH)
    checkpoint = deploy_worker.resolve_checkpoint(str(checkpoint_path))
    worker = deploy_worker.ActWorker(
        checkpoint=checkpoint,
        device=device,
        image_size=(640, 480),
        center_crop_fraction=1.0,
        use_amp=True,
        clamp_actions=True,
        action_clip_margin=0.05,
        n_action_steps=None,
        temporal_ensemble_coeff=None,
        frozen_action_indices=(),
        fixed_action_values={},
        rectify_head_stereo=True,
        head_stereo_profile="cam_20260729",
        head_stereo_cam_calibration=DEFAULT_CALIBRATION,
    )
    return worker, checkpoint


def run_inference(
    *,
    bag_path: Path,
    worker: Any,
    start_ns: int,
    end_ns: int,
    max_age_ns: int,
    max_frames: int | None,
    progress_every: int,
    include_demo_action: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Replay selected topics in timestamp order and invoke the real worker."""

    topics = list(INPUT_TOPICS)
    if include_demo_action:
        topics.extend(DEMO_ACTION_TOPICS)
    total_ticks = int((end_ns - start_ns) // STEP_NS) + 1
    if max_frames is not None:
        total_ticks = min(total_ticks, max_frames)
        end_ns = start_ns + (total_ticks - 1) * STEP_NS
    if total_ticks < 1:
        raise RuntimeError("No 20 Hz ticks in the selected shared interval")

    cache: dict[str, tuple[int, Any]] = {}
    timestamps_ns: list[int] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    demo_actions: list[np.ndarray] = []
    valid_mask = np.zeros(total_ticks, dtype=bool)
    failure_reasons: dict[str, int] = {}
    invalid_ticks = 0
    model_forward_count = 0
    reconnect_resets = 0
    reset_before_next_valid = False
    start_wall = time.perf_counter()
    next_tick_ns = start_ns
    tick_index = 0

    def count_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal model_forward_count
        model_forward_count += 1

    forward_hook = worker.policy.model.register_forward_hook(count_forward)

    def process_tick(tick_ns: int, index: int) -> None:
        nonlocal invalid_ticks, reconnect_resets, reset_before_next_valid
        is_fresh, reason = cache_is_fresh(cache, INPUT_TOPICS, tick_ns, max_age_ns)
        if not is_fresh:
            invalid_ticks += 1
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            reset_before_next_valid = True
            return
        try:
            state = make_state(cache)
            images = {
                name: cache[topic][1]
                for name, topic in CAMERA_TOPICS.items()
            }
            if reset_before_next_valid:
                reset_policy_for_reconnect(worker)
                reconnect_resets += 1
                reset_before_next_valid = False
            predicted_action = np.asarray(worker.select_action(state, images), dtype=np.float32)
            if predicted_action.shape != (ACTION_DIM,):
                raise ValueError(f"model action shape {predicted_action.shape}, expected ({ACTION_DIM},)")
            if not np.isfinite(predicted_action).all():
                raise ValueError("model action contains NaN/Inf")
        except Exception as exc:
            invalid_ticks += 1
            key = f"inference_error:{type(exc).__name__}:{str(exc)[:160]}"
            failure_reasons[key] = failure_reasons.get(key, 0) + 1
            reset_before_next_valid = True
            return

        timestamps_ns.append(tick_ns)
        states.append(state)
        actions.append(predicted_action)
        valid_mask[index] = True
        if include_demo_action:
            if all(topic in cache for topic in DEMO_ACTION_TOPICS):
                try:
                    demo_actions.append(make_demo_action(cache))
                except Exception as exc:
                    failure_reasons[f"demo_action_error:{type(exc).__name__}"] = (
                        failure_reasons.get(f"demo_action_error:{type(exc).__name__}", 0) + 1
                    )
                    demo_actions.append(np.full(ACTION_DIM, np.nan, dtype=np.float32))
            else:
                failure_reasons["demo_action_missing"] = failure_reasons.get("demo_action_missing", 0) + 1
                demo_actions.append(np.full(ACTION_DIM, np.nan, dtype=np.float32))

    try:
        with AnyReader([bag_path]) as reader:
            connections = selected_connections(reader, topics)
            for connection, timestamp_ns_raw, raw in reader.messages(connections=connections):
                timestamp_ns = int(timestamp_ns_raw)
                while next_tick_ns < timestamp_ns and next_tick_ns <= end_ns:
                    process_tick(next_tick_ns, tick_index)
                    tick_index += 1
                    if progress_every and tick_index % progress_every == 0:
                        elapsed = time.perf_counter() - start_wall
                        rate = tick_index / elapsed if elapsed > 0 else float("nan")
                        print(
                            f"[offline-infer] tick={tick_index}/{total_ticks} "
                            f"valid={len(actions)} forwards={model_forward_count} "
                            f"rate={rate:.2f} ticks/s",
                            flush=True,
                        )
                    next_tick_ns += STEP_NS
                if next_tick_ns > end_ns:
                    break

                message = reader.deserialize(raw, connection.msgtype)
                if connection.topic in CAMERA_TOPICS.values():
                    # Retain raw JPEG bytes.  The deployment worker alone applies
                    # top-stereo rectification/crop/left-eye extraction.
                    cache[connection.topic] = (timestamp_ns, bytes(message.data))
                else:
                    cache[connection.topic] = (timestamp_ns, message)

            while next_tick_ns <= end_ns:
                process_tick(next_tick_ns, tick_index)
                tick_index += 1
                if progress_every and tick_index % progress_every == 0:
                    elapsed = time.perf_counter() - start_wall
                    rate = tick_index / elapsed if elapsed > 0 else float("nan")
                    print(
                        f"[offline-infer] tick={tick_index}/{total_ticks} "
                        f"valid={len(actions)} forwards={model_forward_count} "
                        f"rate={rate:.2f} ticks/s",
                        flush=True,
                    )
                next_tick_ns += STEP_NS
    finally:
        forward_hook.remove()

    if tick_index != total_ticks:
        raise RuntimeError(f"Internal tick accounting error: processed {tick_index}, expected {total_ticks}")
    if not actions:
        details = json.dumps(failure_reasons, ensure_ascii=False, indent=2)
        raise RuntimeError(f"No valid model actions were produced. Failures:\n{details}")

    output = {
        "timestamp_s": (np.asarray(timestamps_ns, dtype=np.int64) - start_ns).astype(np.float64) / 1e9,
        "bag_timestamp_ns": np.asarray(timestamps_ns, dtype=np.int64),
        "state": np.stack(states).astype(np.float32, copy=False),
        # ``action`` is intentionally the model output, so replay_zeno_npz_state.py
        # can consume this file without a special-case schema.
        "action": np.stack(actions).astype(np.float32, copy=False),
        "tick_timestamp_s": np.arange(total_ticks, dtype=np.float64) / FPS,
        "valid_mask": valid_mask,
    }
    if include_demo_action:
        output["demo_action"] = np.stack(demo_actions).astype(np.float32, copy=False)

    elapsed_s = time.perf_counter() - start_wall
    summary = {
        "num_timer_ticks": total_ticks,
        "num_model_actions": len(actions),
        "invalid_ticks": invalid_ticks,
        "failure_reasons": failure_reasons,
        "policy_forward_count": model_forward_count,
        "policy_reconnect_resets": reconnect_resets,
        "elapsed_s": elapsed_s,
        "virtual_tick_rate_hz": float(total_ticks / elapsed_s) if elapsed_s > 0 else None,
    }
    return output, summary


def write_outputs(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".json")
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}; pass --overwrite to replace it")

    # Write to a sibling and atomically expose the completed NPZ only after all
    # arrays have been compressed.  A partial run cannot masquerade as a result.
    temporary_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        **arrays,
        source_bag=np.asarray(metadata["source_bag"]),
        checkpoint=np.asarray(metadata["checkpoint"]),
        deployment_contract=np.asarray(metadata["deployment_contract"]),
        sampling_contract=np.asarray(metadata["sampling_contract"]),
        fps=np.asarray(metadata["fps"], dtype=np.float64),
        action_fields=np.asarray(metadata["action_fields"]),
    )
    temporary_path.replace(output_path)
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path, summary_path


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output} (use --overwrite to replace it)")

    print(f"[offline-infer] scanning input streams in {args.bag}", flush=True)
    bounds = scan_bounds(args.bag, INPUT_TOPICS)
    start_ns = max(item.first_ns for item in bounds.values())
    end_ns = min(item.last_ns for item in bounds.values())
    if end_ns < start_ns:
        raise SystemExit("Required deployment streams have no overlapping time range")
    candidate_ticks = int((end_ns - start_ns) // STEP_NS) + 1
    print(
        f"[offline-infer] shared causal interval={(end_ns - start_ns) / 1e9:.3f}s; "
        f"candidate_ticks={candidate_ticks}; fps={FPS:.1f}",
        flush=True,
    )

    worker, checkpoint = build_worker(args.checkpoint_path, args.device)
    print(
        f"[offline-infer] checkpoint={checkpoint}; device={worker.device}; "
        f"ACT n_action_steps={worker.policy.config.n_action_steps}; "
        f"topcam=raw-stereo->cam_20260729-rectify->crop->left-RGB->640x480-letterbox",
        flush=True,
    )
    try:
        arrays, run_summary = run_inference(
            bag_path=args.bag,
            worker=worker,
            start_ns=start_ns,
            end_ns=end_ns,
            max_age_ns=int(args.max_obs_age_s * 1e9),
            max_frames=args.max_frames,
            progress_every=args.progress_every,
            include_demo_action=True,
        )
    finally:
        # Releasing the worker after the output arrays are materialized avoids
        # retaining the DINO model during the NPZ compression step.
        del worker

    metadata: dict[str, Any] = {
        "source_bag": str(args.bag),
        "checkpoint": str(checkpoint),
        "fps": FPS,
        "action_fields": [
            *TORSO_FIELDS,
            *LEFT_ARM_FIELDS,
            *RIGHT_ARM_FIELDS,
            *LEFT_GRIPPER_FIELDS,
            *RIGHT_GRIPPER_FIELDS,
            "base_vx",
            "base_vy",
            "base_rotation",
        ],
        "deployment_contract": (
            "Exact robot8_20260731 raw 3-camera worker: 23D measured state; raw head 2560x720 "
            "left|right JPEG -> cam_20260729 rectification/alignment -> crop (20,0,1240,620) -> "
            "left RGB -> 640x480 letterbox; raw arm JPEGs -> RGB 640x480; AMP, action clamp, no frozen fields."
        ),
        "sampling_contract": (
            "Virtual 20Hz bridge timer. At every tick, use the latest message at or before the tick for "
            "each required model input. Require every input age <= max_obs_age_s; invalid ticks produce no action "
            "and reset ACT before the next valid call."
        ),
        "max_obs_age_s": args.max_obs_age_s,
        "input_bounds": {
            topic: {"first_ns": item.first_ns, "last_ns": item.last_ns, "count": item.count}
            for topic, item in bounds.items()
        },
        "run": run_summary,
    }
    output_path, summary_path = write_outputs(
        args.output, arrays, metadata, overwrite=args.overwrite
    )
    print(
        f"[offline-infer] wrote {output_path} ({arrays['action'].shape[0]} model actions)\n"
        f"[offline-infer] summary {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
