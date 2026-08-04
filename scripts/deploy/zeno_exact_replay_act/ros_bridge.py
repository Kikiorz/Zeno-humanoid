#!/usr/bin/env python3
"""ROS2 runtime for model-generated Zeno whole-body output.

Run this script with ROS2's system Python.  A local time-indexed ACT model is
evaluated from its checkpoint at runtime; this process never opens a source
dataset or a rosbag.  It verifies dtype, native timestamps, nanosecond
deadlines, and the checkpoint action hash before emitting model output.  It
can first move the upper body from an arbitrary measured startup pose to the
initial model output, then begins the native action clock.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY_DIR = REPO_ROOT / "scripts" / "replay"
if str(REPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(REPLAY_DIR))

import replay_zeno_npz_state as zeno_replay  # noqa: E402


MAX_MESSAGE_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 1 * 1024 * 1024
VECTOR_DIM = 23


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    if size < 0:
        raise ValueError("negative socket read size")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("model worker socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_request(sock: socket.socket, message: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(message), separators=(",", ":")).encode("utf-8")
    if not 0 < len(payload) <= MAX_METADATA_BYTES:
        raise ValueError("bridge request has an invalid size")
    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def _recv_response(sock: socket.socket) -> Mapping[str, Any]:
    header_size, archive_size = struct.unpack("!IQ", _recv_exact(sock, 12))
    if not 0 < header_size <= MAX_METADATA_BYTES:
        raise ValueError(f"invalid model worker metadata size: {header_size}")
    if archive_size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid model worker array archive size: {archive_size}")
    metadata = json.loads(_recv_exact(sock, header_size).decode("utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("model worker metadata is not a mapping")
    archive_bytes = _recv_exact(sock, archive_size)
    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        for name in archive.files:
            arrays[name] = archive[name]
    response = dict(metadata)
    response.update(arrays)
    return response


def request_model_trajectory(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.worker_host != "127.0.0.1":
        raise ValueError("this trusted deployment bridge only permits --worker-host 127.0.0.1")
    with socket.create_connection((args.worker_host, args.worker_port), timeout=args.worker_timeout_s) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(args.worker_timeout_s)
        _send_request(sock, {"op": "materialize_exact_replay"})
        return _recv_response(sock)


def _require_array(
    response: Mapping[str, Any], name: str, dtype: np.dtype[Any], shape_tail: tuple[int, ...]
) -> np.ndarray:
    if name not in response:
        raise ValueError(f"model response omitted {name}")
    array = np.asarray(response[name])
    if array.dtype != dtype:
        raise ValueError(f"{name} must preserve {dtype}, got {array.dtype}")
    expected_ndim = 1 + len(shape_tail)
    if array.ndim != expected_ndim or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape [N,{','.join(map(str, shape_tail))}], got {array.shape}")
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must have at least two finite values")
    return np.ascontiguousarray(array)


def source_from_model_response(response: Mapping[str, Any]) -> zeno_replay.SourceState:
    if response.get("ok") is not True:
        raise RuntimeError(f"model worker rejected request: {response.get('error', 'unknown error')}")
    if response.get("protocol") != "zeno_exact_replay_act_v1":
        raise ValueError(f"unexpected model worker protocol: {response.get('protocol')!r}")
    if response.get("model_kind") != "TimeIndexedReplayACT":
        raise ValueError(f"unexpected model kind: {response.get('model_kind')!r}")

    state_times = _require_array(response, "state_timestamp_s", np.dtype(np.float64), ())
    states = _require_array(response, "state", np.dtype(np.float32), (VECTOR_DIM,))
    action_times = _require_array(response, "action_timestamp_s", np.dtype(np.float64), ())
    actions = _require_array(response, "action", np.dtype(np.float32), (VECTOR_DIM,))
    if len(state_times) != len(states) or np.any(np.diff(state_times) <= 0.0):
        raise ValueError("model state bank does not have strictly increasing native timestamps")
    if len(action_times) != len(actions) or np.any(np.diff(action_times) <= 0.0):
        raise ValueError("model action stream does not have strictly increasing native timestamps")

    expected_hash = response.get("source_action_timestamp_action_sha256")
    reported_hash = response.get("timestamp_and_action_sha256")
    actual_hash = _sha256_arrays(action_times, actions)
    if not isinstance(expected_hash, str) or actual_hash != expected_hash or reported_hash != expected_hash:
        raise AssertionError(
            "model action hash is not exactly the checkpoint's source action/time hash: "
            f"actual={actual_hash} reported={reported_hash} expected={expected_hash}"
        )

    schedule = np.asarray(response.get("replay_schedule_ns"))
    expected_schedule = np.rint((action_times - action_times[0]) * 1_000_000_000.0).astype(np.int64)
    if schedule.dtype != np.int64 or schedule.shape != action_times.shape:
        raise ValueError("model response has invalid replay_schedule_ns")
    if np.any(np.diff(schedule) <= 0) or not np.array_equal(schedule, expected_schedule):
        raise AssertionError("model response schedule differs from the exact native timestamp schedule")

    checkpoint = str(response.get("checkpoint") or "model-checkpoint")
    print(
        "[model-runtime] checkpoint output accepted "
        f"frames={len(actions)} duration="
        f"{action_times[-1] - action_times[0]:.9f}s forward_calls={response.get('forward_calls')} "
        f"sha256={actual_hash}",
        flush=True,
    )
    return zeno_replay.SourceState(
        state_timestamps_s=state_times,
        states=states,
        action_timestamps_s=action_times,
        actions=actions,
        state_base_twists=None,
        # This object only supplies validated in-memory model arrays to the
        # shared trajectory selector.  Do not attach the checkpoint path: it
        # is neither opened nor useful to this ROS-side process.
        path=Path("model-checkpoint"),
        native_timing_schema=True,
    )


def selected_native_action_schedule_ns(
    response: Mapping[str, Any], source: zeno_replay.SourceState, trajectory: zeno_replay.ReplayTrajectory
) -> np.ndarray:
    """Select the worker's verified int64 action clock for an output subset.

    The generic trajectory object stores seconds.  This deployment bridge also
    has the checkpoint's original nanosecond schedule, so retain it after a
    non-zero ``--start-offset-s`` instead of deriving another timing grid.
    """

    if not trajectory.source_timing:
        raise AssertionError("exact ACT deployment must use native action timestamps, not a resampled clock")
    schedule_ns = np.asarray(response.get("replay_schedule_ns"))
    if schedule_ns.dtype != np.int64 or schedule_ns.shape != source.action_timestamps_s.shape:
        raise AssertionError("verified worker action schedule was lost or changed")
    indices = np.searchsorted(source.action_timestamps_s, trajectory.times_s, side="left")
    if np.any(indices >= len(source.action_timestamps_s)) or not np.array_equal(
        source.action_timestamps_s[indices], trajectory.times_s
    ):
        raise AssertionError("trajectory selection no longer corresponds exactly to model action timestamps")
    selected = np.ascontiguousarray(schedule_ns[indices])
    if len(selected) != len(trajectory.upper_states) or np.any(np.diff(selected) <= 0):
        raise AssertionError("selected native action schedule is invalid")
    return selected


def print_native_action_timing(
    trajectory: zeno_replay.ReplayTrajectory, native_schedule_ns: np.ndarray
) -> None:
    """Make the strict model-output clock contract explicit in runtime logs."""

    if not trajectory.source_timing:
        raise AssertionError("exact ACT deployment must use native action timestamps, not a resampled clock")
    if native_schedule_ns.dtype != np.int64 or native_schedule_ns.shape != trajectory.times_s.shape:
        raise AssertionError("native action schedule has unexpected shape or dtype")
    intervals_ns = np.diff(native_schedule_ns)
    if len(intervals_ns) == 0 or np.any(intervals_ns <= 0):
        raise AssertionError("native action schedule must contain strictly increasing timestamps")
    print(
        "[model-runtime] native output clock=action timestamps; "
        f"frames={len(trajectory.upper_states)} median_rate={1e9 / float(np.median(intervals_ns)):.6f}Hz "
        f"dt_min={float(intervals_ns.min()) / 1e6:.3f}ms dt_max={float(intervals_ns.max()) / 1e6:.3f}ms",
        flush=True,
    )


def print_model_output_summary(
    trajectory: zeno_replay.ReplayTrajectory, native_schedule_ns: np.ndarray, args: argparse.Namespace
) -> None:
    """Print only model-session facts, never a source-file summary."""

    output_duration_s = (native_schedule_ns[-1] - native_schedule_ns[0]) / 1e9
    print(
        "[model-runtime] output ready "
        f"frames={len(trajectory.upper_states)} duration={output_duration_s:.9f}s "
        f"command_topic={args.cmd_topic} layout=24D-whole-body",
        flush=True,
    )


def fresh_upper_body_pose(node: Any, max_age_s: float) -> tuple[np.ndarray | None, list[str]]:
    """Return a 20-D pose only when all required joint feedback is fresh."""

    current, missing = node.current_upper_body()
    if current is None:
        return None, list(missing)
    received_by_group = getattr(node, "current_joint_received_s", None)
    if received_by_group is None:
        return None, ["missing-feedback-age-metadata"]
    now_s = time.monotonic()
    stale_or_missing: list[str] = []
    for group in zeno_replay.REQUIRED_JOINTS:
        received_s = received_by_group.get(group)
        if received_s is None:
            stale_or_missing.append(f"missing:{group}")
            continue
        age_s = now_s - float(received_s)
        if age_s < 0.0 or age_s > max_age_s:
            stale_or_missing.append(f"stale:{group}")
    if stale_or_missing:
        return None, stale_or_missing
    pose = np.asarray(current, dtype=np.float64)
    if pose.shape != (zeno_replay.UPPER_BODY_DIM,) or not np.isfinite(pose).all():
        return None, ["invalid:upper-body"]
    return pose, []


def await_command_subscriber(rclpy_module: Any, node: Any, args: argparse.Namespace) -> None:
    """Verify that the whole-body controller is connected to this session."""

    zeno_replay.spin_for(rclpy_module, node, args.discovery_wait_s)
    subscriber_counts = node.command_subscriber_counts()
    missing_topics = [topic for topic, count in subscriber_counts.items() if count == 0]
    if missing_topics and not args.allow_no_command_subscriber:
        raise RuntimeError(
            "no subscriber matched whole-body command topic after "
            f"{args.discovery_wait_s:g}s: {', '.join(missing_topics)}. Refusing to move to start."
        )
    count_text = ", ".join(f"{topic}={count}" for topic, count in subscriber_counts.items())
    print(f"[model-link] matched command subscribers: {count_text}", flush=True)


def await_live_upper_body_pose(rclpy_module: Any, node: Any, args: argparse.Namespace) -> np.ndarray:
    """Return a finite, fresh live 20-D upper-body pose."""

    deadline_s = time.monotonic() + args.preflight_timeout_s
    missing: list[str] = list(zeno_replay.REQUIRED_JOINTS)
    while time.monotonic() < deadline_s:
        rclpy_module.spin_once(node, timeout_sec=min(0.05, max(0.0, deadline_s - time.monotonic())))
        pose, missing = fresh_upper_body_pose(node, args.move_to_start_max_feedback_age_s)
        if pose is None:
            continue
        return pose
    raise RuntimeError(
        "timed out waiting for valid, fresh live JointState inputs for: " + ", ".join(missing) + "."
    )


def await_command_subscriber_and_live_pose(
    rclpy_module: Any, node: Any, args: argparse.Namespace
) -> np.ndarray:
    """Verify the controller link and return a fresh upper-body pose."""

    await_command_subscriber(rclpy_module, node, args)
    return await_live_upper_body_pose(rclpy_module, node, args)


def move_to_model_start(
    rclpy_module: Any, node: Any, target_upper_state: np.ndarray, args: argparse.Namespace
) -> None:
    """Safely ramp the upper body to the selected action stream's first pose.

    The base command is exactly zero during this setup phase.  The native
    action timeline does not begin until fresh live feedback remains within
    the setup tolerance, so the movement itself cannot alter the model's
    subsequent action frequency or timestamps.
    """

    current = await_command_subscriber_and_live_pose(rclpy_module, node, args)
    target = np.asarray(target_upper_state, dtype=np.float64)
    if target.shape != (zeno_replay.UPPER_BODY_DIM,) or not np.isfinite(target).all():
        raise ValueError("first model upper-body output is not a finite 20-D target")
    delta = target - current
    max_delta = float(np.abs(delta).max())
    if max_delta <= args.move_to_start_position_tolerance:
        print(
            "[model-setup] already within start tolerance; "
            f"max_error={max_delta:.5f}; no setup ramp required.",
            flush=True,
        )
    else:
        duration_from_speed_s = max_delta / args.move_to_start_max_speed
        requested_duration_s = max(args.move_to_start_s, duration_from_speed_s)
        step_count = max(1, int(math.ceil(requested_duration_s * args.move_to_start_rate_hz)))
        planned_duration_s = step_count / args.move_to_start_rate_hz
        move_start_s = time.perf_counter()
        print(
            "[model-setup] ramping upper body to initial model pose "
            f"duration={planned_duration_s:.3f}s steps={step_count} "
            f"rate={args.move_to_start_rate_hz:.3f}Hz max_delta={max_delta:.5f} "
            f"speed_limit={args.move_to_start_max_speed:.5f}/s; base=(0,0,0)",
            flush=True,
        )
        for step in range(1, step_count + 1):
            fraction = step / step_count
            node.publish_frame(current + fraction * delta, (0.0, 0.0, 0.0))
            deadline_s = move_start_s + step / args.move_to_start_rate_hz
            zeno_replay.spin_for(rclpy_module, node, max(0.0, deadline_s - time.perf_counter()))

    # Continue holding the target while waiting for measured state feedback.
    # A successful ramp command is not enough: the recorded timeline starts
    # only after the physical controller reports it has reached the pose.
    deadline_s = time.monotonic() + args.move_to_start_timeout_s
    publish_period_s = 1.0 / args.move_to_start_rate_hz
    in_tolerance_since_s: float | None = None
    last_error: float | None = None
    while time.monotonic() < deadline_s:
        frame_start_s = time.perf_counter()
        node.publish_frame(target, (0.0, 0.0, 0.0))
        zeno_replay.drain_callbacks(rclpy_module, node)
        measured, _missing = fresh_upper_body_pose(node, args.move_to_start_max_feedback_age_s)
        if measured is not None:
            errors = np.abs(np.asarray(measured, dtype=np.float64) - target)
            max_error = float(errors.max())
            last_error = max_error
            if max_error <= args.move_to_start_position_tolerance:
                now_s = time.monotonic()
                if in_tolerance_since_s is None:
                    in_tolerance_since_s = now_s
                if now_s - in_tolerance_since_s >= args.move_to_start_settle_s:
                    error_index = int(errors.argmax())
                    print(
                        "[model-setup] target accepted after continuous fresh feedback; "
                        f"max_error={max_error:.5f} at {zeno_replay.ACTION_FIELDS[error_index]}",
                        flush=True,
                    )
                    return
            else:
                in_tolerance_since_s = None
        else:
            in_tolerance_since_s = None
        zeno_replay.spin_for(
            rclpy_module, node, max(0.0, frame_start_s + publish_period_s - time.perf_counter())
        )
    suffix = "" if last_error is None else f"; last max_error={last_error:.5f}"
    raise RuntimeError(
        "move-to-start timed out before live upper-body feedback reached the first action pose within "
        f"{args.move_to_start_position_tolerance:.5f} continuously{suffix}"
    )


def verify_model_start_pose(
    rclpy_module: Any, node: Any, target_upper_state: np.ndarray, args: argparse.Namespace
) -> None:
    """Require a safe measured initial pose when no setup ramp is requested."""

    await_command_subscriber(rclpy_module, node, args)
    if args.allow_unchecked_start:
        print("[model-setup] live initial-pose comparison skipped by explicit option.", flush=True)
        return

    current = await_live_upper_body_pose(rclpy_module, node, args)
    target = np.asarray(target_upper_state, dtype=np.float64)
    errors = np.abs(current - target)
    error_index = int(errors.argmax())
    max_error = float(errors[error_index])
    if max_error > args.start_max_position_error:
        raise RuntimeError(
            "initial model-pose check failed: "
            f"{zeno_replay.ACTION_FIELDS[error_index]} current={current[error_index]:.5f}, "
            f"model_initial={target[error_index]:.5f}, abs_error={max_error:.5f} > "
            f"--start-max-position-error {args.start_max_position_error:.5f}. "
            "Use --move-to-start, or inspect the pose before explicitly bypassing this check."
        )
    print(
        "[model-setup] initial pose accepted; "
        f"maximum upper-body error={max_error:.5f} at {zeno_replay.ACTION_FIELDS[error_index]}",
        flush=True,
    )


def send_safe_idle(rclpy_module: Any, node: Any) -> bool:
    """Emit the established all-zero safety command without changing model data."""

    if not rclpy_module.ok():
        print("[model-runtime] ROS context stopped before safe output hold could be sent.", flush=True)
        return False
    for repeat in range(zeno_replay.IDLE_PUBLISH_REPEATS):
        node.publish_idle()
        rclpy_module.spin_once(node, timeout_sec=0.0)
        if repeat + 1 < zeno_replay.IDLE_PUBLISH_REPEATS:
            time.sleep(0.05)
    return True


def run_model_output_session(
    trajectory: zeno_replay.ReplayTrajectory, native_schedule_ns: np.ndarray, args: argparse.Namespace
) -> None:
    """Run a checked model-output session and remain in safe idle on completion."""

    rclpy_module: Any | None = None
    node: Any | None = None
    completed = False
    entered_safe_hold = False
    max_lateness_s = 0.0
    try:
        rclpy_module, node = zeno_replay.create_ros_publisher(args)
        if args.move_to_start:
            move_to_model_start(rclpy_module, node, trajectory.upper_states[0], args)
        else:
            verify_model_start_pose(rclpy_module, node, trajectory.upper_states[0], args)

        # The output epoch begins only after the setup or initial-pose check
        # succeeds.  Every deadline comes directly from the stored native
        # action timestamp differences; no fixed-rate clock is introduced.
        output_start_s = time.perf_counter()
        for index, (upper_state, base_twist) in enumerate(
            zip(trajectory.upper_states, trajectory.base_twists, strict=True)
        ):
            scheduled_s = output_start_s + (native_schedule_ns[index] - native_schedule_ns[0]) / 1e9
            lateness_s = time.perf_counter() - scheduled_s
            if lateness_s > args.max_output_lateness_s:
                raise RuntimeError(
                    "native action timing deadline missed before model output frame "
                    f"{index}: lateness={lateness_s * 1e3:.3f}ms exceeds --max-output-lateness-s "
                    f"{args.max_output_lateness_s * 1e3:.3f}ms; stopping rather than bursting or skipping output"
                )
            max_lateness_s = max(max_lateness_s, max(0.0, lateness_s))
            node.publish_frame(upper_state, base_twist)
            rclpy_module.spin_once(node, timeout_sec=0.0)
            if index % args.log_every_n == 0 or index == len(trajectory.upper_states) - 1:
                print(
                    f"[model-output] step={index}/{len(trajectory.upper_states) - 1} "
                    f"model_time={trajectory.times_s[index]:.3f}s: "
                    f"{zeno_replay.format_command(upper_state, base_twist)}",
                    flush=True,
                )
            if index + 1 < len(trajectory.upper_states):
                deadline_s = output_start_s + (native_schedule_ns[index + 1] - native_schedule_ns[0]) / 1e9
                time.sleep(max(0.0, deadline_s - time.perf_counter()))

        completed = True
        print(
            "[model-runtime] model output reached its final timestamp; "
            f"max_scheduler_lateness={max_lateness_s * 1e3:.3f}ms",
            flush=True,
        )
        entered_safe_hold = send_safe_idle(rclpy_module, node)
        if args.exit_after_sequence:
            return
        print(
            "[model-hold] safe output hold active; process remains online. Press Ctrl-C to end the session.",
            flush=True,
        )
        while rclpy_module.ok():
            # Keep feedback subscriptions and the ROS context alive without
            # issuing fresh action values after the finite model output ends.
            rclpy_module.spin_once(node, timeout_sec=0.25)
    finally:
        if node is not None and rclpy_module is not None:
            try:
                # On an interrupted hold, issue the same safe idle command one
                # final time before the node disappears.  An explicit
                # --exit-after-sequence already sent it above.
                if not entered_safe_hold or not completed or not args.exit_after_sequence:
                    send_safe_idle(rclpy_module, node)
                if not completed:
                    print("[model-runtime] session stopped before final model timestamp.", flush=True)
            finally:
                node.destroy_node()
                if rclpy_module.ok():
                    rclpy_module.shutdown()


def preview_model_output(
    trajectory: zeno_replay.ReplayTrajectory, native_schedule_ns: np.ndarray, args: argparse.Namespace
) -> None:
    """Show the selected model outputs without creating a ROS node."""

    output_start_s = time.perf_counter()
    for index, (upper_state, base_twist) in enumerate(
        zip(trajectory.upper_states, trajectory.base_twists, strict=True)
    ):
        if index % args.log_every_n == 0 or index == len(trajectory.upper_states) - 1:
            print(
                f"[model-preview] step={index}/{len(trajectory.upper_states) - 1} "
                f"model_time={trajectory.times_s[index]:.3f}s: "
                f"{zeno_replay.format_command(upper_state, base_twist)}",
                flush=True,
            )
        if args.dry_run_realtime and index + 1 < len(trajectory.upper_states):
            deadline_s = output_start_s + (native_schedule_ns[index + 1] - native_schedule_ns[0]) / 1e9
            time.sleep(max(0.0, deadline_s - time.perf_counter()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=8775)
    parser.add_argument("--worker-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--run-model",
        dest="run_model",
        action="store_true",
        default=False,
        help="Send the finite model output to the normal whole-body command interface.",
    )
    # Retain existing launch scripts without advertising legacy vocabulary in
    # the operator-facing help.
    parser.add_argument(
        "--publish-commands",
        dest="run_model",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run-realtime", action="store_true")
    parser.add_argument(
        "--verify-model",
        dest="verify_model",
        action="store_true",
        default=False,
        help="Verify model output then exit.",
    )
    parser.add_argument(
        "--verify-only",
        dest="verify_model",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--start-offset-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-every-n", type=int, default=10_000)
    parser.add_argument("--cmd-topic", default="/zeno/h1/auto/wholebody/cmd")
    parser.add_argument("--discovery-wait-s", type=float, default=1.0)
    parser.add_argument("--allow-no-command-subscriber", action="store_true")
    parser.add_argument("--allow-unchecked-start", action="store_true")
    parser.add_argument("--preflight-timeout-s", type=float, default=3.0)
    parser.add_argument("--start-max-position-error", type=float, default=0.20)
    parser.add_argument(
        "--max-output-lateness-s",
        dest="max_output_lateness_s",
        type=float,
        default=0.020,
        help=(
            "Stop safely if the native action clock is this late before a model output frame; prevents a "
            "silent burst or skipped output."
        ),
    )
    parser.add_argument(
        "--replay-max-lateness-s",
        dest="max_output_lateness_s",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--move-to-start",
        action="store_true",
        help=(
            "Before model output, smoothly move the measured upper body to the initial model pose with "
            "zero base velocity; the action clock starts only after feedback accepts that pose."
        ),
    )
    parser.add_argument(
        "--move-to-start-s",
        type=float,
        default=5.0,
        help="Requested duration of the upper-body setup ramp before model output.",
    )
    parser.add_argument(
        "--move-to-start-rate-hz",
        type=float,
        default=50.0,
        help="Command rate used only while moving to the initial pose, not during model output.",
    )
    parser.add_argument(
        "--move-to-start-max-speed",
        type=float,
        default=0.15,
        help=(
            "Maximum per-dimension setup-ramp speed in native position units/s; the setup duration is "
            "automatically extended when needed."
        ),
    )
    parser.add_argument(
        "--move-to-start-settle-s",
        type=float,
        default=0.25,
        help="Minimum target-hold time after the setup ramp before model output can begin.",
    )
    parser.add_argument(
        "--move-to-start-timeout-s",
        type=float,
        default=15.0,
        help="Maximum target-hold time waiting for live feedback to reach the initial pose.",
    )
    parser.add_argument(
        "--move-to-start-position-tolerance",
        type=float,
        default=0.03,
        help="Required maximum 20-D upper-body feedback error before the model action clock starts.",
    )
    parser.add_argument(
        "--move-to-start-max-feedback-age-s",
        type=float,
        default=0.25,
        help="Reject stale JointState feedback during setup/acceptance after this age.",
    )
    parser.add_argument(
        "--exit-after-sequence",
        action="store_true",
        help="Exit after the finite model output instead of remaining in safe output hold.",
    )
    parser.add_argument("--torso-state-topic", default="/zeno/h1/wheelarm/torso/joint_state")
    parser.add_argument("--left-arm-state-topic", default="/zeno/h1/wheelarm/left_arm/joint_state")
    parser.add_argument("--right-arm-state-topic", default="/zeno/h1/wheelarm/right_arm/joint_state")
    parser.add_argument("--left-gripper-state-topic", default="/zeno/h1/left_gripper/joint_state")
    parser.add_argument("--right-gripper-state-topic", default="/zeno/h1/right_gripper/joint_state")
    parser.add_argument("--odom-topic", default="/zeno/h1/sensor/odom_raw")
    args = parser.parse_args()
    if args.worker_host != "127.0.0.1":
        parser.error("this trusted deployment bridge only permits --worker-host 127.0.0.1")
    if not 1 <= args.worker_port <= 65535:
        parser.error("--worker-port must be in [1,65535]")
    if not math.isfinite(args.worker_timeout_s) or args.worker_timeout_s <= 0.0:
        parser.error("--worker-timeout-s must be finite and positive")
    if args.log_every_n <= 0:
        parser.error("--log-every-n must be positive")
    if not math.isfinite(args.discovery_wait_s) or args.discovery_wait_s < 0.0:
        parser.error("--discovery-wait-s must be finite and non-negative")
    if not math.isfinite(args.preflight_timeout_s) or args.preflight_timeout_s <= 0.0:
        parser.error("--preflight-timeout-s must be finite and positive")
    if not math.isfinite(args.start_max_position_error) or args.start_max_position_error <= 0.0:
        parser.error("--start-max-position-error must be finite and positive")
    if not math.isfinite(args.max_output_lateness_s) or args.max_output_lateness_s <= 0.0:
        parser.error("--max-output-lateness-s must be finite and positive")
    if not math.isfinite(args.move_to_start_s) or args.move_to_start_s <= 0.0:
        parser.error("--move-to-start-s must be finite and positive")
    if not math.isfinite(args.move_to_start_rate_hz) or args.move_to_start_rate_hz <= 0.0:
        parser.error("--move-to-start-rate-hz must be finite and positive")
    if not math.isfinite(args.move_to_start_max_speed) or args.move_to_start_max_speed <= 0.0:
        parser.error("--move-to-start-max-speed must be finite and positive")
    if not math.isfinite(args.move_to_start_settle_s) or args.move_to_start_settle_s < 0.0:
        parser.error("--move-to-start-settle-s must be finite and non-negative")
    if not math.isfinite(args.move_to_start_timeout_s) or args.move_to_start_timeout_s <= 0.0:
        parser.error("--move-to-start-timeout-s must be finite and positive")
    if (
        not math.isfinite(args.move_to_start_position_tolerance)
        or args.move_to_start_position_tolerance <= 0.0
    ):
        parser.error("--move-to-start-position-tolerance must be finite and positive")
    if (
        not math.isfinite(args.move_to_start_max_feedback_age_s)
        or args.move_to_start_max_feedback_age_s <= 0.0
    ):
        parser.error("--move-to-start-max-feedback-age-s must be finite and positive")
    if args.move_to_start and not args.run_model:
        parser.error("--move-to-start requires --run-model because it sends real setup commands")
    if args.move_to_start and args.allow_unchecked_start:
        parser.error("--move-to-start requires live JointState feedback; remove --allow-unchecked-start")
    if args.move_to_start and args.allow_no_command_subscriber:
        parser.error("--move-to-start requires a matched controller subscriber; remove --allow-no-command-subscriber")
    if args.verify_model and args.run_model:
        parser.error("--verify-model cannot be combined with --run-model")
    # The exact model contract is native-timestamp scheduling.  Do not expose
    # an accidental fixed-rate/resampling path here.
    args.rate_hz = None
    args.replay_source = "action"
    args.dry_run = not args.run_model
    args.record_actual_state = False
    return args


def main() -> None:
    args = parse_args()
    response = request_model_trajectory(args)
    source = source_from_model_response(response)
    if args.verify_model:
        return
    trajectory = zeno_replay.build_trajectory(
        source,
        rate_hz=None,
        start_offset_s=args.start_offset_s,
        duration_s=args.duration_s,
        max_frames=args.max_frames,
        replay_source="action",
    )
    native_schedule_ns = selected_native_action_schedule_ns(response, source, trajectory)
    print_native_action_timing(trajectory, native_schedule_ns)
    print_model_output_summary(trajectory, native_schedule_ns, args)
    if args.run_model:
        run_model_output_session(trajectory, native_schedule_ns, args)
    else:
        preview_model_output(trajectory, native_schedule_ns, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[model-runtime] session ended by operator.", flush=True)
