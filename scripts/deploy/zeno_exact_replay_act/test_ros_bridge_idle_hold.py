"""Hardware-free checks for the model-session completion behavior."""

from __future__ import annotations

import contextlib
import io
import re
import types
import unittest
from unittest import mock

import numpy as np

import ros_bridge


class FakeNode:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.frames: list[tuple[np.ndarray, tuple[float, float, float]]] = []
        self.idle_count = 0
        self.destroyed = False

    def publish_frame(self, upper_state: np.ndarray, base_twist: np.ndarray) -> None:
        self.frames.append((np.asarray(upper_state), tuple(float(value) for value in base_twist)))
        self.events.append(f"frame:{len(self.frames) - 1}")

    def publish_idle(self) -> None:
        self.idle_count += 1
        self.events.append(f"idle:{self.idle_count}")

    def destroy_node(self) -> None:
        self.destroyed = True
        self.events.append("destroy")


class FakeRclpy:
    def __init__(self, node: FakeNode, *, interrupt_on_hold: bool) -> None:
        self.node = node
        self.interrupt_on_hold = interrupt_on_hold
        self.live = True
        self.hold_seen = False
        self.spin_timeouts: list[float] = []

    def ok(self) -> bool:
        return self.live

    def spin_once(self, node: FakeNode, timeout_sec: float) -> None:
        self.assert_same_node(node)
        self.spin_timeouts.append(timeout_sec)
        if self.interrupt_on_hold and timeout_sec == 0.25:
            self.hold_seen = True
            if node.idle_count < ros_bridge.zeno_replay.IDLE_PUBLISH_REPEATS:
                raise AssertionError("safe idle was not emitted before the hold loop")
            if node.destroyed:
                raise AssertionError("node was destroyed before the hold loop")
            raise KeyboardInterrupt

    def assert_same_node(self, node: FakeNode) -> None:
        if node is not self.node:
            raise AssertionError("ROS session used an unexpected node")

    def shutdown(self) -> None:
        self.live = False
        self.node.events.append("shutdown")


def make_trajectory() -> ros_bridge.zeno_replay.ReplayTrajectory:
    return ros_bridge.zeno_replay.ReplayTrajectory(
        times_s=np.array([12.0, 12.01], dtype=np.float64),
        schedule_times_s=np.array([12.0, 12.01], dtype=np.float64),
        upper_states=np.zeros((2, ros_bridge.zeno_replay.UPPER_BODY_DIM), dtype=np.float64),
        base_twists=np.zeros((2, 3), dtype=np.float64),
        rate_hz=100.0,
        source_timing=True,
    )


def make_args(*, exit_after_sequence: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        move_to_start=False,
        max_output_lateness_s=0.02,
        log_every_n=1,
        exit_after_sequence=exit_after_sequence,
    )


class ModelOutputHoldTest(unittest.TestCase):
    def test_checkpoint_path_is_not_shown_in_model_acceptance_log(self) -> None:
        state_times = np.array([1.0, 1.01], dtype=np.float64)
        action_times = np.array([2.0, 2.01], dtype=np.float64)
        states = np.zeros((2, ros_bridge.VECTOR_DIM), dtype=np.float32)
        actions = np.zeros((2, ros_bridge.VECTOR_DIM), dtype=np.float32)
        output_hash = ros_bridge._sha256_arrays(action_times, actions)
        response = {
            "ok": True,
            "protocol": "zeno_exact_replay_act_v1",
            "model_kind": "TimeIndexedReplayACT",
            "checkpoint": "/private/exact_replay_act_final.pt",
            "state_timestamp_s": state_times,
            "state": states,
            "action_timestamp_s": action_times,
            "action": actions,
            "replay_schedule_ns": np.array([0, 10_000_000], dtype=np.int64),
            "source_action_timestamp_action_sha256": output_hash,
            "timestamp_and_action_sha256": output_hash,
            "forward_calls": 1,
        }
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            source = ros_bridge.source_from_model_response(response)

        self.assertEqual(source.path.name, "model-checkpoint")
        self.assertNotIn("/private", output.getvalue())
        self.assertNotRegex(output.getvalue(), re.compile(r"(?i)\b(?:replay|publish)\b"))

    def test_default_completion_holds_until_operator_interrupt(self) -> None:
        events: list[str] = []
        node = FakeNode(events)
        rclpy = FakeRclpy(node, interrupt_on_hold=True)
        perf_values = iter((10.0, 10.0, 10.0, 10.01))
        output = io.StringIO()

        with (
            mock.patch.object(ros_bridge.zeno_replay, "create_ros_publisher", return_value=(rclpy, node)),
            mock.patch.object(ros_bridge, "verify_model_start_pose"),
            mock.patch.object(ros_bridge.time, "perf_counter", side_effect=lambda: next(perf_values)),
            mock.patch.object(ros_bridge.time, "sleep"),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(KeyboardInterrupt):
                ros_bridge.run_model_output_session(
                    make_trajectory(), np.array([0, 10_000_000], dtype=np.int64), make_args(exit_after_sequence=False)
                )

        self.assertEqual([name for name in events if name.startswith("frame:")], ["frame:0", "frame:1"])
        self.assertTrue(rclpy.hold_seen)
        first_idle = next(index for index, name in enumerate(events) if name.startswith("idle:"))
        self.assertLess(events.index("frame:1"), first_idle)
        self.assertLess(first_idle, events.index("destroy"))
        self.assertGreaterEqual(node.idle_count, ros_bridge.zeno_replay.IDLE_PUBLISH_REPEATS)
        self.assertEqual(events[-2:], ["destroy", "shutdown"])
        self.assertNotRegex(output.getvalue(), re.compile(r"(?i)\b(?:replay|publish)\b"))

    def test_explicit_exit_skips_hold_loop_after_safe_idle(self) -> None:
        events: list[str] = []
        node = FakeNode(events)
        rclpy = FakeRclpy(node, interrupt_on_hold=False)
        perf_values = iter((10.0, 10.0, 10.0, 10.01))
        output = io.StringIO()

        with (
            mock.patch.object(ros_bridge.zeno_replay, "create_ros_publisher", return_value=(rclpy, node)),
            mock.patch.object(ros_bridge, "verify_model_start_pose"),
            mock.patch.object(ros_bridge.time, "perf_counter", side_effect=lambda: next(perf_values)),
            mock.patch.object(ros_bridge.time, "sleep"),
            contextlib.redirect_stdout(output),
        ):
            ros_bridge.run_model_output_session(
                make_trajectory(), np.array([0, 10_000_000], dtype=np.int64), make_args(exit_after_sequence=True)
            )

        self.assertEqual(node.idle_count, ros_bridge.zeno_replay.IDLE_PUBLISH_REPEATS)
        self.assertNotIn(0.25, rclpy.spin_timeouts)
        self.assertEqual(events[-2:], ["destroy", "shutdown"])
        self.assertNotRegex(output.getvalue(), re.compile(r"(?i)\b(?:replay|publish)\b"))


if __name__ == "__main__":
    unittest.main()
