"""Fail-closed mapper for Robot8 V2 physical-base action labels.

The V2 dataset deliberately labels ``action[20:23]`` as the desired physical
body-frame odometry velocity, rather than the robot's low-level whole-body
command.  This module loads the fitted one-step ARX model and, using fresh
odometry, turns that physical desired velocity into a bounded command.

It has no ROS dependency so the math can be unit-tested separately from the
bridge.  Calling code must still enforce fresh odometry and publish idle if a
mapping error occurs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


BASE_DIM = 3


class BaseFeedbackMapperError(ValueError):
    """Raised when an unsafe or malformed V2 mapper cannot be used."""


def _finite_vector(value: Sequence[float], name: str, *, strictly_positive: bool | None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BaseFeedbackMapperError(f"{name} must be a numeric {BASE_DIM}-vector") from exc
    if array.shape != (BASE_DIM,):
        raise BaseFeedbackMapperError(f"{name} must have shape ({BASE_DIM},), got {array.shape}")
    if not np.isfinite(array).all():
        raise BaseFeedbackMapperError(f"{name} must contain only finite values")
    if strictly_positive:
        if np.any(array <= 0.0):
            raise BaseFeedbackMapperError(f"{name} must be strictly positive")
    elif strictly_positive is False and np.any(array < 0.0):
        raise BaseFeedbackMapperError(f"{name} must be non-negative")
    return array


def _finite_matrix(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BaseFeedbackMapperError(f"{name} must be a numeric {BASE_DIM}x{BASE_DIM} matrix") from exc
    if array.shape != (BASE_DIM, BASE_DIM):
        raise BaseFeedbackMapperError(
            f"{name} must have shape ({BASE_DIM}, {BASE_DIM}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise BaseFeedbackMapperError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class BaseMappingResult:
    """Auditable values for one desired-physical-to-command conversion."""

    desired_input: np.ndarray
    desired_limited: np.ndarray
    measured: np.ndarray
    goal: np.ndarray
    raw_command: np.ndarray
    command_limited: np.ndarray
    sent_command: np.ndarray
    desired_saturated: bool
    command_saturated: bool
    slew_saturated: bool

    @property
    def saturated(self) -> bool:
        return self.desired_saturated or self.command_saturated or self.slew_saturated


class PhysicalDesiredBaseMapper:
    """Invert the exported V2 ARX dynamics model with bounded feedback.

    The mapper model is::

        v_next = bias + A @ v_measured + B @ u_command

    For one 20 Hz tick, the bridge chooses a conservative target velocity::

        v_goal = v_measured + gain * (v_desired - v_measured)
        u_raw = inv(B) @ (v_goal - bias - A @ v_measured)

    Separate desired-velocity, low-level-command, and slew limits are applied
    in that order.  ``reset`` makes the next command ramp up from zero.
    """

    def __init__(
        self,
        *,
        bias: np.ndarray,
        state_matrix: np.ndarray,
        command_matrix: np.ndarray,
        command_matrix_inverse: np.ndarray,
        feedback_gain: float,
        desired_limits: np.ndarray,
        command_limits: np.ndarray,
        command_slew_limits: np.ndarray,
        source: Path,
    ) -> None:
        self.bias = bias
        self.state_matrix = state_matrix
        self.command_matrix = command_matrix
        self.command_matrix_inverse = command_matrix_inverse
        self.feedback_gain = feedback_gain
        self.desired_limits = desired_limits
        self.command_limits = command_limits
        self.command_slew_limits = command_slew_limits
        self.source = source
        self._previous_command = np.zeros(BASE_DIM, dtype=np.float64)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        feedback_gain: float | None = None,
        desired_limits: Sequence[float] = (0.16, 0.16, 0.35),
        command_limits: Sequence[float] = (0.15, 0.15, 0.30),
        command_slew_limits: Sequence[float] = (0.50, 0.50, 1.00),
        max_condition_number: float = 50.0,
    ) -> "PhysicalDesiredBaseMapper":
        source = Path(path).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise BaseFeedbackMapperError(f"V2 base mapper does not exist: {source}") from exc
        except json.JSONDecodeError as exc:
            raise BaseFeedbackMapperError(f"V2 base mapper is invalid JSON: {source}") from exc
        if not isinstance(payload, dict):
            raise BaseFeedbackMapperError("V2 base mapper must be a JSON object")

        bias = _finite_vector(payload.get("bias"), "bias", strictly_positive=None)
        state_matrix = _finite_matrix(payload.get("state_matrix"), "state_matrix")
        command_matrix = _finite_matrix(payload.get("command_matrix"), "command_matrix")
        try:
            condition_number = float(np.linalg.cond(command_matrix))
        except np.linalg.LinAlgError as exc:
            raise BaseFeedbackMapperError("command_matrix condition number could not be computed") from exc
        if not math.isfinite(condition_number) or condition_number > float(max_condition_number):
            raise BaseFeedbackMapperError(
                "command_matrix is too poorly conditioned for safe inversion: "
                f"condition={condition_number:.5g}, limit={float(max_condition_number):.5g}"
            )
        try:
            command_matrix_inverse = np.linalg.inv(command_matrix)
        except np.linalg.LinAlgError as exc:
            raise BaseFeedbackMapperError("command_matrix is singular") from exc

        exported_inverse = payload.get("command_matrix_inverse")
        if exported_inverse is not None:
            claimed_inverse = _finite_matrix(exported_inverse, "command_matrix_inverse")
            if not np.allclose(claimed_inverse, command_matrix_inverse, rtol=1e-5, atol=1e-7):
                raise BaseFeedbackMapperError("exported command_matrix_inverse does not match command_matrix")

        selected_gain = payload.get("recommended_feedback_gain", 0.5) if feedback_gain is None else feedback_gain
        try:
            selected_gain = float(selected_gain)
        except (TypeError, ValueError) as exc:
            raise BaseFeedbackMapperError("feedback gain must be a finite scalar") from exc
        if not math.isfinite(selected_gain) or not 0.0 < selected_gain <= 1.0:
            raise BaseFeedbackMapperError("feedback gain must be in (0, 1]")

        return cls(
            bias=bias,
            state_matrix=state_matrix,
            command_matrix=command_matrix,
            command_matrix_inverse=command_matrix_inverse,
            feedback_gain=selected_gain,
            desired_limits=_finite_vector(desired_limits, "desired_limits", strictly_positive=True),
            command_limits=_finite_vector(command_limits, "command_limits", strictly_positive=True),
            command_slew_limits=_finite_vector(
                command_slew_limits, "command_slew_limits", strictly_positive=True
            ),
            source=source,
        )

    def reset(self) -> None:
        """Reset the slew limiter after idle/fault handling."""
        self._previous_command = np.zeros(BASE_DIM, dtype=np.float64)

    def map(self, desired_physical: Sequence[float], measured_physical: Sequence[float], dt_s: float) -> BaseMappingResult:
        if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
            raise BaseFeedbackMapperError(f"dt_s must be finite and positive, got {dt_s!r}")
        desired_input = _finite_vector(desired_physical, "desired_physical", strictly_positive=None)
        measured = _finite_vector(measured_physical, "measured_physical", strictly_positive=None)
        desired_limited = np.clip(desired_input, -self.desired_limits, self.desired_limits)
        goal = measured + self.feedback_gain * (desired_limited - measured)
        raw_command = self.command_matrix_inverse @ (goal - self.bias - self.state_matrix @ measured)
        if not np.isfinite(raw_command).all():
            raise BaseFeedbackMapperError("base-command inversion produced NaN or Inf")
        command_limited = np.clip(raw_command, -self.command_limits, self.command_limits)
        maximum_delta = self.command_slew_limits * float(dt_s)
        sent_command = np.clip(
            command_limited,
            self._previous_command - maximum_delta,
            self._previous_command + maximum_delta,
        )
        self._previous_command = sent_command.copy()
        return BaseMappingResult(
            desired_input=desired_input,
            desired_limited=desired_limited,
            measured=measured,
            goal=goal,
            raw_command=raw_command,
            command_limited=command_limited,
            sent_command=sent_command,
            desired_saturated=not np.allclose(desired_input, desired_limited, rtol=0.0, atol=1e-12),
            command_saturated=not np.allclose(raw_command, command_limited, rtol=0.0, atol=1e-12),
            slew_saturated=not np.allclose(command_limited, sent_command, rtol=0.0, atol=1e-12),
        )
