"""Portable, exact 2026-07-29 Robot8 top-camera processing contract.

The raw ROS head-camera frame is one compressed 2560x720 JPEG containing
``left|right`` unrectified fisheye images.  This module emits the calibrated
left eye used by the current Robot8 deployments:

    raw side-by-side JPEG -> split -> fisheye rectify/alignment ->
    fixed x=20,y=0,w=1240,h=620 crop -> RGB -> letterbox

It intentionally has no dependency on the parent repository.  Keep this file
and its sibling ``stereo_params_20260729_172611.npz`` together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


FIXED_CROP_X = 20
FIXED_CROP_Y = 0
FIXED_CROP_WIDTH = 1240
FIXED_CROP_HEIGHT = 620
EXPECTED_EYE_SIZE = (1280, 720)


class TopStereoRectificationError(ValueError):
    """The raw head frame or cam_20260729 calibration is invalid."""


def _matrix(payload: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in payload.files:
        raise TopStereoRectificationError(f"Calibration is missing {key}")
    value = np.asarray(payload[key], dtype=np.float64)
    if value.shape != shape:
        raise TopStereoRectificationError(
            f"Calibration {key} must have shape {shape}, got {value.shape}"
        )
    return value


class TopStereoRectifier:
    """Rectify raw 2026-07-29 stereo JPEGs and retain calibrated left RGB."""

    def __init__(self, calibration_path: Path) -> None:
        calibration_path = Path(calibration_path).expanduser().resolve()
        if not calibration_path.is_file():
            raise TopStereoRectificationError(
                f"cam_20260729 calibration not found: {calibration_path}"
            )
        try:
            payload = np.load(calibration_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise TopStereoRectificationError(
                f"Unable to load cam_20260729 calibration {calibration_path}: {exc}"
            ) from exc
        try:
            model = str(payload["model"].item())
            image_size = np.asarray(payload["image_size"], dtype=np.int64)
            quality_status = str(payload["quality_status"].item())
        except (KeyError, ValueError) as exc:
            raise TopStereoRectificationError("Invalid cam_20260729 calibration header") from exc
        if model != "fisheye":
            raise TopStereoRectificationError(f"Expected fisheye calibration, got {model!r}")
        if tuple(int(value) for value in image_size) != EXPECTED_EYE_SIZE:
            raise TopStereoRectificationError(
                f"Expected per-eye size {EXPECTED_EYE_SIZE}, got {tuple(image_size)!r}"
            )
        if quality_status != "PASS":
            raise TopStereoRectificationError(
                f"Refusing calibration without PASS quality status, got {quality_status!r}"
            )

        # Validate the full stereo calibration even though this deployment only
        # emits the left eye.  R1/P1 were derived from a joint stereo solve.
        k1 = _matrix(payload, "K1", (3, 3))
        _matrix(payload, "K2", (3, 3))
        d1 = _matrix(payload, "D1", (4, 1))
        _matrix(payload, "D2", (4, 1))
        r1 = _matrix(payload, "R1", (3, 3))
        _matrix(payload, "R2", (3, 3))
        p1 = _matrix(payload, "P1", (3, 4))
        _matrix(payload, "P2", (3, 4))
        payload.close()

        width, height = EXPECTED_EYE_SIZE
        if (
            FIXED_CROP_X + FIXED_CROP_WIDTH > width
            or FIXED_CROP_Y + FIXED_CROP_HEIGHT > height
        ):
            raise TopStereoRectificationError("Fixed crop is outside the calibrated eye image")
        self.calibration_path = calibration_path
        self._left_maps = cv2.fisheye.initUndistortRectifyMap(
            k1,
            d1,
            r1,
            p1,
            (width, height),
            cv2.CV_32FC1,
        )

    @staticmethod
    def _split_raw_bgr(side_by_side_bgr: np.ndarray) -> np.ndarray:
        if (
            side_by_side_bgr is None
            or side_by_side_bgr.ndim != 3
            or side_by_side_bgr.shape[2] != 3
        ):
            raise TopStereoRectificationError("Top stereo frame must be an HxWx3 BGR image")
        height, width = side_by_side_bgr.shape[:2]
        if (width, height) != (EXPECTED_EYE_SIZE[0] * 2, EXPECTED_EYE_SIZE[1]):
            raise TopStereoRectificationError(
                "Expected raw top stereo 2560x720 left|right, "
                f"got {width}x{height}"
            )
        return side_by_side_bgr[:, : EXPECTED_EYE_SIZE[0]]

    def rectify_left_bgr(self, side_by_side_bgr: np.ndarray) -> np.ndarray:
        """Return rectified, fixed-cropped left BGR image (620x1240)."""

        left_raw = self._split_raw_bgr(side_by_side_bgr)
        left_rectified = cv2.remap(
            left_raw,
            self._left_maps[0],
            self._left_maps[1],
            cv2.INTER_LINEAR,
        )
        return left_rectified[
            FIXED_CROP_Y : FIXED_CROP_Y + FIXED_CROP_HEIGHT,
            FIXED_CROP_X : FIXED_CROP_X + FIXED_CROP_WIDTH,
        ].copy()

    def decode_and_rectify_left_rgb(
        self,
        compressed: bytes | bytearray | memoryview,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> np.ndarray | None:
        """Decode raw compressed stereo bytes into calibrated left RGB HWC."""

        if resize_mode != "letterbox":
            raise TopStereoRectificationError(
                "cam_20260729 deployment requires letterbox; stretch would change training geometry"
            )
        encoded = np.frombuffer(compressed, dtype=np.uint8)
        side_by_side_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if side_by_side_bgr is None:
            return None
        left_bgr = self.rectify_left_bgr(side_by_side_bgr)
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        return _letterbox_rgb(left_rgb, output_size)


def _letterbox_rgb(image_rgb: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = (int(output_size[0]), int(output_size[1]))
    if width <= 0 or height <= 0:
        raise TopStereoRectificationError(f"Invalid output size {output_size!r}")
    source_height, source_width = image_rgb.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, int(round(source_width * scale))))
    resized_height = max(1, min(height, int(round(source_height * scale))))
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros((height, width, 3), dtype=image_rgb.dtype)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas
