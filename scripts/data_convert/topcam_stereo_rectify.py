"""Canonical Robot8 top-stereo preprocessing used for training and deployment.

The head camera publishes one JPEG containing two unrectified fisheye images:
left|right, 2560x720 total.  Both eyes are always rectified independently and
remain independent outputs.  They must never be treated as a single raw
fisheye image, and are never concatenated after rectification.

The processing contract is intentionally exact and versioned by the two JSON
files in ``Data/process``:

    2560x720 BGR -> split 1280x720 -> fisheye rectify -> crop 1240x620
    -> emit independent left/right RGB views -> optional model resize.

There is no model-facing centre crop after rectification.  When a 640x480
model input is required, ``letterbox`` preserves the calibrated 2:1 geometry
without crop or aspect-ratio distortion.  This module has no ROS dependency
so the same object can be used by the bag converter and live deployment bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
# These live under version-controlled ``configs/`` rather than ``Data/`` so a
# clean checkout, remote training host, and later deployment host all carry
# the exact geometry contract.  ``Data/process`` remains the colleague-facing
# standalone reference copy.
DEFAULT_CALIBRATION = REPO_ROOT / "configs" / "calibration" / "top_stereo_calibration_basalt_kb4_compat.json"
DEFAULT_PROCESSING = REPO_ROOT / "configs" / "calibration" / "processing_metadata_centered_crop_1240x620.json"


class TopStereoRectificationError(ValueError):
    """The published head image or its calibration contract is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TopStereoRectificationError(f"Top-stereo configuration file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TopStereoRectificationError(f"Invalid JSON in top-stereo configuration {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(payload: dict[str, Any], key: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        value = np.asarray(payload[key], dtype=np.float64)
    except KeyError as exc:
        raise TopStereoRectificationError(f"Missing calibration matrix {key}") from exc
    if value.shape != shape:
        raise TopStereoRectificationError(
            f"Calibration matrix {key} must have shape {shape}, got {value.shape}"
        )
    return value


@dataclass(frozen=True)
class TopStereoContract:
    """Immutable geometry/provenance describing the rectified policy image."""

    calibration_path: Path
    processing_path: Path
    calibration_sha256: str
    processing_sha256: str
    input_width_per_eye: int
    input_height_per_eye: int
    rectified_width_per_eye: int
    rectified_height_per_eye: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    rotation_deg: float

    @property
    def side_by_side_size(self) -> tuple[int, int]:
        return (self.input_width_per_eye * 2, self.input_height_per_eye)

    @property
    def input_size_per_eye(self) -> tuple[int, int]:
        return (self.input_width_per_eye, self.input_height_per_eye)

    @property
    def rectified_size_per_eye(self) -> tuple[int, int]:
        return (self.rectified_width_per_eye, self.rectified_height_per_eye)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": "split_left_right_then_opencv_fisheye_rectify_then_crop_then_independent_left_right_rgb_resize",
            "output_eyes": ["left", "right"],
            "raw_layout": "left|right side-by-side BGR/JPEG",
            "calibration_path": str(self.calibration_path),
            "processing_path": str(self.processing_path),
            "calibration_sha256": self.calibration_sha256,
            "processing_sha256": self.processing_sha256,
            "input_size_per_eye": {
                "width": self.input_width_per_eye,
                "height": self.input_height_per_eye,
            },
            "side_by_side_input_size": {
                "width": self.input_width_per_eye * 2,
                "height": self.input_height_per_eye,
            },
            "rectified_size_per_eye": {
                "width": self.rectified_width_per_eye,
                "height": self.rectified_height_per_eye,
            },
            "crop": {
                "x": self.crop_x,
                "y": self.crop_y,
                "width": self.crop_width,
                "height": self.crop_height,
            },
            # Keep both names because ``spatial_crop`` is the dataset-facing
            # provenance schema used by the 2026-07-29 pipeline, while
            # ``crop`` remains useful to standalone Data/process consumers.
            "spatial_crop": {
                "x": self.crop_x,
                "y": self.crop_y,
                "width": self.crop_width,
                "height": self.crop_height,
            },
            "cropped_size_per_eye": {
                "width": self.crop_width,
                "height": self.crop_height,
            },
            "rotation_deg": self.rotation_deg,
            "interpolation": "cv2.INTER_LINEAR",
        }


class TopStereoRectifier:
    """Map a raw Robot8 top-stereo frame to calibrated left/right views."""

    def __init__(
        self,
        calibration_path: Path = DEFAULT_CALIBRATION,
        processing_path: Path = DEFAULT_PROCESSING,
    ) -> None:
        calibration_path = calibration_path.expanduser().resolve()
        processing_path = processing_path.expanduser().resolve()
        calibration = _read_json(calibration_path)
        processing = _read_json(processing_path)

        try:
            input_width = int(calibration["image_width"])
            input_height = int(calibration["image_height"])
            rectified = processing["rectified_image_size_per_eye"]
            rectified_width = int(rectified["width"])
            rectified_height = int(rectified["height"])
            crop_raw = processing["crop"]
            crop_x = int(crop_raw["x"])
            crop_y = int(crop_raw["y"])
            crop_width = int(crop_raw["width"])
            crop_height = int(crop_raw["height"])
            rotation_deg = float(processing.get("rotation_deg", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise TopStereoRectificationError(f"Invalid top-stereo geometry contract: {exc}") from exc

        rectified_from_calibration = calibration.get("rectified_image_size_per_eye")
        if rectified_from_calibration is not None:
            expected_rectified = (int(rectified_from_calibration["width"]), int(rectified_from_calibration["height"]))
            if (rectified_width, rectified_height) != expected_rectified:
                raise TopStereoRectificationError(
                    "Processing rectified size disagrees with calibration: "
                    f"processing={(rectified_width, rectified_height)}, calibration={expected_rectified}"
                )
        if min(input_width, input_height, rectified_width, rectified_height, crop_width, crop_height) <= 0:
            raise TopStereoRectificationError("Top-stereo dimensions must all be positive")
        if crop_x < 0 or crop_y < 0 or crop_x + crop_width > rectified_width or crop_y + crop_height > rectified_height:
            raise TopStereoRectificationError(
                "Top-stereo crop is outside rectified image: "
                f"crop={(crop_x, crop_y, crop_width, crop_height)}, rectified={(rectified_width, rectified_height)}"
            )

        k_left = _matrix(calibration, "K_left", (3, 3))
        k_right = _matrix(calibration, "K_right", (3, 3))
        r_left = _matrix(calibration, "R1", (3, 3))
        r_right = _matrix(calibration, "R2", (3, 3))
        p_left = _matrix(processing.get("rectification", calibration), "P1", (3, 4))
        p_right = _matrix(processing.get("rectification", calibration), "P2", (3, 4))
        d_left = np.asarray(calibration.get("D_left"), dtype=np.float64).reshape(-1, 1)
        d_right = np.asarray(calibration.get("D_right"), dtype=np.float64).reshape(-1, 1)
        if d_left.shape != (4, 1) or d_right.shape != (4, 1):
            raise TopStereoRectificationError(
                f"Expected four fisheye distortion coefficients, got left={d_left.shape}, right={d_right.shape}"
            )

        # The processing metadata is a deliberate duplicate of P1/P2.  Fail
        # closed if a future edit pairs the wrong metadata with this rig.
        for key, processed, calibrated in (("P1", p_left, _matrix(calibration, "P1", (3, 4))), ("P2", p_right, _matrix(calibration, "P2", (3, 4)))):
            if not np.allclose(processed, calibrated, atol=1e-12, rtol=0.0):
                raise TopStereoRectificationError(
                    f"Processing {key} disagrees with calibration {key}; refusing mismatched stereo geometry"
                )

        output_size = (rectified_width, rectified_height)
        self._left_maps = cv2.fisheye.initUndistortRectifyMap(
            k_left, d_left, r_left, p_left, output_size, cv2.CV_32FC1
        )
        self._right_maps = cv2.fisheye.initUndistortRectifyMap(
            k_right, d_right, r_right, p_right, output_size, cv2.CV_32FC1
        )
        self.contract = TopStereoContract(
            calibration_path=calibration_path,
            processing_path=processing_path,
            calibration_sha256=_sha256(calibration_path),
            processing_sha256=_sha256(processing_path),
            input_width_per_eye=input_width,
            input_height_per_eye=input_height,
            rectified_width_per_eye=rectified_width,
            rectified_height_per_eye=rectified_height,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            rotation_deg=rotation_deg,
        )

    def _split_raw_bgr(self, side_by_side_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if side_by_side_bgr is None or side_by_side_bgr.ndim != 3 or side_by_side_bgr.shape[2] != 3:
            raise TopStereoRectificationError(
                "Top stereo frame must be a decoded HxWx3 BGR image"
            )
        height, width = side_by_side_bgr.shape[:2]
        expected_width, expected_height = self.contract.side_by_side_size
        if (width, height) != (expected_width, expected_height):
            raise TopStereoRectificationError(
                "Expected top stereo raw frame "
                f"{expected_width}x{expected_height} left|right, got {width}x{height}"
            )
        cut = self.contract.input_width_per_eye
        return side_by_side_bgr[:, :cut], side_by_side_bgr[:, cut:]

    def rectify_pair_bgr(self, side_by_side_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return left/right BGR images after exact rectification and crop."""
        left_raw, right_raw = self._split_raw_bgr(side_by_side_bgr)
        left = cv2.remap(left_raw, self._left_maps[0], self._left_maps[1], cv2.INTER_LINEAR)
        right = cv2.remap(right_raw, self._right_maps[0], self._right_maps[1], cv2.INTER_LINEAR)
        c = self.contract
        y_slice = slice(c.crop_y, c.crop_y + c.crop_height)
        x_slice = slice(c.crop_x, c.crop_x + c.crop_width)
        left = left[y_slice, x_slice]
        right = right[y_slice, x_slice]
        if abs(c.rotation_deg) > 1e-9:
            left = _rotate_keep_size(left, c.rotation_deg)
            right = _rotate_keep_size(right, c.rotation_deg)
        return left, right

    def rectify_left_bgr(self, side_by_side_bgr: np.ndarray) -> np.ndarray:
        """Return only the stereo-aligned left eye after the exact fixed crop.

        ``R1/P1`` come from the joint stereo calibration, so this left view is
        already in the aligned coordinate system.  The right raw pixels never
        enter the returned model image or require a right-eye remap when the
        model declares only ``observation.images.head_cam``.
        """
        left_raw, _ = self._split_raw_bgr(side_by_side_bgr)
        left = cv2.remap(left_raw, self._left_maps[0], self._left_maps[1], cv2.INTER_LINEAR)
        c = self.contract
        left = left[
            c.crop_y : c.crop_y + c.crop_height,
            c.crop_x : c.crop_x + c.crop_width,
        ]
        if abs(c.rotation_deg) > 1e-9:
            left = _rotate_keep_size(left, c.rotation_deg)
        return left

    def rectify_pair_rgb(
        self,
        side_by_side_bgr: np.ndarray,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return independent left/right RGB model images from one raw frame."""
        left_bgr, right_bgr = self.rectify_pair_bgr(side_by_side_bgr)
        left_rgb = _resize_rgb(
            cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB),
            output_size,
            resize_mode=resize_mode,
        )
        right_rgb = _resize_rgb(
            cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB),
            output_size,
            resize_mode=resize_mode,
        )
        return left_rgb, right_rgb

    def decode_and_rectify_pair_rgb(
        self,
        compressed: bytes | bytearray | memoryview,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        raw = np.frombuffer(compressed, dtype=np.uint8)
        side_by_side_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if side_by_side_bgr is None:
            return None
        return self.rectify_pair_rgb(side_by_side_bgr, output_size, resize_mode=resize_mode)

    def decode_and_rectify_left_rgb(
        self,
        compressed: bytes | bytearray | memoryview,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> np.ndarray | None:
        """Decode, align, and emit the single model-facing left RGB topcam."""
        raw = np.frombuffer(compressed, dtype=np.uint8)
        side_by_side_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if side_by_side_bgr is None:
            return None
        return _resize_rgb(
            cv2.cvtColor(self.rectify_left_bgr(side_by_side_bgr), cv2.COLOR_BGR2RGB),
            output_size,
            resize_mode=resize_mode,
        )

    def decode_and_rectify_right_rgb(
        self,
        compressed: bytes | bytearray | memoryview,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> np.ndarray | None:
        pair = self.decode_and_rectify_pair_rgb(
            compressed,
            output_size,
            resize_mode=resize_mode,
        )
        return None if pair is None else pair[1]


def _rotate_keep_size(image: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _resize_rgb(image_rgb: np.ndarray, output_size: tuple[int, int], *, resize_mode: str) -> np.ndarray:
    """Resize without changing the calibrated geometry unless explicitly asked."""
    width, height = output_size
    if width <= 0 or height <= 0:
        raise TopStereoRectificationError(f"Invalid requested output size: {output_size}")
    if resize_mode == "stretch":
        return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    if resize_mode != "letterbox":
        raise TopStereoRectificationError(
            f"Unsupported top-stereo resize mode {resize_mode!r}; use 'letterbox' or 'stretch'"
        )

    source_height, source_width = image_rgb.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, int(round(source_width * scale))))
    resized_height = max(1, min(height, int(round(source_height * scale))))
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=image_rgb.dtype)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas
