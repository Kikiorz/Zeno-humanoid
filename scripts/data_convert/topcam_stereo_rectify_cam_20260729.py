"""Exact 2026-07-29 top-stereo rectification contract.

The 2026-07-29 bags publish one *raw*, side-by-side JPEG on the head camera
topic.  The left and right 1280x720 images are unrectified fisheye images.
This module deliberately consumes the calibration supplied by the user under
``scripts/data_convert/cam/`` instead of the older Robot8 calibration.

Contract::

    2560x720 BGR JPEG (left|right)
      -> split into two 1280x720 views
      -> OpenCV fisheye initUndistortRectifyMap/remap using the supplied
         K1/D1/R1/P1 and K2/D2/R2/P2
      -> independent rectified 1280x720 views
      -> fixed valid-field crop x=20, y=0, width=1240, height=620
      -> RGB + optional letterbox to policy resolution

The crop is part of the supplied fixed processing contract, not a generic
center crop. Letterboxing is the default at a 640x480 policy input so the
selected calibrated field is preserved without stretch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = (
    REPO_ROOT / "scripts" / "data_convert" / "cam" / "stereo_params_20260729_172611.npz"
)
FIXED_CROP_X = 20
FIXED_CROP_Y = 0
FIXED_CROP_WIDTH = 1240
FIXED_CROP_HEIGHT = 620


class TopStereoRectificationError(ValueError):
    """The 2026-07-29 head frame or supplied calibration is invalid."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(payload: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in payload.files:
        raise TopStereoRectificationError(f"Calibration is missing {key}")
    value = np.asarray(payload[key], dtype=np.float64)
    if value.shape != shape:
        raise TopStereoRectificationError(
            f"Calibration {key} must have shape {shape}, got {value.shape}"
        )
    return value


@dataclass(frozen=True)
class TopStereoContract:
    calibration_path: Path
    calibration_sha256: str
    input_width_per_eye: int
    input_height_per_eye: int
    rectified_width_per_eye: int
    rectified_height_per_eye: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    stereo_rms_px: float
    vertical_error_p95_px: float
    estimated_baseline_mm: float

    @property
    def side_by_side_size(self) -> tuple[int, int]:
        return (self.input_width_per_eye * 2, self.input_height_per_eye)

    @property
    def input_size_per_eye(self) -> tuple[int, int]:
        return (self.input_width_per_eye, self.input_height_per_eye)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": "cam_20260729",
            "pipeline": (
                "split_left_right_then_cam_20260729_opencv_fisheye_rectify_"
                "then_independent_left_right_rgb_resize"
            ),
            "output_eyes": ["left", "right"],
            "raw_layout": "left|right side-by-side BGR/JPEG",
            "calibration_path": str(self.calibration_path),
            "calibration_sha256": self.calibration_sha256,
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
            "interpolation": "cv2.INTER_LINEAR",
            "calibration_quality": {
                "stereo_rms_px": self.stereo_rms_px,
                "vertical_error_p95_px": self.vertical_error_p95_px,
                "estimated_baseline_mm": self.estimated_baseline_mm,
            },
        }


class TopStereoRectifier:
    """Rectify the 2026-07-29 raw head pair with the supplied NPZ calibration."""

    def __init__(self, calibration_path: Path = DEFAULT_CALIBRATION) -> None:
        calibration_path = calibration_path.expanduser().resolve()
        if not calibration_path.is_file():
            raise TopStereoRectificationError(
                f"2026-07-29 top-stereo calibration not found: {calibration_path}"
            )
        try:
            payload = np.load(calibration_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise TopStereoRectificationError(
                f"Unable to load 2026-07-29 top-stereo NPZ {calibration_path}: {exc}"
            ) from exc
        try:
            model = str(payload["model"].item())
            image_size = np.asarray(payload["image_size"], dtype=np.int64)
        except (KeyError, ValueError) as exc:
            raise TopStereoRectificationError("Invalid 2026-07-29 top-stereo NPZ header") from exc
        if model != "fisheye":
            raise TopStereoRectificationError(f"Expected fisheye model, got {model!r}")
        if image_size.shape != (2,) or np.any(image_size <= 0):
            raise TopStereoRectificationError(
                f"Calibration image_size must be two positive values, got {image_size!r}"
            )
        width, height = (int(image_size[0]), int(image_size[1]))
        k1 = _matrix(payload, "K1", (3, 3))
        k2 = _matrix(payload, "K2", (3, 3))
        d1 = _matrix(payload, "D1", (4, 1))
        d2 = _matrix(payload, "D2", (4, 1))
        r1 = _matrix(payload, "R1", (3, 3))
        r2 = _matrix(payload, "R2", (3, 3))
        p1 = _matrix(payload, "P1", (3, 4))
        p2 = _matrix(payload, "P2", (3, 4))
        try:
            quality_status = str(payload["quality_status"].item())
            rms = float(payload["stereo_rms"].item())
            vertical_p95 = float(payload["vertical_error_p95"].item())
            baseline = float(payload["estimated_baseline_mm"].item())
        except (KeyError, ValueError) as exc:
            raise TopStereoRectificationError("Calibration quality values are missing or invalid") from exc
        if quality_status != "PASS":
            raise TopStereoRectificationError(
                f"Refusing a top-stereo calibration without PASS status, got {quality_status!r}"
            )
        if not np.isfinite([rms, vertical_p95, baseline]).all() or baseline <= 0:
            raise TopStereoRectificationError("Calibration quality metrics are non-finite or invalid")

        output_size = (width, height)
        if (
            FIXED_CROP_X < 0
            or FIXED_CROP_Y < 0
            or FIXED_CROP_X + FIXED_CROP_WIDTH > width
            or FIXED_CROP_Y + FIXED_CROP_HEIGHT > height
        ):
            raise TopStereoRectificationError(
                "The fixed 2026-07-29 crop is outside the calibrated rectified image: "
                f"crop=({FIXED_CROP_X},{FIXED_CROP_Y},{FIXED_CROP_WIDTH},{FIXED_CROP_HEIGHT}) "
                f"image={width}x{height}"
            )
        self._left_maps = cv2.fisheye.initUndistortRectifyMap(
            k1, d1, r1, p1, output_size, cv2.CV_32FC1
        )
        self._right_maps = cv2.fisheye.initUndistortRectifyMap(
            k2, d2, r2, p2, output_size, cv2.CV_32FC1
        )
        self.contract = TopStereoContract(
            calibration_path=calibration_path,
            calibration_sha256=_sha256(calibration_path),
            input_width_per_eye=width,
            input_height_per_eye=height,
            rectified_width_per_eye=width,
            rectified_height_per_eye=height,
            crop_x=FIXED_CROP_X,
            crop_y=FIXED_CROP_Y,
            crop_width=FIXED_CROP_WIDTH,
            crop_height=FIXED_CROP_HEIGHT,
            stereo_rms_px=rms,
            vertical_error_p95_px=vertical_p95,
            estimated_baseline_mm=baseline,
        )

    def _split_raw_bgr(self, side_by_side_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if side_by_side_bgr is None or side_by_side_bgr.ndim != 3 or side_by_side_bgr.shape[2] != 3:
            raise TopStereoRectificationError("Top stereo frame must be a decoded HxWx3 BGR image")
        height, width = side_by_side_bgr.shape[:2]
        expected_width, expected_height = self.contract.side_by_side_size
        if (width, height) != (expected_width, expected_height):
            raise TopStereoRectificationError(
                f"Expected raw top stereo {expected_width}x{expected_height}, got {width}x{height}"
            )
        cut = self.contract.input_width_per_eye
        return side_by_side_bgr[:, :cut], side_by_side_bgr[:, cut:]

    def rectify_pair_bgr(self, side_by_side_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_raw, right_raw = self._split_raw_bgr(side_by_side_bgr)
        left = cv2.remap(left_raw, self._left_maps[0], self._left_maps[1], cv2.INTER_LINEAR)
        right = cv2.remap(right_raw, self._right_maps[0], self._right_maps[1], cv2.INTER_LINEAR)
        return self._crop_rectified(left), self._crop_rectified(right)

    def _crop_rectified(self, image_bgr: np.ndarray) -> np.ndarray:
        crop = self.contract
        return image_bgr[
            crop.crop_y : crop.crop_y + crop.crop_height,
            crop.crop_x : crop.crop_x + crop.crop_width,
        ].copy()

    def rectify_left_bgr(self, side_by_side_bgr: np.ndarray) -> np.ndarray:
        """Return only the aligned rectified left eye.

        ``R1/P1`` were solved jointly with the right-eye calibration, so this
        result is already in the calibrated stereo-aligned coordinate system.
        The single-topcam contract deliberately does not materialize the right
        eye after that alignment has been determined.
        """
        left_raw, _ = self._split_raw_bgr(side_by_side_bgr)
        left = cv2.remap(left_raw, self._left_maps[0], self._left_maps[1], cv2.INTER_LINEAR)
        return self._crop_rectified(left)

    def rectify_pair_rgb(
        self,
        side_by_side_bgr: np.ndarray,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> tuple[np.ndarray, np.ndarray]:
        left_bgr, right_bgr = self.rectify_pair_bgr(side_by_side_bgr)
        return (
            _resize_rgb(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB), output_size, resize_mode=resize_mode),
            _resize_rgb(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB), output_size, resize_mode=resize_mode),
        )

    def rectify_left_rgb(
        self,
        side_by_side_bgr: np.ndarray,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> np.ndarray:
        left_bgr = self.rectify_left_bgr(side_by_side_bgr)
        return _resize_rgb(
            cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB),
            output_size,
            resize_mode=resize_mode,
        )

    def decode_and_rectify_pair_rgb(
        self,
        compressed: bytes | bytearray | memoryview,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        raw = np.frombuffer(compressed, dtype=np.uint8)
        image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return self.rectify_pair_rgb(image_bgr, output_size, resize_mode=resize_mode)

    def decode_and_rectify_left_rgb(
        self,
        compressed: bytes | bytearray | memoryview,
        output_size: tuple[int, int],
        *,
        resize_mode: str = "letterbox",
    ) -> np.ndarray | None:
        """Decode a raw stereo JPEG and emit only the aligned left RGB image."""
        raw = np.frombuffer(compressed, dtype=np.uint8)
        image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return self.rectify_left_rgb(image_bgr, output_size, resize_mode=resize_mode)


def _resize_rgb(image_rgb: np.ndarray, output_size: tuple[int, int], *, resize_mode: str) -> np.ndarray:
    width, height = output_size
    if width <= 0 or height <= 0:
        raise TopStereoRectificationError(f"Invalid requested output size {output_size}")
    if resize_mode == "stretch":
        return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    if resize_mode != "letterbox":
        raise TopStereoRectificationError(
            f"Unsupported resize mode {resize_mode!r}; use letterbox or stretch"
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
