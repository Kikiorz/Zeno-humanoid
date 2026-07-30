#!/usr/bin/env python3
"""Render one raw Robot8 top-stereo JPEG through the deployment camera contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np


EXPECTED_CALIBRATION_SHA256 = "6d08b6a01a1431476c2c3c77bee43e3f8f20888f33940af772cf1963e9f6b342"
# This verifier is intentionally portable: copy this whole directory and the
# sibling camera/ asset directory together; no checkout of the parent repo is
# needed.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the exact cam_20260729 raw stereo -> rectified left RGB -> 640x480 letterbox contract"
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw 2560x720 left|right head-camera JPEG")
    parser.add_argument("--output", type=Path, required=True, help="Output 640x480 RGB visualisation PNG")
    parser.add_argument(
        "--output-rgb-npy",
        type=Path,
        default=None,
        help="Optional output .npy containing uint8 RGB HWC pixels for exact comparison",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 480),
        help="Output size; use the deployment default 640 480 unless the trained model differs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Raw input image does not exist: {input_path}")
    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise SystemExit("--image-size values must be positive")

    calibration_digest = sha256(DEFAULT_CALIBRATION)
    if calibration_digest != EXPECTED_CALIBRATION_SHA256:
        raise SystemExit(
            "Calibration SHA-256 mismatch: expected "
            f"{EXPECTED_CALIBRATION_SHA256}, got {calibration_digest} ({DEFAULT_CALIBRATION})"
        )
    try:
        rectifier = TopStereoRectifier(DEFAULT_CALIBRATION)
        rgb = rectifier.decode_and_rectify_left_rgb(
            input_path.read_bytes(),
            image_size,
            resize_mode="letterbox",
        )
    except TopStereoRectificationError as exc:
        raise SystemExit(f"Top-camera contract check failed: {exc}") from exc
    if rgb is None:
        raise SystemExit(f"Could not decode compressed image: {input_path}")
    expected_shape = (image_size[1], image_size[0], 3)
    if rgb.shape != expected_shape or rgb.dtype != np.uint8:
        raise SystemExit(f"Unexpected rectifier output: {rgb.shape} {rgb.dtype}; expected {expected_shape} uint8")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise SystemExit(f"Could not write output image: {output_path}")
    if args.output_rgb_npy is not None:
        npy_path = args.output_rgb_npy.expanduser().resolve()
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, rgb)
        print(f"RGB NPY: {npy_path}")

    print("cam_20260729 calibration SHA-256:", calibration_digest)
    print("raw input:", input_path)
    print("output RGB shape/dtype:", rgb.shape, rgb.dtype)
    print("output PNG:", output_path)
    print("geometry: raw 2560x720 left|right -> rectify -> left crop x=20,y=0,w=1240,h=620 -> letterbox")


if __name__ == "__main__":
    main()
