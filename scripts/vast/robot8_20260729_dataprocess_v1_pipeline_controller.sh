#!/usr/bin/env bash
# Fresh 2026-07-29 pipeline whose only top-camera source of truth is the
# user-provided Data/process rectification contract.  It delegates lifecycle
# mechanics to the generic controller but uses entirely new dataset, cache,
# analysis, and checkpoint namespaces so no NPZ-calibrated artifact can be
# mistaken for this run.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/2027icra}"
VENV_DIR="${VENV_DIR:-/venv/main}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-robot8_20260729_zeno_h1_auto_cmd_v30_3cam_640x480_topcam_left_dataprocess_v1_all23}"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/Data/lerobot/${SOURCE_REPO_ID}}"
V3_REPO_ID="${V3_REPO_ID:-${SOURCE_REPO_ID}_base_anchor_odom_v3_decoupled_smooth}"
V3_DATASET="${V3_DATASET:-${REPO_ROOT}/Data/lerobot/${V3_REPO_ID}}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/dino_feature_cache}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${CACHE_DIR}/robot8_20260729_topcam_left_dataprocess_v1_shared_dinov3_disk.json}"
CACHE_FILE="${CACHE_FILE:-${CACHE_DIR}/robot8_20260729_topcam_left_dataprocess_v1_shared_dinov3.f16}"
RAW_RUN_ID="${RAW_RUN_ID:-robot8_20260729_act_dinov3_3cam_640x480_topcam_left_dataprocess_v1_all23_decoder7_b32_100k}"
V3_RUN_ID="${V3_RUN_ID:-robot8_20260729_act_dinov3_3cam_640x480_topcam_left_dataprocess_v1_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k}"

# Explicitly point at the files the user supplied rather than their matching
# convenience copies.  These two hashes become a hard provenance gate.
TOPCAM_PROFILE="data_process_20260729"
HEAD_STEREO_CALIBRATION="${HEAD_STEREO_CALIBRATION:-${REPO_ROOT}/Data/process/top_stereo_calibration_basalt_kb4_compat.json}"
HEAD_STEREO_PROCESSING="${HEAD_STEREO_PROCESSING:-${REPO_ROOT}/Data/process/processing_metadata_centered_crop_1240x620.json}"
EXPECTED_TOPCAM_PROFILE="data_process_20260729"
EXPECTED_TOPCAM_PIPELINE="split_left_right_then_opencv_fisheye_rectify_then_crop_then_independent_left_right_rgb_resize"
EXPECTED_TOPCAM_RECTIFIED_HEIGHT=620
EXPECTED_TOPCAM_CALIBRATION_SHA256="53be6cfdc7a82bb3acb119ddcee9d4c504a1e6a89e9c035c54106807844043cc"
EXPECTED_TOPCAM_PROCESSING_SHA256="71633cda55868d1a2af8fadd8035de9cb7ebc180a89ac340461ac4dc5e3fc46e"

export REPO_ROOT VENV_DIR SOURCE_REPO_ID SOURCE_DATASET V3_REPO_ID V3_DATASET
export LEROBOT_ROOT="${REPO_ROOT}/Data/lerobot"
export CACHE_DIR CACHE_MANIFEST CACHE_FILE
export RAW_DATA_DIR="${REPO_ROOT}/Data/2026_07_29"
export ANALYSIS_DIR="${REPO_ROOT}/outputs/analysis/robot8_20260729_dataprocess_v1_base_anchor_odom_v3_decoupled_smooth"
export TOPCAM_PROFILE HEAD_STEREO_CALIBRATION HEAD_STEREO_PROCESSING
export EXPECTED_TOPCAM_PROFILE EXPECTED_TOPCAM_PIPELINE EXPECTED_TOPCAM_RECTIFIED_HEIGHT
export EXPECTED_TOPCAM_CALIBRATION_SHA256 EXPECTED_TOPCAM_PROCESSING_SHA256
export REQUIRE_TOPCAM_CACHE_PROVENANCE=1
export RAW_TRAIN_PROGRAM="robot8_20260729_dataprocess_v1_raw_train"
export V3_TRAIN_PROGRAM="robot8_20260729_dataprocess_v1_v3_train"
export RAW_SUCCESS="${REPO_ROOT}/outputs/train/${RAW_RUN_ID}/TRAINING_SUCCEEDED"
export V3_SUCCESS="${REPO_ROOT}/outputs/train/${V3_RUN_ID}/TRAINING_SUCCEEDED"

exec bash "${REPO_ROOT}/scripts/vast/robot8_20260729_cam20260729_pipeline_controller.sh"
