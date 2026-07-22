#!/usr/bin/env python3
"""Replay data03 through ACT deployment modes and compare base commands.

The script uses the raw ROS bag so image decoding, center crop, resize,
normalization, action clamp, and 20 Hz control calls follow the deployment
worker.  At each virtual bridge tick, it uses the latest *causally available*
message from every topic, matching the bridge's latest-message caches rather
than the offline converter's nearest-timestamp sampling.  ``rtc_off`` is the
deployed ACT default (100-step FIFO).  ACT does not support LeRobot's generic
RTCInferenceEngine, so ``rtc_on`` is the deployment-supported every-tick
alternative: ``n_action_steps=1`` with ACT temporal ensembling (coefficient
0.01).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rosbags.highlevel import AnyReader


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "Data" / "2026_07_21"
RUN_DIR = REPO_ROOT / "outputs" / "train" / "robot8_20260721_act_resnet18_60k_224x224_crop2of3"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "robot8_20260721_data03_act_base_compare"
EXCLUDED_BAGS = {
    "rosbag2_2026_07_21_15_09_46",
    "rosbag2_2026_07_21_16_11_11",
}
CAMERA_NAMES = ("head_cam", "left_arm_cam", "right_arm_cam")
FPS = 20.0
DT = 1.0 / FPS


def import_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CONVERTER = import_module(
    "robot8_data_converter",
    REPO_ROOT / "scripts" / "data_convert" / "convert_zeno_h1_v30.py",
)
DEPLOY_WORKER = import_module(
    "robot8_deploy_worker",
    REPO_ROOT / "scripts" / "deploy" / "robot8_act_dinov3_base_frozen_100k" / "worker.py",
)


@dataclass
class ReplayEpisode:
    """Raw deployment inputs sampled with bridge-style latest-message caches."""

    episode_index: int
    bag_path: Path
    timestamps_s: np.ndarray
    states: np.ndarray
    demo_actions: np.ndarray
    image_messages: dict[str, list[Any]]
    image_indices: dict[str, np.ndarray]

    def image_bytes(self, frame_index: int) -> dict[str, bytes]:
        return {
            name: bytes(self.image_messages[name][self.image_indices[name][frame_index]].data)
            for name in CAMERA_NAMES
        }


def latest_indices(times: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """Indices of messages held by bridge caches at each virtual timer tick.

    ``AutoCmdBridge`` stores every subscription's most recent callback value
    and its 20 Hz timer reads that cache.  Given a bag replayed in timestamp
    order, this is the last message whose timestamp is no later than the
    timer tick; selecting the offline converter's nearest sample could leak a
    future image or state into the model input.
    """
    indices = np.searchsorted(times, sample_times, side="right") - 1
    return np.clip(indices, 0, len(times) - 1)


def get_topic_message(topic_messages: dict[str, list[tuple[int, Any]]], topic: str, index: int) -> Any:
    return topic_messages[topic][index][1]


def build_episode(data_dir: Path, episode_index: int, max_frames: int | None) -> ReplayEpisode:
    bag_paths = CONVERTER.collect_bag_paths(data_dir, exclude_bags=EXCLUDED_BAGS)
    if not 0 <= episode_index < len(bag_paths):
        raise ValueError(f"episode_index must be in [0, {len(bag_paths) - 1}], got {episode_index}")
    bag_path = bag_paths[episode_index]
    topics = CONVERTER.enabled_topics(list(CAMERA_NAMES))
    topic_set = set(topics)

    print(f"Loading episode_index={episode_index} from {bag_path}", flush=True)
    with AnyReader([bag_path]) as reader:
        topic_messages: dict[str, list[tuple[int, Any]]] = {topic: [] for topic in topics}
        connections = [connection for connection in reader.connections if connection.topic in topic_set]
        for connection, timestamp, raw in reader.messages(connections=connections):
            topic_messages[connection.topic].append(
                (timestamp, reader.deserialize(raw, connection.msgtype))
            )

    missing = [topic for topic, messages in topic_messages.items() if not messages]
    if missing:
        raise RuntimeError(f"Required topics missing in {bag_path}: {missing}")

    times = {
        topic: np.asarray([timestamp for timestamp, _ in messages], dtype=np.int64)
        for topic, messages in topic_messages.items()
    }
    start_ns = max(values[0] for values in times.values())
    end_ns = min(values[-1] for values in times.values())
    step_ns = int(1e9 / FPS)
    n_frames = int((end_ns - start_ns) // step_ns) + 1
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    if n_frames < 2:
        raise RuntimeError(f"Episode is too short after alignment: {n_frames} frame(s)")

    sample_times = start_ns + np.arange(n_frames, dtype=np.int64) * step_ns
    indices = {topic: latest_indices(topic_times, sample_times) for topic, topic_times in times.items()}
    states = np.empty((n_frames, 23), dtype=np.float32)
    demo_actions = np.empty((n_frames, 23), dtype=np.float32)

    for frame_index in range(n_frames):
        torso_state = CONVERTER.extract_named_positions(
            get_topic_message(topic_messages, CONVERTER.STATE_TORSO, indices[CONVERTER.STATE_TORSO][frame_index]),
            CONVERTER.TORSO_FIELD_NAMES,
        )
        torso_action = CONVERTER.extract_named_positions(
            get_topic_message(topic_messages, CONVERTER.ACTION_TORSO, indices[CONVERTER.ACTION_TORSO][frame_index]),
            CONVERTER.TORSO_FIELD_NAMES,
        )
        left_arm_state = CONVERTER.extract_named_positions(
            get_topic_message(topic_messages, CONVERTER.STATE_LEFT_ARM, indices[CONVERTER.STATE_LEFT_ARM][frame_index]),
            CONVERTER.LEFT_ARM_NAMES,
        )
        left_arm_action = CONVERTER.extract_named_positions(
            get_topic_message(topic_messages, CONVERTER.ACTION_LEFT_ARM, indices[CONVERTER.ACTION_LEFT_ARM][frame_index]),
            CONVERTER.LEFT_ARM_NAMES,
        )
        right_arm_state = CONVERTER.extract_named_positions(
            get_topic_message(topic_messages, CONVERTER.STATE_RIGHT_ARM, indices[CONVERTER.STATE_RIGHT_ARM][frame_index]),
            CONVERTER.RIGHT_ARM_NAMES,
        )
        right_arm_action = CONVERTER.extract_named_positions(
            get_topic_message(topic_messages, CONVERTER.ACTION_RIGHT_ARM, indices[CONVERTER.ACTION_RIGHT_ARM][frame_index]),
            CONVERTER.RIGHT_ARM_NAMES,
        )
        left_gripper_state = CONVERTER.extract_named_positions(
            get_topic_message(
                topic_messages,
                CONVERTER.STATE_LEFT_GRIPPER,
                indices[CONVERTER.STATE_LEFT_GRIPPER][frame_index],
            ),
            CONVERTER.LEFT_GRIPPER_NAMES,
        )
        left_gripper_action = CONVERTER.extract_named_positions(
            get_topic_message(
                topic_messages,
                CONVERTER.ACTION_LEFT_GRIPPER,
                indices[CONVERTER.ACTION_LEFT_GRIPPER][frame_index],
            ),
            CONVERTER.LEFT_GRIPPER_NAMES,
        )
        right_gripper_state = CONVERTER.extract_named_positions(
            get_topic_message(
                topic_messages,
                CONVERTER.STATE_RIGHT_GRIPPER,
                indices[CONVERTER.STATE_RIGHT_GRIPPER][frame_index],
            ),
            CONVERTER.RIGHT_GRIPPER_NAMES,
        )
        right_gripper_action = CONVERTER.extract_named_positions(
            get_topic_message(
                topic_messages,
                CONVERTER.ACTION_RIGHT_GRIPPER,
                indices[CONVERTER.ACTION_RIGHT_GRIPPER][frame_index],
            ),
            CONVERTER.RIGHT_GRIPPER_NAMES,
        )
        odom = get_topic_message(topic_messages, CONVERTER.ODOM, indices[CONVERTER.ODOM][frame_index])
        twist_cmd = get_topic_message(
            topic_messages, CONVERTER.TWIST_CMD, indices[CONVERTER.TWIST_CMD][frame_index]
        )
        states[frame_index] = np.concatenate(
            [
                torso_state,
                left_arm_state,
                right_arm_state,
                left_gripper_state,
                right_gripper_state,
                CONVERTER.extract_odom_velocity(odom),
            ]
        )
        demo_actions[frame_index] = np.concatenate(
            [
                torso_action,
                left_arm_action,
                right_arm_action,
                left_gripper_action,
                right_gripper_action,
                CONVERTER.extract_twist_velocity(twist_cmd),
            ]
        )

    return ReplayEpisode(
        episode_index=episode_index,
        bag_path=bag_path,
        timestamps_s=np.arange(n_frames, dtype=np.float64) / FPS,
        states=states,
        demo_actions=demo_actions,
        image_messages={name: [message for _, message in topic_messages[CONVERTER.CAMERA_TOPICS[name]]] for name in CAMERA_NAMES},
        image_indices={
            name: indices[CONVERTER.CAMERA_TOPICS[name]] for name in CAMERA_NAMES
        },
    )


def make_worker(checkpoint: Path, *, temporal_ensemble: bool, device: str | None):
    if temporal_ensemble:
        return DEPLOY_WORKER.ActWorker(
            checkpoint=checkpoint,
            device=device,
            image_size=None,
            center_crop_fraction=2.0 / 3.0,
            use_amp=True,
            clamp_actions=True,
            action_clip_margin=0.05,
            n_action_steps=1,
            temporal_ensemble_coeff=0.01,
        )
    return DEPLOY_WORKER.ActWorker(
        checkpoint=checkpoint,
        device=device,
        image_size=None,
        center_crop_fraction=2.0 / 3.0,
        use_amp=True,
        clamp_actions=True,
        action_clip_margin=0.05,
        n_action_steps=None,
        temporal_ensemble_coeff=None,
    )


def replay_mode(worker: Any, episode: ReplayEpisode) -> tuple[np.ndarray, dict[str, float | int]]:
    worker.policy.reset()
    if hasattr(worker.preprocessor, "reset"):
        worker.preprocessor.reset()
    if hasattr(worker.postprocessor, "reset"):
        worker.postprocessor.reset()

    model_forward_count = 0

    def count_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal model_forward_count
        model_forward_count += 1

    hook = worker.policy.model.register_forward_hook(count_forward)
    actions = np.empty_like(episode.demo_actions)
    per_tick_latency_s = np.empty(len(actions), dtype=np.float64)
    try:
        for frame_index in range(len(actions)):
            start = time.perf_counter()
            actions[frame_index] = worker.select_action(
                episode.states[frame_index], episode.image_bytes(frame_index)
            )
            per_tick_latency_s[frame_index] = time.perf_counter() - start
    finally:
        hook.remove()

    metrics: dict[str, float | int] = {
        "model_forward_count": model_forward_count,
        "tick_latency_mean_ms": float(per_tick_latency_s.mean() * 1e3),
        "tick_latency_p95_ms": float(np.quantile(per_tick_latency_s, 0.95) * 1e3),
        "tick_latency_max_ms": float(per_tick_latency_s.max() * 1e3),
    }
    return actions, metrics


def integrate_holonomic_base(actions: np.ndarray) -> np.ndarray:
    """Integrate body-frame vx, vy, wz at 20 Hz into an XY/yaw path."""
    path = np.zeros((len(actions) + 1, 3), dtype=np.float64)
    for index, (vx, vy, wz) in enumerate(actions[:, -3:]):
        x, y, yaw = path[index]
        path[index + 1, 0] = x + DT * (vx * np.cos(yaw) - vy * np.sin(yaw))
        path[index + 1, 1] = y + DT * (vx * np.sin(yaw) + vy * np.cos(yaw))
        path[index + 1, 2] = yaw + DT * wz
    return path


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())


def write_csv(
    output_path: Path,
    timestamps_s: np.ndarray,
    demo_actions: np.ndarray,
    default_actions: np.ndarray,
    ensemble_actions: np.ndarray,
    demo_path: np.ndarray,
    default_path: np.ndarray,
    ensemble_path: np.ndarray,
) -> None:
    header = [
        "time_s",
        "demo_vx",
        "demo_vy",
        "demo_wz",
        "rtc_off_vx",
        "rtc_off_vy",
        "rtc_off_wz",
        "rtc_on_vx",
        "rtc_on_vy",
        "rtc_on_wz",
        "demo_x",
        "demo_y",
        "demo_yaw",
        "rtc_off_x",
        "rtc_off_y",
        "rtc_off_yaw",
        "rtc_on_x",
        "rtc_on_y",
        "rtc_on_yaw",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, timestamp in enumerate(timestamps_s):
            writer.writerow(
                [
                    f"{timestamp:.3f}",
                    *[f"{value:.8f}" for value in demo_actions[index, -3:]],
                    *[f"{value:.8f}" for value in default_actions[index, -3:]],
                    *[f"{value:.8f}" for value in ensemble_actions[index, -3:]],
                    *[f"{value:.8f}" for value in demo_path[index + 1]],
                    *[f"{value:.8f}" for value in default_path[index + 1]],
                    *[f"{value:.8f}" for value in ensemble_path[index + 1]],
                ]
            )


def make_plot(
    output_path: Path,
    step: int,
    episode: ReplayEpisode,
    default_actions: np.ndarray,
    ensemble_actions: np.ndarray,
    default_metrics: dict[str, float | int],
    ensemble_metrics: dict[str, float | int],
) -> dict[str, float]:
    """Write one self-contained PNG without requiring a plotting package.

    The inference environment intentionally only needs the training/runtime
    dependencies.  Pillow is already present there through the camera stack,
    so draw the four diagnostic panels directly rather than requiring a
    separate matplotlib installation just for this offline analysis.
    """
    demo_path = integrate_holonomic_base(episode.demo_actions)
    default_path = integrate_holonomic_base(default_actions)
    ensemble_path = integrate_holonomic_base(ensemble_actions)
    time_s = episode.timestamps_s
    series = (("vx (m/s)", 20), ("vy (m/s)", 21), ("wz (rad/s)", 22))

    # Palette deliberately mirrors standard chart colors: gray = recorded
    # command, blue = exact default deployment FIFO, orange = per-tick mode.
    background = "white"
    black = "#1b1b1b"
    grid = "#d7dce2"
    demo_color = "#5a5a5a"
    off_color = "#1f77b4"
    on_color = "#ff7f0e"
    image_width, image_height = 1800, 1770
    margin_left, margin_right = 155, 70
    header_height = 185
    panel_height, panel_gap = 245, 40
    path_height = 570
    plot_width = image_width - margin_left - margin_right

    image = Image.new("RGB", (image_width, image_height), background)
    draw = ImageDraw.Draw(image)
    font_candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    font_path = next((Path(path) for path in font_candidates if Path(path).is_file()), None)
    if font_path is not None:
        title_font = ImageFont.truetype(str(font_path), 31)
        subtitle_font = ImageFont.truetype(str(font_path), 20)
        label_font = ImageFont.truetype(str(font_path), 19)
        tick_font = ImageFont.truetype(str(font_path), 16)
    else:  # Pillow's fallback is sufficient for headless analysis machines.
        title_font = subtitle_font = label_font = tick_font = ImageFont.load_default()

    def draw_text(position: tuple[float, float], text: str, *, font: Any, fill: str = black, anchor: str | None = None) -> None:
        draw.text(position, text, font=font, fill=fill, anchor=anchor)

    def draw_legend(entries: list[tuple[str, str]], x: int, y: int) -> None:
        cursor_x = x
        for text, color in entries:
            draw.line((cursor_x, y + 11, cursor_x + 36, y + 11), fill=color, width=4)
            draw_text((cursor_x + 45, y), text, font=tick_font)
            bbox = draw.textbbox((cursor_x + 45, y), text, font=tick_font)
            cursor_x = bbox[2] + 34

    def padded_range(values: np.ndarray) -> tuple[float, float]:
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = hi - lo
        if span < 1e-8:
            pad = max(0.05, abs(lo) * 0.10)
        else:
            pad = max(span * 0.08, 1e-4)
        return lo - pad, hi + pad

    def line_points(
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> list[tuple[int, int]]:
        x_span = max(x_hi - x_lo, 1e-9)
        y_span = max(y_hi - y_lo, 1e-9)
        xs = left + (x_values - x_lo) / x_span * width
        ys = top + (y_hi - y_values) / y_span * height
        return list(zip(np.rint(xs).astype(int), np.rint(ys).astype(int), strict=True))

    draw_text(
        (margin_left, 26),
        f"data{episode.episode_index:02d} (episode_index={episode.episode_index}) — ACT checkpoint {step:06d} — causal bridge-cache input at 20 Hz",
        font=title_font,
    )
    draw_text(
        (margin_left, 73),
        "RTC off = deployed ACT FIFO-100; RTC on* = supported per-tick ACT temporal ensemble (n_action_steps=1, coeff=0.01).",
        font=subtitle_font,
    )
    draw_text(
        (margin_left, 105),
        "* Generic LeRobot RTCInferenceEngine is not compatible with this ACT checkpoint API.  Circles = start; crosses = end.",
        font=subtitle_font,
    )
    draw_text(
        (margin_left, 137),
        f"{len(time_s)} control ticks; model forwards: off={default_metrics['model_forward_count']}, on={ensemble_metrics['model_forward_count']}; "
        f"integrated length (m): demo={path_length(demo_path):.2f}, off={path_length(default_path):.2f}, on={path_length(ensemble_path):.2f}",
        font=subtitle_font,
    )

    legend_entries = [
        ("recorded command", demo_color),
        ("RTC off: ACT FIFO-100", off_color),
        ("RTC on*: temporal ensemble", on_color),
    ]

    for panel_index, (label, index) in enumerate(series):
        top = header_height + panel_index * (panel_height + panel_gap)
        values = np.concatenate(
            (episode.demo_actions[:, index], default_actions[:, index], ensemble_actions[:, index])
        )
        y_lo, y_hi = padded_range(values)
        x_lo, x_hi = float(time_s[0]), float(time_s[-1])
        draw.rectangle((margin_left, top, margin_left + plot_width, top + panel_height), outline=black, width=2)
        for tick_index in range(6):
            ratio = tick_index / 5.0
            y = int(round(top + panel_height * ratio))
            value = y_hi - ratio * (y_hi - y_lo)
            draw.line((margin_left, y, margin_left + plot_width, y), fill=grid, width=1)
            draw_text((margin_left - 14, y), f"{value:.2f}", font=tick_font, anchor="rm")
        for tick_index in range(6):
            ratio = tick_index / 5.0
            x = int(round(margin_left + plot_width * ratio))
            draw.line((x, top, x, top + panel_height), fill=grid, width=1)
            if panel_index == len(series) - 1:
                value = x_lo + ratio * (x_hi - x_lo)
                draw_text((x, top + panel_height + 9), f"{value:.1f}", font=tick_font, anchor="ma")
        draw_text((margin_left - 78, top + panel_height / 2), label, font=label_font, anchor="mm")
        if panel_index == len(series) - 1:
            draw_text((margin_left + plot_width / 2, top + panel_height + 38), "time (s)", font=label_font, anchor="ma")
        for values_to_draw, color, width in (
            (episode.demo_actions[:, index], demo_color, 2),
            (default_actions[:, index], off_color, 2),
            (ensemble_actions[:, index], on_color, 2),
        ):
            draw.line(
                line_points(time_s, values_to_draw, x_lo, x_hi, y_lo, y_hi, margin_left, top, plot_width, panel_height),
                fill=color,
                width=width,
            )
        if panel_index == 0:
            draw_legend(legend_entries, margin_left + 12, top + 12)

    path_top = header_height + 3 * (panel_height + panel_gap) + 45
    path_left = margin_left
    path_width = plot_width
    all_path_points = np.concatenate((demo_path[:, :2], default_path[:, :2], ensemble_path[:, :2]), axis=0)
    x_lo, x_hi = padded_range(all_path_points[:, 0])
    y_lo, y_hi = padded_range(all_path_points[:, 1])
    # Preserve an equal XY scale so the accumulated routes are geometrically
    # meaningful instead of stretched to fit the panel.
    x_span, y_span = x_hi - x_lo, y_hi - y_lo
    panel_aspect = path_width / path_height
    if x_span / y_span > panel_aspect:
        target_y_span = x_span / panel_aspect
        midpoint = (y_lo + y_hi) / 2.0
        y_lo, y_hi = midpoint - target_y_span / 2.0, midpoint + target_y_span / 2.0
    else:
        target_x_span = y_span * panel_aspect
        midpoint = (x_lo + x_hi) / 2.0
        x_lo, x_hi = midpoint - target_x_span / 2.0, midpoint + target_x_span / 2.0
    draw.rectangle((path_left, path_top, path_left + path_width, path_top + path_height), outline=black, width=2)
    for tick_index in range(6):
        ratio = tick_index / 5.0
        x = int(round(path_left + path_width * ratio))
        y = int(round(path_top + path_height * ratio))
        x_value = x_lo + ratio * (x_hi - x_lo)
        y_value = y_hi - ratio * (y_hi - y_lo)
        draw.line((x, path_top, x, path_top + path_height), fill=grid, width=1)
        draw.line((path_left, y, path_left + path_width, y), fill=grid, width=1)
        draw_text((x, path_top + path_height + 9), f"{x_value:.2f}", font=tick_font, anchor="ma")
        draw_text((path_left - 14, y), f"{y_value:.2f}", font=tick_font, anchor="rm")
    draw_text((path_left + path_width / 2, path_top + path_height + 36), "integrated world x (m)", font=label_font, anchor="ma")
    draw_text((path_left - 91, path_top + path_height / 2), "integrated world y (m)", font=label_font, anchor="mm")
    draw_text((path_left + 12, path_top + 12), "Integrated base command path", font=label_font)
    draw_legend(legend_entries, path_left + 370, path_top + 12)
    for path, color in ((demo_path, demo_color), (default_path, off_color), (ensemble_path, on_color)):
        points = line_points(path[:, 0], path[:, 1], x_lo, x_hi, y_lo, y_hi, path_left, path_top, path_width, path_height)
        draw.line(points, fill=color, width=3)
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        radius = 6
        draw.ellipse((start_x - radius, start_y - radius, start_x + radius, start_y + radius), fill=color)
        draw.line((end_x - radius, end_y - radius, end_x + radius, end_y + radius), fill=color, width=3)
        draw.line((end_x - radius, end_y + radius, end_x + radius, end_y - radius), fill=color, width=3)

    image.save(output_path)

    return {
        "demo_path_length_m": path_length(demo_path),
        "rtc_off_path_length_m": path_length(default_path),
        "rtc_on_path_length_m": path_length(ensemble_path),
        "demo_final_x_m": float(demo_path[-1, 0]),
        "demo_final_y_m": float(demo_path[-1, 1]),
        "rtc_off_final_x_m": float(default_path[-1, 0]),
        "rtc_off_final_y_m": float(default_path[-1, 1]),
        "rtc_on_final_x_m": float(ensemble_path[-1, 0]),
        "rtc_on_final_y_m": float(ensemble_path[-1, 1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--episode-index", type=int, default=3, help="data03 is episode index 3 (zero-based)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available")
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test limit; omit for full data03")
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[20000, 40000, 60000],
        help="Checkpoint steps to compare",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")
    np.random.seed(0)
    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode = build_episode(args.data_dir, args.episode_index, args.max_frames)
    summary: dict[str, Any] = {
        "mode_definition": {
            "rtc_off": "Exact deployed ACT default: select_action at 20 Hz with n_action_steps=100 FIFO.",
            "rtc_on": "ACT-supported 20 Hz temporal ensemble: n_action_steps=1, temporal_ensemble_coeff=0.01.",
            "generic_rtc": "Not run: LeRobot RTCInferenceEngine is incompatible with ACT predict_action_chunk API.",
        },
        "episode_index": args.episode_index,
        "bag_path": str(episode.bag_path),
        "input_sampling": (
            "Bridge-style latest cache: for every 20 Hz timer tick, the last "
            "message timestamp at or before that tick is used for each topic."
        ),
        "num_frames": len(episode.timestamps_s),
        "duration_s": float(episode.timestamps_s[-1]),
        "fps": FPS,
        "checkpoints": {},
    }

    for step in args.checkpoint_steps:
        checkpoint = RUN_DIR / "checkpoints" / f"{step:06d}" / "pretrained_model"
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        print(f"Replaying checkpoint {step:06d}", flush=True)
        default_worker = make_worker(checkpoint, temporal_ensemble=False, device=args.device)
        ensemble_worker = make_worker(checkpoint, temporal_ensemble=True, device=args.device)
        try:
            default_actions, default_metrics = replay_mode(default_worker, episode)
            ensemble_actions, ensemble_metrics = replay_mode(ensemble_worker, episode)
        finally:
            del default_worker
            del ensemble_worker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        csv_path = args.output_dir / f"checkpoint_{step:06d}_base_signals.csv"
        demo_path = integrate_holonomic_base(episode.demo_actions)
        default_path = integrate_holonomic_base(default_actions)
        ensemble_path = integrate_holonomic_base(ensemble_actions)
        write_csv(
            csv_path,
            episode.timestamps_s,
            episode.demo_actions,
            default_actions,
            ensemble_actions,
            demo_path,
            default_path,
            ensemble_path,
        )
        plot_path = args.output_dir / f"checkpoint_{step:06d}_base_compare.png"
        path_metrics = make_plot(
            plot_path,
            step,
            episode,
            default_actions,
            ensemble_actions,
            default_metrics,
            ensemble_metrics,
        )
        summary["checkpoints"][f"{step:06d}"] = {
            "checkpoint": str(checkpoint),
            "csv": str(csv_path),
            "plot": str(plot_path),
            "rtc_off": default_metrics,
            "rtc_on": ensemble_metrics,
            **path_metrics,
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
