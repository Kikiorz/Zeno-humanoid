#!/usr/bin/env python3
"""Estimate Robot8 bilateral end-effector command-space error from raw EE logs.

This repository does not contain a calibrated Zeno H1 URDF.  The raw bags do,
however, contain the robot-published left/right ``ee_pose`` streams alongside
the corresponding joint states.  This tool learns a *separate empirical FK*
for each arm from non-evaluation bags:

    [torso_lift, torso_waist, arm_j0..arm_j6] -> end-effector xyz

The primary metric fixes the measured torso state from the replay input and
maps each side's recorded/model arm command through the same empirical FK.
This avoids treating the raw torso ``joint_cmd`` coordinate as identical to
the torso ``joint_state`` coordinate (the waist is not).  It is an estimated
*arm command-space* error, not an executed robot trajectory error and not a
substitute for URDF-verified FK.  The held-out state-to-pose error of each
empirical FK is saved alongside every result so the uncertainty is explicit.

The default split holds out the same ten episodes used by the local deployment
replay.  Thus no raw pose samples from the evaluated bags are used to fit the
empirical FK.  The script does not run policy inference; it consumes existing
``*_actions.npz`` replay artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from rosbags.highlevel import AnyReader
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "Data" / "2026_07_21"
DEFAULT_REPLAY_DIR = (
    REPO_ROOT / "outputs" / "analysis" / "robot8_20260721_10episode_all23_local_compare"
)
DEFAULT_EVALUATION_EPISODES = (0, 3, 7, 11, 15, 19, 23, 27, 31, 34)
EXCLUDED_BAGS = {
    "rosbag2_2026_07_21_15_09_46",
    "rosbag2_2026_07_21_16_11_11",
}

TORSO_STATE_TOPIC = "/zeno/h1/wheelarm/torso/joint_state"
LEFT_STATE_TOPIC = "/zeno/h1/wheelarm/left_arm/joint_state"
RIGHT_STATE_TOPIC = "/zeno/h1/wheelarm/right_arm/joint_state"
LEFT_EE_TOPIC = "/zeno/h1/wheelarm/left_arm/ee_pose"
RIGHT_EE_TOPIC = "/zeno/h1/wheelarm/right_arm/ee_pose"
TOPICS = (TORSO_STATE_TOPIC, LEFT_STATE_TOPIC, RIGHT_STATE_TOPIC, LEFT_EE_TOPIC, RIGHT_EE_TOPIC)

TORSO_NAMES = ("torso_lift", "torso_waist")
LEFT_ARM_NAMES = tuple(f"left_arm_j{i}" for i in range(7))
RIGHT_ARM_NAMES = tuple(f"right_arm_j{i}" for i in range(7))
LEFT_ACTION_SLICE = slice(4, 11)
RIGHT_ACTION_SLICE = slice(11, 18)
PREFERRED_MODEL_ORDER = ("normal_090000", "normal_100000", "base3x_100000")
PREFERRED_MODE_ORDER = ("rtc_off", "rtc_on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --replay-dir; writes empirical-FK JSON and CSV there.",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVALUATION_EPISODES),
        help="Replay episode indices held out from empirical-FK calibration.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Use every Nth ~100 Hz EE pose sample to fit/validate empirical FK (default: 5 = ~20 Hz).",
    )
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def bag_name(path: Path) -> str:
    return path.parent.name if path.is_file() else path.name


def collect_bag_paths(data_dir: Path) -> list[Path]:
    """Mirror the converter's valid-bag discovery without importing its CLI module."""
    metadata_dirs = {
        path.parent
        for path in data_dir.rglob("metadata.yaml")
        if path.is_file() and path.stat().st_size > 0
    }
    standalone_mcaps = {
        path for path in data_dir.rglob("*.mcap") if path.parent not in metadata_dirs
    }
    paths = sorted([*metadata_dirs, *standalone_mcaps], key=lambda item: str(item))
    paths = [path for path in paths if bag_name(path) not in EXCLUDED_BAGS]
    if not paths:
        raise FileNotFoundError(f"No usable bags found below {data_dir}")
    return paths


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Return closest timestamp indices for static geometry calibration."""
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(source_times[right] - target_times)
    return np.where(choose_left, left, right)


def joint_positions(message: Any, expected_names: tuple[str, ...]) -> np.ndarray:
    names = [str(name) for name in message.name]
    positions = np.asarray(message.position, dtype=np.float32)
    if names:
        lookup = {name: index for index, name in enumerate(names)}
        missing = [name for name in expected_names if name not in lookup]
        if missing:
            raise ValueError(f"JointState missing {missing}; available names={names}")
        return np.asarray([positions[lookup[name]] for name in expected_names], dtype=np.float32)
    if len(positions) < len(expected_names):
        raise ValueError(
            f"JointState has {len(positions)} positions; expected at least {len(expected_names)}"
        )
    return positions[: len(expected_names)].astype(np.float32, copy=False)


def pose_xyz(message: Any) -> np.ndarray:
    result = np.asarray([message.position.x, message.position.y, message.position.z], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("ee_pose contains non-finite xyz")
    return result


def read_topic_messages(bag_path: Path) -> dict[str, list[tuple[int, Any]]]:
    result: dict[str, list[tuple[int, Any]]] = {topic: [] for topic in TOPICS}
    with AnyReader([bag_path]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in result]
        for connection, timestamp, raw in reader.messages(connections=connections):
            result[connection.topic].append((timestamp, reader.deserialize(raw, connection.msgtype)))
    missing = [topic for topic, messages in result.items() if not messages]
    if missing:
        raise RuntimeError(f"{bag_path}: missing required topics: {missing}")
    return result


def unpack_joint_series(
    messages: list[tuple[int, Any]], expected_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([timestamp for timestamp, _ in messages], dtype=np.int64)
    values = np.stack([joint_positions(message, expected_names) for _, message in messages]).astype(np.float32)
    return times, values


def unpack_pose_series(messages: list[tuple[int, Any]]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([timestamp for timestamp, _ in messages], dtype=np.int64)
    values = np.stack([pose_xyz(message) for _, message in messages]).astype(np.float32)
    return times, values


def arm_samples_from_bag(
    messages: dict[str, list[tuple[int, Any]]], *, side: str, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    torso_t, torso_q = unpack_joint_series(messages[TORSO_STATE_TOPIC], TORSO_NAMES)
    if side == "left":
        arm_t, arm_q = unpack_joint_series(messages[LEFT_STATE_TOPIC], LEFT_ARM_NAMES)
        pose_t, pose = unpack_pose_series(messages[LEFT_EE_TOPIC])
    elif side == "right":
        arm_t, arm_q = unpack_joint_series(messages[RIGHT_STATE_TOPIC], RIGHT_ARM_NAMES)
        pose_t, pose = unpack_pose_series(messages[RIGHT_EE_TOPIC])
    else:
        raise ValueError(f"Unknown side: {side}")

    query_times = pose_t[::stride]
    output_pose = pose[::stride]
    torso_at_pose = torso_q[nearest_indices(torso_t, query_times), :2]
    arm_at_pose = arm_q[nearest_indices(arm_t, query_times)]
    features = np.concatenate((torso_at_pose, arm_at_pose), axis=1).astype(np.float32)
    if features.shape[1] != 9:
        raise AssertionError(f"Expected 9 FK input values, got {features.shape}")
    return features, output_pose


def concat_parts(parts: list[np.ndarray], label: str) -> np.ndarray:
    if not parts:
        raise RuntimeError(f"No samples accumulated for {label}")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


@dataclass
class EmpiricalFK:
    side: str
    model: nn.Module
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    device: torch.device
    train_min: np.ndarray
    train_max: np.ndarray
    train_samples: int
    validation_samples: int
    best_epoch: int
    validation_mse_normalized: float


class CartesianMLP(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def resolve_device(raw_device: str | None) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def train_empirical_fk(
    *,
    side: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    seed: int,
) -> EmpiricalFK:
    if len(train_x) != len(train_y) or len(train_x) < 100:
        raise ValueError(f"Insufficient {side} training samples: {train_x.shape}, {train_y.shape}")
    rng = np.random.default_rng(seed + (0 if side == "left" else 1))
    indices = rng.permutation(len(train_x))
    validation_count = max(512, int(round(len(train_x) * 0.1)))
    validation_count = min(validation_count, len(train_x) - 64)
    train_indices = indices[:-validation_count]
    validation_indices = indices[-validation_count:]
    x_fit = train_x[train_indices]
    y_fit = train_y[train_indices]
    x_mean = x_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.maximum(x_fit.std(axis=0, dtype=np.float64), 1e-5).astype(np.float32)
    y_mean = y_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.maximum(y_fit.std(axis=0, dtype=np.float64), 1e-5).astype(np.float32)

    x_fit_t = torch.from_numpy(normalized(x_fit, x_mean, x_std)).to(device)
    y_fit_t = torch.from_numpy(normalized(y_fit, y_mean, y_std)).to(device)
    x_val_t = torch.from_numpy(normalized(train_x[validation_indices], x_mean, x_std)).to(device)
    y_val_t = torch.from_numpy(normalized(train_y[validation_indices], y_mean, y_std)).to(device)

    model = CartesianMLP(hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = -1
    stale_epochs = 0
    n_fit = len(x_fit_t)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_fit, device=device)
        for start in range(0, n_fit, batch_size):
            selected = permutation[start : start + batch_size]
            prediction = model(x_fit_t[selected])
            loss = torch.nn.functional.mse_loss(prediction, y_fit_t[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(torch.nn.functional.mse_loss(model(x_val_t), y_val_t).item())
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{side} empirical FK did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return EmpiricalFK(
        side=side,
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        train_min=x_fit.min(axis=0).astype(np.float32),
        train_max=x_fit.max(axis=0).astype(np.float32),
        train_samples=len(x_fit),
        validation_samples=len(validation_indices),
        best_epoch=best_epoch,
        validation_mse_normalized=best_loss,
    )


def predict_xyz(fk: EmpiricalFK, features: np.ndarray, batch_size: int = 16384) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != 9:
        raise ValueError(f"Expected (N, 9) features, got {features.shape}")
    outputs: list[np.ndarray] = []
    fk.model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = normalized(features[start : start + batch_size], fk.x_mean, fk.x_std)
            output = fk.model(torch.from_numpy(batch).to(fk.device)).detach().cpu().numpy()
            outputs.append(output * fk.y_std + fk.y_mean)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Metric distribution must be non-empty and finite")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def position_error_metrics(reference_xyz: np.ndarray, predicted_xyz: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(predicted_xyz - reference_xyz, dtype=np.float64)
    distances = np.linalg.norm(delta, axis=1)
    return {
        "distance_m": distribution(distances),
        "component_abs_mae_m": {
            axis: float(np.mean(np.abs(delta[:, index])))
            for index, axis in enumerate(("x", "y", "z"))
        },
    }


def combined_error_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_distances = np.asarray(left.pop("_distances"), dtype=np.float64)
    right_distances = np.asarray(right.pop("_distances"), dtype=np.float64)
    return {
        "left": left,
        "right": right,
        "both_arms_distance_m": distribution(np.concatenate((left_distances, right_distances))),
    }


def metrics_with_hidden_distances(reference_xyz: np.ndarray, predicted_xyz: np.ndarray) -> dict[str, Any]:
    result = position_error_metrics(reference_xyz, predicted_xyz)
    result["_distances"] = np.linalg.norm(predicted_xyz - reference_xyz, axis=1)
    return result


def support_metrics(features: np.ndarray, fk: EmpiricalFK) -> dict[str, float]:
    below = features < fk.train_min
    above = features > fk.train_max
    outside = np.logical_or(below, above)
    feature_span = np.maximum(fk.train_max - fk.train_min, 1e-6)
    lower_excess = np.maximum(fk.train_min - features, 0.0) / feature_span
    upper_excess = np.maximum(features - fk.train_max, 0.0) / feature_span
    relative_excess = np.maximum(lower_excess, upper_excess)
    return {
        "row_any_outside_fraction": float(np.mean(np.any(outside, axis=1))),
        "element_outside_fraction": float(np.mean(outside)),
        "relative_range_excess_p95": float(np.percentile(relative_excess, 95)),
        "relative_range_excess_max": float(np.max(relative_excess)),
    }


def features_from_actions(actions: np.ndarray, side: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 23:
        raise ValueError(f"Expected (N, 23) action array, got {actions.shape}")
    arm_slice = LEFT_ACTION_SLICE if side == "left" else RIGHT_ACTION_SLICE
    return np.concatenate((actions[:, :2], actions[:, arm_slice]), axis=1).astype(np.float32)


def error_for_side(
    *,
    fk: EmpiricalFK,
    demo_actions: np.ndarray,
    predicted_actions: np.ndarray,
    torso_reference_state: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    demo_features = features_from_actions(demo_actions, fk.side)
    predicted_features = features_from_actions(predicted_actions, fk.side)
    if torso_reference_state is not None:
        torso_reference_state = np.asarray(torso_reference_state, dtype=np.float32)
        if torso_reference_state.shape != (len(demo_features), 2):
            raise ValueError(
                f"Expected torso reference state {(len(demo_features), 2)}, got {torso_reference_state.shape}"
            )
        demo_features = demo_features.copy()
        predicted_features = predicted_features.copy()
        demo_features[:, :2] = torso_reference_state
        predicted_features[:, :2] = torso_reference_state
    reference_xyz = predict_xyz(fk, demo_features)
    predicted_xyz = predict_xyz(fk, predicted_features)
    return metrics_with_hidden_distances(reference_xyz, predicted_xyz), support_metrics(predicted_features, fk)


def ordered_names(values: Iterable[str], preferred: tuple[str, ...]) -> list[str]:
    ranks = {name: rank for rank, name in enumerate(preferred)}
    return sorted(values, key=lambda value: (ranks.get(value, len(ranks)), value))


def evaluate_replays(
    *,
    replay_dir: Path,
    episode_indices: list[int],
    fks: dict[str, EmpiricalFK],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        episode_dir = replay_dir / f"episode_{episode_index:02d}"
        summary_path = episode_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing replay summary for episode {episode_index}: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        models = summary.get("models")
        if not isinstance(models, dict):
            raise ValueError(f"{summary_path}: missing object-valued models")
        for model_name in ordered_names(models.keys(), PREFERRED_MODEL_ORDER):
            npz_path = episode_dir / f"{model_name}_actions.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing actions for {episode_index}/{model_name}: {npz_path}")
            with np.load(npz_path, allow_pickle=False) as payload:
                states = np.asarray(payload["states"], dtype=np.float32)
                demo_actions = np.asarray(payload["demo_actions"], dtype=np.float32)
                action_by_mode = {
                    "rtc_off": np.asarray(payload["rtc_off_actions"], dtype=np.float32),
                    "rtc_on": np.asarray(payload["rtc_on_actions"], dtype=np.float32),
                }
            for mode_name in PREFERRED_MODE_ORDER:
                predicted_actions = action_by_mode[mode_name]
                if predicted_actions.shape != demo_actions.shape:
                    raise ValueError(
                        f"{npz_path}: {mode_name} action shape {predicted_actions.shape} != demo {demo_actions.shape}"
                    )
                if states.shape != demo_actions.shape:
                    raise ValueError(f"{npz_path}: state shape {states.shape} != demo {demo_actions.shape}")
                full_left, full_left_support = error_for_side(
                    fk=fks["left"],
                    demo_actions=demo_actions,
                    predicted_actions=predicted_actions,
                )
                full_right, full_right_support = error_for_side(
                    fk=fks["right"],
                    demo_actions=demo_actions,
                    predicted_actions=predicted_actions,
                )
                arm_left, arm_left_support = error_for_side(
                    fk=fks["left"],
                    demo_actions=demo_actions,
                    predicted_actions=predicted_actions,
                    torso_reference_state=states[:, :2],
                )
                arm_right, arm_right_support = error_for_side(
                    fk=fks["right"],
                    demo_actions=demo_actions,
                    predicted_actions=predicted_actions,
                    torso_reference_state=states[:, :2],
                )
                records.append(
                    {
                        "episode_index": episode_index,
                        "num_frames": int(len(demo_actions)),
                        "model": model_name,
                        "mode": mode_name,
                        "full_upper_command": combined_error_metrics(full_left, full_right),
                        "arm_only_current_state_torso": combined_error_metrics(arm_left, arm_right),
                        "input_support": {
                            "demo_command": {
                                "left": support_metrics(features_from_actions(demo_actions, "left"), fks["left"]),
                                "right": support_metrics(features_from_actions(demo_actions, "right"), fks["right"]),
                            },
                            "full_upper_command": {"left": full_left_support, "right": full_right_support},
                            "arm_only_current_state_torso": {"left": arm_left_support, "right": arm_right_support},
                        },
                    }
                )
    return records


def macro_statistics(records: list[dict[str, Any]], target: str) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["mode"])].append(record)
    output: dict[str, dict[str, Any]] = {}
    for model_name in ordered_names({key[0] for key in grouped}, PREFERRED_MODEL_ORDER):
        output[model_name] = {}
        for mode_name in ordered_names(
            {key[1] for key in grouped if key[0] == model_name}, PREFERRED_MODE_ORDER
        ):
            rows = grouped[(model_name, mode_name)]
            arms: dict[str, Any] = {}
            for arm_name in ("left", "right", "both_arms"):
                if arm_name == "both_arms":
                    path = [row[target]["both_arms_distance_m"] for row in rows]
                else:
                    path = [row[target][arm_name]["distance_m"] for row in rows]
                arms[arm_name] = {
                    "per_episode_mean_m": distribution(
                        np.asarray([entry["mean"] for entry in path], dtype=np.float64)
                    ),
                    "per_episode_p95_m": distribution(
                        np.asarray([entry["p95"] for entry in path], dtype=np.float64)
                    ),
                }
            output[model_name][mode_name] = {"n_episodes": len(rows), "arms": arms}
    return output


def pairwise_wins(records: list[dict[str, Any]], target: str) -> dict[str, Any]:
    left_model = "base3x_100000"
    right_model = "normal_100000"
    output: dict[str, Any] = {}
    for mode_name in PREFERRED_MODE_ORDER:
        left = {
            row["episode_index"]: row
            for row in records
            if row["model"] == left_model and row["mode"] == mode_name
        }
        right = {
            row["episode_index"]: row
            for row in records
            if row["model"] == right_model and row["mode"] == mode_name
        }
        pairs = [(left[index], right[index]) for index in sorted(left.keys() & right.keys())]
        per_episode = []
        for left_row, right_row in pairs:
            left_value = left_row[target]["both_arms_distance_m"]["mean"]
            right_value = right_row[target]["both_arms_distance_m"]["mean"]
            per_episode.append(
                {
                    "episode_index": left_row["episode_index"],
                    "base3x_100000_mean_m": left_value,
                    "normal_100000_mean_m": right_value,
                    "difference_m": left_value - right_value,
                }
            )
        deltas = np.asarray([row["difference_m"] for row in per_episode], dtype=np.float64)
        output[mode_name] = {
            "paired_episodes": len(per_episode),
            "base3x_100000_wins": int(np.sum(deltas < 0.0)),
            "normal_100000_wins": int(np.sum(deltas > 0.0)),
            "ties": int(np.sum(deltas == 0.0)),
            "base3x_minus_normal_mean_m": distribution(deltas) if len(deltas) else None,
            "per_episode": per_episode,
        }
    return output


CSV_FIELDS = (
    "episode_index",
    "num_frames",
    "model",
    "mode",
    "full_left_mean_m",
    "full_left_p95_m",
    "full_right_mean_m",
    "full_right_p95_m",
    "full_both_mean_m",
    "full_both_p95_m",
    "arm_only_left_mean_m",
    "arm_only_left_p95_m",
    "arm_only_right_mean_m",
    "arm_only_right_p95_m",
    "arm_only_both_mean_m",
    "arm_only_both_p95_m",
    "full_left_row_any_outside_fraction",
    "full_right_row_any_outside_fraction",
)


def csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        full = row["full_upper_command"]
        arm_only = row["arm_only_current_state_torso"]
        result.append(
            {
                "episode_index": row["episode_index"],
                "num_frames": row["num_frames"],
                "model": row["model"],
                "mode": row["mode"],
                "full_left_mean_m": full["left"]["distance_m"]["mean"],
                "full_left_p95_m": full["left"]["distance_m"]["p95"],
                "full_right_mean_m": full["right"]["distance_m"]["mean"],
                "full_right_p95_m": full["right"]["distance_m"]["p95"],
                "full_both_mean_m": full["both_arms_distance_m"]["mean"],
                "full_both_p95_m": full["both_arms_distance_m"]["p95"],
                "arm_only_left_mean_m": arm_only["left"]["distance_m"]["mean"],
                "arm_only_left_p95_m": arm_only["left"]["distance_m"]["p95"],
                "arm_only_right_mean_m": arm_only["right"]["distance_m"]["mean"],
                "arm_only_right_p95_m": arm_only["right"]["distance_m"]["p95"],
                "arm_only_both_mean_m": arm_only["both_arms_distance_m"]["mean"],
                "arm_only_both_p95_m": arm_only["both_arms_distance_m"]["p95"],
                "full_left_row_any_outside_fraction": row["input_support"]["full_upper_command"]["left"][
                    "row_any_outside_fraction"
                ],
                "full_right_row_any_outside_fraction": row["input_support"]["full_upper_command"]["right"][
                    "row_any_outside_fraction"
                ],
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def calibration_report(fk: EmpiricalFK, heldout_x: np.ndarray, heldout_y: np.ndarray) -> dict[str, Any]:
    prediction = predict_xyz(fk, heldout_x)
    metrics = position_error_metrics(heldout_y, prediction)
    return {
        "train_samples": fk.train_samples,
        "internal_validation_samples": fk.validation_samples,
        "best_epoch": fk.best_epoch,
        "internal_validation_mse_normalized": fk.validation_mse_normalized,
        "heldout_raw_bag_state_to_ee_pose": metrics,
        "heldout_input_support": support_metrics(heldout_x, fk),
    }


def print_macro_summary(result: dict[str, Any]) -> None:
    print("Empirical-FK held-out raw state -> ee_pose calibration:")
    for side in ("left", "right"):
        distance = result["empirical_fk_calibration"][side]["heldout_raw_bag_state_to_ee_pose"]["distance_m"]
        print(f"  {side}: mean={distance['mean'] * 100:.2f} cm, p95={distance['p95'] * 100:.2f} cm")
    print("Estimated arm-only EE error at the measured torso state (both arms, macro mean across episodes):")
    for model_name, modes in result["macro_arm_only_current_state_torso"].items():
        for mode_name, info in modes.items():
            value = info["arms"]["both_arms"]["per_episode_mean_m"]
            print(
                f"  {model_name}/{mode_name}: mean={value['mean'] * 100:.2f} cm, "
                f"median={value['median'] * 100:.2f} cm, p75={value['p75'] * 100:.2f} cm"
            )


def main() -> None:
    args = parse_args()
    if args.sample_stride <= 0:
        raise SystemExit("--sample-stride must be positive")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise SystemExit("--epochs, --patience, --batch-size, and --hidden-dim must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    episode_indices = sorted(set(args.episode_indices))
    data_dir = args.data_dir.expanduser().resolve()
    replay_dir = args.replay_dir.expanduser().resolve()
    output_dir = (args.output_dir or args.replay_dir).expanduser().resolve()
    if not replay_dir.is_dir():
        raise SystemExit(f"--replay-dir does not exist: {replay_dir}")
    bag_paths = collect_bag_paths(data_dir)
    if min(episode_indices) < 0 or max(episode_indices) >= len(bag_paths):
        raise SystemExit(f"episode indices must lie in [0, {len(bag_paths) - 1}], got {episode_indices}")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Discovered {len(bag_paths)} valid bags; holding out episodes {episode_indices}")
    print(f"Training empirical FK on {device} using every {args.sample_stride}th EE pose sample")
    train_x: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    train_y: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    heldout_x: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    heldout_y: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    heldout_set = set(episode_indices)
    for index, bag_path in enumerate(bag_paths):
        messages = read_topic_messages(bag_path)
        target_x, target_y = (heldout_x, heldout_y) if index in heldout_set else (train_x, train_y)
        for side in ("left", "right"):
            features, positions = arm_samples_from_bag(messages, side=side, stride=args.sample_stride)
            target_x[side].append(features)
            target_y[side].append(positions)
        print(f"  {'holdout' if index in heldout_set else 'train  '} episode {index:02d}: {bag_name(bag_path)}", flush=True)

    train_features = {side: concat_parts(train_x[side], f"{side} train features") for side in ("left", "right")}
    train_positions = {side: concat_parts(train_y[side], f"{side} train positions") for side in ("left", "right")}
    heldout_features = {side: concat_parts(heldout_x[side], f"{side} heldout features") for side in ("left", "right")}
    heldout_positions = {side: concat_parts(heldout_y[side], f"{side} heldout positions") for side in ("left", "right")}
    fks = {
        side: train_empirical_fk(
            side=side,
            train_x=train_features[side],
            train_y=train_positions[side],
            device=device,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        for side in ("left", "right")
    }
    calibration = {
        side: calibration_report(fks[side], heldout_features[side], heldout_positions[side])
        for side in ("left", "right")
    }
    records = evaluate_replays(replay_dir=replay_dir, episode_indices=episode_indices, fks=fks)
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "name": "empirical_fk_mlp",
            "description": (
                "Separate learned mappings from torso_lift/torso_waist plus side-specific 7-D joint state "
                "to raw bag EE xyz. The primary arm-only metric fixes measured torso state and compares "
                "side-arm commands. It is estimated command-space Cartesian error, not URDF-verified FK "
                "or actual post-command robot motion."
            ),
            "raw_ee_pose_topics": {"left": LEFT_EE_TOPIC, "right": RIGHT_EE_TOPIC},
            "coordinates": "The geometry_msgs/Pose topics have no frame_id; metrics are treated as publisher-local/body-relative coordinates.",
            "action_mapping": {
                "torso": [0, 1],
                "left_arm": [4, 10],
                "right_arm": [11, 17],
            },
            "full_upper_command": "Diagnostic only: raw torso joint_cmd values are passed directly to a joint_state-calibrated empirical FK. Do not use this as the primary metric until a torso command-to-state calibration or URDF is available.",
            "arm_only_current_state_torso": "Uses the replay input's measured torso state for both reference and prediction, then substitutes only the model side-arm command. This avoids treating torso joint_cmd coordinates as joint_state coordinates and isolates arm-joint contribution.",
        },
        "calibration_split": {
            "valid_bag_count": len(bag_paths),
            "heldout_replay_episode_indices": episode_indices,
            "training_bag_indices": [index for index in range(len(bag_paths)) if index not in heldout_set],
            "sample_stride_at_ee_pose_rate": args.sample_stride,
            "device": str(device),
            "seed": args.seed,
            "model": {"hidden_dim": args.hidden_dim, "max_epochs": args.epochs, "patience": args.patience},
        },
        "empirical_fk_calibration": calibration,
        "metric_definition": {
            "distance_m": "Euclidean distance between empirical-FK(model command) and empirical-FK(recorded demo command).",
            "macro_statistics": "Every replay episode has equal weight.",
        },
        "per_episode": records,
        "macro_full_upper_command": macro_statistics(records, "full_upper_command"),
        "macro_arm_only_current_state_torso": macro_statistics(records, "arm_only_current_state_torso"),
        "pairwise_full_upper_base3x_vs_normal100": pairwise_wins(records, "full_upper_command"),
        "pairwise_arm_only_base3x_vs_normal100": pairwise_wins(records, "arm_only_current_state_torso"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "upper_ee_empirical_fk.json"
    csv_path = output_dir / "upper_ee_empirical_fk_per_episode.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, csv_rows(records))
    print_macro_summary(result)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
