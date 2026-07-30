#!/usr/bin/env python3
"""Aggregate multi-episode Robot8 all-23D deployment replay results.

The input directory is expected to contain one subdirectory per replayed
episode, for example::

    outputs/analysis/robot8_20260721_10episode_all23_local_compare/
      episode_00/summary.json
      episode_00/normal_100000_actions.npz
      episode_00/base3x_100000_actions.npz

``summary.json`` supplies the action-error and latency measurements.  The
corresponding ``*_actions.npz`` files supply the dead-reckoned x/y paths used
to compute endpoint error, path-length error, and ADE.  All aggregate values
are macro statistics: every complete episode contributes one value, regardless
of its number of 20 Hz frames.

The script writes ``aggregate.json`` and ``aggregate_per_episode.csv``.  It is
safe to run after a partial batch: incomplete episode/model entries are skipped
and reported in the JSON ``warnings`` field.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "robot8_20260721_10episode_all23_local_compare"
)

# The source replay emits these modes.  Unknown model names are still included
# (after these known checkpoints) so another checkpoint can be compared later.
PREFERRED_MODEL_ORDER = ("normal_090000", "normal_100000", "base3x_100000")
PREFERRED_MODE_ORDER = ("rtc_off", "rtc_on")
METRICS = (
    "base_mae",
    "full_action_mae",
    "path_endpoint_error_m",
    "path_length_abs_error_m",
    "ade_m",
    "model_forward_count",
    "tick_latency_p95_ms",
)
CSV_FIELDS = (
    "episode_id",
    "episode_index",
    "num_frames",
    "duration_s",
    "model",
    "mode",
    *METRICS,
)
TIE_ABS_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Batch root containing episode_*/summary.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregate.json and aggregate_per_episode.csv; defaults to --input-dir",
    )
    return parser.parse_args()


def episode_sort_key(summary_path: Path) -> tuple[int, int, str]:
    """Sort usual episode_03 directories numerically, then safely fall back."""
    episode_id = summary_path.parent.name
    if episode_id.startswith("episode_"):
        suffix = episode_id[len("episode_") :]
        try:
            return (0, int(suffix), episode_id)
        except ValueError:
            pass
    return (1, 0, episode_id)


def ordered_names(names: set[str] | list[str] | tuple[str, ...], preferred: tuple[str, ...]) -> list[str]:
    rank = {name: index for index, name in enumerate(preferred)}
    return sorted(names, key=lambda name: (rank.get(name, len(rank)), name))


def require_finite_float(mapping: dict[str, Any], key: str, context: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or non-numeric {key} in {context}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} in {context}: {value!r}")
    return value


def optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def find_summary_paths(input_dir: Path) -> list[Path]:
    """Find only summaries directly below episode_* directories in the batch."""
    paths = [path for path in input_dir.glob("episode_*/summary.json") if path.is_file()]
    return sorted(paths, key=episode_sort_key)


def resolve_actions_npz(
    summary_path: Path,
    input_dir: Path,
    model_name: str,
    model_summary: dict[str, Any],
) -> Path:
    """Resolve a model's NPZ even if summary.json saved a relative path."""
    candidates = [summary_path.parent / f"{model_name}_actions.npz"]
    declared_path = model_summary.get("all_actions_npz")
    if isinstance(declared_path, str) and declared_path:
        declared = Path(declared_path).expanduser()
        candidates.extend(
            (
                declared,
                Path.cwd() / declared,
                input_dir / declared,
                summary_path.parent / declared,
                summary_path.parent / declared.name,
            )
        )

    seen: set[Path] = set()
    for candidate in candidates:
        # Avoid duplicate probes while retaining relative paths that may be
        # meaningful when the script is launched from a different directory.
        normalized = candidate.absolute()
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {model_name}_actions.npz next to {summary_path} or at the declared path"
    )


def require_xy_path(array: np.ndarray, name: str, npz_path: Path) -> np.ndarray:
    path = np.asarray(array, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2:
        raise ValueError(
            f"{name} in {npz_path} must have shape (at least 2, at least 2), got {path.shape}"
        )
    xy = path[:, :2]
    if not np.all(np.isfinite(xy)):
        raise ValueError(f"{name} in {npz_path} contains non-finite x/y values")
    return xy


def path_length_m(path_xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())


def path_error_metrics(predicted_xy: np.ndarray, demo_xy: np.ndarray, *, context: str) -> dict[str, float]:
    if predicted_xy.shape != demo_xy.shape:
        raise ValueError(
            f"Path shape mismatch for {context}: predicted {predicted_xy.shape}, demo {demo_xy.shape}"
        )
    point_errors = np.linalg.norm(predicted_xy - demo_xy, axis=1)
    return {
        "path_endpoint_error_m": float(point_errors[-1]),
        "path_length_abs_error_m": float(abs(path_length_m(predicted_xy) - path_length_m(demo_xy))),
        "ade_m": float(point_errors.mean()),
    }


def load_path_metrics(npz_path: Path) -> dict[str, dict[str, float]]:
    """Calculate x/y path metrics for both deployment modes from an NPZ."""
    required = ("demo_path", "rtc_off_path", "rtc_on_path")
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in required}
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"Unable to load required paths from {npz_path}: {error}") from error

    demo_xy = require_xy_path(arrays["demo_path"], "demo_path", npz_path)
    return {
        "rtc_off": path_error_metrics(
            require_xy_path(arrays["rtc_off_path"], "rtc_off_path", npz_path),
            demo_xy,
            context=f"rtc_off in {npz_path}",
        ),
        "rtc_on": path_error_metrics(
            require_xy_path(arrays["rtc_on_path"], "rtc_on_path", npz_path),
            demo_xy,
            context=f"rtc_on in {npz_path}",
        ),
    }


def load_records(input_dir: Path) -> tuple[list[dict[str, Any]], list[str], int]:
    """Read complete per-episode/model/mode records and retain skip reasons."""
    summary_paths = find_summary_paths(input_dir)
    if not summary_paths:
        raise FileNotFoundError(f"No episode_*/summary.json files found under {input_dir}")

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for summary_path in summary_paths:
        episode_id = summary_path.parent.name
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"Skipped {episode_id}: unable to read {summary_path}: {error}")
            continue
        if not isinstance(summary, dict):
            warnings.append(f"Skipped {episode_id}: {summary_path} does not contain a JSON object")
            continue

        models = summary.get("models")
        if not isinstance(models, dict):
            warnings.append(f"Skipped {episode_id}: no object-valued models field in {summary_path}")
            continue

        episode_index = optional_int(summary.get("episode_index"))
        num_frames = optional_int(summary.get("num_frames"))
        duration_s = optional_finite_float(summary.get("duration_s"))

        for model_name in ordered_names(set(models), PREFERRED_MODEL_ORDER):
            model_summary = models[model_name]
            if not isinstance(model_summary, dict):
                warnings.append(f"Skipped {episode_id}/{model_name}: model summary is not an object")
                continue
            try:
                npz_path = resolve_actions_npz(summary_path, input_dir, model_name, model_summary)
                mode_path_metrics = load_path_metrics(npz_path)
            except (FileNotFoundError, ValueError) as error:
                warnings.append(f"Skipped {episode_id}/{model_name}: {error}")
                continue

            for mode_name in PREFERRED_MODE_ORDER:
                mode_summary = model_summary.get(mode_name)
                if not isinstance(mode_summary, dict):
                    warnings.append(
                        f"Skipped {episode_id}/{model_name}/{mode_name}: no object-valued mode metrics"
                    )
                    continue
                try:
                    records.append(
                        {
                            "episode_id": episode_id,
                            "episode_index": episode_index,
                            "num_frames": num_frames,
                            "duration_s": duration_s,
                            "model": model_name,
                            "mode": mode_name,
                            "base_mae": require_finite_float(
                                mode_summary, "base_mae", f"{episode_id}/{model_name}/{mode_name}"
                            ),
                            "full_action_mae": require_finite_float(
                                mode_summary,
                                "full_action_mae",
                                f"{episode_id}/{model_name}/{mode_name}",
                            ),
                            **mode_path_metrics[mode_name],
                            "model_forward_count": require_finite_float(
                                mode_summary,
                                "model_forward_count",
                                f"{episode_id}/{model_name}/{mode_name}",
                            ),
                            "tick_latency_p95_ms": require_finite_float(
                                mode_summary,
                                "tick_latency_p95_ms",
                                f"{episode_id}/{model_name}/{mode_name}",
                            ),
                        }
                    )
                except ValueError as error:
                    warnings.append(f"Skipped {episode_id}/{model_name}/{mode_name}: {error}")

    model_rank = {name: index for index, name in enumerate(PREFERRED_MODEL_ORDER)}
    mode_rank = {name: index for index, name in enumerate(PREFERRED_MODE_ORDER)}
    records.sort(
        key=lambda record: (
            1 if record["episode_index"] is None else 0,
            record["episode_index"] if record["episode_index"] is not None else 0,
            record["episode_id"],
            model_rank.get(record["model"], len(model_rank)),
            record["model"],
            mode_rank.get(record["mode"], len(mode_rank)),
            record["mode"],
        )
    )
    return records, warnings, len(summary_paths)


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty distribution")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def macro_statistics(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["mode"])].append(record)

    result: dict[str, dict[str, dict[str, Any]]] = {}
    model_names = ordered_names({model for model, _ in grouped}, PREFERRED_MODEL_ORDER)
    for model_name in model_names:
        result[model_name] = {}
        mode_names = ordered_names(
            {mode for model, mode in grouped if model == model_name}, PREFERRED_MODE_ORDER
        )
        for mode_name in mode_names:
            group = grouped[(model_name, mode_name)]
            result[model_name][mode_name] = {
                "n_episodes": len(group),
                "metrics": {
                    metric: distribution([float(record[metric]) for record in group])
                    for metric in METRICS
                },
            }
    return result


def record_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    episode_index = record["episode_index"]
    return (
        1 if episode_index is None else 0,
        episode_index if episode_index is not None else 0,
        record["episode_id"],
    )


def make_pairwise_result(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    """Count lower-is-better wins plus paired per-episode metric deltas."""
    pairs = sorted(pairs, key=lambda pair: record_sort_key(pair[0]))
    metric_results: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        deltas = [float(left[metric]) - float(right[metric]) for left, right in pairs]
        left_wins = sum(delta < -TIE_ABS_TOL for delta in deltas)
        right_wins = sum(delta > TIE_ABS_TOL for delta in deltas)
        metric_results[metric] = {
            f"{left_label}_wins": left_wins,
            f"{right_label}_wins": right_wins,
            "ties": len(deltas) - left_wins - right_wins,
            f"{left_label}_minus_{right_label}": distribution(deltas) if deltas else None,
        }

    per_episode = []
    for left, right in pairs:
        per_episode.append(
            {
                "episode_id": left["episode_id"],
                "episode_index": left["episode_index"],
                left_label: {metric: left[metric] for metric in METRICS},
                right_label: {metric: right[metric] for metric in METRICS},
                f"{left_label}_minus_{right_label}": {
                    metric: float(left[metric]) - float(right[metric]) for metric in METRICS
                },
            }
        )
    return {
        "paired_episodes": len(pairs),
        "lower_is_better_for_all_metrics": True,
        "metrics": metric_results,
        "per_episode": per_episode,
    }


def compare_models(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the requested weighted and ordinary 100k checkpoints per mode."""
    left_model = "base3x_100000"
    right_model = "normal_100000"
    available_modes = ordered_names({record["mode"] for record in records}, PREFERRED_MODE_ORDER)
    by_mode: dict[str, dict[str, Any]] = {}
    for mode_name in available_modes:
        left_by_episode = {
            record["episode_id"]: record
            for record in records
            if record["model"] == left_model and record["mode"] == mode_name
        }
        right_by_episode = {
            record["episode_id"]: record
            for record in records
            if record["model"] == right_model and record["mode"] == mode_name
        }
        pairs = [
            (left_by_episode[episode_id], right_by_episode[episode_id])
            for episode_id in left_by_episode.keys() & right_by_episode.keys()
        ]
        by_mode[mode_name] = make_pairwise_result(
            pairs, left_label=left_model, right_label=right_model
        )
    return {
        "comparison": "base3x_100000 versus normal_100000, matched within episode and mode",
        "left_model": left_model,
        "right_model": right_model,
        "by_mode": by_mode,
    }


def compare_temporal_vs_fifo(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare RTC temporal ensemble against deployed FIFO separately per model."""
    by_model: dict[str, dict[str, Any]] = {}
    model_names = ordered_names({record["model"] for record in records}, PREFERRED_MODEL_ORDER)
    for model_name in model_names:
        temporal_by_episode = {
            record["episode_id"]: record
            for record in records
            if record["model"] == model_name and record["mode"] == "rtc_on"
        }
        fifo_by_episode = {
            record["episode_id"]: record
            for record in records
            if record["model"] == model_name and record["mode"] == "rtc_off"
        }
        pairs = [
            (temporal_by_episode[episode_id], fifo_by_episode[episode_id])
            for episode_id in temporal_by_episode.keys() & fifo_by_episode.keys()
        ]
        by_model[model_name] = make_pairwise_result(
            pairs, left_label="temporal", right_label="fifo"
        )
    return {
        "comparison": "rtc_on temporal ensemble versus rtc_off deployed FIFO, matched within episode and model",
        "left_mode": "rtc_on",
        "right_mode": "rtc_off",
        "by_model": by_model,
    }


def csv_text(records: list[dict[str, Any]]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})
    return buffer.getvalue()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir does not exist or is not a directory: {input_dir}")

    records, warnings, summary_count = load_records(input_dir)
    if not records:
        raise SystemExit(
            "No complete episode/model/mode records were found; do not aggregate a batch that is still "
            "writing files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "aggregate.json"
    csv_path = output_dir / "aggregate_per_episode.csv"
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "aggregation": {
            "type": "macro",
            "definition": "Each complete episode contributes one value to every model/mode statistic.",
            "position_metrics": "Endpoint error, path-length absolute error, and ADE use x/y only from actions.npz paths.",
            "mode_labels": {
                "rtc_off": "ACT deployed default: 100-step FIFO action chunks at 20 Hz.",
                "rtc_on": "ACT per-tick temporal ensemble.",
            },
        },
        "metric_units": {
            "base_mae": "normalized action units",
            "full_action_mae": "normalized action units",
            "path_endpoint_error_m": "m",
            "path_length_abs_error_m": "m",
            "ade_m": "m",
            "model_forward_count": "forwards per episode",
            "tick_latency_p95_ms": "ms",
        },
        "episodes": {
            "summary_files_discovered": summary_count,
            "episodes_with_complete_records": len({record["episode_id"] for record in records}),
            "record_count": len(records),
        },
        "macro_by_model_and_mode": macro_statistics(records),
        "per_episode": records,
        "pairwise_wins": {
            "base3x_100000_vs_normal_100000": compare_models(records),
            "temporal_vs_fifo": compare_temporal_vs_fifo(records),
        },
        "warnings": warnings,
    }
    aggregate_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path.write_text(csv_text(records), encoding="utf-8")

    print(f"Wrote {aggregate_path}")
    print(f"Wrote {csv_path}")
    print(
        f"Aggregated {len(records)} model/mode records from "
        f"{len({record['episode_id'] for record in records})} complete episodes."
    )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
