#!/usr/bin/env python3
"""Read-only preflight checks for Zeno ROS bags before LeRobot conversion.

The converter resamples every required topic at 20 Hz.  This tool scans only
message timestamps (it never decodes images or writes data) to flag missing
camera streams, long inter-frame gaps, and cameras that are conspicuously
slower than the requested conversion frame rate.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from rosbags.highlevel import AnyReader

from convert_zeno_h1_v30 import (
    ACTION_LEFT_ARM,
    ACTION_LEFT_GRIPPER,
    ACTION_RIGHT_ARM,
    ACTION_RIGHT_GRIPPER,
    ACTION_TORSO,
    CAMERA_TOPICS,
    FPS,
    ODOM,
    STATE_LEFT_ARM,
    STATE_LEFT_GRIPPER,
    STATE_RIGHT_ARM,
    STATE_RIGHT_GRIPPER,
    STATE_TORSO,
    TWIST_CMD,
    bag_name,
    collect_bag_paths,
    parse_camera_names,
)


NANOSECONDS_PER_SECOND = 1_000_000_000
CAMERA_COUNT_RATIO_WARNING = 0.70
CAMERA_RATE_WARNING_RATIO = 0.50
CAMERA_RATE_ERROR_RATIO = 0.25
GAP_WARNING_FRAME_MULTIPLIER = 3.0
GAP_ERROR_FRAME_MULTIPLIER = 10.0

TOPIC_LABELS = {
    ODOM: "里程计",
    TWIST_CMD: "底盘控制",
    STATE_TORSO: "躯干状态",
    ACTION_TORSO: "躯干控制",
    STATE_LEFT_ARM: "左臂状态",
    ACTION_LEFT_ARM: "左臂控制",
    STATE_RIGHT_ARM: "右臂状态",
    ACTION_RIGHT_ARM: "右臂控制",
    STATE_LEFT_GRIPPER: "左夹爪状态",
    ACTION_LEFT_GRIPPER: "左夹爪控制",
    STATE_RIGHT_GRIPPER: "右夹爪状态",
    ACTION_RIGHT_GRIPPER: "右夹爪控制",
}


@dataclass
class TopicStats:
    """Timestamp-only statistics for one topic in a bag."""

    count: int = 0
    first_ns: int | None = None
    last_ns: int | None = None
    max_gap_ns: int = 0
    estimated_missing_frames: int = 0

    def add(self, timestamp_ns: int, frame_period_ns: int) -> None:
        if self.last_ns is not None:
            gap_ns = timestamp_ns - self.last_ns
            self.max_gap_ns = max(self.max_gap_ns, gap_ns)
            if gap_ns > frame_period_ns:
                self.estimated_missing_frames += max(0, round(gap_ns / frame_period_ns) - 1)
        if self.first_ns is None:
            self.first_ns = timestamp_ns
        self.last_ns = timestamp_ns
        self.count += 1

    @property
    def duration_s(self) -> float:
        if self.first_ns is None or self.last_ns is None:
            return 0.0
        return max(0.0, (self.last_ns - self.first_ns) / NANOSECONDS_PER_SECOND)

    @property
    def rate_hz(self) -> float | None:
        if self.count < 2 or self.duration_s <= 0.0:
            return None
        return (self.count - 1) / self.duration_s

    @property
    def max_gap_s(self) -> float:
        return self.max_gap_ns / NANOSECONDS_PER_SECOND


def camera_topic_label(camera_name: str) -> str:
    """Return a compact Chinese name suitable for a preflight report."""

    return {"head_cam": "头部相机", "left_arm_cam": "左臂相机", "right_arm_cam": "右臂相机"}[camera_name]


def required_topics(camera_names: list[str]) -> list[str]:
    """Mirror the converter's required observation and action topic contract."""

    return [
        *(CAMERA_TOPICS[name] for name in camera_names),
        ODOM,
        TWIST_CMD,
        STATE_TORSO,
        ACTION_TORSO,
        STATE_LEFT_ARM,
        ACTION_LEFT_ARM,
        STATE_RIGHT_ARM,
        ACTION_RIGHT_ARM,
        STATE_LEFT_GRIPPER,
        ACTION_LEFT_GRIPPER,
        STATE_RIGHT_GRIPPER,
        ACTION_RIGHT_GRIPPER,
    ]


def format_rate(rate_hz: float | None) -> str:
    """Format a rate while preserving the difference between no and low data."""

    return "--" if rate_hz is None else f"{rate_hz:.1f} Hz"


def inspect_bag(
    bag_path: Path,
    camera_names: list[str],
    fps: int,
) -> tuple[list[str], list[str], dict[str, object]]:
    """Scan one bag's required topic timestamps without deserializing payloads."""

    topics = required_topics(camera_names)
    topic_set = set(topics)
    frame_period_ns = int(NANOSECONDS_PER_SECOND / fps)
    warning_gap_s = GAP_WARNING_FRAME_MULTIPLIER / fps
    error_gap_s = GAP_ERROR_FRAME_MULTIPLIER / fps
    errors: list[str] = []
    warnings: list[str] = []
    stats = {topic: TopicStats() for topic in topics}

    try:
        with AnyReader([bag_path]) as reader:
            connections = [connection for connection in reader.connections if connection.topic in topic_set]
            available_topics = {connection.topic for connection in connections}
            for topic in topics:
                if topic not in available_topics:
                    errors.append(f"{bag_path.name}: 缺少 {TOPIC_LABELS.get(topic, topic)} topic")
            if connections:
                for connection, timestamp_ns, _raw in reader.messages(connections=connections):
                    stats[connection.topic].add(timestamp_ns, frame_period_ns)
    except Exception as error:
        return [f"{bag_path.name}: 无法读取 Bag（{error}）"], [], {}

    for topic, topic_stats in stats.items():
        if topic_stats.count == 0 and topic in available_topics:
            errors.append(f"{bag_path.name}: {TOPIC_LABELS.get(topic, topic)} 没有消息")

    populated_stats = [topic_stats for topic_stats in stats.values() if topic_stats.count]
    overlap_start = max((topic_stats.first_ns or 0) for topic_stats in populated_stats) if populated_stats else 0
    overlap_end = min((topic_stats.last_ns or 0) for topic_stats in populated_stats) if populated_stats else 0
    overlap_s = max(0.0, (overlap_end - overlap_start) / NANOSECONDS_PER_SECOND)
    expected_frames = int(math.floor(overlap_s * fps)) + 1 if overlap_end > overlap_start else 0
    if populated_stats and overlap_end <= overlap_start:
        errors.append(f"{bag_path.name}: 所有必需 topic 没有可转换的共同时间段")
    elif overlap_s < 1.0:
        warnings.append(f"{bag_path.name}: 共同时间段只有 {overlap_s:.2f}s（约 {expected_frames} 帧）")

    camera_counts = [stats[CAMERA_TOPICS[name]].count for name in camera_names]
    max_camera_count = max(camera_counts, default=0)
    camera_summary: dict[str, object] = {}
    for camera_name in camera_names:
        topic = CAMERA_TOPICS[camera_name]
        topic_stats = stats[topic]
        label = camera_topic_label(camera_name)
        rate_hz = topic_stats.rate_hz
        count_ratio = topic_stats.count / max_camera_count if max_camera_count else 0.0
        camera_summary[camera_name] = {
            "count": topic_stats.count,
            "rate_hz": rate_hz,
            "max_gap_s": topic_stats.max_gap_s,
            "estimated_missing_frames": topic_stats.estimated_missing_frames,
        }
        if topic_stats.count < 2:
            errors.append(f"{bag_path.name}: {label} 只有 {topic_stats.count} 帧")
            continue
        if rate_hz is not None and rate_hz < fps * CAMERA_RATE_ERROR_RATIO:
            errors.append(f"{bag_path.name}: {label} 帧率过低（{format_rate(rate_hz)}）")
        elif rate_hz is not None and rate_hz < fps * CAMERA_RATE_WARNING_RATIO:
            warnings.append(f"{bag_path.name}: {label} 帧率偏低（{format_rate(rate_hz)}）")
        if topic_stats.max_gap_s >= error_gap_s:
            errors.append(
                f"{bag_path.name}: {label} 最大断流 {topic_stats.max_gap_s:.3f}s "
                f"（约缺 {topic_stats.estimated_missing_frames} 帧）"
            )
        elif topic_stats.max_gap_s >= warning_gap_s:
            warnings.append(
                f"{bag_path.name}: {label} 最大帧间隔 {topic_stats.max_gap_s:.3f}s "
                f"（约缺 {topic_stats.estimated_missing_frames} 帧）"
            )
        if count_ratio < CAMERA_COUNT_RATIO_WARNING:
            warnings.append(
                f"{bag_path.name}: {label} 总帧数仅为最多相机的 {count_ratio:.0%} "
                f"（{topic_stats.count}/{max_camera_count}）"
            )

    result = {
        "bag": str(bag_path),
        "overlap_s": round(overlap_s, 3),
        "expected_frames": expected_frames,
        "cameras": camera_summary,
    }
    return errors, warnings, result


def run_preflight(
    data_dir: Path,
    camera_names: list[str],
    fps: int,
    max_bags: int | None,
) -> tuple[list[str], list[str], dict[str, object], list[dict[str, object]]]:
    """Inspect all selected bags and return a compact structured report."""

    errors: list[str] = []
    warnings: list[str] = []
    bag_reports: list[dict[str, object]] = []
    bag_paths = collect_bag_paths(data_dir, max_bags=max_bags)
    if not bag_paths:
        return [f"未找到可检查的 Bag：{data_dir}"], [], {}, bag_reports

    for index, bag_path in enumerate(bag_paths, start=1):
        print(f"[预检 {index}/{len(bag_paths)}] {bag_path}")
        bag_errors, bag_warnings, bag_report = inspect_bag(bag_path, camera_names, fps)
        errors.extend(bag_errors)
        warnings.extend(bag_warnings)
        if bag_report:
            bag_report["index"] = index
            bag_report["name"] = bag_name(bag_path)
            bag_report["errors"] = bag_errors
            bag_report["warnings"] = bag_warnings
            bag_reports.append(bag_report)
            camera_text = "; ".join(
                f"{camera_topic_label(name)}={details['count']}帧/{format_rate(details['rate_hz'])}/"
                f"最大间隔{details['max_gap_s']:.3f}s"
                for name, details in dict(bag_report["cameras"]).items()
            )
            print(
                f"  共同片段={bag_report['overlap_s']:.2f}s，"
                f"预计 {bag_report['expected_frames']} 个 {fps}Hz 帧；{camera_text}"
            )

    summary = {
        "bags": len(bag_paths),
        "checked_bags": len(bag_reports),
        "fps": fps,
        "estimated_frames": sum(int(report["expected_frames"]) for report in bag_reports),
        "error_count": len(errors),
        "warning_count": len(warnings),
    }
    return errors, warnings, summary, bag_reports


def parse_args() -> argparse.Namespace:
    """Define a small CLI so the TUI can run the preflight in its conda env."""

    parser = argparse.ArgumentParser(description="Read-only Zeno ROS bag frame preflight")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cameras", default="head_cam,left_arm_cam,right_arm_cam")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--max-bags", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Run checks and leave a machine-readable final report for the TUI."""

    args = parse_args()
    if args.fps < 1:
        raise SystemExit("--fps must be positive")
    if args.max_bags is not None and args.max_bags < 1:
        raise SystemExit("--max-bags must be positive")
    errors, warnings, summary, bag_reports = run_preflight(
        args.data_dir,
        parse_camera_names(args.cameras),
        args.fps,
        args.max_bags,
    )
    print(
        "PREFLIGHT_RESULT="
        + json.dumps(
            {"errors": errors, "warnings": warnings, "summary": summary, "bags": bag_reports},
            ensure_ascii=False,
        )
    )
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
