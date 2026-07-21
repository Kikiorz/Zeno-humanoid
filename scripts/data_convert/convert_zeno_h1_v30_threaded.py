#!/usr/bin/env python3
"""Convert Zeno H1 ROS2 bags to LeRobot V3.0 with parallel frame building.

This is an additive, thread-pool based variant of ``convert_zeno_h1_v30.py``.
It is intentionally compatible with the topics and JointState names recorded in
``data/rosbag/2026_07_16``:

    Cameras
      /zeno/h1/sensor/head_cam/image/compressed
      /zeno/h1/sensor/left_arm_cam/image/compressed
      /zeno/h1/sensor/right_arm_cam/image/compressed

    State
      /zeno/h1/sensor/odom_raw
      /zeno/h1/wheelarm/{torso,left_arm,right_arm}/joint_state
      /zeno/h1/{left_gripper,right_gripper}/joint_state

    Action
      /zeno/h1/twist/cmd
      /zeno/h1/wheelarm/{torso,left_arm,right_arm}/joint_cmd
      /zeno/h1/{left_gripper,right_gripper}/joint_cmd

The generated state and action vectors are both 23-dimensional:

    [torso_lift, torso_waist, head_pan, head_tilt,
     left_arm_j0..j6, right_arm_j0..j6,
     left_gripper, right_gripper, base_vx, base_vy, base_rotation]

``left_arm_gripper`` and ``right_arm_gripper`` are the actual names found in
the July bags; aliases in the base converter map them to the two canonical
gripper fields above.

Concurrency model
-----------------
ROS bag reads are kept in one thread because ``AnyReader`` is sequential.
After the messages are loaded, a bounded ``ThreadPoolExecutor`` builds frames
in parallel (JPEG decode/crop/resize plus state/action extraction).  A single
main-thread writer calls ``LeRobotDataset.add_frame`` and ``save_episode`` in
source order.  This preserves deterministic episode ordering and avoids the
dataset/video writer races that occur if multiple threads write one dataset.

Example
-------
python scripts/data_convert/convert_zeno_h1_v30_threaded.py \
  --data-dir data/rosbag/2026_07_16 \
  --output-dir data/lerobot \
  --repo-name zeno_h1_20260716_v30 \
  --task zeno_20260716 \
  --frame-workers 8 --prefetch-factor 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import numpy as np
from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rosbags.highlevel import AnyReader

# Keep the topic contract, field order, image processing, and bag discovery in
# one place.  This script never modifies the serial converter.
from convert_zeno_h1_v30 import (
    ACTION_LEFT_ARM,
    ACTION_LEFT_GRIPPER,
    ACTION_RIGHT_ARM,
    ACTION_RIGHT_GRIPPER,
    ACTION_TORSO,
    CAMERA_TOPICS,
    DEFAULT_DATA_DIR,
    DEFAULT_IMG_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPO_NAME,
    DEFAULT_ROBOT_TYPE,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_CRF,
    DEFAULT_VIDEO_FAST_DECODE,
    DEFAULT_VIDEO_GOP,
    FPS,
    LEFT_ARM_NAMES,
    LEFT_GRIPPER_NAMES,
    ODOM,
    RIGHT_ARM_NAMES,
    RIGHT_GRIPPER_NAMES,
    STATE_LEFT_ARM,
    STATE_LEFT_GRIPPER,
    STATE_RIGHT_ARM,
    STATE_RIGHT_GRIPPER,
    STATE_TORSO,
    TORSO_FIELD_NAMES,
    TWIST_CMD,
    build_features,
    collect_bag_paths,
    decode_compressed_image,
    enabled_topics,
    extract_named_positions,
    extract_odom_velocity,
    extract_twist_velocity,
    nearest_idx,
    parse_camera_names,
    resolve_task_label,
    vector_names,
)


T = TypeVar("T")
R = TypeVar("R")


def default_frame_workers() -> int:
    """Choose a conservative default to avoid CPU oversubscription."""
    return max(1, min(8, (os.cpu_count() or 4) // 2))


def ordered_bounded_map(
    executor: ThreadPoolExecutor,
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    max_pending: int,
) -> Iterator[R]:
    """Map work in input order while keeping only ``max_pending`` futures alive.

    ``ThreadPoolExecutor.map`` submits every item eagerly on Python versions
    used by ROS Humble.  A long bag can contain tens of thousands of frames,
    so eager submission needlessly consumes memory.  This helper bounds both
    queued work and decoded-but-not-written RGB images.
    """
    if max_pending < 1:
        raise ValueError(f"max_pending must be positive, got {max_pending}")

    iterator = iter(items)
    pending: dict[int, Future[R]] = {}
    next_submit = 0
    next_yield = 0
    exhausted = False

    while not exhausted or pending:
        while not exhausted and len(pending) < max_pending:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[next_submit] = executor.submit(function, item)
            next_submit += 1

        future = pending.pop(next_yield)
        yield future.result()
        next_yield += 1


def validate_frame_vector(vector: np.ndarray, *, label: str) -> np.ndarray:
    """Reject malformed values before they are written into the dataset."""
    result = np.asarray(vector, dtype=np.float32)
    expected_dim = len(vector_names())
    if result.shape != (expected_dim,):
        raise ValueError(
            f"{label} has shape {result.shape}; expected ({expected_dim},)"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def build_frame(
    timestamp_ns: int,
    *,
    topic_to_msgs: dict[str, list[tuple[int, object]]],
    times: dict[str, np.ndarray],
    camera_names: list[str],
    img_size: tuple[int, int],
    center_crop_fraction: float,
    task_label: str,
) -> dict[str, object] | None:
    """Build one synchronized LeRobot frame from immutable bag message lists.

    This function only reads preloaded messages and is therefore safe to run in
    multiple threads.  It deliberately does *not* call any LeRobot writer API.
    """
    frame: dict[str, object] = {}
    for camera_name in camera_names:
        camera_topic = CAMERA_TOPICS[camera_name]
        image_msg = topic_to_msgs[camera_topic][
            nearest_idx(times[camera_topic], timestamp_ns)
        ][1]
        image = decode_compressed_image(
            image_msg,
            img_size,
            center_crop_fraction,
        )
        if image is None:
            return None
        frame[f"observation.images.{camera_name}"] = image

    odom_msg = topic_to_msgs[ODOM][nearest_idx(times[ODOM], timestamp_ns)][1]
    twist_cmd_msg = topic_to_msgs[TWIST_CMD][
        nearest_idx(times[TWIST_CMD], timestamp_ns)
    ][1]

    torso_state = extract_named_positions(
        topic_to_msgs[STATE_TORSO][nearest_idx(times[STATE_TORSO], timestamp_ns)][1],
        TORSO_FIELD_NAMES,
    )
    torso_action = extract_named_positions(
        topic_to_msgs[ACTION_TORSO][nearest_idx(times[ACTION_TORSO], timestamp_ns)][1],
        TORSO_FIELD_NAMES,
    )
    left_arm_state = extract_named_positions(
        topic_to_msgs[STATE_LEFT_ARM][
            nearest_idx(times[STATE_LEFT_ARM], timestamp_ns)
        ][1],
        LEFT_ARM_NAMES,
    )
    left_arm_action = extract_named_positions(
        topic_to_msgs[ACTION_LEFT_ARM][
            nearest_idx(times[ACTION_LEFT_ARM], timestamp_ns)
        ][1],
        LEFT_ARM_NAMES,
    )
    right_arm_state = extract_named_positions(
        topic_to_msgs[STATE_RIGHT_ARM][
            nearest_idx(times[STATE_RIGHT_ARM], timestamp_ns)
        ][1],
        RIGHT_ARM_NAMES,
    )
    right_arm_action = extract_named_positions(
        topic_to_msgs[ACTION_RIGHT_ARM][
            nearest_idx(times[ACTION_RIGHT_ARM], timestamp_ns)
        ][1],
        RIGHT_ARM_NAMES,
    )
    left_gripper_state = extract_named_positions(
        topic_to_msgs[STATE_LEFT_GRIPPER][
            nearest_idx(times[STATE_LEFT_GRIPPER], timestamp_ns)
        ][1],
        LEFT_GRIPPER_NAMES,
    )
    left_gripper_action = extract_named_positions(
        topic_to_msgs[ACTION_LEFT_GRIPPER][
            nearest_idx(times[ACTION_LEFT_GRIPPER], timestamp_ns)
        ][1],
        LEFT_GRIPPER_NAMES,
    )
    right_gripper_state = extract_named_positions(
        topic_to_msgs[STATE_RIGHT_GRIPPER][
            nearest_idx(times[STATE_RIGHT_GRIPPER], timestamp_ns)
        ][1],
        RIGHT_GRIPPER_NAMES,
    )
    right_gripper_action = extract_named_positions(
        topic_to_msgs[ACTION_RIGHT_GRIPPER][
            nearest_idx(times[ACTION_RIGHT_GRIPPER], timestamp_ns)
        ][1],
        RIGHT_GRIPPER_NAMES,
    )

    state = validate_frame_vector(
        np.concatenate(
            [
                torso_state,
                left_arm_state,
                right_arm_state,
                left_gripper_state,
                right_gripper_state,
                extract_odom_velocity(odom_msg),
            ]
        ),
        label="observation.state",
    )
    action = validate_frame_vector(
        np.concatenate(
            [
                torso_action,
                left_arm_action,
                right_arm_action,
                left_gripper_action,
                right_gripper_action,
                extract_twist_velocity(twist_cmd_msg),
            ]
        ),
        label="action",
    )
    frame["observation.state"] = state
    frame["action"] = action
    frame["task"] = task_label
    return frame


def required_topic_messages(
    topic_to_msgs: dict[str, list[tuple[int, object]]],
    camera_names: list[str],
) -> dict[str, list[tuple[int, object]]]:
    """Return every stream required by the verified July 2026 schema."""
    required = {
        camera_name: topic_to_msgs[CAMERA_TOPICS[camera_name]]
        for camera_name in camera_names
    }
    required.update(
        {
            "odom": topic_to_msgs[ODOM],
            "twist_cmd": topic_to_msgs[TWIST_CMD],
            "state_torso": topic_to_msgs[STATE_TORSO],
            "action_torso": topic_to_msgs[ACTION_TORSO],
            "state_left_arm": topic_to_msgs[STATE_LEFT_ARM],
            "state_right_arm": topic_to_msgs[STATE_RIGHT_ARM],
            "action_left_arm": topic_to_msgs[ACTION_LEFT_ARM],
            "action_right_arm": topic_to_msgs[ACTION_RIGHT_ARM],
            "state_left_gripper": topic_to_msgs[STATE_LEFT_GRIPPER],
            "state_right_gripper": topic_to_msgs[STATE_RIGHT_GRIPPER],
            "action_left_gripper": topic_to_msgs[ACTION_LEFT_GRIPPER],
            "action_right_gripper": topic_to_msgs[ACTION_RIGHT_GRIPPER],
        }
    )
    return required


def write_single_bag_threaded(
    dataset: LeRobotDataset,
    bag_path: Path,
    bag_idx: int,
    total_bags: int,
    *,
    task_label: str,
    fps: int,
    img_size: tuple[int, int],
    camera_names: list[str],
    center_crop_fraction: float,
    max_frames: int | None,
    frame_workers: int,
    prefetch_factor: int,
) -> int:
    """Read a bag and write one episode, parallelizing only frame construction."""
    print(f"\n[Bag {bag_idx}/{total_bags}] Processing: {bag_path}", flush=True)
    bag_start = time.time()
    topics = enabled_topics(camera_names)
    topic_set = set(topics)

    try:
        with AnyReader([bag_path]) as reader:
            topic_to_msgs: dict[str, list[tuple[int, object]]] = {
                topic: [] for topic in topics
            }
            connections = [
                connection for connection in reader.connections if connection.topic in topic_set
            ]
            if not connections:
                print("  No relevant topics found, skipping.", flush=True)
                return 0

            for connection, timestamp_ns, raw in reader.messages(connections=connections):
                message = reader.deserialize(raw, connection.msgtype)
                topic_to_msgs[connection.topic].append((timestamp_ns, message))

            required = required_topic_messages(topic_to_msgs, camera_names)
            missing = [name for name, messages in required.items() if not messages]
            if missing:
                print(f"  Missing required streams {missing}, skipping.", flush=True)
                return 0

            times = {
                topic: np.fromiter(
                    (timestamp for timestamp, _ in messages), dtype=np.int64
                )
                for topic, messages in topic_to_msgs.items()
            }
            t_start = max(times[topic][0] for topic in topics)
            t_end = min(times[topic][-1] for topic in topics)
            if t_end <= t_start:
                print("  No overlapping time range, skipping.", flush=True)
                return 0

            step_ns = int(1e9 / fps)
            frame_count = int((t_end - t_start) // step_ns) + 1
            if max_frames is not None:
                frame_count = min(frame_count, max_frames)
            if frame_count < 2:
                print("  Fewer than two sample frames, skipping.", flush=True)
                return 0

            sample_times = t_start + np.arange(frame_count, dtype=np.int64) * step_ns
            duration_s = (t_end - t_start) / 1e9
            print(
                f"  Duration: {duration_s:.2f}s, candidates: {frame_count}, "
                f"frame-workers: {frame_workers}",
                flush=True,
            )

            def frame_builder(timestamp_ns: np.int64) -> dict[str, object] | None:
                return build_frame(
                    int(timestamp_ns),
                    topic_to_msgs=topic_to_msgs,
                    times=times,
                    camera_names=camera_names,
                    img_size=img_size,
                    center_crop_fraction=center_crop_fraction,
                    task_label=task_label,
                )

            written = 0
            dropped_images = 0
            initial_frames: list[dict[str, object]] = []
            max_pending = frame_workers * prefetch_factor
            with ThreadPoolExecutor(
                max_workers=frame_workers,
                thread_name_prefix="zeno-frame",
            ) as executor:
                for frame in ordered_bounded_map(
                    executor,
                    frame_builder,
                    sample_times,
                    max_pending=max_pending,
                ):
                    if frame is None:
                        dropped_images += 1
                        continue
                    written += 1
                    # Do not mutate the dataset until this bag is known to have
                    # the minimum two frames needed for a valid episode.  This
                    # avoids leaving a one-frame partial episode in LeRobot's
                    # internal buffer when every other image failed to decode.
                    if len(initial_frames) < 2:
                        initial_frames.append(frame)
                        if len(initial_frames) == 2:
                            for initial_frame in initial_frames:
                                dataset.add_frame(initial_frame)
                        continue
                    # LeRobotDataset is not thread-safe; keep all mutations here.
                    dataset.add_frame(frame)

            if written < 2:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                print(
                    f"  Only {written} valid frames (dropped images={dropped_images}), "
                    "skipping episode.",
                    flush=True,
                )
                return 0

            dataset.save_episode()
            elapsed = time.time() - bag_start
            print(
                f"  Saved episode: frames={written}, dropped-images={dropped_images}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            return written
    except Exception as exc:
        # A frame task can fail after valid earlier frames were buffered.  Do
        # not let those partial frames be appended to the following bag.
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        print(f"  Error processing {bag_path}: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--repo-name", default=DEFAULT_REPO_NAME)
    parser.add_argument(
        "--task",
        default=None,
        help="Task label; default is the source bag parent directory.",
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--img-width", type=int, default=None)
    parser.add_argument("--img-height", type=int, default=None)
    parser.add_argument("--center-crop-fraction", type=float, default=1.0)
    parser.add_argument(
        "--cameras",
        default="head_cam,left_arm_cam,right_arm_cam",
        help="Comma-separated subset of: head_cam,left_arm_cam,right_arm_cam.",
    )
    parser.add_argument("--max-bags", type=int, default=None)
    parser.add_argument("--exclude-bags", default="")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--robot-type", default=DEFAULT_ROBOT_TYPE)
    parser.add_argument("--video-codec", "--vcodec", dest="video_codec", default=DEFAULT_VIDEO_CODEC)
    parser.add_argument("--video-crf", type=float, default=DEFAULT_VIDEO_CRF)
    parser.add_argument("--video-gop", type=int, default=DEFAULT_VIDEO_GOP)
    parser.add_argument("--video-fast-decode", type=int, default=DEFAULT_VIDEO_FAST_DECODE)
    parser.add_argument("--video-preset", default=None)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument(
        "--frame-workers",
        type=int,
        default=default_frame_workers(),
        help="Threads for parallel frame building; default is a conservative CPU-based value.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Keep at most frame-workers × this many frame tasks queued.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="Threads used internally by LeRobot image/video writing.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=4,
        help="Processes used internally by LeRobot image/video writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output dataset directory before conversion.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[str], tuple[int, int]]:
    positive = {
        "--fps": args.fps,
        "--img-size": args.img_size,
        "--frame-workers": args.frame_workers,
        "--prefetch-factor": args.prefetch_factor,
        "--image-writer-threads": args.image_writer_threads,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    if args.image_writer_processes < 0:
        raise SystemExit("--image-writer-processes must be zero or positive")
    if args.encoder_threads is not None and args.encoder_threads <= 0:
        raise SystemExit("--encoder-threads must be positive when provided")
    if args.max_bags is not None and args.max_bags <= 0:
        raise SystemExit("--max-bags must be positive when provided")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive when provided")
    if args.video_gop <= 0:
        raise SystemExit("--video-gop must be positive")
    if not 0 < args.center_crop_fraction <= 1:
        raise SystemExit("--center-crop-fraction must be in the range (0, 1]")
    if (args.img_width is None) != (args.img_height is None):
        raise SystemExit("--img-width and --img-height must be specified together")
    if args.img_width is not None and (args.img_width <= 0 or args.img_height <= 0):
        raise SystemExit("--img-width and --img-height must be positive")

    camera_names = parse_camera_names(args.cameras)
    if args.img_width is None:
        image_size = (args.img_size, args.img_size)
    else:
        image_size = (args.img_width, args.img_height)
    return camera_names, image_size


def main() -> None:
    args = parse_args()
    camera_names, image_size = validate_args(args)
    output_path = Path(args.output_dir) / args.repo_name
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    exclusions = {name.strip() for name in args.exclude_bags.split(",") if name.strip()}
    bag_paths = collect_bag_paths(
        args.data_dir,
        max_bags=args.max_bags,
        exclude_bags=exclusions,
    )
    if not bag_paths:
        raise SystemExit("No ROS bag paths found.")

    features = build_features(image_size, camera_names)
    camera_encoder = VideoEncoderConfig(
        vcodec=args.video_codec,
        pix_fmt="yuv420p",
        g=args.video_gop,
        crf=args.video_crf,
        preset=args.video_preset,
        fast_decode=args.video_fast_decode,
    )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        root=output_path,
        robot_type=args.robot_type,
        fps=args.fps,
        features=features,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        camera_encoder=camera_encoder,
        encoder_threads=args.encoder_threads,
    )

    print("\n" + "=" * 68)
    print("Converting Zeno H1 bags to LeRobot V3.0 (threaded frame builder)")
    print(f"  Input:            {args.data_dir}")
    print(f"  Output:           {output_path}")
    print(f"  Bags:             {len(bag_paths)}")
    print(f"  Cameras:          {', '.join(camera_names)}")
    print(f"  State/action:     {len(vector_names())}D [torso, arms, grippers, base]")
    print(f"  Frame workers:    {args.frame_workers} (prefetch={args.prefetch_factor})")
    print(
        "  Writer workers:   "
        f"threads={args.image_writer_threads}, processes={args.image_writer_processes}"
    )
    print("=" * 68)

    total_start = time.time()
    successful = 0
    total_frames = 0
    for bag_idx, bag_path in enumerate(bag_paths, start=1):
        frame_count = write_single_bag_threaded(
            dataset,
            bag_path,
            bag_idx,
            len(bag_paths),
            task_label=resolve_task_label(bag_path, args.task),
            fps=args.fps,
            img_size=image_size,
            camera_names=camera_names,
            center_crop_fraction=args.center_crop_fraction,
            max_frames=args.max_frames,
            frame_workers=args.frame_workers,
            prefetch_factor=args.prefetch_factor,
        )
        if frame_count:
            successful += 1
            total_frames += frame_count

    dataset.finalize()
    elapsed = time.time() - total_start
    print("\n" + "=" * 68)
    print("Conversion complete")
    print(f"  Episodes: {successful}/{len(bag_paths)}")
    print(f"  Frames:   {total_frames}")
    print(f"  Elapsed:  {elapsed:.1f}s")
    print(f"  Output:   {output_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
