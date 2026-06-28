"""
Convert Zeno H1 ROS2 MCAP bags to a LeRobot V3.0 dataset.

This is adapted from convert_co_mobileb_v30.py for bags recorded under:
    /home/zeno-rp/2026CoRL/Data/zeno_bag

Default conversion:
  Cameras:
    /zeno/h1/sensor/head_cam/image/compressed
    /zeno/h1/sensor/left_arm_cam/image/compressed
    /zeno/h1/sensor/right_arm_cam/image/compressed
  State:
    /zeno/h1/sensor/odom_raw
    /zeno/h1/wheelarm/torso/joint_state
    /zeno/h1/wheelarm/left_arm/joint_state
    /zeno/h1/left_gripper/joint_state
    /zeno/h1/wheelarm/right_arm/joint_state
    /zeno/h1/right_gripper/joint_state
  Action:
    /zeno/h1/twist/cmd
    /zeno/h1/wheelarm/torso/joint_cmd
    /zeno/h1/wheelarm/left_arm/joint_cmd
    /zeno/h1/left_gripper/joint_cmd
    /zeno/h1/wheelarm/right_arm/joint_cmd
    /zeno/h1/right_gripper/joint_cmd

The state/action vector is 23D and follows /zeno/h1/auto/wholebody/cmd
without control_mode:
    [torso_lift, torso_waist, head_pan, head_tilt,
     left_arm_j0..j6,
     right_arm_j0..j6,
     left_gripper, right_gripper,
     base_vx, base_vy, base_rotation]

The bags currently contain separated joint_cmd topics rather than
/zeno/h1/auto/wholebody/cmd. The base action is taken from
/zeno/h1/twist/cmd, while the base state is measured from
/zeno/h1/sensor/odom_raw.

Usage:
    python scripts/data_convert/convert_zeno_h1_v30.py \
        --data-dir /home/zeno-rp/2027icra/Data/humanmoid_pick \
        --output-dir /home/zeno-rp/2027icra/Data/lerobot \
        --repo-name humanmoid_pick_zeno_h1_auto_cmd_v30 \
        --task humanmoid_pick
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rosbags.highlevel import AnyReader


DEFAULT_DATA_DIR = Path("/home/zeno-rp/2026CoRL/Data/zeno_bag")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR.parent
DEFAULT_REPO_NAME = "zeno_h1_v30"
DEFAULT_ROBOT_TYPE = "zeno_h1"

CAM_HEAD = "/zeno/h1/sensor/head_cam/image/compressed"
CAM_LEFT_ARM = "/zeno/h1/sensor/left_arm_cam/image/compressed"
CAM_RIGHT_ARM = "/zeno/h1/sensor/right_arm_cam/image/compressed"
ODOM = "/zeno/h1/sensor/odom_raw"
TWIST_CMD = "/zeno/h1/twist/cmd"

STATE_LEFT_ARM = "/zeno/h1/wheelarm/left_arm/joint_state"
STATE_RIGHT_ARM = "/zeno/h1/wheelarm/right_arm/joint_state"
ACTION_LEFT_ARM = "/zeno/h1/wheelarm/left_arm/joint_cmd"
ACTION_RIGHT_ARM = "/zeno/h1/wheelarm/right_arm/joint_cmd"

STATE_LEFT_GRIPPER = "/zeno/h1/left_gripper/joint_state"
STATE_RIGHT_GRIPPER = "/zeno/h1/right_gripper/joint_state"
ACTION_LEFT_GRIPPER = "/zeno/h1/left_gripper/joint_cmd"
ACTION_RIGHT_GRIPPER = "/zeno/h1/right_gripper/joint_cmd"

STATE_TORSO = "/zeno/h1/wheelarm/torso/joint_state"
ACTION_TORSO = "/zeno/h1/wheelarm/torso/joint_cmd"

TORSO_FIELD_NAMES = ["torso_lift", "torso_waist", "head_pan", "head_tilt"]
LEFT_ARM_NAMES = [f"left_arm_j{i}" for i in range(7)]
RIGHT_ARM_NAMES = [f"right_arm_j{i}" for i in range(7)]
LEFT_GRIPPER_NAMES = ["left_gripper"]
RIGHT_GRIPPER_NAMES = ["right_gripper"]
BASE_NAMES = ["base_vx", "base_vy", "base_rotation"]
AUTO_CMD_FIELD_NAMES = (
    TORSO_FIELD_NAMES
    + LEFT_ARM_NAMES
    + RIGHT_ARM_NAMES
    + LEFT_GRIPPER_NAMES
    + RIGHT_GRIPPER_NAMES
    + BASE_NAMES
)

JOINT_NAME_ALIASES = {
    "head_pan": ["torso_head_pan"],
    "head_tilt": ["torso_head_tilt"],
    "left_gripper": ["left_arm_gripper"],
    "right_gripper": ["right_arm_gripper"],
}

FPS = 20
DEFAULT_IMG_SIZE = 224
DEFAULT_VIDEO_CODEC = "h264"
DEFAULT_VIDEO_CRF = 18
DEFAULT_VIDEO_GOP = 2
DEFAULT_VIDEO_FAST_DECODE = 1


def decode_compressed_image(msg, img_size: tuple[int, int]) -> np.ndarray | None:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR)


def nearest_idx(times: np.ndarray, t: int) -> int:
    idx = np.searchsorted(times, t)
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    if abs(times[idx] - t) < abs(t - times[idx - 1]):
        return idx
    return idx - 1


def extract_odom_velocity(msg) -> np.ndarray:
    twist = msg.twist.twist
    return np.array(
        [twist.linear.x, twist.linear.y, twist.angular.z],
        dtype=np.float32,
    )


def extract_twist_velocity(msg) -> np.ndarray:
    return np.array(
        [msg.linear.x, msg.linear.y, msg.angular.z],
        dtype=np.float32,
    )


def extract_named_positions(msg, expected_names: list[str]) -> np.ndarray:
    positions = list(msg.position)
    names = [str(name) for name in msg.name]

    if names:
        by_name = {name: float(pos) for name, pos in zip(names, positions)}
        values = []
        missing = []
        for name in expected_names:
            aliases = [name, *JOINT_NAME_ALIASES.get(name, [])]
            matched = next((alias for alias in aliases if alias in by_name), None)
            if matched is None:
                missing.append(name)
                continue
            values.append(by_name[matched])
        if missing:
            raise ValueError(
                f"missing joint names {missing}; message has {names}; "
                f"aliases={JOINT_NAME_ALIASES}"
            )
        return np.array(values, dtype=np.float32)

    if len(positions) < len(expected_names):
        raise ValueError(
            f"expected {len(expected_names)} positions, got {len(positions)}"
        )
    return np.array(positions[: len(expected_names)], dtype=np.float32)


def enabled_topics() -> list[str]:
    return [
        CAM_HEAD,
        CAM_LEFT_ARM,
        CAM_RIGHT_ARM,
        ODOM,
        TWIST_CMD,
        STATE_TORSO,
        ACTION_TORSO,
        STATE_LEFT_ARM,
        STATE_RIGHT_ARM,
        ACTION_LEFT_ARM,
        ACTION_RIGHT_ARM,
        STATE_LEFT_GRIPPER,
        STATE_RIGHT_GRIPPER,
        ACTION_LEFT_GRIPPER,
        ACTION_RIGHT_GRIPPER,
    ]


def collect_bag_paths(data_dir: str | Path, max_bags: int | None = None) -> list[Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: data directory not found: {data_path}")
        return []

    if (data_path / "metadata.yaml").exists():
        bag_paths = [data_path]
    else:
        bag_paths = sorted({path.parent for path in data_path.rglob("metadata.yaml")})

    if not bag_paths:
        bag_paths = sorted(data_path.rglob("*.mcap"))
    if not bag_paths:
        bag_paths = sorted(data_path.rglob("*.bag"))

    if max_bags is not None:
        bag_paths = bag_paths[:max_bags]

    if bag_paths:
        print(f"Found {len(bag_paths)} bag path(s)")
    else:
        print(f"No ROS bag paths found in {data_path}")

    return bag_paths


def resolve_task_label(bag_path: Path, task_override: str | None) -> str:
    if task_override:
        return task_override
    if bag_path.is_dir() and bag_path.parent.name:
        return bag_path.parent.name
    if bag_path.parent.name:
        return bag_path.parent.name
    return "zeno_h1"


def vector_names() -> list[str]:
    return list(AUTO_CMD_FIELD_NAMES)


def build_features(img_size: tuple[int, int]) -> dict:
    names = vector_names()
    dim = len(names)

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": names,
        },
        "action": {
            "dtype": "float32",
            "shape": (dim,),
            "names": names,
        },
        "observation.images.head_cam": {
            "dtype": "video",
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.left_arm_cam": {
            "dtype": "video",
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_arm_cam": {
            "dtype": "video",
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channels"],
        },
    }


def process_single_bag(
    bag_path: Path,
    bag_idx: int,
    total_bags: int,
    *,
    task_label: str,
    fps: int,
    img_size: tuple[int, int],
    max_frames: int | None,
) -> list[dict] | None:
    print(f"\n[Bag {bag_idx}/{total_bags}] Processing: {bag_path}")
    bag_start = time.time()
    topics = enabled_topics()
    topic_set = set(topics)

    try:
        with AnyReader([bag_path]) as reader:
            topic_to_msgs = {topic: [] for topic in topics}
            connections = [c for c in reader.connections if c.topic in topic_set]
            if not connections:
                print("  No relevant topics found, skipping.")
                return None

            for conn, t, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, conn.msgtype)
                topic_to_msgs[conn.topic].append((t, msg))

            required = {
                "head_cam": topic_to_msgs[CAM_HEAD],
                "left_arm_cam": topic_to_msgs[CAM_LEFT_ARM],
                "right_arm_cam": topic_to_msgs[CAM_RIGHT_ARM],
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

            for name, msgs in required.items():
                if not msgs:
                    print(f"  Missing {name}, skipping.")
                    return None

            times = {
                topic: np.array([t for t, _ in msgs], dtype=np.int64)
                for topic, msgs in topic_to_msgs.items()
            }
            time_arrays = [times[topic] for topic in topics]
            t_start = max(arr[0] for arr in time_arrays)
            t_end = min(arr[-1] for arr in time_arrays)
            if t_end <= t_start:
                print("  No overlapping time range, skipping.")
                return None

            step_ns = int(1e9 / fps)
            n_frames = int((t_end - t_start) // step_ns) + 1
            if max_frames is not None:
                n_frames = min(n_frames, max_frames)

            duration_s = (t_end - t_start) / 1e9
            if n_frames < 2:
                print(f"  Too short ({duration_s:.2f}s), skipping.")
                return None

            sample_times = t_start + np.arange(n_frames, dtype=np.int64) * step_ns
            print(
                f"  Duration: {duration_s:.2f}s, frames: {n_frames}, "
                f"fps: {fps}, dim: {len(vector_names())}"
            )

            frames = []
            for t in sample_times:
                img_head = decode_compressed_image(
                    topic_to_msgs[CAM_HEAD][nearest_idx(times[CAM_HEAD], t)][1],
                    img_size,
                )
                img_left = decode_compressed_image(
                    topic_to_msgs[CAM_LEFT_ARM][nearest_idx(times[CAM_LEFT_ARM], t)][1],
                    img_size,
                )
                img_right = decode_compressed_image(
                    topic_to_msgs[CAM_RIGHT_ARM][nearest_idx(times[CAM_RIGHT_ARM], t)][1],
                    img_size,
                )
                if any(img is None for img in [img_head, img_left, img_right]):
                    continue

                odom_msg = topic_to_msgs[ODOM][nearest_idx(times[ODOM], t)][1]
                twist_cmd_msg = topic_to_msgs[TWIST_CMD][
                    nearest_idx(times[TWIST_CMD], t)
                ][1]
                base_state = extract_odom_velocity(odom_msg)
                base_action = extract_twist_velocity(twist_cmd_msg)

                torso_state = extract_named_positions(
                    topic_to_msgs[STATE_TORSO][nearest_idx(times[STATE_TORSO], t)][1],
                    TORSO_FIELD_NAMES,
                )
                torso_action = extract_named_positions(
                    topic_to_msgs[ACTION_TORSO][nearest_idx(times[ACTION_TORSO], t)][1],
                    TORSO_FIELD_NAMES,
                )
                left_arm_state = extract_named_positions(
                    topic_to_msgs[STATE_LEFT_ARM][
                        nearest_idx(times[STATE_LEFT_ARM], t)
                    ][1],
                    LEFT_ARM_NAMES,
                )
                right_arm_state = extract_named_positions(
                    topic_to_msgs[STATE_RIGHT_ARM][
                        nearest_idx(times[STATE_RIGHT_ARM], t)
                    ][1],
                    RIGHT_ARM_NAMES,
                )
                left_arm_action = extract_named_positions(
                    topic_to_msgs[ACTION_LEFT_ARM][
                        nearest_idx(times[ACTION_LEFT_ARM], t)
                    ][1],
                    LEFT_ARM_NAMES,
                )
                right_arm_action = extract_named_positions(
                    topic_to_msgs[ACTION_RIGHT_ARM][
                        nearest_idx(times[ACTION_RIGHT_ARM], t)
                    ][1],
                    RIGHT_ARM_NAMES,
                )

                left_gripper_state = extract_named_positions(
                    topic_to_msgs[STATE_LEFT_GRIPPER][
                        nearest_idx(times[STATE_LEFT_GRIPPER], t)
                    ][1],
                    LEFT_GRIPPER_NAMES,
                )
                left_gripper_action = extract_named_positions(
                    topic_to_msgs[ACTION_LEFT_GRIPPER][
                        nearest_idx(times[ACTION_LEFT_GRIPPER], t)
                    ][1],
                    LEFT_GRIPPER_NAMES,
                )
                right_gripper_state = extract_named_positions(
                    topic_to_msgs[STATE_RIGHT_GRIPPER][
                        nearest_idx(times[STATE_RIGHT_GRIPPER], t)
                    ][1],
                    RIGHT_GRIPPER_NAMES,
                )
                right_gripper_action = extract_named_positions(
                    topic_to_msgs[ACTION_RIGHT_GRIPPER][
                        nearest_idx(times[ACTION_RIGHT_GRIPPER], t)
                    ][1],
                    RIGHT_GRIPPER_NAMES,
                )

                state_parts = [
                    torso_state,
                    left_arm_state,
                    right_arm_state,
                    left_gripper_state,
                    right_gripper_state,
                    base_state,
                ]
                action_parts = [
                    torso_action,
                    left_arm_action,
                    right_arm_action,
                    left_gripper_action,
                    right_gripper_action,
                    base_action,
                ]

                frames.append(
                    {
                        "observation.images.head_cam": img_head,
                        "observation.images.left_arm_cam": img_left,
                        "observation.images.right_arm_cam": img_right,
                        "observation.state": np.concatenate(state_parts).astype(
                            np.float32
                        ),
                        "action": np.concatenate(action_parts).astype(np.float32),
                        "task": task_label,
                    }
                )

            elapsed = time.time() - bag_start
            print(f"  Done: {len(frames)} frames in {elapsed:.1f}s")
            return frames if frames else None

    except Exception as exc:
        print(f"  Error processing {bag_path}: {exc}")
        import traceback

        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Zeno H1 ROS2 MCAP bags to a LeRobot V3.0 dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing ROS2 bag directories, .mcap files, or .bag files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output root directory",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=DEFAULT_REPO_NAME,
        help="Dataset repo name",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task label override; defaults to the bag parent directory",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Target frame rate")
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="Resize images to img-size x img-size",
    )
    parser.add_argument(
        "--max-bags",
        type=int,
        default=None,
        help="Optional limit on number of bags to convert",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit per bag for smoke tests",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=DEFAULT_ROBOT_TYPE,
        help="robot_type passed into LeRobotDataset.create",
    )
    parser.add_argument(
        "--vcodec",
        "--video-codec",
        dest="video_codec",
        type=str,
        default=DEFAULT_VIDEO_CODEC,
        help=(
            "Video codec for camera MP4s. Use h264 for fast training decode; "
            "libsvtav1/av1 saves space but is slower to decode."
        ),
    )
    parser.add_argument(
        "--video-crf",
        type=float,
        default=DEFAULT_VIDEO_CRF,
        help="Video quality. Lower is better/larger. For h264, 18 is visually high quality.",
    )
    parser.add_argument(
        "--video-gop",
        type=int,
        default=DEFAULT_VIDEO_GOP,
        help="Video GOP/keyframe interval. Small values speed random training reads.",
    )
    parser.add_argument(
        "--video-fast-decode",
        type=int,
        default=DEFAULT_VIDEO_FAST_DECODE,
        help="Enable codec fast-decode tuning when supported; 1 is recommended for training.",
    )
    parser.add_argument(
        "--video-preset",
        type=str,
        default=None,
        help="Optional codec preset, for example veryfast/fast/medium for h264.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Optional encoder threads passed to LeRobot video encoder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before conversion",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.img_size <= 0:
        raise SystemExit("--img-size must be positive")
    if args.max_bags is not None and args.max_bags <= 0:
        raise SystemExit("--max-bags must be positive when provided")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive when provided")
    if args.video_gop <= 0:
        raise SystemExit("--video-gop must be positive")
    if args.encoder_threads is not None and args.encoder_threads <= 0:
        raise SystemExit("--encoder-threads must be positive when provided")

    img_size = (args.img_size, args.img_size)
    output_path = Path(args.output_dir) / args.repo_name

    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    bag_paths = collect_bag_paths(args.data_dir, max_bags=args.max_bags)
    total_bags = len(bag_paths)
    if total_bags == 0:
        print("No bag paths to process.")
        return

    features = build_features(img_size)
    camera_encoder = VideoEncoderConfig(
        vcodec=args.video_codec,
        pix_fmt="yuv420p",
        g=args.video_gop,
        crf=args.video_crf,
        preset=args.video_preset,
        fast_decode=args.video_fast_decode,
    )
    create_kwargs = {
        "repo_id": args.repo_name,
        "root": output_path,
        "robot_type": args.robot_type,
        "fps": args.fps,
        "features": features,
        "use_videos": True,
        "image_writer_threads": 8,
        "image_writer_processes": 4,
        "camera_encoder": camera_encoder,
        "encoder_threads": args.encoder_threads,
    }

    dataset = LeRobotDataset.create(**create_kwargs)

    total_start = time.time()
    dim = len(vector_names())
    print(f"\n{'=' * 60}")
    print("Converting Zeno H1 to LeRobot V3.0")
    print(f"  Input:    {args.data_dir}")
    print(f"  Output:   {output_path}")
    print(f"  Bags:     {total_bags}")
    print(f"  FPS:      {args.fps}")
    print(f"  ImgSize:  {img_size}")
    print(
        "  Video:    "
        f"codec={camera_encoder.vcodec}, crf={camera_encoder.crf}, "
        f"gop={camera_encoder.g}, fast_decode={camera_encoder.fast_decode}"
    )
    print(f"  Dim:      {dim}D")
    print("  Layout:   /zeno/h1/auto/wholebody/cmd[1..23]")
    print("  Note:     base state uses odom_raw; base action uses twist/cmd")
    print("  Cameras:  head_cam, left_arm_cam, right_arm_cam")
    print(f"{'=' * 60}")

    successful = 0
    for bag_idx, bag_path in enumerate(bag_paths, 1):
        task_label = resolve_task_label(bag_path, args.task)
        result = process_single_bag(
            bag_path,
            bag_idx,
            total_bags,
            task_label=task_label,
            fps=args.fps,
            img_size=img_size,
            max_frames=args.max_frames,
        )
        if result is not None:
            for frame in result:
                dataset.add_frame(frame)
            dataset.save_episode()
            successful += 1
            print(f"  [Bag {bag_idx}/{total_bags}] Saved episode: task={task_label}")
        del result

    dataset.finalize()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"  Episodes: {successful}/{total_bags}")
    print(f"  Time:     {total_elapsed:.1f}s")
    print(f"  Output:   {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
