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

    # For runs where the left arm camera was not publishing:
    python scripts/data_convert/convert_zeno_h1_v30.py \
        --data-dir /home/zeno-rp/2027icra/Data/2026_07_09 \
        --output-dir /home/zeno-rp/2027icra/Data/lerobot \
        --repo-name robot8_20260709_zeno_h1_auto_cmd_v30_head_right \
        --task robot8_20260709 \
        --cameras head_cam,right_arm_cam
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
CAMERA_TOPICS = {
    "head_cam": CAM_HEAD,
    "left_arm_cam": CAM_LEFT_ARM,
    "right_arm_cam": CAM_RIGHT_ARM,
}
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
AUTO_CMD_FIELD_TO_INDEX = {
    field_name: index for index, field_name in enumerate(AUTO_CMD_FIELD_NAMES)
}

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


def center_crop_image(img: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 1.0:
        return img

    height, width = img.shape[:2]
    crop_width = max(1, min(width, int(round(width * fraction))))
    crop_height = max(1, min(height, int(round(height * fraction))))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    return img[y0 : y0 + crop_height, x0 : x0 + crop_width]


def decode_compressed_image(
    msg,
    img_size: tuple[int, int],
    center_crop_fraction: float = 1.0,
) -> np.ndarray | None:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_bgr = center_crop_image(img_bgr, center_crop_fraction)
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


ROBOT_TOPICS = [
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


def parse_camera_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise SystemExit("--cameras must include at least one camera name")

    unknown = [name for name in names if name not in CAMERA_TOPICS]
    if unknown:
        known = ", ".join(CAMERA_TOPICS)
        raise SystemExit(f"Unknown camera name(s): {unknown}. Known cameras: {known}")

    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def parse_vector_field_names(raw: str) -> list[str]:
    """Parse optional 23D state/action fields that should be held at zero.

    Keeping the vector shape intact is important: existing ACT checkpoints,
    normalizers, and the robot bridge all use the fixed 23D command ABI.  A
    frozen field is therefore represented by a zero value rather than by
    removing a dimension.
    """
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in AUTO_CMD_FIELD_TO_INDEX]
    if unknown:
        known = ", ".join(AUTO_CMD_FIELD_NAMES)
        raise SystemExit(f"Unknown --frozen-fields value(s): {unknown}. Known fields: {known}")
    return list(dict.fromkeys(names))


def enabled_topics(camera_names: list[str]) -> list[str]:
    return [CAMERA_TOPICS[name] for name in camera_names] + ROBOT_TOPICS


def bag_name(path: Path) -> str:
    return path.parent.name if path.is_file() else path.name


def collect_bag_paths(
    data_dir: str | Path,
    max_bags: int | None = None,
    exclude_bags: set[str] | None = None,
) -> list[Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: data directory not found: {data_path}")
        return []

    if data_path.is_file():
        bag_paths = [data_path]
    elif (
        (data_path / "metadata.yaml").is_file()
        and (data_path / "metadata.yaml").stat().st_size > 0
    ):
        bag_paths = [data_path]
    else:
        # An interrupted rosbag may leave an empty metadata.yaml next to a valid
        # MCAP. AnyReader cannot open that directory, so fall back to the MCAP.
        metadata_dirs = {
            path.parent
            for path in data_path.rglob("metadata.yaml")
            if path.is_file() and path.stat().st_size > 0
        }
        standalone_mcaps = {
            path
            for path in data_path.rglob("*.mcap")
            if path.parent not in metadata_dirs
        }
        bag_paths = sorted(
            [*metadata_dirs, *standalone_mcaps],
            key=lambda path: str(path),
        )
    if not bag_paths:
        bag_paths = sorted(data_path.rglob("*.bag"))

    if exclude_bags:
        discovered_names = {bag_name(path) for path in bag_paths}
        unknown_exclusions = sorted(exclude_bags - discovered_names)
        if unknown_exclusions:
            print(
                "Warning: requested exclusions were not found: "
                + ", ".join(unknown_exclusions)
            )
        for path in bag_paths:
            if bag_name(path) in exclude_bags:
                print(f"Excluding bag: {bag_name(path)}")
        bag_paths = [
            path for path in bag_paths if bag_name(path) not in exclude_bags
        ]

    empty_files = [
        path for path in bag_paths if path.is_file() and path.stat().st_size == 0
    ]
    for path in empty_files:
        print(f"Skipping empty bag file: {path}")
    if empty_files:
        empty_set = set(empty_files)
        bag_paths = [path for path in bag_paths if path not in empty_set]

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


def build_features(img_size: tuple[int, int], camera_names: list[str]) -> dict:
    names = vector_names()
    dim = len(names)

    features = {
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
    }

    for camera_name in camera_names:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channels"],
        }

    return features


def process_single_bag(
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
    frozen_indices: tuple[int, ...],
) -> list[dict] | None:
    print(f"\n[Bag {bag_idx}/{total_bags}] Processing: {bag_path}")
    bag_start = time.time()
    topics = enabled_topics(camera_names)
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
                name: topic_to_msgs[CAMERA_TOPICS[name]] for name in camera_names
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
            dropped_images = 0
            for t in sample_times:
                frame = {}
                failed_image = False
                for camera_name in camera_names:
                    camera_topic = CAMERA_TOPICS[camera_name]
                    img = decode_compressed_image(
                        topic_to_msgs[camera_topic][nearest_idx(times[camera_topic], t)][1],
                        img_size,
                        center_crop_fraction,
                    )
                    if img is None:
                        failed_image = True
                        break
                    frame[f"observation.images.{camera_name}"] = img
                if failed_image:
                    dropped_images += 1
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

                state = np.concatenate(state_parts).astype(np.float32)
                action = np.concatenate(action_parts).astype(np.float32)
                if frozen_indices:
                    state[list(frozen_indices)] = 0.0
                    action[list(frozen_indices)] = 0.0

                frame.update(
                    {
                        "observation.state": state,
                        "action": action,
                        "task": task_label,
                    }
                )
                frames.append(frame)

            elapsed = time.time() - bag_start
            print(
                f"  Done: {len(frames)} frames in {elapsed:.1f}s; "
                f"dropped-images={dropped_images}"
            )
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
        help=(
            "Resize images to img-size x img-size unless --img-width/--img-height "
            "are provided"
        ),
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=None,
        help="Resize images to this width; use with --img-height for non-square output",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=None,
        help="Resize images to this height; use with --img-width for non-square output",
    )
    parser.add_argument(
        "--center-crop-fraction",
        type=float,
        default=1.0,
        help=(
            "Keep this centered fraction of original width and height before resize; "
            "1.0 disables cropping"
        ),
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="head_cam,left_arm_cam,right_arm_cam",
        help=(
            "Comma-separated camera names to include. Known names: "
            "head_cam,left_arm_cam,right_arm_cam"
        ),
    )
    parser.add_argument(
        "--frozen-fields",
        type=str,
        default="",
        help=(
            "Comma-separated 23D state/action fields to set to zero in the generated dataset. "
            "Example: torso_lift,torso_waist"
        ),
    )
    parser.add_argument(
        "--max-bags",
        type=int,
        default=None,
        help="Optional limit on number of bags to convert",
    )
    parser.add_argument(
        "--exclude-bags",
        type=str,
        default="",
        help=(
            "Comma-separated bag directory names to exclude after data inspection, "
            "for example rosbag2_..._a,rosbag2_..._b"
        ),
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
    if (args.img_width is None) != (args.img_height is None):
        raise SystemExit("--img-width and --img-height must be provided together")
    if args.img_width is not None and args.img_width <= 0:
        raise SystemExit("--img-width must be positive")
    if args.img_height is not None and args.img_height <= 0:
        raise SystemExit("--img-height must be positive")
    if not (0 < args.center_crop_fraction <= 1.0):
        raise SystemExit("--center-crop-fraction must be in the range (0, 1]")
    if args.max_bags is not None and args.max_bags <= 0:
        raise SystemExit("--max-bags must be positive when provided")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive when provided")
    if args.video_gop <= 0:
        raise SystemExit("--video-gop must be positive")
    if args.encoder_threads is not None and args.encoder_threads <= 0:
        raise SystemExit("--encoder-threads must be positive when provided")
    camera_names = parse_camera_names(args.cameras)
    frozen_fields = parse_vector_field_names(args.frozen_fields)
    frozen_indices = tuple(AUTO_CMD_FIELD_TO_INDEX[name] for name in frozen_fields)
    exclude_bags = {
        name.strip() for name in args.exclude_bags.split(",") if name.strip()
    }

    if args.img_width is not None and args.img_height is not None:
        img_size = (args.img_width, args.img_height)
    else:
        img_size = (args.img_size, args.img_size)
    output_path = Path(args.output_dir) / args.repo_name

    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    bag_paths = collect_bag_paths(
        args.data_dir,
        max_bags=args.max_bags,
        exclude_bags=exclude_bags,
    )
    total_bags = len(bag_paths)
    if total_bags == 0:
        raise SystemExit("No non-empty bag paths to process.")

    features = build_features(img_size, camera_names)
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
    print(f"  Crop:     center fraction={args.center_crop_fraction}")
    print(
        "  Video:    "
        f"codec={camera_encoder.vcodec}, crf={camera_encoder.crf}, "
        f"gop={camera_encoder.g}, fast_decode={camera_encoder.fast_decode}"
    )
    print(f"  Dim:      {dim}D")
    print("  Layout:   /zeno/h1/auto/wholebody/cmd[1..23]")
    print("  Note:     base state uses odom_raw; base action uses twist/cmd")
    print(f"  Cameras:  {', '.join(camera_names)}")
    print(f"  Frozen:   {', '.join(frozen_fields) if frozen_fields else 'none'}")
    print(f"{'=' * 60}")

    successful = 0
    failed_bags = []
    for bag_idx, bag_path in enumerate(bag_paths, 1):
        task_label = resolve_task_label(bag_path, args.task)
        result = process_single_bag(
            bag_path,
            bag_idx,
            total_bags,
            task_label=task_label,
            fps=args.fps,
            img_size=img_size,
            camera_names=camera_names,
            center_crop_fraction=args.center_crop_fraction,
            max_frames=args.max_frames,
            frozen_indices=frozen_indices,
        )
        if result is not None:
            for frame in result:
                dataset.add_frame(frame)
            dataset.save_episode()
            successful += 1
            print(f"  [Bag {bag_idx}/{total_bags}] Saved episode: task={task_label}")
        else:
            failed_bags.append(bag_name(bag_path))
        del result

    dataset.finalize()

    total_elapsed = time.time() - total_start
    complete = successful == total_bags
    print(f"\n{'=' * 60}")
    print("Conversion complete!" if complete else "Conversion incomplete!")
    print(f"  Episodes: {successful}/{total_bags}")
    if failed_bags:
        print(f"  Failed:   {', '.join(failed_bags)}")
    print(f"  Time:     {total_elapsed:.1f}s")
    print(f"  Output:   {output_path}")
    print(f"{'=' * 60}")

    if not complete:
        raise SystemExit(
            f"Conversion failed for {len(failed_bags)} of {total_bags} bag(s)."
        )


if __name__ == "__main__":
    main()
