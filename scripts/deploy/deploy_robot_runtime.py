#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
ROS_ENV_VARS = {
    "AMENT_PREFIX_PATH",
    "COLCON_PREFIX_PATH",
    "CMAKE_PREFIX_PATH",
    "CYCLONEDDS_URI",
    "FASTDDS_DEFAULT_PROFILES_FILE",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "PKG_CONFIG_PATH",
    "RMW_IMPLEMENTATION",
    "ROS_AUTOMATIC_DISCOVERY_RANGE",
    "ROS_DISTRO",
    "ROS_DOMAIN_ID",
    "ROS_LOCALHOST_ONLY",
    "ROS_PACKAGE_PATH",
    "ROS_PYTHON_VERSION",
    "ROS_STATIC_PEERS",
    "ROS_VERSION",
}
ROS_PATH_MARKERS = ("/opt/ros/",)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def add_bool_flags(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def resolve_repo_path(path: Path, repo_root: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def checkpoint_exists(path: Path) -> bool:
    return (
        (path / "model.safetensors").is_file()
        or (path / "pretrained_model" / "model.safetensors").is_file()
        or (path / "checkpoints").is_dir()
    )


def executable_exists(executable: str) -> bool:
    return shutil.which(executable) is not None or Path(executable).expanduser().is_file()


def filter_ros_path_entries(value: str) -> str:
    entries = [
        entry
        for entry in value.split(os.pathsep)
        if entry and not any(marker in entry for marker in ROS_PATH_MARKERS)
    ]
    return os.pathsep.join(entries)


def worker_environment(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in ROS_ENV_VARS:
        env.pop(name, None)
    for name in ("PATH", "LD_LIBRARY_PATH"):
        if name not in env:
            continue
        filtered = filter_ros_path_entries(env[name])
        if filtered:
            env[name] = filtered
        else:
            env.pop(name, None)

    lerobot_src = repo_root / "third_party" / "lerobot" / "src"
    env["PYTHONPATH"] = str(lerobot_src)
    env.setdefault("HF_HOME", str(repo_root / ".hf_home"))
    env.setdefault("HF_LEROBOT_HOME", str(repo_root / ".hf_lerobot"))
    env.setdefault("HF_DATASETS_CACHE", str(Path(env["HF_HOME"]) / "datasets"))
    return env


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def add_worker_args(parser: argparse.ArgumentParser, default_run_id: str, default_port: int) -> None:
    repo_root = Path(os.environ.get("REPO_ROOT", DEFAULT_REPO_ROOT)).expanduser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--conda-bin", default=os.environ.get("CONDA_BIN", "conda"))
    parser.add_argument("--conda-env", default=os.environ.get("CONDA_ENV", "lerobot-qrp312"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", default_run_id))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--worker-host", default=os.environ.get("WORKER_HOST", "127.0.0.1"))
    parser.add_argument("--worker-port", type=int, default=int(os.environ.get("WORKER_PORT", default_port)))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("IMAGE_SIZE", "224")))
    parser.add_argument("--action-clip-margin", type=float, default=float(os.environ.get("ACTION_CLIP_MARGIN", "0.05")))
    add_bool_flags(parser, "use-amp", env_bool("USE_AMP", True), "Use CUDA autocast in the model worker.")
    add_bool_flags(parser, "clamp-actions", env_bool("CLAMP_ACTIONS", True), "Clamp actions to checkpoint action bounds.")


def parse_worker_args(robot: str, default_run_id: str, default_port: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Start {robot} ACT+DINOv3 model worker only.")
    add_worker_args(parser, default_run_id, default_port)
    return parser.parse_args()


def worker_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    repo_root = args.repo_root.expanduser().resolve()
    run_dir = repo_root / "outputs" / "train" / args.run_id
    checkpoint_path = (
        resolve_repo_path(args.checkpoint_path, repo_root)
        if args.checkpoint_path
        else run_dir / "checkpoints" / "100000" / "pretrained_model"
    )
    stats_path = resolve_repo_path(args.stats_path, repo_root) if args.stats_path else None
    return {
        "repo_root": repo_root,
        "deploy_dir": repo_root / "scripts" / "deploy",
        "checkpoint_path": checkpoint_path,
        "stats_path": stats_path,
    }


def worker_command(args: argparse.Namespace, paths: dict[str, Path | None]) -> list[str]:
    worker_script = paths["deploy_dir"] / "human_new_pick_act_dinov2_worker.py"
    cmd = [
        args.conda_bin,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(worker_script),
        "--host",
        args.worker_host,
        "--port",
        str(args.worker_port),
        "--checkpoint-path",
        str(paths["checkpoint_path"]),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--use-amp" if args.use_amp else "--no-use-amp",
        "--clamp-actions" if args.clamp_actions else "--no-clamp-actions",
        "--action-clip-margin",
        str(args.action_clip_margin),
    ]
    if paths["stats_path"] is not None:
        cmd.extend(["--stats-path", str(paths["stats_path"])])
    return cmd


def run_worker(robot: str, args: argparse.Namespace) -> int:
    paths = worker_paths(args)
    if not checkpoint_exists(paths["checkpoint_path"]):
        print(f"checkpoint not found: {paths['checkpoint_path']}", file=sys.stderr)
        return 1
    if paths["stats_path"] is not None and not paths["stats_path"].is_file():
        print(f"stats file not found: {paths['stats_path']}", file=sys.stderr)
        return 1
    if not executable_exists(args.conda_bin):
        print(
            f"conda executable not found: {args.conda_bin}. "
            "Set CONDA_BIN or pass --conda-bin /path/to/conda.",
            file=sys.stderr,
        )
        return 1

    print("=" * 60)
    print(f"Start {robot} ACT+DINOv3 model worker")
    print(f"checkpoint: {paths['checkpoint_path']}")
    print(f"normalizer: {'checkpoint processor files' if paths['stats_path'] is None else paths['stats_path']}")
    print(f"worker:     {args.worker_host}:{args.worker_port}")
    print("bridge:     start the ROS2 bridge in a separate terminal")
    print("=" * 60)

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(worker_command(args, paths), env=worker_environment(paths["repo_root"]))
        return proc.wait()
    except KeyboardInterrupt:
        terminate(proc)
        return 130


def add_bridge_args(parser: argparse.ArgumentParser, default_port: int) -> None:
    repo_root = Path(os.environ.get("REPO_ROOT", DEFAULT_REPO_ROOT)).expanduser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--ros-setup", type=Path, default=Path(os.environ.get("ROS_SETUP", "/opt/ros/humble/setup.bash")))
    parser.add_argument("--ros-python", default=os.environ.get("ROS_PYTHON", "/usr/bin/python3"))
    parser.add_argument("--worker-host", default=os.environ.get("WORKER_HOST", "127.0.0.1"))
    parser.add_argument("--worker-port", type=int, default=int(os.environ.get("WORKER_PORT", default_port)))
    parser.add_argument("--worker-timeout-s", type=float, default=float(os.environ.get("WORKER_TIMEOUT_S", "1.0")))
    add_bool_flags(
        parser,
        "publish-commands",
        env_bool("PUBLISH_COMMANDS", False),
        "Actually publish /zeno/h1/auto/wholebody/cmd. Default is dry-run.",
    )
    add_bool_flags(
        parser,
        "publish-idle-on-stale",
        env_bool("PUBLISH_IDLE_ON_STALE", True),
        "Publish idle command when observations are stale.",
    )
    parser.add_argument("--control-mode", type=float, default=float(os.environ.get("CONTROL_MODE", "1.0")))
    parser.add_argument("--rate-hz", type=float, default=float(os.environ.get("RATE_HZ", "20.0")))
    parser.add_argument("--max-obs-age-s", type=float, default=float(os.environ.get("MAX_OBS_AGE_S", "0.5")))
    parser.add_argument("--log-every-n", type=int, default=int(os.environ.get("LOG_EVERY_N", "20")))
    parser.add_argument("--cmd-topic", default=os.environ.get("CMD_TOPIC", "/zeno/h1/auto/wholebody/cmd"))
    parser.add_argument("--head-cam-topic", default=os.environ.get("HEAD_CAM_TOPIC", "/zeno/h1/sensor/head_cam/image/compressed"))
    parser.add_argument("--left-arm-cam-topic", default=os.environ.get("LEFT_ARM_CAM_TOPIC", "/zeno/h1/sensor/left_arm_cam/image/compressed"))
    parser.add_argument("--right-arm-cam-topic", default=os.environ.get("RIGHT_ARM_CAM_TOPIC", "/zeno/h1/sensor/right_arm_cam/image/compressed"))
    parser.add_argument("--odom-topic", default=os.environ.get("ODOM_TOPIC", "/zeno/h1/sensor/odom_raw"))
    parser.add_argument("--torso-state-topic", default=os.environ.get("TORSO_STATE_TOPIC", "/zeno/h1/wheelarm/torso/joint_state"))
    parser.add_argument("--left-arm-state-topic", default=os.environ.get("LEFT_ARM_STATE_TOPIC", "/zeno/h1/wheelarm/left_arm/joint_state"))
    parser.add_argument("--right-arm-state-topic", default=os.environ.get("RIGHT_ARM_STATE_TOPIC", "/zeno/h1/wheelarm/right_arm/joint_state"))
    parser.add_argument("--left-gripper-state-topic", default=os.environ.get("LEFT_GRIPPER_STATE_TOPIC", "/zeno/h1/left_gripper/joint_state"))
    parser.add_argument("--right-gripper-state-topic", default=os.environ.get("RIGHT_GRIPPER_STATE_TOPIC", "/zeno/h1/right_gripper/joint_state"))


def parse_bridge_args(robot: str, default_port: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Start {robot} ROS2 auto command bridge only.")
    add_bridge_args(parser, default_port)
    return parser.parse_args()


def bridge_command(robot: str, args: argparse.Namespace) -> list[str]:
    repo_root = args.repo_root.expanduser().resolve()
    bridge_script = repo_root / "scripts" / "deploy" / "human_new_pick_ros2_auto_cmd_bridge.py"
    tokens = [
        args.ros_python,
        str(bridge_script),
        "--ros-args",
        "-r",
        f"__node:={robot}_auto_cmd_bridge",
        "-p",
        f"worker_host:={args.worker_host}",
        "-p",
        f"worker_port:={args.worker_port}",
        "-p",
        f"worker_timeout_s:={args.worker_timeout_s}",
        "-p",
        f"publish_commands:={str(args.publish_commands).lower()}",
        "-p",
        f"publish_idle_on_stale:={str(args.publish_idle_on_stale).lower()}",
        "-p",
        f"control_mode:={args.control_mode}",
        "-p",
        f"rate_hz:={args.rate_hz}",
        "-p",
        f"max_obs_age_s:={args.max_obs_age_s}",
        "-p",
        f"log_every_n:={args.log_every_n}",
        "-p",
        f"cmd_topic:={args.cmd_topic}",
        "-p",
        f"head_cam_topic:={args.head_cam_topic}",
        "-p",
        f"left_arm_cam_topic:={args.left_arm_cam_topic}",
        "-p",
        f"right_arm_cam_topic:={args.right_arm_cam_topic}",
        "-p",
        f"odom_topic:={args.odom_topic}",
        "-p",
        f"torso_state_topic:={args.torso_state_topic}",
        "-p",
        f"left_arm_state_topic:={args.left_arm_state_topic}",
        "-p",
        f"right_arm_state_topic:={args.right_arm_state_topic}",
        "-p",
        f"left_gripper_state_topic:={args.left_gripper_state_topic}",
        "-p",
        f"right_gripper_state_topic:={args.right_gripper_state_topic}",
    ]
    command = f"source {shlex.quote(str(args.ros_setup))} && exec " + " ".join(shlex.quote(token) for token in tokens)
    return ["/bin/bash", "-lc", command]


def run_bridge(robot: str, args: argparse.Namespace) -> int:
    if not args.ros_setup.expanduser().is_file():
        print(f"ROS setup not found: {args.ros_setup}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(f"Start {robot} ROS2 auto command bridge")
    print(f"worker:    {args.worker_host}:{args.worker_port}")
    print(f"publish:   {args.publish_commands}")
    print(f"cmd_topic: {args.cmd_topic}")
    print("model:     start the ACT+DINOv3 worker in a separate terminal")
    print("=" * 60)
    if not args.publish_commands:
        print("DRY-RUN mode. Add --publish-commands only after checking topics and actions.")

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(bridge_command(robot, args))
        return proc.wait()
    except KeyboardInterrupt:
        terminate(proc)
        return 130
