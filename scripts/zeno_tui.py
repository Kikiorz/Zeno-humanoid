#!/usr/bin/env python3
"""Terminal menu for common Zeno humanoid workspace operations.

The menu intentionally uses only the Python standard library.  Add new
operations to ``MENU_ITEMS`` as the workspace tooling grows.
"""

from __future__ import annotations

import argparse
import curses
import ipaddress
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ROS_SETUP_SCRIPT = Path("/opt/ros/humble/setup.bash")
REPLAY_SCRIPT = WORKSPACE_ROOT / "scripts/replay/replay_zeno_episode.py"
ROSBAG_ROOT = WORKSPACE_ROOT / "data/rosbag"
LEROBOT_ROOT = WORKSPACE_ROOT / "data/lerobot"
CONDA_ENV_PATH = Path("/home/zeno-yanan22/miniconda3/envs/lerobot-qrp312")
SERIAL_CONVERTER_SCRIPT = WORKSPACE_ROOT / "scripts/data_convert/convert_zeno_h1_v30.py"
THREADED_CONVERTER_SCRIPT = WORKSPACE_ROOT / "scripts/data_convert/convert_zeno_h1_v30_threaded.py"
VALIDATOR_SCRIPT = WORKSPACE_ROOT / "scripts/data_convert/validate_lerobot_dataset.py"
PREFLIGHT_SCRIPT = WORKSPACE_ROOT / "scripts/data_convert/preflight_zeno_bags.py"
ROBOT8_3CAM_TRAIN_SCRIPT = WORKSPACE_ROOT / "scripts/train_robot8_20260708_20260709_act_dinov3_base_frozen_20260709.sh"
LEROBOT_SOURCE_ROOT = WORKSPACE_ROOT / "third_party/lerobot/src"
TRAIN_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs/train"
DEPLOY_ROOT = WORKSPACE_ROOT / "scripts/deploy"
CONVERSION_MANIFEST_NAME = ".zeno_conversion.json"
CANCELLED_EXIT_CODE = 125
ALL_BAGS_DISCARDED_EXIT_CODE = 126
ROBOT_HOST_ENV = "ZENO_ROBOT_HOST"
DEFAULT_ROBOT_HOST = "orangepi@192.168.10.108"
DEFAULT_ROBOT_PASSWORD = "orangepi"
MODE_REFRESH_INTERVAL_S = 1.0
MENU_INPUT_TIMEOUT_MS = 200
DEPLOYMENT_LOG_LIMIT = 500
DEFAULT_WORKER_HOST = "127.0.0.1"
DEFAULT_WORKER_PORT = 8768
THREE_CAMERA_NAMES = ("head_cam", "left_arm_cam", "right_arm_cam")
HEAD_RIGHT_CAMERA_NAMES = ("head_cam", "right_arm_cam")
AUTO_ROBOT_USER = "orangepi"
MAX_WIRED_NEIGHBORS = 8
MAX_WIRED_SCAN_HOSTS = 254
WIRED_SCAN_WORKERS = 64
WIRED_SCAN_TIMEOUT_S = 0.1
WIRED_SCAN_CACHE_S = 15.0

# The monitor runs this read-only probe on the Orange Pi. It reads the running
# driver's ROS networking environment, then samples the persistent robot mode,
# both arm joint arrays, torso motors, and chassis odometry in one ROS node.
ROBOT_MODE_REMOTE_COMMAND = r"""
driver_pid="$(pgrep -f '/zeno_h1_driver --ros-args' | head -n 1)"
if [ -z "$driver_pid" ]; then
    printf driver-offline
    exit 0
fi

driver_environment_value() {
    tr '\0' '\n' < "/proc/$driver_pid/environ" | sed -n "s/^$1=//p" | head -n 1
}

source /opt/ros/humble/setup.bash
source /home/orangepi/ros_workspace/ws_zeno_h1_driver/install/setup.bash
export ROS_DOMAIN_ID="$(driver_environment_value ROS_DOMAIN_ID)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
export ROS_LOCALHOST_ONLY="$(driver_environment_value ROS_LOCALHOST_ONLY)"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="$(driver_environment_value RMW_IMPLEMENTATION)"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="$(driver_environment_value CYCLONEDDS_URI)"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///home/orangepi/.ros/cyclonedds.xml}"

exec /usr/bin/python3 - <<'PY'
import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32
from zeno_h1_msgs.msg import ArmMotorsState, TorsoMotorState


def arm_positions(message):
    return [float(getattr(message, f"arm_j{index}").position) for index in range(7)]


def torso_positions(message):
    return {
        "lift": float(message.lift.position),
        "waist": float(message.waist.position),
        "head_pan": float(message.head_pan.position),
        "head_tilt": float(message.head_tilt.position),
    }


def base_state(message):
    orientation = message.pose.pose.orientation
    yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )
    twist = message.twist.twist
    position = message.pose.pose.position
    return {
        "x": float(position.x),
        "y": float(position.y),
        "yaw": float(yaw),
        "vx": float(twist.linear.x),
        "vy": float(twist.linear.y),
        "wz": float(twist.angular.z),
    }


rclpy.init()
node = Node("zeno_tui_telemetry_probe")
samples = {}
cameras = {}
camera_counts = {}
camera_first_at = {}
node.create_subscription(Int32, "/zeno/h1/robot_model_status", lambda msg: samples.__setitem__("mode", int(msg.data)), 10)
node.create_subscription(ArmMotorsState, "/zeno/h1/wheelarm/left_arm/motor_state", lambda msg: samples.__setitem__("left_arm", arm_positions(msg)), 10)
node.create_subscription(ArmMotorsState, "/zeno/h1/wheelarm/right_arm/motor_state", lambda msg: samples.__setitem__("right_arm", arm_positions(msg)), 10)
node.create_subscription(TorsoMotorState, "/zeno/h1/wheelarm/torso/motor_state", lambda msg: samples.__setitem__("torso", torso_positions(msg)), 10)
node.create_subscription(Odometry, "/zeno/h1/sensor/odom_raw", lambda msg: samples.__setitem__("base", base_state(msg)), 10)


def camera_sample(name, message):
    # Only retain metadata.  Image bytes are deliberately not sent over SSH.
    now = time.monotonic()
    camera_counts[name] = camera_counts.get(name, 0) + 1
    first_at = camera_first_at.setdefault(name, now)
    elapsed = now - first_at
    frame_rate_hz = (camera_counts[name] - 1) / elapsed if elapsed > 0.0 else None
    cameras[name] = {
        "bytes": len(message.data),
        "format": str(message.format),
        "hz": frame_rate_hz,
    }


camera_topics = {
    "head_cam": "/zeno/h1/sensor/head_cam/image/compressed",
    "left_arm_cam": "/zeno/h1/sensor/left_arm_cam/image/compressed",
    "right_arm_cam": "/zeno/h1/sensor/right_arm_cam/image/compressed",
}
for camera_name, topic in camera_topics.items():
    node.create_subscription(
        CompressedImage,
        topic,
        lambda message, name=camera_name: camera_sample(name, message),
        10,
    )

deadline = time.monotonic() + 3.0
required = {"mode", "left_arm", "right_arm", "torso", "base"}
state_ready_at = None
try:
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if not required.issubset(samples):
            continue
        if state_ready_at is None:
            state_ready_at = time.monotonic()
        # Sample a full second after state becomes available.  That is long
        # enough to derive a useful per-camera frame rate without allowing a
        # missing image topic to hold the monitor until its full timeout.
        if time.monotonic() - state_ready_at >= 1.0:
            break
    samples["missing"] = sorted(required.difference(samples))
    samples["cameras"] = cameras
    print(json.dumps(samples, separators=(",", ":"), allow_nan=False))
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
""".strip()

# A TUI deployment owns the worker and bridge together.  The worker is put in
# its own process session, so this shell can terminate the model process when
# bridge exits or the operator stops deployment with Ctrl-C.  File paths are
# positional arguments rather than interpolated into this shell program.
DEPLOYMENT_LAUNCH_SCRIPT = r"""
set -e
source "$1"

worker_pid=""
cleanup() {
    if [ -n "$worker_pid" ] && kill -0 "$worker_pid" 2>/dev/null; then
        kill -TERM -- "-$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

printf '%s\n' "正在启动模型 worker，首次加载 checkpoint 可能需要一些时间……"
setsid conda run --no-capture-output -p "$2" python "$3" --host "$5" --port "$6" &
worker_pid="$!"

bridge_args=(--worker-host "$5" --worker-port "$6" --log-full-action)
if [ "$7" = "publish" ]; then
    bridge_args+=(--publish-commands)
fi
printf '%s\n' "正在启动 ROS2 bridge；按 Ctrl-C 停止 bridge 和 worker。"
/usr/bin/python3 "$4" "${bridge_args[@]}"
""".strip()


@dataclass(frozen=True)
class MenuItem:
    """One selectable operation in the terminal UI."""

    label: str
    description: str
    action: Callable[..., int] | None = None
    uses_curses: bool = False


@dataclass(frozen=True)
class ConversionDataset:
    """A selectable directory containing one or more ROS bag episodes."""

    path: Path
    bag_count: int


@dataclass(frozen=True)
class Choice:
    """One item in a curses sub-menu."""

    label: str
    description: str
    value: object


@dataclass(frozen=True)
class CommandResult:
    """Exit state and captured output from a command rendered inside the TUI."""

    returncode: int
    lines: list[str]


@dataclass(frozen=True)
class RobotModeStatus:
    """A read-only, best-effort view of the robot driver mode."""

    state: str
    label: str
    detail: str
    telemetry: "RobotTelemetry | None" = None


@dataclass(frozen=True)
class CameraStatus:
    """One compressed camera's most recent callback during a probe."""

    name: str
    received: bool
    byte_count: int | None = None
    image_format: str = ""
    frame_rate_hz: float | None = None


@dataclass(frozen=True)
class RobotTelemetry:
    """Latest position and odometry values sampled from the robot driver."""

    left_arm: tuple[float, ...] | None = None
    right_arm: tuple[float, ...] | None = None
    torso: tuple[float, float, float, float] | None = None
    base: tuple[float, float, float, float, float, float] | None = None
    cameras: tuple[CameraStatus, ...] | None = None


class RobotModeMonitor:
    """Refresh robot mode in the background without blocking curses input."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = RobotModeStatus("loading", "读取中", "正在读取机器人状态")
        self._refreshing = False
        self._next_refresh_at = 0.0
        self._updated_at: float | None = None

    def refresh_if_due(self, *, force: bool = False) -> bool:
        """Start one query when idle and due, returning whether it was started."""

        with self._lock:
            if self._refreshing:
                return False
            if not force and time.monotonic() < self._next_refresh_at:
                return False
            self._refreshing = True

        thread = threading.Thread(target=self._refresh, name="zeno-mode-monitor", daemon=True)
        thread.start()
        return True

    def snapshot(self) -> tuple[RobotModeStatus, float | None]:
        """Return the latest completed status and its age in seconds."""

        with self._lock:
            age = None if self._updated_at is None else max(0.0, time.monotonic() - self._updated_at)
            return self._status, age

    def _refresh(self) -> None:
        try:
            status = detect_robot_mode()
        except Exception:
            status = RobotModeStatus("unavailable", "不可达", "后台模式查询失败")
        with self._lock:
            self._status = status
            self._refreshing = False
            self._updated_at = time.monotonic()
            self._next_refresh_at = time.monotonic() + MODE_REFRESH_INTERVAL_S


@dataclass(frozen=True)
class ExistingConversion:
    """A completed LeRobot dataset associated with a selected ROS source."""

    path: Path
    summary: str


@dataclass(frozen=True)
class PreflightBag:
    """One inspected Bag, represented by its converter-compatible exclusion name."""

    index: int
    name: str
    expected_frames: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TrainingDataset:
    """A finalized local LeRobot dataset selectable for policy training."""

    path: Path
    summary: str
    camera_count: int


@dataclass(frozen=True)
class TrainingRecipe:
    """One curated policy configuration exposed by the training TUI."""

    key: str
    label: str
    description: str
    run_label: str


@dataclass(frozen=True)
class DeploymentProfile:
    """One worker and ROS bridge pair selectable from the deployment page."""

    key: str
    label: str
    description: str
    worker_script: Path
    bridge_script: Path
    camera_count: int


BACK = object()
QUIT_TO_MENU = object()
_auto_discovered_robot_host: str | None = None
_last_wired_scan_at = 0.0
_cached_wired_ssh_hosts: tuple[str, ...] = ()


def active_wired_interfaces() -> tuple[str, ...]:
    """Return active Ethernet interfaces without selecting Wi-Fi or loopback."""

    interfaces: list[str] = []
    network_root = Path("/sys/class/net")
    try:
        candidates = tuple(network_root.iterdir())
    except OSError:
        return ()

    for interface in candidates:
        if (interface / "wireless").exists():
            continue
        try:
            if (interface / "type").read_text(encoding="utf-8").strip() != "1":
                continue
            carrier = (interface / "carrier").read_text(encoding="utf-8").strip()
            if carrier == "1":
                interfaces.append(interface.name)
        except OSError:
            continue
    return tuple(sorted(interfaces))


def wired_ipv4_networks(interfaces: tuple[str, ...]) -> tuple[ipaddress.IPv4Network, ...]:
    """Read the directly connected Ethernet IPv4 ranges from ``ip``."""

    networks: list[ipaddress.IPv4Network] = []
    for interface in interfaces:
        try:
            completed = subprocess.run(
                ["ip", "-j", "-4", "addr", "show", "dev", interface],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.5,
                check=False,
            )
            entries = json.loads(completed.stdout) if completed.returncode == 0 else []
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
            continue
        for entry in entries:
            for address in entry.get("addr_info", []):
                if address.get("family") != "inet" or address.get("scope") != "global":
                    continue
                try:
                    network = ipaddress.ip_interface(
                        f"{address['local']}/{address['prefixlen']}"
                    ).network
                except (KeyError, ValueError):
                    continue
                # A /16 or wider network could contain thousands of unrelated
                # hosts. Probe only the local /24 around this computer instead.
                if network.num_addresses > MAX_WIRED_SCAN_HOSTS + 2:
                    network = ipaddress.ip_interface(f"{address['local']}/24").network
                networks.append(network)
    return tuple(dict.fromkeys(networks))


def is_ssh_port_open(address: ipaddress.IPv4Address) -> bool:
    """Perform one short TCP probe; no credentials or commands are sent."""

    try:
        with socket.create_connection((str(address), 22), timeout=WIRED_SCAN_TIMEOUT_S):
            return True
    except OSError:
        return False


def discover_wired_ssh_hosts(interfaces: tuple[str, ...]) -> tuple[str, ...]:
    """Discover SSH peers in the local wired range, with a short-lived cache."""

    global _last_wired_scan_at, _cached_wired_ssh_hosts

    now = time.monotonic()
    if now - _last_wired_scan_at < WIRED_SCAN_CACHE_S:
        return _cached_wired_ssh_hosts
    _last_wired_scan_at = now

    addresses: list[ipaddress.IPv4Address] = []
    for network in wired_ipv4_networks(interfaces):
        addresses.extend(network.hosts())
        if len(addresses) >= MAX_WIRED_SCAN_HOSTS:
            break
    addresses = addresses[:MAX_WIRED_SCAN_HOSTS]
    if not addresses:
        _cached_wired_ssh_hosts = ()
        return ()

    with ThreadPoolExecutor(max_workers=min(WIRED_SCAN_WORKERS, len(addresses))) as executor:
        open_addresses = [
            address
            for address, is_open in zip(addresses, executor.map(is_ssh_port_open, addresses))
            if is_open
        ]
    _cached_wired_ssh_hosts = tuple(f"{AUTO_ROBOT_USER}@{address}" for address in open_addresses)
    return _cached_wired_ssh_hosts


def auto_robot_host_candidates() -> tuple[str, ...]:
    """Find likely Orange Pi SSH targets on the connected wired network.

    ``orangepi.local`` uses local mDNS when it is available. Existing neighbour
    entries are tried next. If that has no answer, the local IPv4 range of the
    wired interface is probed for SSH, with at most 254 addresses.
    """

    interfaces = active_wired_interfaces()
    if not interfaces:
        return ()

    candidates = [f"{AUTO_ROBOT_USER}@orangepi.local"]
    for interface in interfaces:
        if len(candidates) >= MAX_WIRED_NEIGHBORS + 1:
            break
        try:
            completed = subprocess.run(
                ["ip", "neigh", "show", "dev", interface],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if completed.returncode != 0:
            continue
        for line in completed.stdout.splitlines():
            match = re.match(r"^(\d{1,3}(?:\.\d{1,3}){3})\s+", line)
            if match and "FAILED" not in line:
                candidates.append(f"{AUTO_ROBOT_USER}@{match.group(1)}")
            if len(candidates) >= MAX_WIRED_NEIGHBORS + 1:
                break

    candidates.extend(discover_wired_ssh_hosts(interfaces))

    return tuple(dict.fromkeys(candidates))


def query_robot_mode(host: str) -> RobotModeStatus:
    """Query one SSH target for its published mode and live motion state."""

    askpass_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="zeno_tui_askpass_",
            delete=False,
            encoding="utf-8",
        ) as askpass_file:
            askpass_file.write("#!/bin/sh\nprintf '%s\\n' \"$ZENO_TUI_SSH_PASSWORD\"\n")
            askpass_path = Path(askpass_file.name)
        askpass_path.chmod(0o700)
        environment = os.environ.copy()
        environment.update(
            {
                "DISPLAY": "zeno-tui",
                "SSH_ASKPASS": str(askpass_path),
                "SSH_ASKPASS_REQUIRE": "force",
                "ZENO_TUI_SSH_PASSWORD": DEFAULT_ROBOT_PASSWORD,
            }
        )
        completed = subprocess.run(
            [
                "setsid",
                "--wait",
                "ssh",
                "-o",
                "BatchMode=no",
                "-o",
                "NumberOfPasswordPrompts=1",
                "-o",
                "ConnectTimeout=1",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "LogLevel=ERROR",
                host,
                ROBOT_MODE_REMOTE_COMMAND,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12.0,
            check=False,
            env=environment,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return RobotModeStatus("unavailable", "不可达", "SSH 状态查询失败")
    finally:
        if askpass_path is not None:
            try:
                askpass_path.unlink()
            except OSError:
                pass

    if completed.returncode != 0:
        error = completed.stderr.lower()
        if "permission denied" in error:
            return RobotModeStatus(
                "authentication-required",
                "SSH 需认证",
                f"为 {host} 配置 SSH 公钥后可自动监测",
            )
        if "host key verification failed" in error:
            return RobotModeStatus(
                "host-key-unverified",
                "SSH 主机密钥未确认",
                "主机密钥与已记录的 Orange Pi 不一致，请人工确认",
            )
        return RobotModeStatus("unavailable", "不可达", "SSH 状态查询失败")

    if completed.stdout.strip() == "driver-offline":
        return RobotModeStatus("offline", "驱动未运行", "未检测到 zeno_h1_driver 进程")

    payload: dict[str, object] | None = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        return RobotModeStatus("unavailable", "模式状态不可读", "未收到机器人遥测数据")

    telemetry = telemetry_from_payload(payload)
    mode = payload.get("mode")
    if mode == 1:
        return RobotModeStatus(
            state="work",
            label="工作模式",
            detail="/zeno/h1/robot_model_status = 1",
            telemetry=telemetry,
        )
    if mode == 0:
        return RobotModeStatus(
            state="stowed",
            label="收纳模式",
            detail="/zeno/h1/robot_model_status = 0（双臂零力矩）",
            telemetry=telemetry,
        )
    missing = payload.get("missing")
    detail = "未收到 /zeno/h1/robot_model_status"
    if isinstance(missing, list) and missing:
        detail = f"缺少遥测：{', '.join(str(item) for item in missing)}"
    return RobotModeStatus("unavailable", "模式状态不可读", detail, telemetry=telemetry)


def finite_vector(value: object, expected_size: int) -> tuple[float, ...] | None:
    """Convert a JSON numeric array to a finite, fixed-size tuple."""

    if not isinstance(value, list) or len(value) != expected_size:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(item == item and abs(item) != float("inf") for item in result):
        return None
    return result


def camera_statuses_from_payload(payload: dict[str, object]) -> tuple[CameraStatus, ...] | None:
    """Parse compact camera metadata from the remote ROS probe."""

    camera_data = payload.get("cameras")
    if not isinstance(camera_data, dict):
        return None

    statuses: list[CameraStatus] = []
    for name in ("head_cam", "left_arm_cam", "right_arm_cam"):
        sample = camera_data.get(name)
        if not isinstance(sample, dict):
            statuses.append(CameraStatus(name=name, received=False))
            continue
        byte_count = sample.get("bytes")
        if not isinstance(byte_count, int) or byte_count < 0:
            byte_count = None
        image_format = sample.get("format")
        frame_rate_hz = sample.get("hz")
        if isinstance(frame_rate_hz, bool) or not isinstance(frame_rate_hz, (int, float)):
            frame_rate_hz = None
        elif not (0.0 <= float(frame_rate_hz) <= 1000.0):
            frame_rate_hz = None
        statuses.append(
            CameraStatus(
                name=name,
                received=True,
                byte_count=byte_count,
                image_format=str(image_format) if image_format is not None else "",
                frame_rate_hz=float(frame_rate_hz) if frame_rate_hz is not None else None,
            )
        )
    return tuple(statuses)


def telemetry_from_payload(payload: dict[str, object]) -> RobotTelemetry | None:
    """Parse the remote ROS probe's JSON payload without trusting its shape."""

    torso_data = payload.get("torso")
    torso = None
    if isinstance(torso_data, dict):
        try:
            torso_values = tuple(
                float(torso_data[field]) for field in ("lift", "waist", "head_pan", "head_tilt")
            )
        except (KeyError, TypeError, ValueError):
            torso_values = ()
        if len(torso_values) == 4 and all(value == value and abs(value) != float("inf") for value in torso_values):
            torso = torso_values

    base_data = payload.get("base")
    base = None
    if isinstance(base_data, dict):
        try:
            base_values = tuple(float(base_data[field]) for field in ("x", "y", "yaw", "vx", "vy", "wz"))
        except (KeyError, TypeError, ValueError):
            base_values = ()
        if len(base_values) == 6 and all(value == value and abs(value) != float("inf") for value in base_values):
            base = base_values

    telemetry = RobotTelemetry(
        left_arm=finite_vector(payload.get("left_arm"), 7),
        right_arm=finite_vector(payload.get("right_arm"), 7),
        torso=torso,
        base=base,
        cameras=camera_statuses_from_payload(payload),
    )
    return (
        telemetry
        if any((telemetry.left_arm, telemetry.right_arm, telemetry.torso, telemetry.base, telemetry.cameras is not None))
        else None
    )


def detect_robot_mode(host: str | None = None) -> RobotModeStatus:
    """Read the configured Orange Pi driver's published work/stow mode."""

    global _auto_discovered_robot_host

    configured_host = host if host is not None else os.environ.get(ROBOT_HOST_ENV)
    return query_robot_mode(configured_host or DEFAULT_ROBOT_HOST)


def robot_mode_attribute(mode: RobotModeStatus) -> int:
    """Return the display attribute for the current mode state."""

    if mode.state == "work":
        return curses.color_pair(4) | curses.A_BOLD
    if mode.state in {"stowed", "loading"}:
        return curses.color_pair(5) | curses.A_BOLD
    return curses.color_pair(3) | curses.A_BOLD


def format_joint_vector(values: tuple[float, ...] | None) -> str:
    """Format one arm's joint positions compactly for a terminal row."""

    if values is None:
        return "--"
    return "[" + ", ".join(f"{value:+.3f}" for value in values) + "] rad"


def format_camera_status(camera: CameraStatus) -> str:
    """Format one camera's no-image metadata without expanding the TUI row."""

    labels = {
        "head_cam": "头部",
        "left_arm_cam": "左臂",
        "right_arm_cam": "右臂",
    }
    label = labels.get(camera.name, camera.name)
    if not camera.received:
        return f"{label}：未收到图像"
    byte_text = "--" if camera.byte_count is None else f"{camera.byte_count / 1024:.0f}KB"
    format_text = camera.image_format or "compressed"
    rate_text = "-- Hz" if camera.frame_rate_hz is None else f"{camera.frame_rate_hz:.1f} Hz"
    return f"{label}：正常 {format_text} {byte_text} {rate_text}"


def camera_status_attribute(telemetry: RobotTelemetry | None) -> int:
    """Highlight missing camera frames without obscuring normal telemetry."""

    if telemetry is None or telemetry.cameras is None:
        return curses.color_pair(5) | curses.A_BOLD
    if any(not camera.received for camera in telemetry.cameras):
        return curses.color_pair(3) | curses.A_BOLD
    if any(camera.frame_rate_hz is None for camera in telemetry.cameras):
        return curses.color_pair(5) | curses.A_BOLD
    return curses.A_DIM


def telemetry_display_lines(telemetry: RobotTelemetry | None) -> tuple[str, ...]:
    """Build the compact camera, upper-body, torso, and base display."""

    if telemetry is None:
        return (
            "相机：等待图像状态",
            "上肢左臂 J0-J6: --",
            "上肢右臂 J0-J6: --",
            "躯干：lift=--  waist=--  head_pan=--  head_tilt=--",
            "底盘：x=--  y=--  yaw=-- | vx=--  vy=--  wz=--",
        )

    cameras = telemetry.cameras
    if cameras is None:
        camera_line = "相机：未收到相机遥测"
    else:
        camera_line = "相机：" + " | ".join(format_camera_status(camera) for camera in cameras)

    lines = [
        camera_line,
        f"上肢左臂 J0-J6: {format_joint_vector(telemetry.left_arm)}",
        f"上肢右臂 J0-J6: {format_joint_vector(telemetry.right_arm)}",
    ]
    if telemetry.torso is None:
        lines.append("躯干：lift=--  waist=--  head_pan=--  head_tilt=--")
    else:
        lift, waist, head_pan, head_tilt = telemetry.torso
        lines.append(
            f"躯干：lift={lift:+.3f}  waist={waist:+.3f}  "
            f"head_pan={head_pan:+.3f}  head_tilt={head_tilt:+.3f}"
        )
    if telemetry.base is None:
        lines.append("底盘：x=--  y=--  yaw=-- | vx=--  vy=--  wz=--")
    else:
        x, y, yaw, vx, vy, wz = telemetry.base
        lines.append(
            f"底盘：x={x:+.3f}m  y={y:+.3f}m  yaw={yaw:+.3f}rad | "
            f"vx={vx:+.3f}m/s  vy={vy:+.3f}m/s  wz={wz:+.3f}rad/s"
        )
    return tuple(lines)


def run_cyclonedds_setup() -> int:
    """Run the existing interactive CycloneDDS setup script unchanged."""

    script = WORKSPACE_ROOT / "setup_cyclone_perm.sh"
    if not script.is_file():
        print(f"Error: setup script not found: {script}", file=sys.stderr)
        return 1

    # The setup script opens a new interactive shell when it is executed with
    # ``bash script.sh``.  Source it from this short-lived child shell instead:
    # its prompts still run normally, while a completed operation returns here
    # and the TUI can immediately redraw.  ``$1`` is a positional parameter,
    # so the resolved path is never interpolated into shell code.
    completed = subprocess.run(
        ["bash", "-c", 'source "$1"', "bash", str(script)],
        cwd=WORKSPACE_ROOT,
        check=False,
    )
    return completed.returncode


def draw_choice_menu(
    screen: curses.window,
    title: str,
    subtitle: str,
    choices: list[Choice],
    selected: int,
    status: str = "",
    *,
    allow_back: bool = True,
) -> None:
    """Render a scrollable sub-menu with the same appearance as the main menu."""

    screen.erase()
    rows, columns = screen.getmaxyx()
    footer = "Use 1-9 or Up/Down and Enter. b: back; q: main menu." if allow_back else (
        "Use 1-9 or Up/Down and Enter. q: main menu."
    )
    try:
        screen.addnstr(1, 2, title, max(columns - 4, 0), curses.A_BOLD)
        screen.addnstr(2, 2, subtitle, max(columns - 4, 0), curses.A_DIM)

        visible_count = max(1, (rows - 9) // 3)
        first = min(max(0, selected - visible_count // 2), max(0, len(choices) - visible_count))
        last = min(len(choices), first + visible_count)
        for index in range(first, last):
            choice = choices[index]
            row = 5 + (index - first) * 3
            attribute = curses.color_pair(2) | curses.A_BOLD if index == selected else curses.color_pair(1)
            screen.addnstr(row, 4, f"{index + 1}. {choice.label}", max(columns - 8, 0), attribute)
            screen.addnstr(row + 1, 6, choice.description, max(columns - 10, 0), curses.A_DIM)

        if len(choices) > visible_count:
            screen.addnstr(rows - 4, 2, f"Showing {first + 1}-{last} of {len(choices)} options.", max(columns - 4, 0), curses.A_DIM)
        screen.addnstr(rows - 3, 2, footer, max(columns - 4, 0), curses.A_DIM)
        if status:
            screen.addnstr(rows - 2, 2, status, max(columns - 4, 0), curses.A_BOLD)
    except curses.error:
        pass
    screen.refresh()


def select_choice(
    screen: curses.window,
    title: str,
    subtitle: str,
    choices: list[Choice],
    *,
    allow_back: bool = True,
) -> object:
    """Choose a numbered curses menu option, go back, or return to the main menu."""

    if not choices:
        return QUIT_TO_MENU

    selected = 0
    while True:
        draw_choice_menu(screen, title, subtitle, choices, selected, allow_back=allow_back)
        key = screen.getch()
        if key in (ord("q"), ord("Q"), 27):
            return QUIT_TO_MENU
        if allow_back and key in (ord("b"), ord("B")):
            return BACK
        if ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(choices):
                return choices[index].value
        elif key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(choices)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(choices)
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return choices[selected].value


def prompt_text(
    screen: curses.window,
    title: str,
    subtitle: str,
    prompt: str,
    *,
    default: str | None = None,
    validation_message: str = "",
) -> object:
    """Show a styled text field; exact ``b`` goes back and exact ``q`` returns home."""

    status = validation_message
    while True:
        screen.erase()
        rows, columns = screen.getmaxyx()
        default_text = f"Default: {default}" if default is not None else "A value is required."
        try:
            screen.addnstr(1, 2, title, max(columns - 4, 0), curses.A_BOLD)
            screen.addnstr(2, 2, subtitle, max(columns - 4, 0), curses.A_DIM)
            screen.addnstr(5, 2, default_text, max(columns - 4, 0), curses.A_DIM)
            screen.addnstr(7, 2, prompt, max(columns - 4, 0), curses.color_pair(1))
            screen.addnstr(rows - 3, 2, "Enter: continue; b: back; q: main menu.", max(columns - 4, 0), curses.A_DIM)
            if status:
                screen.addnstr(rows - 2, 2, status, max(columns - 4, 0), curses.A_BOLD)
            screen.move(8, 2)
            screen.clrtoeol()
            screen.addstr(8, 2, "> ")
            screen.refresh()
        except curses.error:
            pass

        curses.curs_set(1)
        curses.echo()
        try:
            raw = screen.getstr(8, 4, max(columns - 6, 1)).decode(errors="replace").strip()
        except curses.error:
            raw = ""
        finally:
            curses.noecho()
            curses.curs_set(0)

        lowered = raw.lower()
        if lowered in {"q", "quit"}:
            return QUIT_TO_MENU
        if lowered in {"b", "back"}:
            return BACK
        if raw:
            return raw
        if default is not None:
            return default
        status = "Please enter a value, or press b/q."


def run_terminal_command(screen: curses.window, command: list[str], heading: str) -> int:
    """Temporarily hand the terminal to a long-running child command."""

    curses.def_prog_mode()
    curses.endwin()
    try:
        print(f"\n{heading}\n")
        return subprocess.run(command, cwd=WORKSPACE_ROOT, check=False).returncode
    finally:
        input("\nPress Enter to return to the menu...")
        curses.reset_prog_mode()
        screen.refresh()


def is_problem_line(line: str) -> bool:
    """Identify lines that should be red while a conversion is running."""

    lowered = line.lower()
    if line.startswith("[error]") or "traceback" in lowered:
        return True
    if any(token in lowered for token in (" error", "failed", "exception", "missing required")):
        return True
    match = re.search(r"dropped-images=(\d+)", lowered)
    return match is not None and int(match.group(1)) > 0


def draw_process_output(screen: curses.window, title: str, lines: list[str], status: str = "") -> None:
    """Render live command output in the same terminal style as menu pages."""

    screen.erase()
    rows, columns = screen.getmaxyx()
    try:
        screen.addnstr(1, 2, title, max(columns - 4, 0), curses.A_BOLD)
        screen.addnstr(2, 2, "Live output — please wait for this operation to finish.", max(columns - 4, 0), curses.A_DIM)
        visible_count = max(1, rows - 8)
        for offset, line in enumerate(lines[-visible_count:]):
            attribute = curses.color_pair(3) | curses.A_BOLD if is_problem_line(line) else curses.color_pair(1)
            screen.addnstr(4 + offset, 2, line, max(columns - 4, 0), attribute)
        if status:
            attribute = curses.color_pair(3) | curses.A_BOLD if is_problem_line(status) else curses.A_BOLD
            screen.addnstr(rows - 2, 2, status, max(columns - 4, 0), attribute)
    except curses.error:
        pass
    screen.refresh()


def run_curses_command(
    screen: curses.window,
    command: list[str],
    title: str,
    environment_overrides: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command and stream stdout/stderr into the TUI."""

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if environment_overrides:
        environment.update(environment_overrides)
    process = subprocess.Popen(
        command,
        cwd=WORKSPACE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    lines: list[str] = []
    assert process.stdout is not None
    while True:
        raw = process.stdout.readline()
        if raw:
            fresh_lines = [line.strip() for line in raw.replace("\r", "\n").splitlines() if line.strip()]
            lines.extend(fresh_lines)
            del lines[:-500]
            draw_process_output(screen, title, lines)
            continue
        if process.poll() is not None:
            break
    returncode = process.wait()
    draw_process_output(screen, title, lines, f"Command finished with exit code {returncode}.")
    return CommandResult(returncode=returncode, lines=lines)


DEPLOYMENT_ROOT = WORKSPACE_ROOT / "scripts" / "deploy"
DEPLOYMENT_PROFILES = (
    DeploymentProfile(
        key="robot8_20260716_3cam",
        label="Robot10 · 2026-07-16 三相机（当前）",
        description="ACT + DINOv3 Base（冻结），100k checkpoint，居中裁剪 2/3。",
        worker_script=DEPLOYMENT_ROOT / "robot8_20260716_3cam_act_dinov3_base_frozen_100k" / "worker.py",
        bridge_script=DEPLOYMENT_ROOT / "robot8_20260716_3cam_act_dinov3_base_frozen_100k" / "bridge.py",
        camera_count=3,
    ),
    DeploymentProfile(
        key="robot8_20260709_head_right",
        label="Robot10 · 2026-07-09 头部+右臂",
        description="ACT + DINOv3 Base（冻结），20k checkpoint，使用头部与右臂两路相机。",
        worker_script=DEPLOYMENT_ROOT / "robot8_20260709_head_right_act_dinov3_base_frozen_20k" / "worker.py",
        bridge_script=DEPLOYMENT_ROOT / "robot8_20260709_head_right_act_dinov3_base_frozen_20k" / "bridge.py",
        camera_count=2,
    ),
    DeploymentProfile(
        key="robot8_base_3cam",
        label="Robot10 · 基线三相机",
        description="ACT + DINOv3 Base（冻结），100k checkpoint，使用三路相机。",
        worker_script=DEPLOYMENT_ROOT / "robot8_act_dinov3_base_frozen_100k" / "worker.py",
        bridge_script=DEPLOYMENT_ROOT / "robot8_act_dinov3_base_frozen_100k" / "bridge.py",
        camera_count=3,
    ),
    DeploymentProfile(
        key="robot4_base_3cam",
        label="Robot4 · 基线三相机",
        description="ACT + DINOv3 Base（冻结），100k checkpoint，使用三路相机。",
        worker_script=DEPLOYMENT_ROOT / "robot4_act_dinov3_base_frozen_100k" / "worker.py",
        bridge_script=DEPLOYMENT_ROOT / "robot4_act_dinov3_base_frozen_100k" / "bridge.py",
        camera_count=3,
    ),
    DeploymentProfile(
        key="robot5_base_3cam",
        label="Robot5 · 基线三相机",
        description="ACT + DINOv3 Base（冻结），100k checkpoint，使用三路相机。",
        worker_script=DEPLOYMENT_ROOT / "robot5_act_dinov3_base_frozen_100k" / "worker.py",
        bridge_script=DEPLOYMENT_ROOT / "robot5_act_dinov3_base_frozen_100k" / "bridge.py",
        camera_count=3,
    ),
)


def available_deployment_profiles() -> tuple[DeploymentProfile, ...]:
    """Return only deployment pairs that are complete in this checkout."""

    return tuple(
        profile
        for profile in DEPLOYMENT_PROFILES
        if profile.worker_script.is_file() and profile.bridge_script.is_file()
    )


def select_deployment_profile(screen: curses.window) -> object:
    """Select a checked-in worker/bridge deployment pair."""

    profiles = available_deployment_profiles()
    return select_choice(
        screen,
        "部署策略",
        "选择要同时启动的推理 worker 与 ROS bridge。",
        [
            Choice(
                profile.label,
                f"{profile.description} 需要 {profile.camera_count} 路相机。",
                profile,
            )
            for profile in profiles
        ],
    )


def prompt_worker_port(screen: curses.window, validation_message: str = "") -> object:
    """Ask for the local TCP port shared by the paired worker and bridge."""

    status = validation_message
    while True:
        selected = prompt_text(
            screen,
            "部署服务端口",
            "worker 和 bridge 在本机通过 127.0.0.1 通信。",
            "端口号：",
            default=str(DEFAULT_WORKER_PORT),
            validation_message=status,
        )
        if selected in (BACK, QUIT_TO_MENU):
            return selected
        value = str(selected)
        if value.isdigit() and 1 <= int(value) <= 65535:
            return int(value)
        status = "请输入 1 到 65535 之间的端口号。"


def choose_deployment_mode(screen: curses.window, profile: DeploymentProfile) -> object:
    """Make command publication an explicit deployment choice."""

    return select_choice(
        screen,
        "选择部署模式",
        f"{profile.label}：bridge 将订阅 {profile.camera_count} 路相机。",
        [
            Choice(
                "预览模式",
                "worker 正常推理；bridge 只记录动作，不向机器人发布命令。",
                False,
            ),
            Choice(
                "真机发布模式",
                "bridge 会传入 --publish-commands 并向机器人发布控制指令。",
                True,
            ),
        ],
    )


def review_deployment(
    screen: curses.window,
    profile: DeploymentProfile,
    worker_port: int,
    publish_commands: bool,
) -> object:
    """Show the exact deployment effect before spawning either process."""

    mode = "真机发布" if publish_commands else "预览"
    action_description = (
        "启动后 bridge 会向 /zeno/h1/auto/wholebody/cmd 发布真实控制指令。"
        if publish_commands
        else "启动后仅执行推理与日志记录，不发布真实控制指令。"
    )
    return select_choice(
        screen,
        "确认部署",
        f"{profile.label} | 本机端口 {worker_port} | {mode}模式",
        [Choice("启动 worker 与 bridge", action_description, "start")],
    )


def build_deployment_commands(
    profile: DeploymentProfile,
    worker_port: int,
    publish_commands: bool,
) -> tuple[list[str], list[str]]:
    """Build paired local commands without requiring a sourced parent shell."""

    worker_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-p",
        str(CONDA_ENV_PATH),
        "python",
        str(profile.worker_script),
        "--host",
        DEFAULT_WORKER_HOST,
        "--port",
        str(worker_port),
    ]
    bridge_python_command = [
        "/usr/bin/python3",
        str(profile.bridge_script),
        "--worker-host",
        DEFAULT_WORKER_HOST,
        "--worker-port",
        str(worker_port),
        "--log-full-action",
    ]
    if publish_commands:
        bridge_python_command.append("--publish-commands")

    bridge_command = [
        "bash",
        "-lc",
        f"source {shlex.quote(str(ROS_SETUP_SCRIPT))} && exec {shlex.join(bridge_python_command)}",
    ]
    return worker_command, bridge_command


def camera_names_for_profile(profile: DeploymentProfile) -> tuple[str, ...]:
    """Return the image streams expected by one deployment profile."""

    return HEAD_RIGHT_CAMERA_NAMES if profile.camera_count == 2 else THREE_CAMERA_NAMES


def deployment_camera_status_line(lines: list[str], camera_names: tuple[str, ...]) -> str:
    """Summarize the latest bridge freshness report for every required camera."""

    labels = {
        "head_cam": "头部",
        "left_arm_cam": "左臂",
        "right_arm_cam": "右臂",
    }
    waiting = "相机：" + " | ".join(f"{labels.get(name, name)} 等待图像" for name in camera_names)
    for line in reversed(lines):
        if not line.startswith("[bridge]"):
            continue
        if "Observation missing/stale:" in line:
            statuses = {name: "正常" for name in camera_names}
            for name in camera_names:
                if f"{name} missing " in line:
                    statuses[name] = "未收到"
                    continue
                stale_match = re.search(rf"{re.escape(name)} stale age=([0-9.]+s)", line)
                if stale_match:
                    statuses[name] = f"停滞 {stale_match.group(1)}"
            return "相机：" + " | ".join(
                f"{labels.get(name, name)} {statuses[name]}" for name in camera_names
            )
        if "action[" in line:
            return "相机：" + " | ".join(
                f"{labels.get(name, name)} 正常" for name in camera_names
            )
    return waiting


def deployment_camera_status_attribute(camera_line: str) -> int:
    """Assign a high-visibility color to stale or absent deployment images."""

    if "未收到" in camera_line or "停滞" in camera_line:
        return curses.color_pair(3) | curses.A_BOLD
    if "等待图像" in camera_line:
        return curses.color_pair(5) | curses.A_BOLD
    return curses.A_DIM


def draw_deployment_output(
    screen: curses.window,
    lines: list[str],
    status: str = "",
    *,
    camera_names: tuple[str, ...] = (),
) -> None:
    """Render live worker/bridge output while retaining a responsive stop key."""

    screen.erase()
    rows, columns = screen.getmaxyx()
    try:
        screen.addnstr(1, 2, "部署运行中", max(columns - 4, 0), curses.A_BOLD)
        screen.addnstr(
            2,
            2,
            "worker 与 bridge 正在运行。按 q、Esc 或 Ctrl-C 会停止两者并返回菜单。",
            max(columns - 4, 0),
            curses.A_DIM,
        )
        log_start_row = 4
        visible_count = max(1, rows - 8)
        if camera_names:
            camera_line = deployment_camera_status_line(lines, camera_names)
            screen.addnstr(
                3,
                2,
                camera_line,
                max(columns - 4, 0),
                deployment_camera_status_attribute(camera_line),
            )
            log_start_row = 5
            visible_count = max(1, rows - 9)
        for offset, line in enumerate(lines[-visible_count:]):
            attribute = curses.color_pair(3) | curses.A_BOLD if is_problem_line(line) else curses.color_pair(1)
            screen.addnstr(log_start_row + offset, 2, line, max(columns - 4, 0), attribute)
        if status:
            attribute = curses.color_pair(3) | curses.A_BOLD if is_problem_line(status) else curses.A_BOLD
            screen.addnstr(rows - 2, 2, status, max(columns - 4, 0), attribute)
    except curses.error:
        pass
    screen.refresh()


def stop_deployment_process(process: subprocess.Popen[str]) -> None:
    """Stop one process group, including the child started by conda or bash."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        process.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def run_deployment_processes(
    screen: curses.window,
    worker_command: list[str],
    bridge_command: list[str],
    camera_names: tuple[str, ...],
) -> int:
    """Run and stream the paired services until the operator stops them."""

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        worker = subprocess.Popen(
            worker_command,
            cwd=WORKSPACE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
    except OSError:
        return 1

    try:
        bridge = subprocess.Popen(
            bridge_command,
            cwd=WORKSPACE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
    except OSError:
        stop_deployment_process(worker)
        return 1

    processes = {"worker": worker, "bridge": bridge}
    lines: list[str] = []
    selector = selectors.DefaultSelector()
    for service, process in processes.items():
        assert process.stdout is not None
        selector.register(process.stdout, selectors.EVENT_READ, service)

    cancelled = False
    screen.timeout(MENU_INPUT_TIMEOUT_MS)
    try:
        while True:
            for key, _ in selector.select(timeout=0.1):
                output = key.fileobj.readline()
                if not output:
                    try:
                        selector.unregister(key.fileobj)
                    except KeyError:
                        pass
                    continue
                fresh_lines = [line.strip() for line in output.replace("\r", "\n").splitlines() if line.strip()]
                lines.extend(f"[{key.data}] {line}" for line in fresh_lines)
                del lines[:-DEPLOYMENT_LOG_LIMIT]

            completed = [service for service, process in processes.items() if process.poll() is not None]
            if completed:
                names = "、".join(completed)
                draw_deployment_output(
                    screen,
                    lines,
                    f"{names} 已退出，正在停止其余服务。",
                    camera_names=camera_names,
                )
                returncode = next(processes[service].returncode for service in completed)
                return returncode if returncode not in (None, 0) else 1

            draw_deployment_output(screen, lines, camera_names=camera_names)
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 27, 3):
                cancelled = True
                draw_deployment_output(screen, lines, "正在停止 worker 与 bridge。", camera_names=camera_names)
                return CANCELLED_EXIT_CODE
    finally:
        selector.close()
        for process in processes.values():
            stop_deployment_process(process)
        if cancelled:
            draw_deployment_output(screen, lines, "worker 与 bridge 已停止。", camera_names=camera_names)


def run_policy_deployment(screen: curses.window) -> int:
    """Configure and start the selected local worker/bridge deployment pair."""

    if not CONDA_ENV_PATH.is_dir() or shutil.which("conda") is None or not ROS_SETUP_SCRIPT.is_file():
        return 1

    step = "profile"
    profile: DeploymentProfile | None = None
    worker_port: int | None = None
    publish_commands: bool | None = None
    while True:
        if step == "profile":
            selected = select_deployment_profile(screen)
            if selected in (BACK, QUIT_TO_MENU):
                return CANCELLED_EXIT_CODE
            profile = selected  # type: ignore[assignment]
            step = "port"
        elif step == "port":
            selected = prompt_worker_port(screen)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "profile"
                continue
            worker_port = int(selected)
            step = "mode"
        elif step == "mode":
            assert profile is not None
            selected = choose_deployment_mode(screen, profile)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "port"
                continue
            publish_commands = bool(selected)
            step = "review"
        else:
            assert profile is not None and worker_port is not None and publish_commands is not None
            selected = review_deployment(screen, profile, worker_port, publish_commands)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "mode"
                continue
            if selected == "start":
                worker_command, bridge_command = build_deployment_commands(
                    profile,
                    worker_port,
                    publish_commands,
                )
                return run_deployment_processes(
                    screen,
                    worker_command,
                    bridge_command,
                    camera_names_for_profile(profile),
                )


def reported_dropped_frames(lines: list[str]) -> int:
    """Collect source-frame drops reported by either converter implementation."""

    return sum(int(match.group(1)) for line in lines for match in re.finditer(r"dropped-images=(\d+)", line.lower()))


def build_preflight_command(dataset: ConversionDataset, max_bags: int | None) -> list[str]:
    """Build the timestamp-only Bag preflight command for the chosen scope."""

    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-p",
        str(CONDA_ENV_PATH),
        "python",
        str(PREFLIGHT_SCRIPT),
        "--data-dir",
        str(dataset.path),
    ]
    if max_bags is not None:
        command.extend(["--max-bags", str(max_bags)])
    return command


def preflight_result_from_output(lines: list[str]) -> dict[str, object] | None:
    """Extract the structured last-line report emitted by the Bag preflight."""

    prefix = "PREFLIGHT_RESULT="
    for line in reversed(lines):
        if line.startswith(prefix):
            try:
                result = json.loads(line.removeprefix(prefix))
            except json.JSONDecodeError:
                return None
            return result if isinstance(result, dict) else None
    return None


def preflight_bags_from_result(result: dict[str, object]) -> tuple[PreflightBag, ...]:
    """Parse ordered, converter-compatible Bag identifiers from a preflight result."""

    raw_bags = result.get("bags")
    if not isinstance(raw_bags, list):
        return ()

    bags: list[PreflightBag] = []
    for fallback_index, raw_bag in enumerate(raw_bags, start=1):
        if not isinstance(raw_bag, dict):
            continue
        raw_index = raw_bag.get("index")
        index = raw_index if isinstance(raw_index, int) and raw_index > 0 else fallback_index
        name = raw_bag.get("name")
        if not isinstance(name, str) or not name:
            continue
        expected_frames = raw_bag.get("expected_frames")
        if not isinstance(expected_frames, int) or expected_frames < 0:
            expected_frames = 0
        raw_errors = raw_bag.get("errors")
        raw_warnings = raw_bag.get("warnings")
        errors = tuple(str(message) for message in raw_errors) if isinstance(raw_errors, list) else ()
        warnings = tuple(str(message) for message in raw_warnings) if isinstance(raw_warnings, list) else ()
        bags.append(PreflightBag(index, name, expected_frames, errors, warnings))
    return tuple(bags)


def draw_preflight_bag_selection(
    screen: curses.window,
    bags: tuple[PreflightBag, ...],
    retained_indices: set[int],
    selected: int,
    status: str = "",
) -> None:
    """Render an indexed multi-select list of inspected Bags."""

    screen.erase()
    rows, columns = screen.getmaxyx()
    try:
        screen.addnstr(1, 2, "按编号保留 Bag", max(columns - 4, 0), curses.A_BOLD)
        screen.addnstr(
            2,
            2,
            "上下键移动，Space 切换；a 全选，n 全不选，Enter 确认。",
            max(columns - 4, 0),
            curses.A_DIM,
        )
        visible_count = max(1, (rows - 9) // 2)
        first = min(max(0, selected - visible_count // 2), max(0, len(bags) - visible_count))
        last = min(len(bags), first + visible_count)
        for position in range(first, last):
            bag = bags[position]
            row = 5 + (position - first) * 2
            retained = bag.index in retained_indices
            marker = "[x]" if retained else "[ ]"
            attribute = curses.color_pair(2) | curses.A_BOLD if position == selected else curses.color_pair(1)
            screen.addnstr(row, 4, f"{marker} {bag.index}. {bag.name}", max(columns - 8, 0), attribute)
            if bag.errors:
                detail = f"异常 {len(bag.errors)} 项；预计 {bag.expected_frames} 帧"
                detail_attribute = curses.color_pair(3) | curses.A_BOLD
            elif bag.warnings:
                detail = f"警告 {len(bag.warnings)} 项；预计 {bag.expected_frames} 帧"
                detail_attribute = curses.color_pair(5) | curses.A_BOLD
            else:
                detail = f"预检正常；预计 {bag.expected_frames} 帧"
                detail_attribute = curses.A_DIM
            screen.addnstr(row + 1, 6, detail, max(columns - 10, 0), detail_attribute)
        screen.addnstr(rows - 3, 2, f"已保留 {len(retained_indices)}/{len(bags)} 个 Bag。b：返回；q：主菜单。", max(columns - 4, 0), curses.A_DIM)
        if status:
            screen.addnstr(rows - 2, 2, status, max(columns - 4, 0), curses.A_BOLD)
    except curses.error:
        pass
    screen.refresh()


def select_preflight_bags_by_number(screen: curses.window, bags: tuple[PreflightBag, ...]) -> object:
    """Select an arbitrary non-empty subset of Bags after frame preflight."""

    if not bags:
        return BACK
    retained_indices: set[int] = set()
    selected = 0
    status = ""
    by_index = {bag.index: bag for bag in bags}
    while True:
        draw_preflight_bag_selection(screen, bags, retained_indices, selected, status)
        key = screen.getch()
        if key in (ord("q"), ord("Q"), 27):
            return QUIT_TO_MENU
        if key in (ord("b"), ord("B")):
            return BACK
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(bags)
            continue
        if key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(bags)
            continue
        if key == ord(" "):
            bag_index = bags[selected].index
            if bag_index in retained_indices:
                retained_indices.remove(bag_index)
            else:
                retained_indices.add(bag_index)
            status = ""
            continue
        if key in (ord("a"), ord("A")):
            retained_indices = set(by_index)
            status = ""
            continue
        if key in (ord("n"), ord("N")):
            retained_indices.clear()
            status = ""
            continue
        if ord("1") <= key <= ord("9"):
            bag_index = key - ord("0")
            if bag_index in by_index:
                if bag_index in retained_indices:
                    retained_indices.remove(bag_index)
                else:
                    retained_indices.add(bag_index)
                status = ""
            continue
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            if retained_indices:
                return tuple(bag for bag in bags if bag.index in retained_indices)
            status = "请至少保留一个 Bag，或返回选择“全部丢弃”。"


def choose_preflight_bag_retention(screen: curses.window, bags: tuple[PreflightBag, ...]) -> object:
    """Choose to skip, retain all, or retain a numbered subset without deleting Bags."""

    selected = select_choice(
        screen,
        "选择 Bag 保留方式",
        "“丢弃”只是不纳入本次转换，原始 ROS Bag 不会被删除。",
        [
            Choice("全部丢弃", "不创建数据集，保留所有原始 Bag 文件。", "discard_all"),
            Choice(f"全部保留（{len(bags)} 个）", "将预检范围内的全部 Bag 纳入转换。", "keep_all"),
            Choice("按编号保留", "在下一页勾选要保留的 Bag 编号。", "select_by_number"),
        ],
    )
    if selected in (BACK, QUIT_TO_MENU):
        return selected
    if selected == "discard_all":
        return "discard_all"
    if selected == "keep_all":
        return bags
    return select_preflight_bags_by_number(screen, bags)


def errors_for_retained_preflight_bags(
    retained_bags: tuple[PreflightBag, ...],
    all_bags: tuple[PreflightBag, ...],
    all_errors: list[str],
) -> list[str]:
    """Keep only errors relevant to Bags selected for conversion."""

    attributed_errors = {error for bag in all_bags for error in bag.errors}
    retained_errors = [error for bag in retained_bags for error in bag.errors]
    retained_errors.extend(error for error in all_errors if error not in attributed_errors)
    return list(dict.fromkeys(retained_errors))


def validation_result_from_output(lines: list[str]) -> dict[str, object] | None:
    """Extract the validator's structured last-line report."""

    prefix = "VALIDATION_RESULT="
    for line in reversed(lines):
        if line.startswith(prefix):
            try:
                return json.loads(line.removeprefix(prefix))
            except json.JSONDecodeError:
                return None
    return None


def show_preflight_report(
    screen: curses.window,
    errors: list[str],
    warnings: list[str],
    summary: dict[str, object],
) -> object:
    """Present frame-loss checks before a conversion is allowed to start."""

    screen.erase()
    rows, columns = screen.getmaxyx()
    passed = not errors
    title = "转换前数据预检：通过" if passed else "转换前数据预检：发现明显异常"
    try:
        title_attribute = curses.color_pair(4) | curses.A_BOLD if passed else curses.color_pair(3) | curses.A_BOLD
        screen.addnstr(1, 2, title, max(columns - 4, 0), title_attribute)
        screen.addnstr(
            2,
            2,
            "仅扫描 Bag 时间戳，不读取图像内容，也不会写入或修改数据。",
            max(columns - 4, 0),
            curses.A_DIM,
        )
        if summary:
            details = ", ".join(f"{key}={value}" for key, value in summary.items())
            screen.addnstr(3, 2, details, max(columns - 4, 0), curses.A_DIM)
        report_lines = [("错误", message) for message in errors] + [("警告", message) for message in warnings]
        if not report_lines:
            report_lines = [("正常", "未发现相机缺流、明显低帧率或长帧间隔。")]
        visible_count = max(1, rows - 10)
        for offset, (level, message) in enumerate(report_lines[-visible_count:]):
            attribute = curses.color_pair(3) | curses.A_BOLD if level == "错误" else curses.color_pair(1)
            screen.addnstr(5 + offset, 2, f"[{level}] {message}", max(columns - 4, 0), attribute)
        screen.addnstr(rows - 2, 2, "Enter：继续；b：返回；q：主菜单。", max(columns - 4, 0), curses.A_DIM)
    except curses.error:
        pass
    screen.refresh()
    while True:
        key = screen.getch()
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return "continue"
        if key in (ord("b"), ord("B")):
            return BACK
        if key in (ord("q"), ord("Q"), 27):
            return QUIT_TO_MENU


def choose_preflight_error_recovery(screen: curses.window) -> object:
    """Require an explicit decision before converting a dataset with errors."""

    return select_choice(
        screen,
        "保留的 Bag 仍有预检异常",
        "预检发现缺流、严重低帧率或长时间断流。建议取消保留对应编号。",
        [
            Choice("重新选择保留 Bag", "不创建数据集，返回按编号保留页面。", "reselect"),
            Choice("忽略异常并继续", "仍会创建新数据集；异常可能导致低质量或重复图像帧。", "continue"),
        ],
    )


def show_validation_report(screen: curses.window, errors: list[str], warnings: list[str], summary: dict[str, object]) -> None:
    """Show the final validation result, emphasizing every error in red."""

    screen.erase()
    rows, columns = screen.getmaxyx()
    passed = not errors
    title = "Conversion Validation Passed" if passed else "Conversion Validation Found Problems"
    try:
        title_attribute = curses.color_pair(2) | curses.A_BOLD if passed else curses.color_pair(3) | curses.A_BOLD
        screen.addnstr(1, 2, title, max(columns - 4, 0), title_attribute)
        if summary:
            details = ", ".join(f"{key}={value}" for key, value in summary.items())
            screen.addnstr(2, 2, details, max(columns - 4, 0), curses.A_DIM)
        report_lines = [("ERROR", message) for message in errors] + [("WARNING", message) for message in warnings]
        if not report_lines:
            report_lines = [("OK", "Frame counts, timestamps, action/state values, and decoded camera videos are aligned.")]
        visible_count = max(1, rows - 8)
        for offset, (level, message) in enumerate(report_lines[-visible_count:]):
            attribute = curses.color_pair(3) | curses.A_BOLD if level == "ERROR" else curses.color_pair(1)
            screen.addnstr(4 + offset, 2, f"[{level}] {message}", max(columns - 4, 0), attribute)
        screen.addnstr(rows - 2, 2, "Press Enter, b, or q to return to the main menu.", max(columns - 4, 0), curses.A_DIM)
    except curses.error:
        pass
    screen.refresh()
    while screen.getch() not in (curses.KEY_ENTER, ord("\n"), ord("\r"), ord("b"), ord("B"), ord("q"), ord("Q"), 27):
        pass


def select_replay_bag(screen: curses.window) -> object:
    """Browse ``data/rosbag`` and choose a bag file before replaying it."""

    if not ROSBAG_ROOT.is_dir():
        return QUIT_TO_MENU

    current = ROSBAG_ROOT
    supported_suffixes = {".mcap", ".bag", ".db3"}
    while True:
        choices: list[Choice] = []
        if (current / "metadata.yaml").is_file():
            choices.append(
                Choice(
                    "Replay this bag folder",
                    "Use this ROS bag directory directly.",
                    ("replay", current),
                )
            )
        for path in sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if path.is_dir():
                contained_bags = sorted(
                    child for child in path.iterdir() if child.is_file() and child.suffix.lower() in supported_suffixes
                )
                if len(contained_bags) == 1:
                    choices.append(
                        Choice(
                            path.name,
                            f"Replay {contained_bags[0].name}.",
                            ("replay", contained_bags[0]),
                        )
                    )
                else:
                    choices.append(Choice(f"[folder] {path.name}", "Open this folder.", ("open", path)))
            elif path.suffix.lower() in supported_suffixes:
                choices.append(Choice(f"[file] {path.name}", "Replay this ROS bag file.", ("replay", path)))
        if not choices:
            choices.append(Choice("No replayable items", "Press b to return to the previous folder.", ("empty", current)))

        relative = "." if current == ROSBAG_ROOT else str(current.relative_to(ROSBAG_ROOT))
        selected = select_choice(
            screen,
            "Replay a ROS Bag",
            f"Browse data/rosbag/{relative}; choose a bag file to replay.",
            choices,
        )
        if selected is QUIT_TO_MENU:
            return QUIT_TO_MENU
        if selected is BACK:
            if current == ROSBAG_ROOT:
                return BACK
            current = current.parent
            continue
        action, path = selected  # type: ignore[misc]
        if action == "open":
            current = path
        elif action == "empty":
            continue
        else:
            return path


def choose_replay_publish_mode(screen: curses.window) -> object:
    """Choose dry-run or explicitly confirm motion-command publication."""

    while True:
        mode = select_choice(
            screen,
            "Replay Mode",
            "Dry-run is safe; publishing sends motion commands to the robot.",
            [
                Choice("Dry-run", "Preview the replay without publishing commands.", False),
                Choice("Publish commands", "Send replay commands to the autonomous command topic.", True),
            ],
        )
        if mode in (BACK, QUIT_TO_MENU, False):
            return mode
        confirmation = select_choice(
            screen,
            "Confirm Command Publication",
            "This will publish robot motion commands. Keep the emergency stop available.",
            [
                Choice("Start publishing", "Confirm and start this replay with command publication enabled.", True),
                Choice("Cancel publication", "Return to replay mode selection without publishing.", False),
            ],
        )
        if confirmation is QUIT_TO_MENU:
            return QUIT_TO_MENU
        if confirmation is BACK or confirmation is False:
            continue
        return True


def choose_replay_motion_mode(screen: curses.window) -> object:
    """Choose which part of the robot is allowed to move during replay."""

    return select_choice(
        screen,
        "Replay Motion Mode",
        "Choose the moving subsystem before deciding whether to publish commands.",
        [
            Choice(
                "Full replay (no freeze)",
                "Replay every recorded torso, arm, gripper, and base command without constraints.",
                "full",
            ),
            Choice(
                "Freeze base (replay upper body)",
                "Commands zero base vx/vy/yaw rate; torso, arms, head, and grippers replay.",
                "base-frozen",
            ),
            Choice(
                "Base only (freeze upper body)",
                "Holds the first torso/arm/head/gripper target; only base vx/vy/yaw rate replays.",
                "base-only",
            ),
        ],
    )


def run_dataset_replay(screen: curses.window) -> int:
    """Replay a selected bag, with all TUI choices rendered in curses."""

    if not REPLAY_SCRIPT.is_file() or not ROS_SETUP_SCRIPT.is_file():
        return 1

    bag = select_replay_bag(screen)
    if bag in (BACK, QUIT_TO_MENU):
        return CANCELLED_EXIT_CODE
    motion_mode = choose_replay_motion_mode(screen)
    if motion_mode is QUIT_TO_MENU:
        return CANCELLED_EXIT_CODE
    if motion_mode is BACK:
        return run_dataset_replay(screen)
    publish = choose_replay_publish_mode(screen)
    if publish is QUIT_TO_MENU:
        return CANCELLED_EXIT_CODE
    if publish is BACK:
        return run_dataset_replay(screen)

    command = [
        "bash",
        "-c",
        'source "$1" && shift && exec "$@"',
        "bash",
        str(ROS_SETUP_SCRIPT),
        "/usr/bin/python3",
        str(REPLAY_SCRIPT),
        "--bag",
        str(bag),
        "--motion-mode",
        str(motion_mode),
    ]
    if publish:
        command.append("--publish")
    return run_terminal_command(screen, command, "Loading ROS 2 Humble and starting replay...")


def list_conversion_datasets() -> tuple[ConversionDataset, ...]:
    """List immediate ROS-bag dataset directories with their episode counts.

    Keeping this at one level lets an operator choose, for example,
    ``Test(2026_07_16)`` as a whole dataset instead of scanning through every
    individual bag in the main menu.
    """

    if not ROSBAG_ROOT.is_dir():
        return ()

    datasets = []
    for path in sorted((entry for entry in ROSBAG_ROOT.iterdir() if entry.is_dir()), key=lambda entry: entry.name):
        bag_count = sum(1 for _ in path.rglob("metadata.yaml"))
        if bag_count:
            datasets.append(ConversionDataset(path=path, bag_count=bag_count))
    return tuple(datasets)


def existing_dataset_summary(path: Path) -> str:
    """Read a compact, safe-to-display summary from a finalized LeRobot dataset."""

    try:
        info = json.loads((path / "meta/info.json").read_text(encoding="utf-8"))
        return (
            f"{path.name}: {info.get('total_episodes', '?')} episode(s), "
            f"{info.get('total_frames', '?')} frames, {info.get('fps', '?')} Hz"
        )
    except (OSError, json.JSONDecodeError):
        return f"{path.name}: finalized dataset metadata found"


def find_existing_conversions(dataset: ConversionDataset) -> tuple[ExistingConversion, ...]:
    """Find finalized outputs matching this source by name or source manifest."""

    if not LEROBOT_ROOT.is_dir():
        return ()

    source_path = str(dataset.path.resolve())
    default_path = LEROBOT_ROOT / default_repo_name(dataset)
    matches: dict[Path, ExistingConversion] = {}
    for candidate in (path for path in LEROBOT_ROOT.iterdir() if path.is_dir()):
        if not (candidate / "meta/info.json").is_file():
            continue
        matches_default_name = candidate == default_path
        matches_manifest = False
        manifest_path = candidate / CONVERSION_MANIFEST_NAME
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                matches_manifest = manifest.get("source_dataset") == source_path
            except (OSError, json.JSONDecodeError):
                pass
        if matches_default_name or matches_manifest:
            matches[candidate] = ExistingConversion(candidate, existing_dataset_summary(candidate))
    return tuple(sorted(matches.values(), key=lambda item: item.path.name))


def next_available_repo_name(base_name: str) -> str:
    """Return a non-existing sibling name without ever overwriting a dataset."""

    if not (LEROBOT_ROOT / base_name).exists():
        return base_name
    suffix = 2
    while (LEROBOT_ROOT / f"{base_name}_v{suffix}").exists():
        suffix += 1
    return f"{base_name}_v{suffix}"


def handle_existing_conversions(screen: curses.window, existing: tuple[ExistingConversion, ...]) -> object:
    """Let the operator skip a known conversion or intentionally create a new one."""

    summaries = "; ".join(item.summary for item in existing)
    return select_choice(
        screen,
        "Existing LeRobot Conversion Found",
        summaries,
        [
            Choice("Use existing dataset", "Skip conversion and return to the main menu.", "use_existing"),
            Choice("Convert again with a new name", "Keep the existing dataset and create a distinct output.", "convert_again"),
        ],
    )


def select_conversion_dataset(screen: curses.window) -> object:
    """Choose a dataset root below ``data/rosbag``."""

    datasets = list_conversion_datasets()
    choices = []
    for dataset in datasets:
        suffix = "bag" if dataset.bag_count == 1 else "bags"
        choices.append(
            Choice(
                label=str(dataset.path.relative_to(WORKSPACE_ROOT)),
                description=f"Convert this dataset ({dataset.bag_count} {suffix}).",
                value=dataset,
            )
        )
    return select_choice(screen, "Convert ROS Bags to LeRobot", "Select an input dataset below data/rosbag.", choices)


def choose_converter_script(screen: curses.window) -> object:
    """Choose the serial or threaded existing conversion implementation."""

    if not SERIAL_CONVERTER_SCRIPT.is_file() or not THREADED_CONVERTER_SCRIPT.is_file():
        return QUIT_TO_MENU
    return select_choice(
        screen,
        "Choose a Converter",
        "Both converters preserve the same topic layout and LeRobot output format.",
        [
            Choice("Threaded frame builder", "Parallelizes frame decoding and construction; recommended.", THREADED_CONVERTER_SCRIPT),
            Choice("Original serial frame builder", "Builds frames one at a time; useful for comparison and debugging.", SERIAL_CONVERTER_SCRIPT),
        ],
    )


def choose_image_arguments(screen: curses.window) -> object:
    """Choose a numbered image-processing profile."""

    return select_choice(
        screen,
        "Choose an Image Profile",
        "Use the Robot10 profile for the current 640x480 DINOv3 deployment.",
        [
            Choice("Converter default", "224x224 with no crop.", []),
            Choice("Robot10 3-camera profile", "640x480 with centered 2/3 crop.", ["--img-width", "640", "--img-height", "480", "--center-crop-fraction", "0.6666667"]),
        ],
    )


def default_repo_name(dataset: ConversionDataset) -> str:
    """Derive a filesystem- and Hub-safe LeRobot dataset name."""

    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", dataset.path.name).strip("._-").lower()
    return f"{cleaned or 'zeno_h1'}_zeno_h1_v30"


def prompt_repo_name(screen: curses.window, dataset: ConversionDataset, default: str) -> object:
    """Request a valid, non-destructive LeRobot output dataset name."""

    while True:
        name = prompt_text(
            screen,
            "Name the LeRobot Dataset",
            "A new subdirectory will be created below data/lerobot; existing data is never overwritten.",
            "Dataset name:",
            default=default,
        )
        if name in (BACK, QUIT_TO_MENU):
            return name
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(name)):
            return name


def prompt_custom_bag_count(screen: curses.window, bag_count: int) -> object:
    """Ask for a numeric bag limit while retaining the shared b/q behavior."""

    while True:
        value = prompt_text(
            screen,
            "Choose a Bag Limit",
            f"Enter an integer from 1 through {bag_count}.",
            "Number of bags:",
            default=None,
        )
        if value in (BACK, QUIT_TO_MENU):
            return value
        if str(value).isdigit() and 1 <= int(str(value)) <= bag_count:
            return int(str(value))


def choose_max_bags(screen: curses.window, dataset: ConversionDataset) -> object:
    """Select all bags, a smoke-test subset, or an explicit numeric limit."""

    while True:
        choice = select_choice(
            screen,
            "Choose How Many Bags to Convert",
            f"The selected dataset contains {dataset.bag_count} bags.",
            [
                Choice(f"All {dataset.bag_count} bags", "Convert the complete selected dataset.", None),
                Choice("First 1 bag", "Quick smoke test before a full conversion.", 1),
                Choice("Custom number", "Enter an exact number of bags to convert.", "custom"),
            ],
        )
        if choice in (BACK, QUIT_TO_MENU) or choice is None or isinstance(choice, int):
            return choice
        custom = prompt_custom_bag_count(screen, dataset.bag_count)
        if custom is BACK:
            continue
        return custom


def build_conversion_command(
    dataset: ConversionDataset,
    converter: Path,
    image_arguments: list[str],
    repo_name: str,
    max_bags: int | None,
    excluded_bag_names: tuple[str, ...] = (),
) -> list[str]:
    """Build the exact conda command without exposing overwrite behavior."""

    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-p",
        str(CONDA_ENV_PATH),
        "python",
        str(converter),
        "--data-dir",
        str(dataset.path),
        "--output-dir",
        str(LEROBOT_ROOT),
        "--repo-name",
        repo_name,
        "--task",
        dataset.path.name,
        *image_arguments,
    ]
    if max_bags is not None:
        command.extend(["--max-bags", str(max_bags)])
    if excluded_bag_names:
        command.extend(["--exclude-bags", ",".join(excluded_bag_names)])
    return command


def write_conversion_manifest(
    output_path: Path,
    dataset: ConversionDataset,
    converter: Path,
    image_arguments: list[str],
    max_bags: int | None,
    excluded_bag_names: tuple[str, ...] = (),
) -> None:
    """Persist the source identity so later TUI runs can avoid duplicate work."""

    manifest = {
        "source_dataset": str(dataset.path.resolve()),
        "source_bag_count": dataset.bag_count,
        "max_bags": max_bags,
        "converter": converter.name,
        "image_arguments": image_arguments,
        "excluded_bag_names": list(excluded_bag_names),
    }
    (output_path / CONVERSION_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def review_conversion(
    screen: curses.window,
    output_path: Path,
    max_bags: int | None,
    retained_bag_count: int | None,
) -> object:
    """Give the operator one final start option; b reaches earlier choices."""

    if retained_bag_count is not None:
        bag_text = f"保留 {retained_bag_count} 个 Bag"
    else:
        bag_text = "all bags" if max_bags is None else f"first {max_bags} bag(s)"
    return select_choice(
        screen,
        "Ready to Convert",
        f"Output: {output_path} | Scope: {bag_text}",
        [
            Choice("Start conversion", "Create the new LeRobot dataset now.", "start"),
        ],
    )


def run_dataset_conversion(screen: curses.window) -> int:
    """Configure and launch conversion through styled, backtrackable sub-menus."""

    if not CONDA_ENV_PATH.is_dir() or shutil.which("conda") is None or not PREFLIGHT_SCRIPT.is_file():
        return 1

    step = "dataset"
    dataset: ConversionDataset | None = None
    converter: Path | None = None
    image_arguments: list[str] = []
    repo_name = ""
    max_bags: int | None = None
    excluded_bag_names: tuple[str, ...] = ()
    retained_bag_count: int | None = None

    while True:
        if step == "dataset":
            selected = select_conversion_dataset(screen)
            if selected in (BACK, QUIT_TO_MENU):
                return CANCELLED_EXIT_CODE
            dataset = selected  # type: ignore[assignment]
            repo_name = default_repo_name(dataset)
            step = "existing_check"
        elif step == "existing_check":
            assert dataset is not None
            existing = find_existing_conversions(dataset)
            if existing:
                selected = handle_existing_conversions(screen, existing)
                if selected is QUIT_TO_MENU:
                    return CANCELLED_EXIT_CODE
                if selected is BACK:
                    step = "dataset"
                    continue
                if selected == "use_existing":
                    return 0
                repo_name = next_available_repo_name(default_repo_name(dataset))
            step = "converter"
        elif step == "converter":
            selected = choose_converter_script(screen)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "dataset"
                continue
            converter = selected  # type: ignore[assignment]
            step = "image_profile"
        elif step == "image_profile":
            selected = choose_image_arguments(screen)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "converter"
                continue
            image_arguments = selected  # type: ignore[assignment]
            step = "repo_name"
        elif step == "repo_name":
            assert dataset is not None
            selected = prompt_repo_name(screen, dataset, repo_name)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "image_profile"
                continue
            repo_name = str(selected)
            step = "max_bags"
        elif step == "max_bags":
            assert dataset is not None
            selected = choose_max_bags(screen, dataset)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "repo_name"
                continue
            max_bags = selected  # type: ignore[assignment]
            excluded_bag_names = ()
            retained_bag_count = None
            step = "preflight"
        elif step == "preflight":
            assert dataset is not None
            preflight = run_curses_command(
                screen,
                build_preflight_command(dataset, max_bags),
                "预检 ROS Bag 帧完整性",
            )
            report = preflight_result_from_output(preflight.lines)
            errors: list[str] = []
            warnings: list[str] = []
            summary: dict[str, object] = {}
            preflight_bags: tuple[PreflightBag, ...] = ()
            if report is None:
                errors.append("预检未生成可读取的结果")
                if preflight.returncode != 0:
                    errors.append(f"预检退出码：{preflight.returncode}")
            else:
                errors.extend(str(message) for message in report.get("errors", []))
                warnings.extend(str(message) for message in report.get("warnings", []))
                raw_summary = report.get("summary", {})
                if isinstance(raw_summary, dict):
                    summary = {str(key): value for key, value in raw_summary.items()}
                preflight_bags = preflight_bags_from_result(report)
                if not preflight_bags:
                    errors.append("预检未返回可选择的 Bag 编号")
                if preflight.returncode not in (0, 2):
                    errors.append(f"预检异常退出，退出码：{preflight.returncode}")

            dismissed = show_preflight_report(screen, errors, warnings, summary)
            if dismissed is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if dismissed is BACK:
                step = "max_bags"
                continue
            while True:
                selected = choose_preflight_bag_retention(screen, preflight_bags)
                if selected is QUIT_TO_MENU:
                    return CANCELLED_EXIT_CODE
                if selected is BACK:
                    step = "max_bags"
                    break
                if selected == "discard_all":
                    return ALL_BAGS_DISCARDED_EXIT_CODE

                retained_bags = selected  # type: ignore[assignment]
                assert isinstance(retained_bags, tuple)
                retained_errors = errors_for_retained_preflight_bags(
                    retained_bags,
                    preflight_bags,
                    errors,
                )
                if retained_errors:
                    recovery = choose_preflight_error_recovery(screen)
                    if recovery is QUIT_TO_MENU:
                        return CANCELLED_EXIT_CODE
                    if recovery is BACK:
                        step = "max_bags"
                        break
                    if recovery == "reselect":
                        continue

                retained_names = {bag.name for bag in retained_bags}
                excluded_bag_names = tuple(
                    bag.name for bag in preflight_bags if bag.name not in retained_names
                )
                retained_bag_count = len(retained_bags)
                step = "review"
                break
            if step != "review":
                continue
        else:
            assert dataset is not None and converter is not None
            output_path = LEROBOT_ROOT / repo_name
            if output_path.exists():
                step = "repo_name"
                continue
            selected = review_conversion(screen, output_path, max_bags, retained_bag_count)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "max_bags"
                continue
            if selected == "start":
                LEROBOT_ROOT.mkdir(parents=True, exist_ok=True)
                command = build_conversion_command(
                    dataset,
                    converter,
                    image_arguments,
                    repo_name,
                    max_bags,
                    excluded_bag_names,
                )
                conversion = run_curses_command(screen, command, "Converting ROS Bags to LeRobot")
                errors: list[str] = []
                warnings: list[str] = []
                summary: dict[str, object] = {}
                if conversion.returncode != 0:
                    errors.append(f"converter exited with code {conversion.returncode}")
                dropped_frames = reported_dropped_frames(conversion.lines)
                if dropped_frames:
                    errors.append(f"converter reported {dropped_frames} dropped image frame(s)")

                if output_path.is_dir() and VALIDATOR_SCRIPT.is_file():
                    validation_command = [
                        "conda",
                        "run",
                        "--no-capture-output",
                        "-p",
                        str(CONDA_ENV_PATH),
                        "python",
                        str(VALIDATOR_SCRIPT),
                        "--dataset-path",
                        str(output_path),
                    ]
                    validation = run_curses_command(screen, validation_command, "Validating LeRobot Dataset")
                    report = validation_result_from_output(validation.lines)
                    if report is None:
                        errors.append("validator did not produce a readable report")
                    else:
                        errors.extend(str(message) for message in report.get("errors", []))
                        warnings.extend(str(message) for message in report.get("warnings", []))
                        summary = {str(key): value for key, value in dict(report.get("summary", {})).items()}
                    if validation.returncode != 0 and report is None:
                        errors.append(f"validator exited with code {validation.returncode}")
                else:
                    errors.append("converted dataset or validation script is missing")

                if not errors:
                    write_conversion_manifest(
                        output_path,
                        dataset,
                        converter,
                        image_arguments,
                        max_bags,
                        excluded_bag_names,
                    )
                show_validation_report(screen, errors, warnings, summary)
                return 0 if not errors else 2


VANILLA_ACT_RECIPE = TrainingRecipe(
    key="vanilla_act",
    label="Vanilla ACT",
    description="LeRobot 默认 ACT：ResNet-18 视觉骨干，100k steps，batch size 8。",
    run_label="vanilla_act",
)
ROBOT8_3CAM_DINOV3_RECIPE = TrainingRecipe(
    key="robot8_3cam_dinov3",
    label="Robot10 三相机 ACT + DINOv3 Base（冻结）",
    description="DINOv3 ViT-B/16 冻结，100k steps；batch 32 起，OOM 时自动降档。",
    run_label="robot8_3cam_act_dinov3_base_frozen",
)
TRAINING_RECIPES = (VANILLA_ACT_RECIPE, ROBOT8_3CAM_DINOV3_RECIPE)


def read_training_dataset(path: Path) -> TrainingDataset | None:
    """Read the metadata needed to safely present a local LeRobot dataset."""

    try:
        info = json.loads((path / "meta/info.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    features = info.get("features", {})
    if not isinstance(features, dict):
        return None
    camera_count = sum(name.startswith("observation.images.") for name in features)
    summary = (
        f"{info.get('total_episodes', '?')} episode(s), "
        f"{info.get('total_frames', '?')} frames, {info.get('fps', '?')} Hz, "
        f"{camera_count} camera(s)."
    )
    return TrainingDataset(path=path, summary=summary, camera_count=camera_count)


def list_training_datasets() -> tuple[TrainingDataset, ...]:
    """List finalized LeRobot datasets directly below ``data/lerobot``."""

    if not LEROBOT_ROOT.is_dir():
        return ()

    datasets = []
    for path in sorted((entry for entry in LEROBOT_ROOT.iterdir() if entry.is_dir()), key=lambda entry: entry.name):
        dataset = read_training_dataset(path)
        if dataset is not None:
            datasets.append(dataset)
    return tuple(datasets)


def select_training_dataset(screen: curses.window) -> object:
    """Choose one finalized local dataset for policy training."""

    datasets = list_training_datasets()
    choices = [
        Choice(
            label=str(dataset.path.relative_to(WORKSPACE_ROOT)),
            description=dataset.summary,
            value=dataset,
        )
        for dataset in datasets
    ]
    return select_choice(
        screen,
        "Train a LeRobot Policy",
        "Select a finalized dataset below data/lerobot.",
        choices,
    )


def select_training_recipe(screen: curses.window, dataset: TrainingDataset) -> object:
    """Choose between the two curated ACT configurations."""

    choices = [
        Choice(recipe.label, recipe.description, recipe)
        for recipe in TRAINING_RECIPES
    ]
    return select_choice(
        screen,
        "Choose a Training Recipe",
        f"Dataset: {dataset.path.name} ({dataset.camera_count} camera(s)).",
        choices,
    )


def choose_three_camera_recovery(screen: curses.window, dataset: TrainingDataset) -> object:
    """Explain the Robot10 recipe's required three visual inputs."""

    return select_choice(
        screen,
        "Robot10 Three-camera Recipe Requires Three Cameras",
        f"{dataset.path.name} declares {dataset.camera_count} observation.images.* feature(s).",
        [
            Choice("Choose another dataset", "Select a dataset containing exactly three camera features.", "dataset"),
            Choice("Choose Vanilla ACT", "Return to recipe selection and use the default ACT configuration.", "recipe"),
        ],
    )


def clean_run_name(value: str) -> str:
    """Create a concise filesystem-safe component for a local training run."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "dataset"


def new_training_run(recipe: TrainingRecipe, dataset: TrainingDataset) -> tuple[str, str | None]:
    """Return a collision-free output directory name and optional Robot10 suffix."""

    dataset_name = clean_run_name(dataset.path.name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if recipe is ROBOT8_3CAM_DINOV3_RECIPE:
        model_tag = f"tui_{dataset_name}"
        suffix = timestamp
        run_id = f"robot8_{model_tag}_act_dinov3_base_frozen_100k_640x480_crop2of3_{suffix}"
        index = 2
        while (TRAIN_OUTPUT_ROOT / run_id).exists():
            suffix = f"{timestamp}_{index}"
            run_id = f"robot8_{model_tag}_act_dinov3_base_frozen_100k_640x480_crop2of3_{suffix}"
            index += 1
        return run_id, suffix

    base = f"{dataset_name}_{recipe.run_label}_{timestamp}"
    run_id = base
    index = 2
    while (TRAIN_OUTPUT_ROOT / run_id).exists():
        run_id = f"{base}_{index}"
        index += 1
    return run_id, None


def lerobot_training_environment() -> dict[str, str]:
    """Set the local fork and cache roots for an in-place LeRobot invocation."""

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = str(LEROBOT_SOURCE_ROOT)
    if current_pythonpath:
        pythonpath = f"{pythonpath}{os.pathsep}{current_pythonpath}"
    hf_home = os.environ.get("HF_HOME", str(WORKSPACE_ROOT / ".hf_home"))
    return {
        "PYTHONPATH": pythonpath,
        "HF_HOME": hf_home,
        "HF_LEROBOT_HOME": os.environ.get("HF_LEROBOT_HOME", str(WORKSPACE_ROOT / ".hf_lerobot")),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", str(Path(hf_home) / "datasets")),
    }


def build_training_command(
    recipe: TrainingRecipe,
    dataset: TrainingDataset,
    run_id: str,
    robot8_run_suffix: str | None,
) -> tuple[list[str], dict[str, str]]:
    """Build a local, non-Hub training command for the selected recipe."""

    if recipe is ROBOT8_3CAM_DINOV3_RECIPE:
        assert robot8_run_suffix is not None
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-p",
            str(CONDA_ENV_PATH),
            "env",
            "CONDA_ENV=",
            f"DATASET_REPO_ID={dataset.path.name}",
            f"DATASET_ROOT={dataset.path}",
            f"MODEL_TAG=tui_{clean_run_name(dataset.path.name)}",
            f"RUN_SUFFIX={robot8_run_suffix}",
            "RUN_PARALLEL=false",
            "bash",
            str(ROBOT8_3CAM_TRAIN_SCRIPT),
        ]
        return command, {}

    output_dir = TRAIN_OUTPUT_ROOT / run_id
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-p",
        str(CONDA_ENV_PATH),
        "python",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset.path.name}",
        f"--dataset.root={dataset.path}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.use_amp=true",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={run_id}",
        "--steps=100000",
        "--batch_size=8",
        "--num_workers=4",
        "--save_freq=20000",
        "--log_freq=200",
        "--wandb.enable=false",
    ]
    return command, lerobot_training_environment()


def review_training(
    screen: curses.window,
    recipe: TrainingRecipe,
    dataset: TrainingDataset,
    run_id: str,
) -> object:
    """Give the operator one explicit start choice before training begins."""

    return select_choice(
        screen,
        "Ready to Start Training",
        f"Recipe: {recipe.label} | Dataset: {dataset.path.name} | Output: outputs/train/{run_id}",
        [
            Choice("Start training", "Training writes checkpoints locally and does not push to Hugging Face.", "start"),
        ],
    )


def run_dataset_training(screen: curses.window) -> int:
    """Choose a local dataset and run one of the curated ACT training recipes."""

    if not CONDA_ENV_PATH.is_dir() or shutil.which("conda") is None:
        return 1
    if not ROBOT8_3CAM_TRAIN_SCRIPT.is_file() or not LEROBOT_SOURCE_ROOT.is_dir():
        return 1

    step = "dataset"
    dataset: TrainingDataset | None = None
    recipe: TrainingRecipe | None = None

    while True:
        if step == "dataset":
            selected = select_training_dataset(screen)
            if selected in (BACK, QUIT_TO_MENU):
                return CANCELLED_EXIT_CODE
            dataset = selected  # type: ignore[assignment]
            step = "recipe"
        elif step == "recipe":
            assert dataset is not None
            selected = select_training_recipe(screen, dataset)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "dataset"
                continue
            recipe = selected  # type: ignore[assignment]
            if recipe is ROBOT8_3CAM_DINOV3_RECIPE and dataset.camera_count != 3:
                step = "three_camera_recovery"
            else:
                step = "review"
        elif step == "three_camera_recovery":
            assert dataset is not None
            selected = choose_three_camera_recovery(screen, dataset)
            if selected in (QUIT_TO_MENU, BACK):
                return CANCELLED_EXIT_CODE
            step = str(selected)
        else:
            assert dataset is not None and recipe is not None
            run_id, robot8_run_suffix = new_training_run(recipe, dataset)
            selected = review_training(screen, recipe, dataset, run_id)
            if selected is QUIT_TO_MENU:
                return CANCELLED_EXIT_CODE
            if selected is BACK:
                step = "recipe"
                continue
            if selected == "start":
                TRAIN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
                command, environment = build_training_command(recipe, dataset, run_id, robot8_run_suffix)
                return run_curses_command(screen, command, f"Training {recipe.label}", environment).returncode


MENU_ITEMS = (
    MenuItem(
        label="配置 CycloneDDS 永久网络",
        description="运行：bash setup_cyclone_perm.sh",
        action=run_cyclonedds_setup,
    ),
    MenuItem(
        label="回放 ROS Bag 数据集",
        description="选择数据集；自动启动 ROS 2，默认仅预览、不发布控制命令。",
        action=run_dataset_replay,
        uses_curses=True,
    ),
    MenuItem(
        label="转换 ROS Bag 为 LeRobot 数据集",
        description="选择 data/rosbag 数据；输出新的数据集到 data/lerobot。",
        action=run_dataset_conversion,
        uses_curses=True,
    ),
    MenuItem(
        label="训练 ACT 策略",
        description="选择本地 LeRobot 数据集；可选 Vanilla ACT 或 Robot10 三相机 DINOv3 配方。",
        action=run_dataset_training,
        uses_curses=True,
    ),
    MenuItem(
        label="部署策略 worker + bridge",
        description="选择已有部署配置；同时启动本地推理 worker 与 ROS bridge。",
        action=run_policy_deployment,
        uses_curses=True,
    ),
    MenuItem(label="预留功能", description="等待后续功能接入。"),
)


def draw_menu(
    screen: curses.window,
    selected: int,
    status: str,
    robot_mode: RobotModeStatus,
    status_age_s: float | None,
) -> None:
    """Render the menu, adapting gracefully to small terminal windows."""

    screen.erase()
    rows, columns = screen.getmaxyx()
    title = "Zeno 人形机器人工具箱"
    subtitle = f"按 1–{len(MENU_ITEMS)} 或方向键选择，Enter 确认，q 退出。"
    age_text = "--" if status_age_s is None else f"{status_age_s:.1f}s"

    try:
        screen.addnstr(1, 2, title, max(columns - 4, 0), curses.A_BOLD)
        screen.addnstr(2, 2, subtitle, max(columns - 4, 0), curses.A_DIM)
        screen.addnstr(
            4,
            2,
            (
                f"机器人模式监测：{robot_mode.label} — {robot_mode.detail}"
                f"（状态年龄：{age_text}；r: 刷新）"
            ),
            max(columns - 4, 0),
            robot_mode_attribute(robot_mode),
        )
        for offset, line in enumerate(telemetry_display_lines(robot_mode.telemetry)):
            attribute = camera_status_attribute(robot_mode.telemetry) if offset == 0 else curses.A_DIM
            screen.addnstr(5 + offset, 2, line, max(columns - 4, 0), attribute)

        for index, item in enumerate(MENU_ITEMS):
            row = 10 + index * 2
            if row >= rows - 3:
                break
            attribute = (
                curses.color_pair(2) | curses.A_BOLD
                if index == selected
                else curses.color_pair(1)
            )
            screen.addnstr(row, 4, f"{index + 1}. {item.label}", max(columns - 8, 0), attribute)
            if row + 1 < rows - 3:
                screen.addnstr(row + 1, 6, item.description, max(columns - 10, 0), curses.A_DIM)

        if status:
            screen.addnstr(rows - 2, 2, status, max(columns - 4, 0), curses.A_BOLD)
    except curses.error:
        # A terminal can be resized between getmaxyx and addnstr.  The next
        # input event will redraw it at the new dimensions.
        pass
    screen.refresh()


def run_item(screen: curses.window, item: MenuItem) -> str:
    """Run a TUI-native action or suspend curses for an external terminal flow."""

    if item.action is None:
        return "该功能暂未开放。"

    if item.uses_curses:
        exit_code = item.action(screen)
    else:
        curses.def_prog_mode()
        curses.endwin()
        print(f"\nRunning: {item.description}\n")
        exit_code = item.action()
        curses.reset_prog_mode()
        screen.refresh()

    if exit_code == ALL_BAGS_DISCARDED_EXIT_CODE:
        return "已丢弃本次转换范围内的所有 Bag；原始数据未删除。"
    if exit_code == CANCELLED_EXIT_CODE:
        return "操作已取消。"
    if exit_code == 0:
        return "操作已完成。"
    return f"操作结束，退出码：{exit_code}。"


def main_loop(screen: curses.window) -> None:
    """Handle menu navigation and operation dispatch."""

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_YELLOW, -1)

    curses.curs_set(0)
    screen.keypad(True)
    screen.timeout(MENU_INPUT_TIMEOUT_MS)
    selected = 0
    status = ""
    mode_monitor = RobotModeMonitor()
    mode_monitor.refresh_if_due(force=True)

    while True:
        mode_monitor.refresh_if_due()
        robot_mode, status_age_s = mode_monitor.snapshot()

        draw_menu(screen, selected, status, robot_mode, status_age_s)
        key = screen.getch()

        if key in (ord("q"), ord("Q"), 27):
            return
        if key in (ord("r"), ord("R")):
            mode_monitor.refresh_if_due(force=True)
        elif ord("1") <= key <= ord(str(len(MENU_ITEMS))):
            selected = key - ord("1")
            screen.timeout(-1)
            try:
                status = run_item(screen, MENU_ITEMS[selected])
            finally:
                screen.timeout(MENU_INPUT_TIMEOUT_MS)
            mode_monitor.refresh_if_due(force=True)
        elif key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(MENU_ITEMS)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(MENU_ITEMS)
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            screen.timeout(-1)
            try:
                status = run_item(screen, MENU_ITEMS[selected])
            finally:
                screen.timeout(MENU_INPUT_TIMEOUT_MS)
            mode_monitor.refresh_if_due(force=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TUI launcher for Zeno workspace operations.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("This toolkit needs an interactive terminal.", file=sys.stderr)
        return 2
    if not os.environ.get("TERM"):
        print("This toolkit needs the TERM environment variable set.", file=sys.stderr)
        return 2

    try:
        curses.wrapper(main_loop)
    except curses.error as error:
        print(f"Unable to start terminal UI: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
