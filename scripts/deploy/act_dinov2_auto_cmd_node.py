#!/usr/bin/env python3
"""
ROS2 deployment node for the humanmoid_pick ACT DINOv2 policy.

Input observation layout matches the LeRobot dataset:
  observation.state: 23D auto-cmd payload without control_mode
  observation.images.{head_cam,left_arm_cam,right_arm_cam}: 3x224x224 RGB

Output command layout:
  /zeno/h1/auto/wholebody/cmd Float64MultiArray, 24D
  [0] control_mode, [1..23] policy action
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Float64MultiArray
except ImportError as exc:  # pragma: no cover - import is environment-specific.
    raise SystemExit(
        "ROS2 Python packages are missing. Run this script inside your ROS2 environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402


NORM_EPS = 1e-6
ACTION_DIM = 23
COMMAND_DIM = 24

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "train"
DEFAULT_RUN_PREFIX = "humanmoid_pick_act_dinov2_auto_cmd23"

CMD_TOPIC = "/zeno/h1/auto/wholebody/cmd"
CAM_HEAD_TOPIC = "/zeno/h1/sensor/head_cam/image/compressed"
CAM_LEFT_ARM_TOPIC = "/zeno/h1/sensor/left_arm_cam/image/compressed"
CAM_RIGHT_ARM_TOPIC = "/zeno/h1/sensor/right_arm_cam/image/compressed"
ODOM_TOPIC = "/zeno/h1/sensor/odom_raw"
STATE_TORSO_TOPIC = "/zeno/h1/wheelarm/torso/joint_state"
STATE_LEFT_ARM_TOPIC = "/zeno/h1/wheelarm/left_arm/joint_state"
STATE_RIGHT_ARM_TOPIC = "/zeno/h1/wheelarm/right_arm/joint_state"
STATE_LEFT_GRIPPER_TOPIC = "/zeno/h1/left_gripper/joint_state"
STATE_RIGHT_GRIPPER_TOPIC = "/zeno/h1/right_gripper/joint_state"

TORSO_FIELDS = ["torso_lift", "torso_waist", "head_pan", "head_tilt"]
LEFT_ARM_FIELDS = [f"left_arm_j{i}" for i in range(7)]
RIGHT_ARM_FIELDS = [f"right_arm_j{i}" for i in range(7)]
LEFT_GRIPPER_FIELDS = ["left_gripper"]
RIGHT_GRIPPER_FIELDS = ["right_gripper"]
BASE_FIELDS = ["base_vx", "base_vy", "base_rotation"]
ACTION_FIELDS = (
    TORSO_FIELDS
    + LEFT_ARM_FIELDS
    + RIGHT_ARM_FIELDS
    + LEFT_GRIPPER_FIELDS
    + RIGHT_GRIPPER_FIELDS
    + BASE_FIELDS
)

JOINT_NAME_ALIASES = {
    "head_pan": ["torso_head_pan"],
    "head_tilt": ["torso_head_tilt"],
    "left_gripper": ["left_arm_gripper"],
    "right_gripper": ["right_arm_gripper"],
}

IMAGE_KEYS = {
    "head_cam": "observation.images.head_cam",
    "left_arm_cam": "observation.images.left_arm_cam",
    "right_arm_cam": "observation.images.right_arm_cam",
}

STATE_MEAN = [
    0.06629907019526637,
    -0.05590814082025135,
    0.11646821813029283,
    -0.6266132368715587,
    0.057981648629609396,
    0.045238616382455014,
    -0.04236767465822407,
    0.08916511262711597,
    -0.04190582294810735,
    0.007063568155413458,
    0.002217752039731857,
    -0.06180571561676185,
    -0.06819940049314288,
    0.0540851632584586,
    1.0941904276011145,
    0.02510407164758101,
    0.09202907995133942,
    -0.003169012435936132,
    0.08175789400053053,
    1.213155426491627,
    0.012292727681802864,
    0.018929421771833183,
    0.0030569169527993587,
]
STATE_STD = [
    0.15409339708025388,
    0.06973948412041121,
    0.1567311718276111,
    4.5253110222064015,
    0.06598179155410612,
    0.04575642793076321,
    0.05023995200247108,
    0.13098848817304856,
    0.060494029800189794,
    0.06458363415167175,
    0.08812887392272124,
    0.32200123709527795,
    0.04918729363212187,
    0.07198849763681336,
    0.7306975594472097,
    0.07365033358428291,
    0.11591193587736273,
    0.22251401856895184,
    0.019694628447070185,
    0.9797960658418878,
    0.03605304428207425,
    0.044003668821573624,
    0.04177032635031798,
]
STATE_MIN = [
    -0.46292057633399963,
    -0.325970858335495,
    0.050164032727479935,
    -12.5,
    -0.1348516047000885,
    -0.14553292095661163,
    -0.2010619342327118,
    0.013194689527153969,
    -0.2447165697813034,
    -0.18940261006355286,
    -0.612840473651886,
    -1.2941558361053467,
    -0.14286258816719055,
    -0.2079734355211258,
    -0.03769911080598831,
    -0.24090181291103363,
    -0.2782863974571228,
    -0.7230868935585022,
    0.06618601083755493,
    0.07724879682064056,
    -0.08328412473201752,
    -0.019552122801542282,
    -0.3921944499015808,
]
STATE_MAX = [
    0.5548561811447144,
    0.19054703414440155,
    1.2407492399215698,
    12.49542236328125,
    0.5125123858451843,
    0.23899443447589874,
    0.10430087894201279,
    1.8767874240875244,
    0.11387044936418533,
    0.21648737788200378,
    0.6689173579216003,
    1.0660333633422852,
    0.134470134973526,
    0.24190263450145721,
    2.425309419631958,
    0.30117493867874146,
    0.3583962619304657,
    1.0400930643081665,
    1.725986123085022,
    2.1730754375457764,
    0.17503295838832855,
    0.2400037944316864,
    0.5401923656463623,
]

ACTION_MEAN = [
    -0.10365233838721921,
    0.005942048270958422,
    -0.05605904515871872,
    0.06571494646222371,
    0.0639829378394914,
    0.04378487446630782,
    -0.04039570103033145,
    0.0579352438273034,
    -0.04256340922866348,
    0.009985926210418945,
    -0.000880431673688093,
    -0.08291973692791084,
    -0.11328279076740563,
    0.044875480293795914,
    1.1287692557671676,
    0.028022327834596628,
    0.09329303081958389,
    0.06336851246769414,
    0.0003146091873079551,
    1.6830681658332192,
    0.012292727681802864,
    0.018929421771833183,
    0.0030569169527993587,
]
ACTION_STD = [
    0.18232188102310073,
    0.05465406304543733,
    0.07004855443912195,
    0.15955475716193016,
    0.09811029229217647,
    0.04872495984705921,
    0.06100737530421967,
    0.1415972983946157,
    0.06144951709354732,
    0.06482474507664222,
    0.08863812066931381,
    0.33707291526284777,
    0.09921649491013156,
    0.07615506879940818,
    0.7328309614754072,
    0.07947317622954973,
    0.12042855074696196,
    0.2125301371944121,
    0.02708957437238535,
    1.4588431248692368,
    0.03605304428207425,
    0.044003668821573624,
    0.04177032635031798,
]
ACTION_MIN = [
    -0.599999189376831,
    -0.08729226887226105,
    -0.3251272737979889,
    -0.6976150870323181,
    -0.14012131094932556,
    -0.14061851799488068,
    -0.23400533199310303,
    -4.1348270662243294e-16,
    -0.24915514886379242,
    -0.19362764060497284,
    -0.6066493988037109,
    -1.3754724264144897,
    -0.34757131338119507,
    -0.25229406356811523,
    -6.38047221330656e-17,
    -0.2511748969554901,
    -0.2860746681690216,
    -0.7242950201034546,
    0.0,
    0.0,
    -0.08328412473201752,
    -0.019552122801542282,
    -0.3921944499015808,
]
ACTION_MAX = [
    0.0,
    0.3146280348300934,
    0.19017180800437927,
    0.6016148924827576,
    1.1035407781600952,
    0.2538188099861145,
    0.12104932963848114,
    1.9046839475631714,
    0.11672401428222656,
    0.22020936012268066,
    0.6801795959472656,
    1.0807044506072998,
    0.14945131540298462,
    0.25256162881851196,
    2.4273970127105713,
    0.3349626362323761,
    0.36604490876197815,
    1.0446527004241943,
    2.950000047683716,
    2.950000047683716,
    0.17503295838832855,
    0.2400037944316864,
    0.5401923656463623,
]

HEAD_CAM_MEAN = [0.49867124790837025, 0.5207219249645205, 0.4707800516668092]
HEAD_CAM_STD = [0.004661754117081969, 0.0030363431985702665, 0.0014945382795570614]
LEFT_ARM_CAM_MEAN = [0.5278932394937336, 0.5470216758385095, 0.5114780027802325]
LEFT_ARM_CAM_STD = [0.0048412498545484005, 0.004683366412521938, 0.005658771356362135]
RIGHT_ARM_CAM_MEAN = [0.21465043091970087, 0.2286127294677534, 0.22356414038153136]
RIGHT_ARM_CAM_STD = [0.09429959684155328, 0.10075341821000729, 0.088536258845907]


def _safe_std(values: list[float]) -> list[float]:
    return [max(abs(float(value)), NORM_EPS) for value in values]


DEPLOY_STATS = {
    "observation.state": {
        "mean": STATE_MEAN,
        "std": _safe_std(STATE_STD),
        "min": STATE_MIN,
        "max": STATE_MAX,
    },
    "action": {
        "mean": ACTION_MEAN,
        "std": _safe_std(ACTION_STD),
        "min": ACTION_MIN,
        "max": ACTION_MAX,
    },
    "observation.images.head_cam": {
        "mean": HEAD_CAM_MEAN,
        "std": _safe_std(HEAD_CAM_STD),
    },
    "observation.images.left_arm_cam": {
        "mean": LEFT_ARM_CAM_MEAN,
        "std": _safe_std(LEFT_ARM_CAM_STD),
    },
    "observation.images.right_arm_cam": {
        "mean": RIGHT_ARM_CAM_MEAN,
        "std": _safe_std(RIGHT_ARM_CAM_STD),
    },
}


def _latest_default_run_dir() -> Path:
    if not DEFAULT_OUTPUT_DIR.is_dir():
        return DEFAULT_OUTPUT_DIR / DEFAULT_RUN_PREFIX
    runs = sorted(
        path for path in DEFAULT_OUTPUT_DIR.glob(f"{DEFAULT_RUN_PREFIX}_*") if path.is_dir()
    )
    if runs:
        return runs[-1]
    return DEFAULT_OUTPUT_DIR / DEFAULT_RUN_PREFIX


def _resolve_checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()

    if (candidate / "model.safetensors").is_file():
        return candidate
    if (candidate / "pretrained_model" / "model.safetensors").is_file():
        return candidate / "pretrained_model"

    checkpoints_dir = candidate / "checkpoints"
    if checkpoints_dir.is_dir():
        checkpoints: list[tuple[int, str, Path]] = []
        for step_dir in checkpoints_dir.iterdir():
            pretrained = step_dir / "pretrained_model"
            if not (step_dir.is_dir() and (pretrained / "model.safetensors").is_file()):
                continue
            try:
                step = int(step_dir.name)
            except ValueError:
                step = -1
            checkpoints.append((step, step_dir.name, pretrained))
        if checkpoints:
            checkpoints.sort(key=lambda item: (item[0], item[1]))
            return checkpoints[-1][2]

    raise FileNotFoundError(
        "Could not find model.safetensors. Pass a run dir, checkpoint dir, "
        f"or pretrained_model dir. Got: {candidate}"
    )


def _decode_compressed_image(msg: CompressedImage, image_size: int) -> np.ndarray | None:
    buffer = np.frombuffer(msg.data, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))


def _extract_joint_positions(msg: JointState, fields: list[str]) -> np.ndarray | None:
    if len(msg.position) == 0:
        return None

    if msg.name:
        name_to_position = {
            name: float(position) for name, position in zip(msg.name, msg.position, strict=False)
        }
        values = []
        for field in fields:
            names = [field, *JOINT_NAME_ALIASES.get(field, [])]
            value = next((name_to_position[name] for name in names if name in name_to_position), None)
            if value is None:
                return None
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    if len(msg.position) < len(fields):
        return None
    return np.asarray(msg.position[: len(fields)], dtype=np.float32)


def _odom_velocity(msg: Odometry) -> np.ndarray:
    twist = msg.twist.twist
    return np.asarray(
        [twist.linear.x, twist.linear.y, twist.angular.z],
        dtype=np.float32,
    )


def _set_norm_eps(pipeline, eps: float) -> None:
    for step in getattr(pipeline, "steps", []):
        if hasattr(step, "eps"):
            step.eps = eps


def _as_ros_param_path(value: str) -> str:
    return str(Path(value).expanduser()) if value else value


class ActDinoV2AutoCmdNode(Node):
    def __init__(self) -> None:
        super().__init__("act_dinov2_auto_cmd_node")

        self.declare_parameter("checkpoint_path", str(_latest_default_run_dir()))
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("use_amp", True)
        self.declare_parameter("publish_commands", False)
        self.declare_parameter("publish_idle_on_stale", True)
        self.declare_parameter("control_mode", 1.0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("image_size", 224)
        self.declare_parameter("max_obs_age_s", 0.5)
        self.declare_parameter("clamp_actions", True)
        self.declare_parameter("action_clip_margin", 0.05)
        self.declare_parameter("log_every_n", 20)
        self.declare_parameter("cmd_topic", CMD_TOPIC)
        self.declare_parameter("head_cam_topic", CAM_HEAD_TOPIC)
        self.declare_parameter("left_arm_cam_topic", CAM_LEFT_ARM_TOPIC)
        self.declare_parameter("right_arm_cam_topic", CAM_RIGHT_ARM_TOPIC)
        self.declare_parameter("odom_topic", ODOM_TOPIC)
        self.declare_parameter("torso_state_topic", STATE_TORSO_TOPIC)
        self.declare_parameter("left_arm_state_topic", STATE_LEFT_ARM_TOPIC)
        self.declare_parameter("right_arm_state_topic", STATE_RIGHT_ARM_TOPIC)
        self.declare_parameter("left_gripper_state_topic", STATE_LEFT_GRIPPER_TOPIC)
        self.declare_parameter("right_gripper_state_topic", STATE_RIGHT_GRIPPER_TOPIC)

        checkpoint_param = self.get_parameter("checkpoint_path").value
        self.checkpoint_path = _resolve_checkpoint_path(_as_ros_param_path(str(checkpoint_param)))
        self.device = str(self.get_parameter("device").value)
        self.use_amp = bool(self.get_parameter("use_amp").value)
        self.publish_commands = bool(self.get_parameter("publish_commands").value)
        self.publish_idle_on_stale = bool(self.get_parameter("publish_idle_on_stale").value)
        self.control_mode = float(self.get_parameter("control_mode").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.max_obs_age_s = float(self.get_parameter("max_obs_age_s").value)
        self.clamp_actions = bool(self.get_parameter("clamp_actions").value)
        self.action_clip_margin = float(self.get_parameter("action_clip_margin").value)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))

        self.image_cache: dict[str, tuple[np.ndarray, float]] = {}
        self.joint_cache: dict[str, tuple[JointState, float]] = {}
        self.odom_cache: tuple[Odometry, float] | None = None
        self.subscriptions_keepalive = []
        self.infer_count = 0
        self.stale_publish_count = 0

        self.action_min = np.asarray(ACTION_MIN, dtype=np.float32)
        self.action_max = np.asarray(ACTION_MAX, dtype=np.float32)
        action_range = np.maximum(self.action_max - self.action_min, NORM_EPS)
        self.action_low = self.action_min - self.action_clip_margin * action_range
        self.action_high = self.action_max + self.action_clip_margin * action_range

        self.policy, self.preprocessor, self.postprocessor = self._load_policy()

        cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.publisher = self.create_publisher(Float64MultiArray, cmd_topic, 10)
        self._create_subscriptions()
        self.timer = self.create_timer(1.0 / self.rate_hz, self._timer_cb)

        mode = "PUBLISH" if self.publish_commands else "DRY-RUN"
        self.get_logger().info(
            f"Loaded {self.checkpoint_path}; mode={mode}; device={self.device}; "
            f"rate={self.rate_hz:.1f} Hz"
        )

    def _load_policy(self):
        config = PreTrainedConfig.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
        )
        config.device = self.device

        # The checkpoint already contains DINOv2 weights; disable fresh timm downloads.
        if hasattr(config, "dinov2_pretrained"):
            config.dinov2_pretrained = False

        policy_class = get_policy_class(config.type)
        policy = policy_class.from_pretrained(
            self.checkpoint_path,
            config=config,
            local_files_only=True,
        )
        policy.reset()

        preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=DEPLOY_STATS)
        _set_norm_eps(preprocessor, NORM_EPS)
        _set_norm_eps(postprocessor, NORM_EPS)
        return policy, preprocessor, postprocessor

    def _create_subscriptions(self) -> None:
        image_topics = {
            "head_cam": str(self.get_parameter("head_cam_topic").value),
            "left_arm_cam": str(self.get_parameter("left_arm_cam_topic").value),
            "right_arm_cam": str(self.get_parameter("right_arm_cam_topic").value),
        }
        for name, topic in image_topics.items():
            self.subscriptions_keepalive.append(
                self.create_subscription(
                    CompressedImage,
                    topic,
                    self._image_callback(name),
                    10,
                )
            )

        joint_topics = {
            "torso": str(self.get_parameter("torso_state_topic").value),
            "left_arm": str(self.get_parameter("left_arm_state_topic").value),
            "right_arm": str(self.get_parameter("right_arm_state_topic").value),
            "left_gripper": str(self.get_parameter("left_gripper_state_topic").value),
            "right_gripper": str(self.get_parameter("right_gripper_state_topic").value),
        }
        for name, topic in joint_topics.items():
            self.subscriptions_keepalive.append(
                self.create_subscription(
                    JointState,
                    topic,
                    self._joint_callback(name),
                    10,
                )
            )

        odom_topic = str(self.get_parameter("odom_topic").value)
        self.subscriptions_keepalive.append(
            self.create_subscription(Odometry, odom_topic, self._odom_callback, 10)
        )

    def _image_callback(self, name: str) -> Callable[[CompressedImage], None]:
        def callback(msg: CompressedImage) -> None:
            image = _decode_compressed_image(msg, self.image_size)
            if image is None:
                self.get_logger().warn(f"Failed to decode {name} image")
                return
            self.image_cache[name] = (image, time.monotonic())

        return callback

    def _joint_callback(self, name: str) -> Callable[[JointState], None]:
        def callback(msg: JointState) -> None:
            self.joint_cache[name] = (msg, time.monotonic())

        return callback

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom_cache = (msg, time.monotonic())

    def _build_state(self) -> np.ndarray | None:
        required_joints = {
            "torso": TORSO_FIELDS,
            "left_arm": LEFT_ARM_FIELDS,
            "right_arm": RIGHT_ARM_FIELDS,
            "left_gripper": LEFT_GRIPPER_FIELDS,
            "right_gripper": RIGHT_GRIPPER_FIELDS,
        }

        parts = []
        for name, fields in required_joints.items():
            cached = self.joint_cache.get(name)
            if cached is None:
                return None
            msg, _ = cached
            positions = _extract_joint_positions(msg, fields)
            if positions is None:
                self.get_logger().warn(f"Missing joint fields for {name}: {fields}")
                return None
            parts.append(positions)

        if self.odom_cache is None:
            return None
        odom_msg, _ = self.odom_cache
        parts.append(_odom_velocity(odom_msg))

        state = np.concatenate(parts).astype(np.float32)
        if state.shape != (ACTION_DIM,):
            self.get_logger().warn(f"Unexpected state shape: {state.shape}")
            return None
        if not np.isfinite(state).all():
            self.get_logger().warn("State contains NaN or Inf")
            return None
        return state

    def _is_cache_fresh(self) -> bool:
        now = time.monotonic()
        for name in IMAGE_KEYS:
            cached = self.image_cache.get(name)
            if cached is None or now - cached[1] > self.max_obs_age_s:
                return False

        for name in ["torso", "left_arm", "right_arm", "left_gripper", "right_gripper"]:
            cached = self.joint_cache.get(name)
            if cached is None or now - cached[1] > self.max_obs_age_s:
                return False

        if self.odom_cache is None or now - self.odom_cache[1] > self.max_obs_age_s:
            return False
        return True

    def _build_observation(self) -> dict[str, torch.Tensor] | None:
        if not self._is_cache_fresh():
            return None

        state = self._build_state()
        if state is None:
            return None

        observation: dict[str, torch.Tensor] = {
            "observation.state": torch.from_numpy(state),
        }
        for name, key in IMAGE_KEYS.items():
            image, _ = self.image_cache[name]
            observation[key] = torch.from_numpy(image)
        return observation

    def _select_action(self, observation: dict[str, torch.Tensor]) -> np.ndarray | None:
        batch = self.preprocessor(observation)
        device_type = self.device.split(":", maxsplit=1)[0]
        autocast_enabled = self.use_amp and device_type == "cuda"

        with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=autocast_enabled):
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)

        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)

        action_np = np.squeeze(action_np).astype(np.float32)
        if action_np.shape != (ACTION_DIM,):
            self.get_logger().warn(f"Unexpected action shape: {action_np.shape}")
            return None
        if not np.isfinite(action_np).all():
            self.get_logger().warn("Action contains NaN or Inf")
            return None

        if self.clamp_actions:
            action_np = np.clip(action_np, self.action_low, self.action_high)
        return action_np

    def _publish_command(self, action: np.ndarray, control_mode: float | None = None) -> None:
        msg = Float64MultiArray()
        command = np.zeros(COMMAND_DIM, dtype=np.float64)
        command[0] = self.control_mode if control_mode is None else control_mode
        command[1:] = action.astype(np.float64)
        msg.data = command.tolist()

        if self.publish_commands:
            self.publisher.publish(msg)

    def _publish_idle(self) -> None:
        if not (self.publish_commands and self.publish_idle_on_stale):
            return
        self.stale_publish_count += 1
        self._publish_command(np.zeros(ACTION_DIM, dtype=np.float32), control_mode=0.0)
        if self.stale_publish_count % self.log_every_n == 1:
            self.get_logger().warn("Observation is stale; publishing idle command")

    def _timer_cb(self) -> None:
        observation = self._build_observation()
        if observation is None:
            self._publish_idle()
            return

        self.stale_publish_count = 0
        action = self._select_action(observation)
        if action is None:
            self._publish_idle()
            return

        self._publish_command(action)
        self.infer_count += 1
        if self.infer_count % self.log_every_n == 1:
            sample = ", ".join(
                f"{name}={value:.4f}" for name, value in zip(ACTION_FIELDS[:6], action[:6], strict=False)
            )
            self.get_logger().info(f"action[{self.infer_count}]: {sample}, ...")


def main() -> None:
    if len(ACTION_FIELDS) != ACTION_DIM:
        raise RuntimeError(f"ACTION_FIELDS has {len(ACTION_FIELDS)} fields, expected {ACTION_DIM}")
    if any(not math.isfinite(value) for value in STATE_STD + ACTION_STD):
        raise RuntimeError("Normalization std contains NaN or Inf")

    rclpy.init()
    node = ActDinoV2AutoCmdNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
