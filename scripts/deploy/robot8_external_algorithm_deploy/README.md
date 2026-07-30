# Robot8 外部算法部署接入说明（2026-07-29 相机链路）

这个目录是给“算法由你们自己实现、但需要控制同一台 Robot8”的接入说明。

推荐的边界很简单：**保留本仓库的 ROS bridge 和相机处理；把模型 worker 换成你们自己的本机 TCP 服务。** 这样你们不需要理解或修改本仓库的 ACT / LeRobot 代码，只要每次收到一组观测后返回与现有部署相同的 23 维原始 action 即可。

本目录提供：

- `external_algorithm_server_example.py`：可运行的本机 worker 模板。只在其中的 `ExternalPolicy.predict()` 接入你们的算法；未接入时它会安全地返回失败，绝不会用全零 action 冒充正常控制。
- `verify_topcam_contract.py`：用一帧原始头部 JPEG 验证去畸变、固定裁剪和 letterbox 是否与训练/现有部署一致。

> 给接入方的 Codex：请先完整阅读本文件，再只修改 `external_algorithm_server_example.py` 中标出的 `ExternalPolicy` 区域或新建同协议的 worker。不要改 ROS bridge 的 action 顺序、不要替换头部相机的几何链路、不要把 TCP 端口暴露到公网。

## 1. 最短接入路径

克隆时请带上 submodule；某些现有部署脚本依赖 `third_party/lerobot`。

```bash
git clone --recurse-submodules <此仓库地址>
cd 2027icra
git submodule update --init --recursive
```

你们自己的算法环境至少需要 `numpy` 和 `opencv-python`，并应满足自己的 GPU/模型依赖。ROS bridge 必须使用机器人上的 ROS 2 Python，而算法 worker 可以使用自己的 conda / venv 环境。

1. 在 `external_algorithm_server_example.py` 的 `ExternalPolicy` 中加载你们的模型，并实现 `predict(observation)`。
2. 启动你们的本机 worker（默认只能监听 `127.0.0.1:8768`）：

   ```bash
   cd <repo-root>
   python scripts/deploy/robot8_external_algorithm_deploy/external_algorithm_server_example.py
   ```

3. 另开终端，先运行 **dry-run** bridge。此时不会向真机发布控制命令：

   ```bash
   cd <repo-root>
   source /opt/ros/humble/setup.bash
   /usr/bin/python3 \
     scripts/deploy/robot8_20260729_tearoom_3cam_act_dinov3_topcam_left_cam20260729/bridge.py \
     --worker-host 127.0.0.1 --worker-port 8768 --log-full-action
   ```

4. 确认图像、state、action 顺序、单位、限幅均正确后，才显式加入 `--publish-commands`。

   ```bash
   /usr/bin/python3 \
     scripts/deploy/robot8_20260729_tearoom_3cam_act_dinov3_topcam_left_cam20260729/bridge.py \
     --worker-host 127.0.0.1 --worker-port 8768 \
     --publish-commands --log-full-action
   ```

bridge 固定以 **20 Hz** 工作。它在观测超过 0.5 s、worker 连接失败、worker 返回错误、action 长度不对或包含 NaN/Inf 时丢弃该轮输出并发布 idle。你们的算法仍必须把单轮推理控制在约 50 ms 内，并且自行处理动作的安全限幅与平滑。

## 2. 固定 TCP ABI：你们的算法只需遵守这一层

ROS bridge 和 worker 使用本机 TCP。每条消息是：

```text
4-byte big-endian payload length + Python pickle payload
```

这不是网络安全协议，**只能绑定在 localhost，绝不能暴露到局域网或公网**。

每个 20 Hz 周期，bridge 向 worker 发送：

```python
{
    "state": list[float],          # 23 个原始、未归一化的值
    "images": {
        "head_cam": bytes,         # 原始 2560x720 双目鱼眼 CompressedImage.data
        "left_arm_cam": bytes,     # 原始 CompressedImage.data
        "right_arm_cam": bytes,    # 原始 CompressedImage.data
    },
}
```

你们的 worker 必须在同一连接上返回：

```python
{
    "ok": True,
    "action": list[float],         # **恰好 23 个**、有限、原始未归一化的控制值
    "latency_s": 0.012,            # 可选但推荐；单位秒
}
```

遇到任何不能安全推理的情况，返回失败而不是猜测动作：

```python
{"ok": False, "error": "具体原因"}
```

模板已经实现了这一 framing、输入检查、图像预处理和 action 长度/有限性检查。通常只需改 `ExternalPolicy.__init__()`、`ExternalPolicy.predict()`，以及你们自己的模型文件路径。

## 3. 23D state 与 action 的严格顺序

`state` 与 `action` 使用同一顺序。不要重排、归一化后直接输出、插入额外维度，或把 action chunk 当作单步 action 返回。

| 索引 | 名称 | state 来源 | action 语义 |
| --- | --- | --- | --- |
| 0 | `torso_lift` | torso `JointState.position` | torso lift command |
| 1 | `torso_waist` | torso `JointState.position` | torso waist command |
| 2 | `head_pan` | torso `JointState.position`（兼容别名 `torso_head_pan`） | head pan command |
| 3 | `head_tilt` | torso `JointState.position`（兼容别名 `torso_head_tilt`） | head tilt command |
| 4–10 | `left_arm_j0` … `left_arm_j6` | left arm `JointState.position` | left arm joint commands |
| 11–17 | `right_arm_j0` … `right_arm_j6` | right arm `JointState.position` | right arm joint commands |
| 18 | `left_gripper` | left gripper `JointState.position`（兼容别名 `left_arm_gripper`） | left gripper command |
| 19 | `right_gripper` | right gripper `JointState.position`（兼容别名 `right_arm_gripper`） | right gripper command |
| 20 | `base_vx` | odom `twist.linear.x` | low-level base `linear.x` command |
| 21 | `base_vy` | odom `twist.linear.y` | low-level base `linear.y` command |
| 22 | `base_rotation` | odom `twist.angular.z` | low-level base `angular.z` command |

前 20 维 state 是原始关节位置；最后三维 state 是 odometry 速度。action 也必须是原始控制量，**不是**训练时的归一化量，也不是 V2 标签所称的“期望物理速度”。

bridge 会将你们返回的 23D action 封装为 24D ROS 消息并发布：

```text
topic: /zeno/h1/auto/wholebody/cmd
type:  std_msgs/msg/Float64MultiArray
data:  [control_mode, action[0], action[1], ..., action[22]]
```

其中 `control_mode=1.0` 是 active，`0.0` 是 idle。现有桥接层仅检查“23D 且有限”；原 ACT worker 还有基于训练集范围的 clamp，但外部算法没有。因此**限幅、速度限制、加速度/jerk 限制、碰撞安全由你们的算法负责**。

## 4. 头部相机：必须使用 cam_20260729 去畸变链路

这是最容易出错、也不能替换的部分。bridge 发给 worker 的 `head_cam` 是一张未经处理的 ROS compressed JPEG。解码后必须为 **2560×720、H×W×3 BGR**，画面从左到右严格为：

```text
原始 JPEG (BGR, 2560x720)
        left raw fisheye 1280x720 | right raw fisheye 1280x720
```

外部算法若使用头部视觉，必须复用此仓库的实现和标定：

```text
scripts/data_convert/topcam_stereo_rectify_cam_20260729.py
scripts/data_convert/cam/stereo_params_20260729_172611.npz
```

标定文件 SHA-256 必须为：

```text
6d08b6a01a1431476c2c3c77bee43e3f8f20888f33940af772cf1963e9f6b342
```

唯一正确的图像几何链路是：

```text
raw 2560x720 left|right JPEG
  -> split: left [:, :1280] and right [:, 1280:]
  -> OpenCV fisheye rectify/alignment with the 2026-07-29 NPZ
     (left uses K1/D1/R1/P1; the pair was jointly calibrated)
  -> fixed valid-field crop on the rectified left image:
     x=20, y=0, width=1240, height=620
  -> BGR -> RGB
  -> preserve aspect ratio, letterbox to 640x480 with cv2.INTER_LINEAR
  -> uint8 RGB HWC (480, 640, 3)
```

1240×620 在 640×480 下会变成 **640×320**，并填充为 `(480, 640, 3)` 的全零 RGB canvas，所以顶部和底部各有 **80 px 黑边**。这是训练时的几何，不是可选视觉增强。

直接复用的 API 如下；模板也已经这样做：

```python
from pathlib import Path
import sys

repo_root = Path("<repo-root>")
sys.path.insert(0, str(repo_root / "scripts" / "data_convert"))
from topcam_stereo_rectify_cam_20260729 import TopStereoRectifier

rectifier = TopStereoRectifier(
    repo_root / "scripts" / "data_convert" / "cam" / "stereo_params_20260729_172611.npz"
)
head_rgb = rectifier.decode_and_rectify_left_rgb(
    raw_compressed_jpeg_bytes,
    (640, 480),
    resize_mode="letterbox",
)
# head_rgb.dtype == uint8; head_rgb.shape == (480, 640, 3); RGB
```

如果你们的模型需要 float tensor，则在这一步之后再按自身训练约定做数值归一化。例如当前 ACT 模型在交给 checkpoint 内 normalizer 前使用 `float32 / 255.0`、CHW，形状为 `(3, 480, 640)`。你们可以使用不同的数值归一化，但**不能改变上述去畸变、裁剪、左眼选择或 letterbox 几何**。

左/右臂相机不走这条鱼眼去畸变链。现有三相机部署对它们使用：原始 compressed bytes -> BGR decode -> RGB -> 直接 resize 到 640×480（无 crop）。模板也实现了这一默认约定；若你们自己的模型训练时采用不同的臂部相机尺度，请只在 `ExternalPolicy` 内适配，保持头部相机链路不变。

### 明确禁止

- 不要把左半张预裁剪 JPEG 作为 `head_cam` 发给 worker。
- 不要预先把图片去畸变后再交给模板，否则会二次校正。
- 不要把左右眼重新拼成“模型输入图”，也不要把右眼当作单眼模型的第二个视觉 key。
- 不要把固定裁剪替换为 center crop，或把 1240×620 直接拉伸到 640×480。
- 不要误用旧的 `topcam_stereo_rectify.py` / `data_process_20260729` JSON 相机链路；本轮必须是 `topcam_stereo_rectify_cam_20260729.py` + NPZ。

## 5. 在真机前验证相机链路

先核验标定文件：

```bash
cd <repo-root>
sha256sum scripts/data_convert/cam/stereo_params_20260729_172611.npz
```

然后选一帧来自 `/zeno/h1/sensor/head_cam/image/compressed` 的**原始** 2560×720 JPEG，运行：

```bash
python scripts/deploy/robot8_external_algorithm_deploy/verify_topcam_contract.py \
  --input /path/to/raw_head_stereo.jpg \
  --output /tmp/robot8_topcam_left_640x480.png \
  --output-rgb-npy /tmp/robot8_topcam_left_640x480.npy
```

输出 PNG 应为 640×480；可见内容应为校正后的左眼，且上下存在 80 px 黑边。NPY 是 RGB `uint8` HWC 数组，可用来与外部算法实际收到的输入逐元素比对。

## 6. ROS 话题与启动安全检查

默认 bridge 使用如下话题；需要时可通过 bridge 参数覆盖，但请先确认机器人版本一致。

| 角色 | 默认话题 | ROS 类型 |
| --- | --- | --- |
| head camera | `/zeno/h1/sensor/head_cam/image/compressed` | `sensor_msgs/msg/CompressedImage` |
| left arm camera | `/zeno/h1/sensor/left_arm_cam/image/compressed` | `sensor_msgs/msg/CompressedImage` |
| right arm camera | `/zeno/h1/sensor/right_arm_cam/image/compressed` | `sensor_msgs/msg/CompressedImage` |
| odometry | `/zeno/h1/sensor/odom_raw` | `nav_msgs/msg/Odometry` |
| torso state | `/zeno/h1/wheelarm/torso/joint_state` | `sensor_msgs/msg/JointState` |
| left arm state | `/zeno/h1/wheelarm/left_arm/joint_state` | `sensor_msgs/msg/JointState` |
| right arm state | `/zeno/h1/wheelarm/right_arm/joint_state` | `sensor_msgs/msg/JointState` |
| left gripper state | `/zeno/h1/left_gripper/joint_state` | `sensor_msgs/msg/JointState` |
| right gripper state | `/zeno/h1/right_gripper/joint_state` | `sensor_msgs/msg/JointState` |
| command output | `/zeno/h1/auto/wholebody/cmd` | `std_msgs/msg/Float64MultiArray` |

真机流程必须遵守：

1. 校验 calibration SHA、相机 output shape、state/action 23D 顺序。
2. 先不加 `--publish-commands`，看 worker 日志和 bridge 的 `--log-full-action`。
3. 在机器人安全姿态、周围无障碍、速度/关节范围已在你们算法中限幅时，才加 `--publish-commands`。
4. 任意异常、视觉 stale、模型异常或 Ctrl-C 时，bridge 会尝试发布 24D idle command；不要把这当作唯一的安全措施。

## 7. 给接入方 Codex 的完成标准

在声称“部署完成”之前，请逐项验证：

- [ ] worker 仅监听 `127.0.0.1`，没有对外暴露 pickle TCP 服务；
- [ ] 每个 request 都验证 state 为 23 个有限浮点数，action 也为 23 个有限浮点数；
- [ ] `head_cam` 从原始 2560×720 left|right JPEG 经 **cam_20260729 NPZ** 处理成左眼 RGB 640×480 letterbox；
- [ ] 没有 center crop、没有拉伸、没有二次去畸变、没有右眼视觉输入；
- [ ] 外部算法返回的是原始 low-level action，不是归一化值、action chunk 或 V2 物理速度标签；
- [ ] dry-run 时 bridge 能连续以 20 Hz 接收有效 response；
- [ ] 真实发布前已在外部算法中配置关节、夹爪、底盘的安全限幅与平滑。

现有可参考的正式部署实现是：

```text
scripts/deploy/robot8_20260729_tearoom_3cam_act_dinov3_topcam_left_cam20260729/
scripts/deploy/robot8_act_dinov3_base_frozen_100k/
```

它们是此 ABI 和相机几何的权威参考；外部算法只替换 worker 内部，不替换 bridge 或相机标定。
