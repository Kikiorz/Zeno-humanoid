# Robot8：给外部算法的独立部署包

**这个文件夹可以整体复制给对方，对方不需要 clone 本仓库，也不需要安装 ACT、LeRobot 或下载我们的模型。**

对方只需要把自己的算法接到 `external_algorithm_server_example.py` 里，并让算法每次输出与 Robot8 相同的 **23 个原始 action**。其余的机器人通信、20 Hz 调度、图像去畸变和 ROS 发布都已经在这个文件夹中。

## 先理解两个程序

```text
Robot ROS topics
      │
      ▼
robot8_ros_bridge.py       ← bridge：负责和机器人说话，通常不要改
      │  本机 TCP，20 Hz
      ▼
external_algorithm_server_example.py
      │  去畸变头部相机 + 调用你们的算法
      ▼
你们自己的模型
      │  返回 23D action
      └────────────────────→ bridge 发布给机器人
```

### bridge 是什么？

`robot8_ros_bridge.py` 是机器人侧的“小翻译器”：

1. 从 ROS 收集三路相机、关节 state 和 odometry；
2. 每秒 20 次把这些观测发给 worker；
3. 收到 worker 的 23D action 后，封装成 Robot8 的 24D ROS 命令；
4. 默认只打印（dry-run），只有显式加 `--publish-commands` 才会控制真机；
5. 图像/state 过期、worker 出错、action 非法时，不发 active command，并在真机发布模式下发送 idle。

### worker 是什么？

`external_algorithm_server_example.py` 是模型侧的“小适配器”：

1. 收到 bridge 的 state 与原始 JPEG；
2. 对头部双目鱼眼执行固定的去畸变流程；
3. 调用你们自己的模型；
4. 返回一个 23D action。

**对方通常只改 worker 里 `ExternalPolicy` 的 `__init__()` 和 `predict()`。不要把 ROS 订阅、发布或相机几何塞进模型代码。**

## 文件夹内容

```text
robot8_external_algorithm_deploy/
├── README.md                              # 本说明
├── robot8_ros_bridge.py                   # 独立 ROS2 bridge（不依赖本仓库）
├── external_algorithm_server_example.py   # 你们改这里接入模型
├── verify_topcam_contract.py               # 检查去畸变结果
├── requirements-worker.txt                 # worker 最小 Python 依赖
└── camera/
    ├── topcam_stereo_rectify_cam_20260729.py
    └── stereo_params_20260729_172611.npz  # 已附带的标定文件
```

复制时一定复制**整个**文件夹，尤其不要漏掉 `camera/`。

```bash
# 例：从 U 盘、scp、rsync 或文件共享复制整个目录都可以
rsync -a robot8_external_algorithm_deploy/ /target/path/robot8_external_algorithm_deploy/
```

## 最短启动流程

### 1. 在 worker 里接入你们的算法

打开 `external_algorithm_server_example.py`，找到：

```python
class ExternalPolicy:
    def __init__(self) -> None:
        # 在这里加载你们自己的模型
        pass

    def predict(self, observation: RobotObservation) -> np.ndarray:
        # 输入：observation.state 和三个 RGB 图像
        # 输出：一个原始、有限的 23D action
        raise NotImplementedError
```

把它改为你们的模型调用。例如模型最终需要的内容是：

```text
observation.state               float32, shape (23,)
observation.head_cam_rgb        uint8 RGB, shape (480, 640, 3)
observation.left_arm_cam_rgb    uint8 RGB, shape (480, 640, 3)
observation.right_arm_cam_rgb   uint8 RGB, shape (480, 640, 3)
```

`predict()` 必须返回一个能转为 `float32` 的 `(23,)` 数组/列表。它必须是**原始控制量**，不是归一化值，不是 `[chunk, 23]` 的动作序列。

请在你们自己的模型里做关节、夹爪和底盘的限幅、速度限制与平滑。bridge 只会检查“23 个有限数”，不知道你们的安全范围。

### 2. 启动 worker

在含有你们模型依赖的 Python 环境中：

```bash
cd /target/path/robot8_external_algorithm_deploy
python -m pip install -r requirements-worker.txt

# 先确认你已经在 ExternalPolicy 中接入了真实模型
python external_algorithm_server_example.py
```

worker 只监听 `127.0.0.1:8768`。TCP 使用 Python pickle，只能在同一台可信电脑上运行，**不能暴露到局域网或公网**。

### 3. 启动 bridge：先 dry-run

另开终端，在机器人 ROS 环境中运行。bridge 必须使用 ROS 的系统 Python，不要用模型的 conda Python：

```bash
cd /target/path/robot8_external_algorithm_deploy
source /opt/ros/humble/setup.bash
/usr/bin/python3 robot8_ros_bridge.py --log-full-action
```

这一步不会向机器人发布命令，只会打印收到的 action。确认图像、23D 顺序、数值范围、延迟均正确后，才运行：

```bash
/usr/bin/python3 robot8_ros_bridge.py --log-full-action --publish-commands
```

bridge 固定为 **20 Hz**，worker 单轮 deadline 是 45 ms（最大允许 50 ms）。如果模型太慢，bridge 会拒绝该轮而不是积压旧动作。

## worker 收什么、回什么？

bridge 每一轮发给 worker：

```python
{
    "state": [23 个 float],
    "images": {
        "head_cam": 原始头部双目 JPEG bytes,
        "left_arm_cam": 原始左臂 JPEG bytes,
        "right_arm_cam": 原始右臂 JPEG bytes,
    },
}
```

worker 每一轮回给 bridge：

```python
{"ok": True, "action": [23 个 float], "latency_s": 0.012}
```

如果无法安全推理，worker 应回：

```python
{"ok": False, "error": "原因"}
```

模板已实现这层通信；通常不需要自己改 socket 代码。

## 23D 的顺序：state 和 action 完全相同

```text
 0  torso_lift
 1  torso_waist
 2  head_pan
 3  head_tilt
 4–10  left_arm_j0 ... left_arm_j6
11–17  right_arm_j0 ... right_arm_j6
18  left_gripper
19  right_gripper
20  base_vx
21  base_vy
22  base_rotation
```

- state 的前 20 个来自相应 `JointState.position`；最后三个来自 odom 的 `linear.x`、`linear.y`、`angular.z`。
- action 的 23 个值也必须按这套顺序，且是 Robot8 的**原始 low-level command**。
- bridge 会自动发布 `[1.0, action[0], ..., action[22]]` 到 `/zeno/h1/auto/wholebody/cmd`。`1.0` 是 active；出错时发布 `[0.0, 0, ..., 0]` idle。

## 头部相机：这是必须保留的去畸变流程

`head_cam` 不是普通单目图。ROS 中它是一张原始的 **2560×720** JPEG：左、右两张 1280×720 鱼眼图水平拼接。

```text
原始 2560x720 left|right JPEG
  -> 分开两眼
  -> 用 camera/stereo_params_20260729_172611.npz 做 fisheye 去畸变/立体对齐
  -> 只取校正后的左眼
  -> 固定裁剪 x=20, y=0, width=1240, height=620
  -> BGR 转 RGB
  -> 保持比例 letterbox 到 640x480
```

最终 `head_cam_rgb` 是 RGB `uint8 (480, 640, 3)`。原图 1240×620 在 640×480 下会变成 640×320，因此上下各有 **80 px 黑边**。这是模型使用的几何；不能改成拉伸或 center crop。

请严格遵守：

- 传给 worker 的头图必须是**原始** 2560×720 left|right JPEG；不要预裁剪或预去畸变。
- 不要二次去畸变、不要只传原始左半图、不要把右眼作为第二张模型图。
- 不要把固定裁剪替换为 center crop；不要将 1240×620 直接拉伸为 640×480。
- 左/右臂相机不是鱼眼：模板对它们执行 raw JPEG → RGB → 直接 resize 640×480。

标定文件已经附带，SHA-256 必须是：

```text
6d08b6a01a1431476c2c3c77bee43e3f8f20888f33940af772cf1963e9f6b342
```

验证一帧原始头图：

```bash
cd /target/path/robot8_external_algorithm_deploy
sha256sum camera/stereo_params_20260729_172611.npz
python verify_topcam_contract.py \
  --input /path/to/raw_2560x720_left_right.jpg \
  --output /tmp/robot8_topcam_left.png \
  --output-rgb-npy /tmp/robot8_topcam_left.npy
```

输出 PNG 应为校正后的左眼、尺寸 640×480，并有上下黑边。

## ROS 话题

| 作用 | 默认话题 |
| --- | --- |
| 头部相机 | `/zeno/h1/sensor/head_cam/image/compressed` |
| 左臂相机 | `/zeno/h1/sensor/left_arm_cam/image/compressed` |
| 右臂相机 | `/zeno/h1/sensor/right_arm_cam/image/compressed` |
| odometry | `/zeno/h1/sensor/odom_raw` |
| torso state | `/zeno/h1/wheelarm/torso/joint_state` |
| left arm state | `/zeno/h1/wheelarm/left_arm/joint_state` |
| right arm state | `/zeno/h1/wheelarm/right_arm/joint_state` |
| left gripper state | `/zeno/h1/left_gripper/joint_state` |
| right gripper state | `/zeno/h1/right_gripper/joint_state` |
| 24D command output | `/zeno/h1/auto/wholebody/cmd` |

若机器人上话题不同，可通过 `robot8_ros_bridge.py --help` 查看相应 `--*-topic` 参数；修改话题名不应改变 23D 顺序或相机去畸变链路。

## 给接入方 Codex 的最后检查

- [ ] 我复制的是整个文件夹，包含 `camera/` 和 NPZ；
- [ ] 我只在 `ExternalPolicy` 中接入了自己的模型；
- [ ] 我的模型输入使用了模板产生的左眼 RGB 去畸变图；
- [ ] 我返回的是 23 个原始、有限的 action；
- [ ] 我先完成 dry-run，再显式使用 `--publish-commands`；
- [ ] 我已在自己的算法中完成限幅与平滑。
