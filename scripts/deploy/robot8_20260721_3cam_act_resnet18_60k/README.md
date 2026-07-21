# robot8 2026-07-21 三相机 ACT + ResNet18 60K 部署

该目录部署 2026-07-21 数据训练得到的最终 60K checkpoint，并复用仓库中
已经验证过的 robot8 ACT worker 与 ROS2 bridge。

默认配置：

- checkpoint：`outputs/train/robot8_20260721_act_resnet18_60k_224x224_crop2of3/checkpoints/060000/pretrained_model`
- 模型：ACT + ResNet18，`chunk_size=100`、`n_action_steps=100`
- 相机：`head_cam,left_arm_cam,right_arm_cam`
- 图像预处理：原图宽、高分别居中裁剪至 `2/3`，再缩放至 `224x224`
- state/action：23 维，与训练数据字段顺序完全一致
- 控制频率：20 Hz
- worker 超时：1 秒；推理完成后会再次检查观测新鲜度
- 动作限制：默认裁剪至训练集 action min/max 外扩 5% 的范围
- bridge 默认是 dry-run；只有显式传入 `--publish-commands` 才发布真机命令

## 启动

终端 A 启动 Conda 推理 worker：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260721_3cam_act_resnet18_60k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

看到下面一类日志后再启动 bridge：

```text
[robot8 worker] ... cameras=head_cam,left_arm_cam,right_arm_cam; image_size=(224, 224); center_crop_fraction=0.6666667; ... listening=127.0.0.1:8768
```

终端 B 先以 ROS2 dry-run 启动：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260721_3cam_act_resnet18_60k
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

dry-run 会订阅真机观测并执行推理，但不会发布控制命令。确认三路相机、关节状态、
里程计和 23 维动作均正常后，停止 dry-run，再显式启用真机发布：

```bash
/usr/bin/python3 bridge.py --publish-commands
```

## ROS2 接口

订阅：

```text
/zeno/h1/sensor/head_cam/image/compressed
/zeno/h1/sensor/left_arm_cam/image/compressed
/zeno/h1/sensor/right_arm_cam/image/compressed
/zeno/h1/sensor/odom_raw
/zeno/h1/wheelarm/torso/joint_state
/zeno/h1/wheelarm/left_arm/joint_state
/zeno/h1/wheelarm/right_arm/joint_state
/zeno/h1/left_gripper/joint_state
/zeno/h1/right_gripper/joint_state
```

发布：

```text
topic: /zeno/h1/auto/wholebody/cmd
type:  std_msgs/msg/Float64MultiArray
```

命令共 24 个字段：

```text
[control_mode,
 torso_lift, torso_waist, head_pan, head_tilt,
 left_arm_j0..j6, right_arm_j0..j6,
 left_gripper, right_gripper,
 base_vx, base_vy, base_rotation]
```

模型输入 state 是上述命令去掉 `control_mode` 后的 23 维顺序；其中末尾三维状态来自
里程计速度，末尾三维动作对应底盘 `vx`、`vy` 和旋转速度。

## 启用发布前必须确认

本次原始 bags 记录的是分开的 `joint_cmd` 和 `/zeno/h1/twist/cmd`，没有记录组合后的
`/zeno/h1/auto/wholebody/cmd`。因此，下面这些接收端约定必须在当前机器人运行环境中确认，
不能仅由训练数据证明：

```bash
ros2 topic info -v /zeno/h1/auto/wholebody/cmd
ros2 topic type /zeno/h1/auto/wholebody/cmd
ros2 topic hz /zeno/h1/sensor/head_cam/image/compressed
ros2 topic hz /zeno/h1/wheelarm/torso/joint_state
ros2 topic hz /zeno/h1/sensor/odom_raw
```

必须确认：

- 接收者确实使用 `Float64MultiArray`，并严格采用上面的 24 字段顺序。
- `control_mode=0` 是安全 idle、`control_mode=1` 是自动控制，mode 0 会忽略后续零值。
- `torso_lift`、其余关节、夹爪和底盘速度的单位及正方向与训练数据一致。夹爪是原始
  joint position，并不是默认的 `[0,1]` 开度。
- 里程计速度和底盘命令使用同一个 base 坐标系，`vx/vy/wz` 轴方向一致。
- 各传感器 publisher 与 bridge subscription 的 QoS 兼容。
- bridge 使用各 topic 的最新样本并限制 age 不超过 0.5 秒，但不按 ROS header 做硬同步；
  应确认三路相机、关节和里程计的实际频率与延迟足够接近。

checkpoint 的 action clamp 只限制到训练数据范围外扩 5%，不等同于机器人硬件极限、
速度/加速度限制或碰撞保护；这些限制和命令超时 watchdog 必须由底层控制器独立保证。

## 真机前检查

1. 保持 bridge 为默认 dry-run，并保持急停可用。
2. 确认 head 原图为 `1280x720`，左右腕原图为 `640x480`，方向与训练时一致。
3. bridge 日志中不应出现 camera/joint/odom 的 `missing` 或 `stale`。
4. 使用 `--log-full-action` 检查 23 维输出均为有限值，关节顺序、方向和量级正确。
5. 首次真机发布从低风险姿态和空旷环境开始。

如果观测过期、worker 超时或动作包含非有限值，bridge 不会发布模型动作；启用真机发布
时会发送 `control_mode=0` 的 idle 命令，并断开 worker 以清空旧的 ACT 动作队列。退出
bridge 时也会尽力发送一次 idle；真机控制端仍应配置独立的命令超时 watchdog。checkpoint
已包含模型、ResNet18、normalizer 和 unnormalizer 权重，部署不需要训练数据集，也不需要
联网下载视觉 backbone。

## 可选参数

临时验证其他 checkpoint：

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /absolute/path/to/pretrained_model
```

默认 ACT 会连续执行一个 100 步动作块，在 20 Hz 下约为 5 秒开环。观测中断会清空旧块，
但正常执行期间新观测不会改变当前块。若要每一步结合新观测并启用 temporal ensemble，
可在 dry-run 中先预热并验证：

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --n-action-steps 1 --temporal-ensemble-coeff 0.01
```

所有 bridge 参数可通过下面命令查看：

```bash
/usr/bin/python3 bridge.py --help
```
