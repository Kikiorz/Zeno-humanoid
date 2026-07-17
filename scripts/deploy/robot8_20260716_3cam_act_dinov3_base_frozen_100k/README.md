# robot8 2026-07-16 三相机 ACT+DINOv3 100k 部署

该目录部署 2026-07-16 数据训练得到的 100k checkpoint，并复用
`scripts/deploy/robot8_act_dinov3_base_frozen_100k` 中已经验证过的 worker 和 ROS2 bridge。

默认配置：

- checkpoint：`outputs/train/robot8_20260716_act_dinov3_base_frozen_100k_640x480_crop2of3_curated30/checkpoints/100000/pretrained_model`
- 相机：`head_cam,left_arm_cam,right_arm_cam`
- 图像预处理：原图宽、高分别居中裁剪至 `2/3`，再缩放为 `640x480`
- 模型：ACT + DINOv3 ViT-B/16，训练时冻结视觉 backbone
- state/action：23 维全身向量
- ACT：`chunk_size=100`、`n_action_steps=100`、不使用 temporal ensemble
- worker 超时：`5s`
- bridge 默认 dry-run；只有显式传入 `--publish-commands` 才会发布机器人命令

## 启动

终端 A，启动 conda 推理 worker：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260716_3cam_act_dinov3_base_frozen_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

终端 B，先以 ROS2 dry-run 启动并检查三路图像、状态和动作日志：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260716_3cam_act_dinov3_base_frozen_100k
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

确认三路相机、全部关节状态、里程计与动作都正常后，才启用真机命令：

```bash
/usr/bin/python3 bridge.py --publish-commands
```

发布 topic 为 `/zeno/h1/auto/wholebody/cmd`，消息类型为
`std_msgs/msg/Float64MultiArray`，共 24 个字段：

```text
[control_mode,
 torso_lift, torso_waist, head_pan, head_tilt,
 left_arm_j0..j6, right_arm_j0..j6,
 left_gripper, right_gripper,
 base_vx, base_vy, base_rotation]
```

## 真机前检查

- head 原始图像应为 `1280x720`，左右腕原始图像应为 `640x480`。
- bridge 日志应显示三路相机均无 missing/stale。
- 先保持 dry-run，确认输出是 23 维有限值且方向、量级合理。
- 真机首次发布时保持急停可用，并由低风险姿态开始。

如需临时验证其他 checkpoint，可显式覆盖默认路径：

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /home/zeno-rp/2027icra/outputs/train/robot8_20260716_act_dinov3_base_frozen_100k_640x480_crop2of3_curated30/checkpoints/080000/pretrained_model
```

模型、DINOv3 backbone、normalizer 和 unnormalizer 权重均已保存在
`pretrained_model` 中；运行部署不需要训练数据集，也不需要联网下载视觉 backbone。
