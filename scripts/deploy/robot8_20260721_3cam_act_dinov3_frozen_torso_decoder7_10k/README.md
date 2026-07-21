# robot8 2026-07-21 DINOv3 ACT decoder-7（普通冻结）部署

默认 checkpoint 为训练完成后的 `010000/pretrained_model`。

- 输入图像：三相机、640×480、无 crop（`center_crop_fraction=1.0`）
- 频率：20 Hz
- 冻结字段：`torso_lift`、`torso_waist`
- worker 在归一化前将这两个 state 字段置为 `0.0`，模型输出后再次置为 `0.0`
- bridge 发布前再把 ROS 24D 命令的 `command[1]` 与 `command[2]` 置为 `0.0`；`command[0]` 的 control mode 保持为 `1.0`
- worker 默认使用 CUDA AMP 推理

先启动 worker：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260721_3cam_act_dinov3_frozen_torso_decoder7_10k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

再以 dry-run 启动 ROS bridge：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

确认三路相机、状态、冻结字段均正确后，才显式添加 `--publish-commands`。注意：这里的 `0.0` 是绝对关节命令零位，不是“保持当前位置”。
