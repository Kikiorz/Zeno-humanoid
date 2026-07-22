# robot8 2026-07-21 DINOv3 ACT decoder-7（底盘 3× 加权）部署

默认 checkpoint 为训练完成后的 `010000/pretrained_model`。这个模型与普通冻结模型的结构、数据、频率和部署预处理完全相同；唯一区别是训练时 `base_vx`、`base_vy`、`base_rotation` 的 L1 动作损失各为其他活动维度的 3 倍。

- 输入图像：三相机、640×480、无 crop（`center_crop_fraction=1.0`）
- 频率：20 Hz
- 冻结字段：`torso_lift`、`torso_waist`
- worker 将冻结 state 输入置为 `0.0`，与冻结训练数据完全一致
- worker 和 bridge 将未训练的 active action 输出固定为原始未冻结数据均值：`torso_lift=-0.0013542304`、`torso_waist=-0.0655177758`
- ROS 24D 命令中固定的是 `command[1]` 与 `command[2]`；不会改写 `command[0]` control mode。idle（control mode `0.0`）仍使用全零 action
- worker 默认使用 CUDA AMP 推理

先启动 worker：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260721_3cam_act_dinov3_frozen_torso_base3x_decoder7_10k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

再保持 dry-run 启动 bridge：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

确认 state 输入掩码与固定 action 基线后才添加 `--publish-commands`。固定均值也是绝对命令，首次真机使用前请先 dry-run 确认关节零位和方向。
