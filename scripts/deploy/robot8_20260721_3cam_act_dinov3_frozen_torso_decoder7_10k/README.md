# robot8 2026-07-21 DINOv3 ACT decoder-7（普通冻结）部署

默认 checkpoint 为训练完成后的 `010000/pretrained_model`。

- 输入图像：三相机、640×480、无 crop（`center_crop_fraction=1.0`）
- 频率：20 Hz
- 冻结字段：`torso_lift`、`torso_waist`
- worker 在归一化前将这两个 **state 输入**置为 `0.0`，与冻结训练数据一致
- 两个模型未训练的 **active action 输出**固定为原始未冻结数据的 action 均值：`torso_lift=-0.0013542304`、`torso_waist=-0.0655177758`
- bridge 发布前会再次写入同一组均值，形成独立安全边界；ROS 24D 命令中对应 `command[1]`、`command[2]`，`command[0]` control mode 保持为 `1.0`
- 当 bridge 发布 idle（control mode `0.0`）时，仍发送全零 action，避免把固定基线用于停机命令
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

确认三路相机、state 输入掩码与固定 action 基线均正确后，才显式添加 `--publish-commands`。注意：固定均值也是绝对关节命令；首次真机使用前应在 dry-run 日志中确认底层关节零位和方向。
