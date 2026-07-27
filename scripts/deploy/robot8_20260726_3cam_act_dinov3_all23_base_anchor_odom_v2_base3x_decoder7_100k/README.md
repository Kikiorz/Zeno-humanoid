# Robot8 2026-07-26 DINOv3 ACT V2（平滑物理底盘标签）部署

这是 V2 底盘 3× loss 模型的专用部署目录。它不能套用旧 all-23D/base3x bridge：V2 的 `action[20:23]` 是期望的实际 body odom 速度，不是低层 `/twist/cmd`。

默认模型目录：

```text
outputs/train/robot8_20260726_act_dinov3_3cam_640x480_nocrop_all23_base_anchor_odom_v2_base3x_decoder7_b32x2_100k
```

bridge 默认加载同一 V2 数据集导出的 `dynamics_feedback_mapper.json`。每次发布前它读取最新 odom，并以：

```text
v_goal = v_measured + 0.5 * (v_desired - v_measured)
u = inv(B) * (v_goal - bias - A * v_measured)
```

把 `u` 写入后三个低层命令维度。期望物理速度、低层命令、及其变化率分别限幅；odom、mapper 或 worker 出错时 fail-closed 并发送 idle。

V2 bridge 还要求单次 worker/control cycle 不超过 0.10 s，socket timeout 也默认 0.10 s；不能满足时它会 idle，而不是假装在 20 Hz 控制。若三相机 DINO 推理达不到该条件，应先优化/重构部署闭环，不能直接放宽超时来持续发布旧命令。

离线用本批数据模拟时，安全的低层 `±[0.15, 0.15, 0.30]` 限幅会触发约 9.7% 帧，且总限幅/变化率触发约 12.9% 帧。因此它是安全起点，不是“端点一定保持”的保证；bridge 会记录 `desired/measured/raw_command/sent_command` 和每类饱和标志，必须据此做真机台架标定。

先启动 worker（默认 `n_action_steps=1`，避免执行 5 秒陈旧视觉输出）：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260726_3cam_act_dinov3_all23_base_anchor_odom_v2_base3x_decoder7_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

先只 dry-run：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

仅在逐轴零点、阶跃和斜坡测试确认坐标系、方向、限幅和 20 Hz 运行后，才显式启用：

```bash
/usr/bin/python3 bridge.py --publish-commands
```

这是速度反馈，不是外层位姿/终点控制器；即使离线标签终点严格重建，也不能据此承诺真机终点精度。若任务要求终点精度，需要再加 pose/goal 闭环。不要把此 V2 checkpoint 放进旧 bridge 直接发布。
