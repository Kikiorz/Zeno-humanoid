# robot8 2026-07-21 DINOv3 ACT decoder-7（普通 all-23D）部署

这份代码部署未冻结的普通 all-23D 模型。它与底盘加权版本的结构、数据、相机和频率完全一致；训练目标中每个动作维度权重均为 `1`。

默认训练 run 目录为：

```text
outputs/train/robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_decoder7_ddp128_100k
```

worker 会自动选择其中 `checkpoints/<step>/pretrained_model` 里编号最高且包含 `model.safetensors` 的完整 checkpoint。只需将新的完整 checkpoint 放进同一 run 目录，不必修改部署代码。

当前已由私有 Hugging Face 快照同步到本机的版本是 `035000`：
[QRP123/robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_decoder7_ddp128_100k](https://huggingface.co/QRP123/robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_decoder7_ddp128_100k)。

- 三相机：`head_cam,left_arm_cam,right_arm_cam`
- 图像：640×480、无 crop（`center_crop_fraction=1.0`）
- 控制频率：20 Hz
- state/action：完整 23 维；`torso_lift`、`torso_waist` 不冻结，使用真实 state 输入，也不固定输出
- DINOv3 在训练时冻结；checkpoint 已包含视觉骨干和归一化器，部署不需要训练数据或特征缓存
- 默认启用 CUDA AMP 推理与 action range clamp（训练范围外扩 5%）

## 启动

终端 A：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260721_3cam_act_dinov3_all23_decoder7_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

终端 B 先 dry-run：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

确认三路相机、23 维 state、20 Hz 推理和动作方向正确后，才显式启用真机发布：

```bash
/usr/bin/python3 bridge.py --publish-commands
```

默认 bridge 不会发布真机命令；观测过期、worker 超时或非有限动作时，会丢弃动作并发送 idle。ROS topic、24 维命令顺序和真机前安全检查与仓库中其他 robot8 deploy 目录一致。

## 固定某个 checkpoint

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /home/zeno-rp/2027icra/outputs/train/<run>/checkpoints/<step>/pretrained_model
```
