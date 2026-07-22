# robot8 2026-07-21 DINOv3 ACT decoder-7（底盘 3× 加权，all-23D）部署

这份代码部署全 23 维、未冻结的底盘加权模型。与普通 all-23D 版本的部署逻辑完全一致；唯一训练差异是 `base_vx`、`base_vy`、`base_rotation` 三个动作维度的 L1 loss 权重均为其他维度的 3 倍。

默认训练 run 目录为：

```text
outputs/train/robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_base3x_decoder7_ddp128_100k
```

worker 自动选择该 run 下 `checkpoints/<step>/pretrained_model` 中编号最高且完整的模型。后续下载新 checkpoint 到此目录后，默认部署自动使用新版本。

当前已由私有 Hugging Face 快照同步到本机的版本是 `030000`：
[QRP123/robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_base3x_decoder7_100k](https://huggingface.co/QRP123/robot8_20260721_act_dinov3_3cam_640x480_nocrop_all23_base3x_decoder7_ddp128_100k)。

- 三相机：`head_cam,left_arm_cam,right_arm_cam`
- 图像：640×480、无 crop（`center_crop_fraction=1.0`）
- 控制频率：20 Hz
- state/action：完整 23 维；腰部升降、腰部俯仰都使用真实输入和模型输出，不冻结、不置零、不改写均值
- DINOv3 仅在训练中冻结；部署使用 checkpoint 内的骨干与归一化器，不需要特征缓存
- 默认使用 CUDA AMP，动作裁剪到训练 action 范围外扩 5%

## 启动

终端 A：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260721_3cam_act_dinov3_all23_base3x_decoder7_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

终端 B 先 dry-run：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

确认输入和动作均正确后，才添加：

```bash
/usr/bin/python3 bridge.py --publish-commands
```

bridge 默认 dry-run；观测过期、worker 超时或动作无效时不会发布模型动作，并会发送 idle。ROS 接口、24 维命令顺序和真机前安全检查与仓库已有 robot8 deploy 一致。

## 固定某个 checkpoint

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /home/zeno-rp/2027icra/outputs/train/<run>/checkpoints/<step>/pretrained_model
```
