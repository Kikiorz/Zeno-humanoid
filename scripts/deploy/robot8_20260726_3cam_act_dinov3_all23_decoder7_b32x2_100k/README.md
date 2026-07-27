# Robot8 2026-07-26 DINOv3 ACT decoder-7（原始底盘命令标签）部署

该目录部署刚完成的普通 all-23D 基线：DINOv3 ViT-B 冻结训练、三路相机、640×480 无 crop、ACT decoder 7 层、每卡 batch 32 的 100k-step 模型。

默认模型目录：

```text
outputs/train/robot8_20260726_act_dinov3_3cam_640x480_nocrop_all23_decoder7_b32x2_100k
```

worker 自动选择其中编号最高且完整的 checkpoint，当前为 `100000`。部署使用真实 23D state，不冻结腰部升降或俯仰；图像为 `head_cam,left_arm_cam,right_arm_cam`，直接 resize 到 640×480，不裁剪。

这套模型的 `action[20:23]` 仍是采集时的原始低层底盘命令，因此可以按现有 whole-body command 协议直接发布。它与 V2 平滑数据模型不同：不要把 V2 checkpoint 放在本目录的 bridge 下运行。

启动 worker：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260726_3cam_act_dinov3_all23_decoder7_b32x2_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

先 dry-run：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

确认三路图像、23D state、动作方向与 20 Hz 行为正确后，才显式发布：

```bash
/usr/bin/python3 bridge.py --publish-commands
```

观测过期、worker 超时或非有限动作时，bridge 会丢弃模型动作并发布 idle。固定某个 checkpoint 时，将 worker 的 `--checkpoint-path` 指向 `checkpoints/<step>/pretrained_model`。
