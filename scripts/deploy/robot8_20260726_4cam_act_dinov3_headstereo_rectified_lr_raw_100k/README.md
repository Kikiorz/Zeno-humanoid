# Robot8 2026-07-26 四相机原始底盘命令 ACT 部署

这个目录只对应新的四相机模型：

```text
outputs/train/robot8_20260726_act_dinov3_4cam_640x480_headstereo_rectified_crop_lr_all23_decoder7_b32_100k
```

不要在这里放旧的三相机或未去畸变 checkpoint；`head_cam_right` 必须是模型的第四路视觉输入。

视觉契约

`/zeno/h1/sensor/head_cam/image/compressed` 的一张 JPEG 是 `2560×720`、左眼在左且右眼在右。worker 固定执行与训练完全相同的链路：

```text
raw stereo JPEG 2560×720
  → split: left/right 1280×720
  → each eye OpenCV fisheye rectification (P1 / P2)
  → each eye crop x=20, y=0, width=1240, height=620
  → BGR→RGB; aspect-ratio-preserving letterbox to 640×480
```

因此每只头部眼最终先缩放为 `640×320`，再在上下各补 80 像素黑边；不会再裁剪、拉伸或把左右眼拼回一张图。模型特征的固定含义为：

| Feature | 内容 |
| --- | --- |
| `head_cam` | 去畸变、裁剪后的左眼 |
| `head_cam_right` | 去畸变、裁剪后的右眼 |
| `left_arm_cam` | 左臂相机，直接 640×480 resize、无 crop |
| `right_arm_cam` | 右臂相机，直接 640×480 resize、无 crop |

bridge 默认以 20 Hz 发送这四个逻辑相机。两个 head feature 都来自同一条原始头部 stereo topic；worker 对同一原始帧产生一对独立的校正眼图。

启动 worker：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260726_4cam_act_dinov3_headstereo_rectified_lr_raw_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

先 dry-run bridge：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

确认四路图像、23D state、动作方向与 20 Hz 时序后，才显式发布：

```bash
/usr/bin/python3 bridge.py --publish-commands
```

该模型的最后三个 action 是采集时的原始低层底盘命令，会按现有 whole-body command 协议直接发布；不要给它加入 V3 的 physical-velocity feedback mapper。
