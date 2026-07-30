# Robot8 2026-07-26 四相机 V3 平滑底盘 ACT 部署

这个目录只对应新的四相机 V3 模型：

```text
outputs/train/robot8_20260726_act_dinov3_4cam_640x480_headstereo_rectified_crop_lr_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k
```

不要放入三相机、未去畸变或原始底盘命令 checkpoint。这个模型必须同时得到 `head_cam`（左眼）和 `head_cam_right`（右眼）。

视觉契约

原始 `/zeno/h1/sensor/head_cam/image/compressed` 是左|右并排的 `2560×720` JPEG。worker 对每个原始帧固定执行：

```text
split 2560×720 → left/right 1280×720
→ 每眼独立 OpenCV fisheye rectification (P1/P2)
→ 每眼裁剪 x=20, y=0, width=1240, height=620
→ BGR→RGB，等比例 letterbox 到 640×480
```

`1240×620` 的每只眼以不变形方式变成 `640×320`，上下各有 80 像素黑边。没有额外 crop、没有 stretch、没有把左右眼重新拼接。固定特征映射为：

| Feature | 内容 |
| --- | --- |
| `head_cam` | 校正、裁剪、letterbox 后的左眼 |
| `head_cam_right` | 校正、裁剪、letterbox 后的右眼 |
| `left_arm_cam` | 左臂相机，直接 640×480 resize、无 crop |
| `right_arm_cam` | 右臂相机，直接 640×480 resize、无 crop |

bridge 默认将四个逻辑相机以 20 Hz 送至 worker；两路 head feature 使用同一条 raw stereo topic，worker 从同一 JPEG 产生独立的左/右模型输入。

V3 底盘契约

V3 的最后三维 action 是期望的真实 body odom 速度，不是可直接发布的 `/twist/cmd`。本目录的 bridge 默认并且强制使用 V3 数据生成的 mapper：

```text
Data/lerobot/robot8_20260726_zeno_h1_auto_cmd_v30_4cam_640x480_headstereo_rectified_crop_lr_all23_base_anchor_odom_v3_decoupled_smooth/meta/base_anchor_odom_v3_decoupled_smooth/dynamics_feedback_mapper.json
```

worker 默认 `n_action_steps=1`，每个 20 Hz tick 重新评估视觉。bridge 对 worker 超时或过期观测 fail-closed 并发送 idle；默认 `worker-timeout-s` 和 `max-obs-age-s` 都是 0.10 秒。不要删除 mapper 或直接把 V3 action tail 发布到 whole-body command。

启动 worker：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260726_4cam_act_dinov3_headstereo_rectified_lr_v3_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

先 dry-run：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

只在验证四路图像、左右眼几何、每轴底盘方向、mapper 限幅以及稳定 20 Hz 后，才发布真实命令：

```bash
/usr/bin/python3 bridge.py --publish-commands
```
