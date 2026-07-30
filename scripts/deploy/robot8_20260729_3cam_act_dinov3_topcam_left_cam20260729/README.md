# Robot8 2026-07-29 左眼 Topcam ACT 部署

这个目录适用于本轮普通标签和 V3 标签的三相机 checkpoint。视觉输入固定为：

```text
head_cam,left_arm_cam,right_arm_cam
```

其中 `head_cam` 不是原始鱼眼 JPEG。worker 对每一帧严格执行训练时的处理：

```text
raw 2560x720 left|right JPEG
  -> split left/right
  -> new 2026-07-29 NPZ fisheye rectification/alignment (left K1/D1/R1/P1)
  -> fixed crop x=20, y=0, width=1240, height=620
  -> retain only rectified left RGB; letterbox to 640x480
  -> observation.images.head_cam
```

右眼不作为 topic、数据集特征或模型输入。`R1/P1` 来自双目联合标定，因此保留下来的左眼已经处于校正后的双目对齐坐标。

普通模型默认目录：

```text
outputs/train/robot8_20260729_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_b32_100k
```

使用 V3 模型时显式指定其训练目录：

```bash
python worker.py --checkpoint-path \
  /home/zeno-rp/2027icra/outputs/train/robot8_20260729_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k
```

启动 worker：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260729_3cam_act_dinov3_topcam_left_cam20260729
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

先 dry-run bridge，再显式发布控制：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
/usr/bin/python3 bridge.py --publish-commands
```
