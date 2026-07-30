# Robot8 2026-07-29 Data/process 左眼 Topcam ACT 部署

此目录只适用于 `topcam_left_dataprocess_v1` 的普通标签与 V3 标签模型。视觉输入严格固定为：

```text
head_cam,left_arm_cam,right_arm_cam
```

其中 `head_cam` 不是原始鱼眼 JPEG。训练、离线验证和 worker 对每一帧走同一条链路：

```text
raw 2560×720 BGR/JPEG (left|right)
  → split to 1280×720 per eye
  → Data/process 的 OpenCV fisheye 去畸变 + 双目对齐
     (top_stereo_calibration_basalt_kb4_compat.json +
      processing_metadata_centered_crop_1240x620.json)
  → rectified 1280×620
  → fixed crop x=20, y=0, width=1240, height=620
  → only the aligned left eye, BGR→RGB
  → letterbox to 640×480 (no generic crop / no stretch)
  → observation.images.head_cam
```

右眼只参与标定的联合几何；它不是 dataset feature、ROS topic 或模型输入。`R1/P1` 来自双目联合标定，因此保存下来的左眼已处于对齐坐标。为保证从 GitHub 新拉的仓库可直接运行，代码读取 `configs/calibration/` 中已版本化的两份 JSON；它们与原始 `Data/process/` 文件逐字一致。

普通模型路径：

```text
outputs/train/robot8_20260729_act_dinov3_3cam_640x480_topcam_left_dataprocess_v1_all23_decoder7_b32_100k
```

V3 模型路径：

```text
outputs/train/robot8_20260729_act_dinov3_3cam_640x480_topcam_left_dataprocess_v1_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k
```

## 启动 worker

`worker.py` 默认使用普通模型。它只监听本机 TCP 请求，不会直接向 ROS 或机器人发布动作。

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260729_3cam_act_dinov3_topcam_left_dataprocess_v1
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

启动 V3 模型时，显式指定其最终 checkpoint：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260729_3cam_act_dinov3_topcam_left_dataprocess_v1
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /home/zeno-rp/2027icra/outputs/train/robot8_20260729_act_dinov3_3cam_640x480_topcam_left_dataprocess_v1_all23_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_100k/checkpoints/100000/pretrained_model
```

普通模型的显式最终 checkpoint 等价于：

```text
outputs/train/robot8_20260729_act_dinov3_3cam_640x480_topcam_left_dataprocess_v1_all23_decoder7_b32_100k/checkpoints/100000/pretrained_model
```

## 安全启动顺序

先 dry-run bridge；确认三路图像、23D state、20 Hz 频率和 action 日志后，才显式发布控制：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
/usr/bin/python3 bridge.py --publish-commands
```

bridge 默认只记录动作，只有传入 `--publish-commands` 才会向真机发布。原始头图保持
`2560×720` 左右拼接形式传给 worker；worker 自己完成 Data/process 标定去畸变、双目对齐、
固定裁切、左目选择和 640×480 letterbox。因此 bridge 不创建右 topcam topic，也不把右目
送入模型。

为了避免把训练视觉约定悄悄改掉，此专用 wrapper 会拒绝覆盖 `--cameras`、`--rate-hz`、
`--head-cam-topic`、`--image-size`、crop 或任一头部标定参数；唯一常用的模型选择参数是
`--checkpoint-path`。若收到的头图不是原始 `2560×720` 左右拼接帧，rectifier 会直接失败，
不会把已经去畸变的图二次处理。

## 从 Hugging Face 恢复最终模型

最终普通和 V3 模型分别会存入私有 Hub 仓库：

```text
QRP123/robot8-20260729-act-dinov3-topcam-left-all23-100k
QRP123/robot8-20260729-act-dinov3-topcam-left-v3-smooth-100k
```

在已登录 Hugging Face 的本机仓库根目录执行：

```bash
python scripts/sync_robot8_20260729_dataprocess_v1_hf.py download --only both
```

该命令只下载部署所需的最终 `pretrained_model` 文件，目标位置正是上文两条
`outputs/train/.../checkpoints/100000/pretrained_model` 路径；不会下载训练时的优化器状态。
下载包是现有仓库式部署的一部分：仍需在本仓库中运行，以复用共享 worker、Data/process
预处理和 LeRobot 依赖；HF 仓库也备份了两份 deploy wrapper 与标定 JSON。
