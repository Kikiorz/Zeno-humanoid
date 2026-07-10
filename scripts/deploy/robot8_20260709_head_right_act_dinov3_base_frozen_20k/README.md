# robot8 2026-07-09 Head+Right ACT+DINOv3 20k Deployment

This wrapper deploys the current test checkpoint for the 2026-07-09 robot8 data.
It reuses the shared robot8 ACT+DINOv3 worker and bridge, with safer defaults for
this specific model:

- checkpoint: `outputs/train/robot8_20260709_head_right_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709/checkpoints/020000/pretrained_model`
- cameras: `head_cam,right_arm_cam`
- state/action: full 23D whole-body vector, including left arm and left gripper
- mode: dry-run unless `--publish-commands` is passed to `bridge.py`

Terminal A, conda inference worker:

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260709_head_right_act_dinov3_base_frozen_20k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

Terminal B, ROS2 bridge dry-run:

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260709_head_right_act_dinov3_base_frozen_20k
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
```

This bridge intentionally does not subscribe to `left_arm_cam`. It still
subscribes to left arm and left gripper state topics and publishes the model's
full 23D action unchanged.

When robot8 is ready to receive commands:

```bash
/usr/bin/python3 bridge.py --publish-commands
```

To test a later checkpoint, keep this deploy wrapper and override the worker:

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /home/zeno-rp/2027icra/outputs/train/robot8_20260709_head_right_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709/checkpoints/040000/pretrained_model
```
