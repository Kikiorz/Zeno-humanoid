# robot8 ACT+DINOv3 Base Frozen Deployment

Default checkpoint:

`outputs/train/robot8_20260708_3cam_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709/checkpoints/100000/pretrained_model`

This directory defaults to the 2026-07-08 robot8 ACT+DINOv3 Base Frozen checkpoint
with head, left-arm, and right-arm cameras.
It uses a two-process layout:

- `worker.py`: conda inference process, loads ACT + DINOv3 from the checkpoint config.
- `bridge.py`: ROS2 process, subscribes robot observations and publishes the 24-field whole-body command.

Run two terminals on robot8.

Terminal A, conda inference worker:

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_act_dinov3_base_frozen_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

Terminal B, ROS2 bridge:

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_act_dinov3_base_frozen_100k
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py
```

For the 2026-07-09 head+right model, point the worker at that checkpoint and start the bridge with
only those two image streams:

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /home/zeno-rp/2027icra/outputs/train/robot8_20260709_head_right_act_dinov3_base_frozen_100k_640x480_crop2of3_20260709/checkpoints/100000/pretrained_model
```

```bash
/usr/bin/python3 bridge.py --cameras head_cam,right_arm_cam
```

The bridge is dry-run by default. It logs inferred actions but does not publish commands.
When robot8 is ready, run:

```bash
/usr/bin/python3 bridge.py --publish-commands
```

For the 2026-07-09 head+right model with command publishing:

```bash
/usr/bin/python3 bridge.py --cameras head_cam,right_arm_cam --publish-commands
```

For full action logging during dry-run:

```bash
/usr/bin/python3 bridge.py --log-full-action
```

The published command is `std_msgs/msg/Float64MultiArray` on `/zeno/h1/auto/wholebody/cmd`:

`[control_mode, torso_lift, torso_waist, head_pan, head_tilt, left_arm_j0..j6, right_arm_j0..j6, left_gripper, right_gripper, base_vx, base_vy, base_rotation]`

Normalization is loaded from the checkpoint. Visual inputs use the saved ImageNet/DINO mean/std
`[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`; state/action normalization and action
unnormalization use the checkpoint processor files. The deploy machine does not need the training
dataset.

Default inference behavior:

- `rate_hz=20`
- `worker_port=8768`
- `use_amp=true`
- action clamp is enabled from checkpoint action min/max with `0.05` margin
- `n_action_steps` and `temporal_ensemble_coeff` default to the checkpoint config

To test temporal ensembling, override the worker explicitly:

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py --n-action-steps 1 --temporal-ensemble-coeff 0.01
```
