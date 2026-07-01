# robot4 ACT+DINOv3 Base FullFT Deployment

Default checkpoint:

`outputs/train/robot4_20260623_act_dinov3_base_fullft/checkpoints/080000/pretrained_model`

This directory now defaults to the robot4 2026-06-23 VAST Base FullFT checkpoint.
It uses a two-process layout:

- `worker.py`: conda inference process, loads ACT + DINOv3 from the checkpoint config.
- `bridge.py`: ROS2 process, subscribes robot observations and publishes the 24-field whole-body command.

Run two terminals on robot4.

Terminal A, conda inference worker:

```bash
cd /path/to/2027icra/scripts/deploy/robot4_act_dinov3_base_frozen_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

Terminal B, ROS2 bridge:

```bash
cd /path/to/2027icra/scripts/deploy/robot4_act_dinov3_base_frozen_100k
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py
```

The bridge is dry-run by default. It logs inferred actions but does not publish commands.
When robot4 is ready, run:

```bash
/usr/bin/python3 bridge.py --publish-commands
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
- `worker_port=8764`
- `use_amp=true`
- action clamp is enabled from checkpoint action min/max with `0.05` margin
- temporal ensemble is off by default

Optional temporal ensemble test:

```bash
conda run --no-capture-output -n lerobot-qrp312 python worker.py --n-action-steps 1 --temporal-ensemble-coeff 0.01
```
