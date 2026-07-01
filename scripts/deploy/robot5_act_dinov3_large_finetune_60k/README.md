# robot5 ACT+DINOv3 Base FullFT Deployment

Default checkpoint:

`outputs/train/robot5_20260623_act_dinov3_base_fullft/checkpoints/080000/pretrained_model`

This directory now defaults to the robot5 2026-06-23 VAST Base FullFT checkpoint.
The directory name is historical; the worker loads architecture details from the checkpoint config.

- `worker.py`: conda inference process, loads ACT + DINOv3 from the checkpoint config.
- `bridge.py`: ROS2 process, subscribes robot observations and publishes the 24-field whole-body command.

Run two terminals on the robot.

Terminal A, conda inference worker:

```bash
cd /path/to/2027icra/scripts/deploy/robot5_act_dinov3_large_finetune_60k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

Terminal B, ROS2 bridge:

```bash
cd /path/to/2027icra/scripts/deploy/robot5_act_dinov3_large_finetune_60k
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py
```

The bridge is dry-run by default. Add `--publish-commands` only when the robot is ready.
Use `--checkpoint-path` on `worker.py` to test another checkpoint when needed.

The published command is `std_msgs/msg/Float64MultiArray` on `/zeno/h1/auto/wholebody/cmd`:

`[control_mode, torso_lift, torso_waist, head_pan, head_tilt, left_arm_j0..j6, right_arm_j0..j6, left_gripper, right_gripper, base_vx, base_vy, base_rotation]`

Normalization is loaded from the checkpoint. Visual inputs use the saved ImageNet/DINO mean/std;
state/action normalization and action unnormalization use the checkpoint processor files. The deploy
machine does not need the training dataset.

Default inference behavior:

- `rate_hz=20`
- `worker_port=8765`
- `use_amp=true`
- action clamp is enabled from checkpoint action min/max with `0.05` margin
- temporal ensemble is off by default
