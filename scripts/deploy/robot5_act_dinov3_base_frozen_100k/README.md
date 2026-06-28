# robot5 ACT+DINOv3 Base Frozen Deployment

Run two terminals on the robot.

Terminal A, conda inference worker:

```bash
cd /path/to/2027icra/scripts/deploy/robot5_act_dinov3_base_frozen_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

Terminal B, ROS2 bridge:

```bash
cd /path/to/2027icra/scripts/deploy/robot5_act_dinov3_base_frozen_100k
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py
```

The bridge is dry-run by default. Add `--publish-commands` only when the robot is ready.

Normalization is loaded from the checkpoint. Visual inputs use the saved ImageNet/DINO mean/std
`[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`; state/action normalization and action
unnormalization use the checkpoint processor files. The deploy machine does not need the training
dataset.
