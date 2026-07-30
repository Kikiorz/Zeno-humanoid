# Robot8 TeaRoom 2026-07-29 ResNet-18 ACT deployment

This package deploys the local three-camera TeaRoom ResNet-18 ACT model. It
uses these logical model cameras:

```text
head_cam,left_arm_cam,right_arm_cam
```

`head_cam` must be the original ROS compressed **2560x720 side-by-side
left|right fisheye JPEG**, exactly as it was before dataset conversion. Do not
rectify, crop, resize, or split it in the ROS bridge. The model worker uses the
same shared `cam` implementation that converted the training data:

```text
raw 2560x720 left|right JPEG
  -> split left/right 1280x720 images
  -> cam_20260729 NPZ fisheye rectification/alignment (K1/D1/R1/P1)
  -> fixed left crop x=20, y=0, width=1240, height=620
  -> retain the calibrated left RGB eye only
  -> 640x480 letterbox
  -> observation.images.head_cam
```

The right top-camera eye is not a model input. The package pins this geometry
and validates the required NPZ hash before it loads a policy, so a mismatched
camera path cannot silently change the deployment visual distribution.

The default run root is:

```text
outputs/train/robot8_20260729_tearoom_act_resnet18_3cam_640x480_topcam_left_cam20260729_all23_b8_10k
```

The worker automatically uses the newest complete numbered checkpoint under
that directory. During the current run it can use `005000`; after completion
it will select `010000` automatically.

Start the worker from the repository checkout:

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260729_tearoom_3cam_act_resnet18_topcam_left_cam20260729
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

Then start the ROS bridge in a second terminal. First use dry-run mode and
inspect the returned actions; add `--publish-commands` only after the topics
and commands are confirmed.

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
/usr/bin/python3 bridge.py --publish-commands --log-full-action
```

This is an all-23D normal-label policy. Do not add `--frozen-fields`,
`--frozen-action-values`, or a base-command mapper unless you deliberately
want to change the trained action semantics. The ROS bridge publishes command
control mode `1` for active commands and `0` when idling.
