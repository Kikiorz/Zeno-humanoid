# Robot8 TeaRoom 2026-07-29 normal ACT+DINOv3 deployment

This wrapper deploys the normal (raw 23D command) TeaRoom model.  It uses the
same three logical cameras as training:

```text
head_cam,left_arm_cam,right_arm_cam
```

`head_cam` must remain the original ROS compressed **2560x720 side-by-side
left|right fisheye JPEG**.  The worker applies the exact training-time chain:

```text
raw 2560x720 left|right JPEG
  -> split left/right 1280x720 images
  -> cam_20260729 NPZ fisheye rectification/alignment (K1/D1/R1/P1)
  -> fixed left crop x=20, y=0, width=1240, height=620
  -> retain left RGB only
  -> 640x480 letterbox
  -> observation.images.head_cam
```

The right eye is not a model input.  Do not send a pre-cropped image, a
pre-rectified image, or the two-eye composite to `head_cam`: that would apply a
different visual contract from training.

The default checkpoint root is:

```text
outputs/train/robot8_20260729_tearoom_act_dinov3_3cam_640x480_topcam_left_cam20260729_all23_decoder7_ddp64_b32_w6_60k
```

The worker automatically selects the latest complete numbered checkpoint below
that directory.  A downloaded 10k snapshot therefore works without editing the
wrapper.

Start the model worker from this repository:

```bash
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260729_tearoom_3cam_act_dinov3_topcam_left_cam20260729
conda run --no-capture-output -n lerobot-qrp312 python worker.py
```

Then use a second terminal for a dry-run ROS bridge.  Add
`--publish-commands` only after the topic names and returned actions look
correct:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
/usr/bin/python3 bridge.py --publish-commands --log-full-action
```

This is the normal-label model: do not pass `--frozen-fields`,
`--frozen-action-values`, or `--base-command-mapper`.  Those options belong to
other label variants and would change this model's input/output semantics.

The required calibrated NPZ is
`scripts/data_convert/cam/stereo_params_20260729_172611.npz` (SHA-256
`6d08b6a01a1431476c2c3c77bee43e3f8f20888f33940af772cf1963e9f6b342`).
