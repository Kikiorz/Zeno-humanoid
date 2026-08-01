# Robot8 2026-07-31 V3 smooth-base ACT+DINOv3 deployment

This directory is only for the V3 endpoint-anchored, decoupled-smoothed base
label model. Its final local run root is:

~~~
outputs/train/robot8_20260731_act_dinov3_3cam_640x480_topcam_left_cam20260729_all15_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_bf16_nogc_resume_100k_onthefly
~~~

## Fixed visual contract

The model receives head_cam,left_arm_cam,right_arm_cam at 20 Hz. head_cam must
be the original compressed 2560x720 left|right fisheye image. The worker
performs the same training-time processing on every frame:

~~~
raw stereo -> split eyes -> cam_20260729 rectify/align -> crop (20,0,1240,620)
-> left RGB only -> 640x480 letterbox (no generic crop and no stretch)
~~~

The right eye is not a model feature. Do not send a pre-rectified, cropped, or
single-eye image to the raw head topic; the wrapper rejects geometry overrides.

## V3 base contract

action[20:23] is desired physical body odom velocity [vx, vy, wz], not the
low-level base command. It must never be sent directly to
/zeno/h1/auto/wholebody/cmd. The paired bridge always loads the sibling
dynamics_feedback_mapper.json, reads current odometry, maps physical desired
velocity to a bounded low-level command, and fails closed to idle if odometry,
observations, or the 20 Hz inference cycle are stale.

The V3 worker also forces n_action_steps=1: each 20 Hz observation produces a
fresh desired velocity instead of consuming a stale ACT action queue.

## Start

Start the worker:

~~~
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260731_3cam_act_dinov3_topcam_left_cam20260729_v3_smooth_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
~~~

Then first dry-run the bridge:

~~~
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
~~~

Only after checking camera freshness, mapper logs, base-axis sign, limits and
20 Hz behaviour on the robot should commands be enabled:

~~~
/usr/bin/python3 bridge.py --publish-commands --log-full-action
~~~

Use the downloaded V3-70k intermediate checkpoint with the same wrapper:

~~~
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /home/zeno-rp/2027icra/outputs/train/robot8_20260731_act_dinov3_3cam_640x480_topcam_left_cam20260729_all15_base_anchor_odom_v3_decoupled_smooth_ops3_to_base3_to_equal_decoder7_b32_bf16_nogc_resume_100k_onthefly/checkpoints/070000/pretrained_model
~~~

The included mapper SHA-256 is
229acccc3caae1cd8ef2230fa51822191a7585158c115450bf96f964ea5e69dc.
