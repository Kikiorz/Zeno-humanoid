# Robot8 2026-07-31 normal ACT+DINOv3 deployment

This directory is only for the normal/raw-command 23D model. Its final local
run root is:

~~~
outputs/train/robot8_20260731_act_dinov3_3cam_640x480_topcam_left_cam20260729_all15_decoder7_b32_bf16_nogc_resume_100k_onthefly
~~~

The bridge supplies exactly three logical cameras at 20 Hz:

~~~
head_cam,left_arm_cam,right_arm_cam
~~~

head_cam must be the original ROS compressed 2560x720 left|right fisheye JPEG,
not an already-processed image. The worker enforces the exact training-time
chain:

~~~
raw 2560x720 left|right JPEG
  -> split into 1280x720 eyes
  -> cam_20260729 fisheye rectification/alignment (K1/D1/R1/P1)
  -> fixed crop x=20, y=0, width=1240, height=620
  -> retain only the aligned left RGB eye
  -> 640x480 letterbox (no crop, no stretch)
  -> observation.images.head_cam
~~~

The right eye is used by the calibrated stereo solve but is not a model input.
The wrapper rejects visual-geometry overrides so an already-rectified or
single-eye input cannot be accidentally processed a second time.

## Start

Start the worker first:

~~~
cd /home/zeno-rp/2027icra/scripts/deploy/robot8_20260731_3cam_act_dinov3_topcam_left_cam20260729_raw_100k
conda run --no-capture-output -n lerobot-qrp312 python worker.py
~~~

In a second terminal, always dry-run the ROS bridge before enabling commands:

~~~
source /opt/ros/humble/setup.bash
/usr/bin/python3 bridge.py --log-full-action
/usr/bin/python3 bridge.py --publish-commands --log-full-action
~~~

For the downloaded intermediate RAW-40k checkpoint, keep the same deployment
contract and only replace the checkpoint path:

~~~
conda run --no-capture-output -n lerobot-qrp312 python worker.py \
  --checkpoint-path /home/zeno-rp/2027icra/outputs/train/robot8_20260731_act_dinov3_3cam_640x480_topcam_left_cam20260729_all15_decoder7_b32_bf16_nogc_resume_100k_onthefly/checkpoints/040000/pretrained_model
~~~

This model's last three action values remain the recorded low-level base
command convention, so it uses the ordinary 23D bridge. Do not use the V3
feedback mapper with this model.
