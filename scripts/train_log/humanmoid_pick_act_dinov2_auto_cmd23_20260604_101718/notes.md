# humanmoid_pick_act_dinov2_auto_cmd23_20260604_101718

- Dataset action/state layout: /zeno/h1/auto/wholebody/cmd fields [1..23].
- Field [0] control_mode is not trained; deployment should set it to 1.
- Base action is converted from /zeno/h1/sensor/odom_raw velocity because the bags do not contain /zeno/h1/auto/wholebody/cmd.
- Model checkpoints are written under: /home/zeno-rp/2027icra/outputs/train/humanmoid_pick_act_dinov2_auto_cmd23_20260604_101718
