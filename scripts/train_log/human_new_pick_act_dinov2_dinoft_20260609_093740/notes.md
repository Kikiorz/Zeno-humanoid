# human_new_pick_act_dinov2_dinoft_20260609_093740

- Dataset action/state layout: /zeno/h1/auto/wholebody/cmd fields [1..23].
- Field [0] control_mode is not trained; deployment should set it to 1.
- Base state uses /zeno/h1/sensor/odom_raw velocity; base action uses /zeno/h1/twist/cmd.
- Model checkpoints are written under: /home/zeno-rp/2027icra/outputs/train/human_new_pick_act_dinov2_dinoft_20260609_093740
