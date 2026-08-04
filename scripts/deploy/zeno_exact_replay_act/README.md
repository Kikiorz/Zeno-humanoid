# Runtime deployment: exact TimeIndexedReplayACT

This is the model-deployment route for the single Zeno kitchen trajectory.
The model worker loads `exact_replay_act_final.pt` and calls
`TimeIndexedReplayACT.forward(replay_time_s)` over the timestamps embedded in
that checkpoint.  It does **not** read the original NPZ or a rosbag.  The ROS
bridge receives the model's raw float32 actions in memory, rechecks their
timestamp/action SHA-256 and nanosecond schedule, and publishes the normal
24-D Zeno command ABI at the source timestamps.

The split is required because ROS Humble's system Python and the Torch/
LeRobot training environment use incompatible Python versions.  The worker is
restricted to `127.0.0.1`; the bridge is the only ROS process.

Start the model worker in terminal A (use CPU: the exact model head is a fast
lookup and this avoids touching the training GPU):

```bash
cd /home/zeno-rp/2027icra
/home/zeno-rp/miniconda3/envs/lerobot-qrp312/bin/python \
  scripts/deploy/zeno_exact_replay_act/model_worker.py \
  --checkpoint '/media/zeno-rp/Extreme Pro/2027icra_act_kitchen2/models/zeno_kitchen2_edited_exact_native_replay_act_strict_fp32_rawtoken0/exact_replay_act_final.pt' \
  --device cpu --port 8775
```

In terminal B, first verify that a live call through the checkpoint returns
the expected full trajectory; this publishes nothing:

```bash
cd /home/zeno-rp/2027icra
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/deploy/zeno_exact_replay_act/ros_bridge.py \
  --worker-port 8775 --verify-only
```

When the normal robot controller and five JointState topics are running, use
the standard deployment command below.  It first moves the upper body from
the current measured pose to the first model action, holds it until feedback
is within the requested tolerance, and only then begins replay:

```bash
cd /home/zeno-rp/2027icra
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/deploy/zeno_exact_replay_act/ros_bridge.py \
  --worker-port 8775 --publish-commands --move-to-start
```

`--move-to-start` is a pre-replay setup phase: it sends a 50 Hz linear
upper-body ramp with base velocity fixed at `(0, 0, 0)`.  The requested
duration is 5 seconds, but it is automatically made longer if needed to keep
every command dimension at or below `0.15` native units/s
(`--move-to-start-max-speed`).  It does not resample, insert, or modify any
replay action.  The replay epoch starts only after fresh JointState feedback
has remained within `0.03` of the first action for 0.25 seconds (adjust with
`--move-to-start-position-tolerance` / `--move-to-start-settle-s` if the
controller's normal feedback noise requires it).  Use
`--move-to-start-s 8` or a longer value for a slower setup motion.

Once that epoch starts, every model action is published on the original
native action nanosecond clock (about 300 Hz here; each individual source `dt`
is preserved, not converted to a fixed rate).  The bridge refuses setup when
no controller subscriber or fresh live JointState feedback is present; do not
use `--allow-unchecked-start` or `--allow-no-command-subscriber` with
`--move-to-start`.  On completion or `Ctrl-C`, it publishes three all-zero
idle commands.  It also aborts (then sends idle) rather than silently burst or
drop actions if a native deadline is more than 20 ms late; tune only with
`--replay-max-lateness-s` after measuring the robot computer.  The model
action values and planned timestamps are exact; a normal Linux/Python/ROS
process still cannot promise zero physical DDS scheduling jitter at this rate.
