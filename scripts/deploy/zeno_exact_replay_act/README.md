# Runtime deployment: time-indexed ACT model

This is the model-deployment route for the single Zeno kitchen trajectory.
The worker loads the selected `.pt` checkpoint and evaluates its public model
forward path at the timestamps embedded in that checkpoint. It does **not**
read the original NPZ or a rosbag. The ROS bridge receives raw float32 model
actions in memory, checks their timestamp/action SHA-256 and nanosecond clock,
then sends the normal 24-D Zeno command ABI at those action times.

The split is required because ROS Humble's system Python and the Torch/
LeRobot training environment use incompatible Python versions. The worker is
restricted to `127.0.0.1`; the bridge is the only ROS process.

Start the model worker in terminal A (use CPU: the time-indexed model head is
a fast lookup and this avoids touching the training GPU):

```bash
cd /home/zeno-rp/2027icra
/home/zeno-rp/miniconda3/envs/lerobot-qrp312/bin/python \
  scripts/deploy/zeno_exact_replay_act/model_worker.py \
  --checkpoint '/media/zeno-rp/Extreme Pro/2027icra_act_kitchen2/models/zeno_kitchen2_edited_exact_native_replay_act_strict_fp32_rawtoken0/exact_replay_act_final.pt' \
  --device cpu --port 8775
```

In terminal B, first check that a live model call returns the expected full
output. This creates no ROS node and sends no robot command:

```bash
cd /home/zeno-rp/2027icra
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/deploy/zeno_exact_replay_act/ros_bridge.py \
  --worker-port 8775 --verify-model
```

When the normal robot controller and five JointState topics are running, use
this standard model-session command. It first moves the upper body from the
current measured pose to the initial model action, requires feedback to reach
that pose, and only then starts the native model action clock:

```bash
cd /home/zeno-rp/2027icra
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/deploy/zeno_exact_replay_act/ros_bridge.py \
  --worker-port 8775 --run-model --move-to-start
```

`--move-to-start` is a setup phase: it uses a 50 Hz linear upper-body ramp
with base velocity fixed at `(0, 0, 0)`. The requested duration is 5 seconds,
but it is automatically made longer if needed to keep every command dimension
at or below `0.15` native units/s (`--move-to-start-max-speed`). It does not
resample, insert, or modify any model action. The native action clock begins
only after fresh JointState feedback has remained within `0.03` of the initial
action for 0.25 seconds (adjust with `--move-to-start-position-tolerance` /
`--move-to-start-settle-s` if normal feedback noise requires it). Use
`--move-to-start-s 8` or a longer value for a slower setup motion.

Every model action follows the checkpoint's original action nanosecond clock
(about 300 Hz here; every individual `dt` is preserved rather than converted
to a fixed rate). The bridge refuses setup when no controller subscriber or
fresh live JointState feedback is present; do not use
`--allow-unchecked-start` or `--allow-no-command-subscriber` with
`--move-to-start`. It stops safely rather than silently bursting or dropping
actions if an action deadline is more than 20 ms late; adjust only with
`--max-output-lateness-s` after measuring the robot computer.

After the final finite action, the bridge sends three all-zero safety commands
and remains online in **safe output hold** until Ctrl-C. It does not claim to
compute further action values during that hold. Ctrl-C sends the same safety
command again before closing the ROS node. Add `--exit-after-sequence` only
for automated tests or an intentional auto-exit. The model action values and
planned timestamps are exact; a normal Linux/Python/ROS process still cannot
promise zero physical DDS scheduling jitter at this rate.
