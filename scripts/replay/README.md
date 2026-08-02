# Zeno NPZ Replay

`replay_zeno_npz_state.py` replays an existing Zeno trajectory through the
normal deployment command topic.  It publishes at 20 Hz:

```text
/zeno/h1/auto/wholebody/cmd
std_msgs/msg/Float64MultiArray
[1.0, upper_body_20, base_twist_3]
```

It is a ROS2 command tool.  Use it only after the robot controller and all
state topics are running.

## Input NPZ contract

The input needs these arrays:

```text
timestamp_s  float64 [N]
state        float32/float64 [N, 23]
action       float32/float64 [N, 23]
```

The 23-D order is:

```text
torso_lift, torso_waist, head_pan, head_tilt,
left_arm_j0..j6, right_arm_j0..j6,
left_gripper, right_gripper,
base_vx, base_vy, base_rotation
```

In the default `--replay-source state` mode, replay publishes
`state[:20]` for the upper body and `action[20:23]` for the base.  Therefore
the upper body follows the recorded feedback pose, while the base follows the
recorded `/zeno/h1/twist/cmd` tail.  Measured `state[20:23]` odometry velocity
is never sent as a command.

Use `--replay-source action` only for a model-output NPZ, where every 23-D
action value is a command target.

## Replay normally

First preview without publishing:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/replay/replay_zeno_npz_state.py \
  --npz /path/to/trajectory.npz \
  --dry-run
```

Then replay on the robot (publishing is the default):

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/replay/replay_zeno_npz_state.py \
  --npz /path/to/trajectory.npz
```

Before the first command, the script waits for a real controller subscriber
and compares the live 20-D joint state to the first target.  It refuses to
start if the largest error exceeds `0.20` unless explicitly overridden with
`--allow-unchecked-start`.  It sends three all-zero idle commands on normal
exit or interruption.

## Replay and record the real robot state

Add `--record-actual-state` to one replay command.  This never changes or
overwrites the input NPZ.  It creates a **new** NPZ after replay finishes:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/replay/replay_zeno_npz_state.py \
  --npz /path/to/trajectory.npz \
  --record-actual-state \
  --actual-state-output /home/zeno-rp/2027icra/Data/replay/trajectory_actual_state.npz
```

The capture shares the replay process and the same 20 Hz loop, so it does not
add an extra subscriber to the auto-command topic or affect replay's
subscriber preflight.  Each fresh capture row contains:

```text
timestamp_s           capture time relative to replay start
state [N, 23]         actual live joint feedback + measured odom velocity
action [N, 23]        exact target sent by replay at that frame (no leading mode 1.0)
source_timestamp_s    input NPZ time of the command
replay_frame_index    replay frame index for alignment
state_cache_age_s     age of torso, left arm, right arm, left gripper,
                      right gripper, and odom caches
```

State caches must be fresh (default maximum age `0.25` s).  If a state topic
is missing or stale, that row is skipped rather than mixing an old feedback
value with a new target.  Check the adjacent JSON summary for skipped-row
counts.  A capture can therefore contain fewer rows than replay frames; align
by `replay_frame_index`, not merely array position.

`--record-actual-state` requires a real publishing run and intentionally
rejects `--dry-run`.  The script refuses an output path equal to the input
NPZ; use `--overwrite-actual-state` only to replace a previous capture file.

## Useful safety options

```bash
# Insert 0.5 seconds of linear upper-body ramp only for a step > 0.30 rad.
# This changes replay timing deliberately.
--transition-s 0.5 --transition-threshold-rad 0.30

# Tighten the initial live-state vs first-command check.
--start-max-position-error 0.10

# Require more recent feedback for capture rows.
--actual-state-max-age-s 0.10
```

The replay script has an initial-pose check and safe idle on exit, but it does
not impose per-joint hardware limits, velocity limits, or a mid-run tracking
error stop.  Hardware-side limits and tracking protection remain active; use
the captured actual-state NPZ to diagnose whether a target was reached.
