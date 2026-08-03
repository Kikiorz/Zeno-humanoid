# Zeno NPZ Replay

`replay_zeno_npz_state.py` sends an existing Zeno trajectory through the
normal deployment command topic:

```text
/zeno/h1/auto/wholebody/cmd
std_msgs/msg/Float64MultiArray
[1.0, upper_body_20, base_twist_3]
```

It is a ROS2 command tool. Use it only after the robot controller and all
state topics are running.

## Timing

The default is `--rate-hz source`: it publishes on the original timestamps of
the selected input stream rather than forcing 20 Hz.

- `--replay-source state` follows the native `state` timestamps. For the
  current bag conversion this is approximately 100 Hz.
- `--replay-source action` follows the native `action` timestamps. For the
  current bag conversion this is approximately 300 Hz.
- `--rate-hz 20` (or another positive value) deliberately resamples onto a
  fixed-rate clock. Use this only when a controller needs that behavior.

The scheduler uses the actual timestamp intervals, not merely their median
frequency. This preserves timing jitter and any nonuniform intervals from the
recording.

## Input NPZ contracts

The older, shared-clock format remains supported:

```text
timestamp_s  float64 [N]
state        float32/float64 [N, 23]
action       float32/float64 [N, 23]
```

The new native-clock format is used by
`scripts/data_convert/convert_zeno_bag_to_replay_npz.py` and preserves the
two source rates independently:

```text
state_timestamp_s  float64 [Ns]
state              float32 [Ns, 23]
state_base_twist   float32 [Ns, 3]  # recorded twist/cmd, causal-aligned
action_timestamp_s float64 [Na]
action             float32 [Na, 23]
```

The 23-D order is:

```text
torso_lift, torso_waist, head_pan, head_tilt,
left_arm_j0..j6, right_arm_j0..j6,
left_gripper, right_gripper,
base_vx, base_vy, base_rotation
```

In the default `--replay-source state` mode, the upper body comes from
`state[:20]`, while the base command comes from `state_base_twist` when it is
available (otherwise causal `action[20:23]`, for compatibility). Measured
`state[20:23]` odometry velocity is never sent as a command.

`--replay-source action` sends the full `action[:23]` command and is the
correct mode for a model-output NPZ.

## Replay normally

Preview without publishing:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/replay/replay_zeno_npz_state.py \
  --npz /path/to/trajectory_native.npz \
  --replay-source state \
  --dry-run
```

Replay the original native state timing on the robot (publishing is the
default):

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/replay/replay_zeno_npz_state.py \
  --npz /path/to/trajectory_native.npz \
  --replay-source state
```

To intentionally use fixed 20 Hz:

```bash
/usr/bin/python3 scripts/replay/replay_zeno_npz_state.py \
  --npz /path/to/trajectory_native.npz \
  --replay-source state \
  --rate-hz 20
```

Before the first command, the script waits for a real controller subscriber
and compares the live 20-D joint state to the first target. It refuses to
start if the largest error exceeds `0.20` unless explicitly overridden with
`--allow-unchecked-start`. It sends three all-zero idle commands on normal
exit or interruption.

`--transition-s` intentionally changes timing by inserting upper-body ramp
frames. It is therefore rejected in default source-timing mode; give an
explicit `--rate-hz` when using it.

## Replay and record the real robot state

Add `--record-actual-state` to a real replay run. The input NPZ is never
changed; a new capture NPZ is written after replay finishes:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/replay/replay_zeno_npz_state.py \
  --npz /path/to/trajectory_native.npz \
  --record-actual-state \
  --actual-state-output /home/zeno-rp/2027icra/Data/replay/trajectory_actual_state.npz
```

Each fresh capture row contains:

```text
timestamp_s           capture time relative to replay start
state [N, 23]         actual live joint feedback + measured odom velocity
action [N, 23]        exact target sent by replay (no leading mode 1.0)
source_timestamp_s    input NPZ time of that command
replay_frame_index    replay frame index for alignment
state_cache_age_s     age of torso, arms, grippers, and odom caches
```

State caches must be fresh (default maximum age `0.25` s). If a state topic
is missing or stale, that row is skipped. At action replay's roughly 300 Hz,
the captured feedback may repeat a most-recent 100 Hz JointState; that is
expected and is visible in the cache ages. Align by `replay_frame_index`, not
only array position.

The replay script provides initial-pose checking and safe idle on exit. It
does not impose per-joint hardware limits, velocity limits, or a mid-run
tracking-error stop; hardware-side limits and tracking protection remain
active.
