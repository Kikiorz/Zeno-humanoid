# robot4 / robot5 ACT+DINOv3 部署说明

部署电脑不需要训练数据，也不需要 `Data/lerobot/.../meta/stats.json`。归一化、反归一化和 action clamp 的 min/max 默认都从 checkpoint 目录里的 processor 文件读取。

## 1. 拉代码

新电脑第一次拉仓库：

```bash
cd /home/zeno-rp
git clone --recurse-submodules git@github.com:Kikiorz/Zeno-humanoid.git 2027icra
cd /home/zeno-rp/2027icra
git checkout develop
git submodule update --init --recursive
```

已经拉过仓库的电脑更新代码：

```bash
cd /home/zeno-rp/2027icra
git pull
git submodule update --init --recursive
```

确认使用子库里的 lerobot：

```bash
ls /home/zeno-rp/2027icra/third_party/lerobot/src/lerobot
```

## 2. 放模型

每台部署电脑只需要放对应 robot 的 `pretrained_model` 目录。

robot4 默认模型路径：

```text
/home/zeno-rp/2027icra/outputs/train/robot4_20260623_act_dinov3_base_dim768/checkpoints/100000/pretrained_model
```

robot5 默认模型路径：

```text
/home/zeno-rp/2027icra/outputs/train/robot5_20260623_act_dinov3_base_dim768/checkpoints/100000/pretrained_model
```

模型目录至少要包含这些文件：

```text
config.json
model.safetensors
train_config.json
policy_preprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

如果模型不放在默认路径，启动时加 `--checkpoint-path /path/to/pretrained_model`。

## 3. robot4 电脑启动

先 dry-run 检查，不会真正发控制指令：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy
./deploy_robot4_act_dinov3.py
```

确认终端输出里看到：

```text
normalizer: checkpoint processor files
```

确认 worker 日志里看到：

```text
action_stats=checkpoint policy_postprocessor_step_0_unnormalizer_processor.safetensors: action.min/action.max
```

确认 topic 和动作正常后，再允许发布控制：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy
./deploy_robot4_act_dinov3.py --publish-commands
```

robot4 默认 worker 端口是 `8764`，ROS2 node 名是 `robot4_auto_cmd_bridge`。

## 4. robot5 电脑启动

先 dry-run 检查：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy
./deploy_robot5_act_dinov3.py
```

确认终端输出里看到：

```text
normalizer: checkpoint processor files
```

确认 worker 日志里看到：

```text
action_stats=checkpoint policy_postprocessor_step_0_unnormalizer_processor.safetensors: action.min/action.max
```

确认 topic 和动作正常后，再允许发布控制：

```bash
cd /home/zeno-rp/2027icra/scripts/deploy
./deploy_robot5_act_dinov3.py --publish-commands
```

robot5 默认 worker 端口是 `8765`，ROS2 node 名是 `robot5_auto_cmd_bridge`。

## 5. 常用参数

如果 checkpoint 不在默认路径：

```bash
./deploy_robot4_act_dinov3.py --checkpoint-path /path/to/pretrained_model
./deploy_robot5_act_dinov3.py --checkpoint-path /path/to/pretrained_model
```

如果 conda 环境名不同：

```bash
./deploy_robot4_act_dinov3.py --conda-env your_env_name
```

如果 ROS2 setup 路径不同：

```bash
./deploy_robot4_act_dinov3.py --ros-setup /opt/ros/humble/setup.bash
```

如果机器人上的 topic 名不同，可以覆盖对应 topic：

```bash
./deploy_robot4_act_dinov3.py \
  --head-cam-topic /zeno/h1/sensor/head_cam/image/compressed \
  --left-arm-cam-topic /zeno/h1/sensor/left_arm_cam/image/compressed \
  --right-arm-cam-topic /zeno/h1/sensor/right_arm_cam/image/compressed \
  --odom-topic /zeno/h1/sensor/odom_raw
```

## 6. 日志位置

默认日志：

```text
/home/zeno-rp/2027icra/outputs/logs/deploy/
```

主要看两个文件：

```text
robot4_robot4_20260623_act_dinov3_base_dim768_worker.log
robot4_robot4_20260623_act_dinov3_base_dim768_bridge.log
robot5_robot5_20260623_act_dinov3_base_dim768_worker.log
robot5_robot5_20260623_act_dinov3_base_dim768_bridge.log
```

worker 日志负责模型加载和归一化来源；bridge 日志负责 ROS2 topic、观测是否 stale、推理延迟和控制发布状态。
