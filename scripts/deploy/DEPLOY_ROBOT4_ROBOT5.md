# robot4 / robot5 ACT+DINOv3 部署说明

部署电脑不需要训练数据，也不需要 `Data/lerobot/.../meta/stats.json`。归一化、反归一化和 action clamp 的 min/max 默认都从 checkpoint 目录里的 processor 文件读取。

脚本默认会根据自身位置自动推导项目根目录：`scripts/deploy/../..`。因此项目可以放在任意用户目录下，最外层项目名也可以不同。下面的 `$REPO_ROOT` 只是为了写命令方便。

## 1. 拉代码

新电脑第一次拉仓库：

```bash
mkdir -p ~/work
cd ~/work
PROJECT_DIR=zeno_humanoid_deploy
git clone --recurse-submodules git@github.com:Kikiorz/Zeno-humanoid.git "$PROJECT_DIR"
cd "$PROJECT_DIR"
git checkout develop
git submodule update --init --recursive
export REPO_ROOT="$(pwd)"
```

已经拉过仓库的电脑更新代码：

```bash
cd /path/to/your/project
export REPO_ROOT="$(pwd)"
git pull
git submodule update --init --recursive
```

确认使用子库里的 lerobot：

```bash
ls "$REPO_ROOT/third_party/lerobot/src/lerobot"
```

## 2. 放模型

每台部署电脑只需要放对应 robot 的 `pretrained_model` 目录。

robot4 默认模型路径：

```text
$REPO_ROOT/outputs/train/robot4_20260623_act_dinov3_base_dim768/checkpoints/100000/pretrained_model
```

robot5 默认模型路径：

```text
$REPO_ROOT/outputs/train/robot5_20260623_act_dinov3_base_dim768/checkpoints/100000/pretrained_model
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

模型 worker 和 ROS2 bridge 是两个独立进程，分别在两个终端里启动。模型 worker 用 conda；ROS2 bridge 用系统 ROS Python。通常不用在当前 shell 里手动 `source /opt/ros/...`，bridge 脚本会自己 source ROS。

终端 A：启动模型推理 worker。

```bash
cd "$REPO_ROOT/scripts/deploy"
./deploy_robot4_act_dinov3.py
```

确认终端输出里看到：

```text
normalizer: checkpoint processor files
```

确认 worker 终端里看到：

```text
action_stats=checkpoint policy_postprocessor_step_0_unnormalizer_processor.safetensors: action.min/action.max
```

终端 B：启动 ROS2 bridge。默认 dry-run，不会真正发控制指令：

```bash
cd "$REPO_ROOT/scripts/deploy"
./deploy_robot4_ros2_auto_cmd_bridge.py
```

确认 topic 和动作正常后，在终端 B 里允许发布控制：

```bash
cd "$REPO_ROOT/scripts/deploy"
./deploy_robot4_ros2_auto_cmd_bridge.py --publish-commands
```

robot4 默认 worker 端口是 `8764`，ROS2 node 名是 `robot4_auto_cmd_bridge`。

## 4. robot5 电脑启动

终端 A：启动模型推理 worker。

```bash
cd "$REPO_ROOT/scripts/deploy"
./deploy_robot5_act_dinov3.py
```

确认终端输出里看到：

```text
normalizer: checkpoint processor files
```

确认 worker 终端里看到：

```text
action_stats=checkpoint policy_postprocessor_step_0_unnormalizer_processor.safetensors: action.min/action.max
```

终端 B：启动 ROS2 bridge。默认 dry-run，不会真正发控制指令：

```bash
cd "$REPO_ROOT/scripts/deploy"
./deploy_robot5_ros2_auto_cmd_bridge.py
```

确认 topic 和动作正常后，在终端 B 里允许发布控制：

```bash
cd "$REPO_ROOT/scripts/deploy"
./deploy_robot5_ros2_auto_cmd_bridge.py --publish-commands
```

robot5 默认 worker 端口是 `8765`，ROS2 node 名是 `robot5_auto_cmd_bridge`。

## 5. 常用参数

`--checkpoint-path`、`--stats-path` 如果传相对路径，会按项目根目录 `$REPO_ROOT` 解析。

如果 checkpoint 不在默认路径：

```bash
./deploy_robot4_act_dinov3.py --checkpoint-path outputs/train/your_robot4_run/checkpoints/100000/pretrained_model
./deploy_robot5_act_dinov3.py --checkpoint-path outputs/train/your_robot5_run/checkpoints/100000/pretrained_model
```

如果 conda 环境名不同：

```bash
./deploy_robot4_act_dinov3.py --conda-env your_env_name
```

如果实际效果发散或明显滞后，先测试每一步都重新推理，避免执行 checkpoint 默认的 100 步 action queue：

```bash
./deploy_robot4_act_dinov3.py --n-action-steps 1
```

如果要测试 ACT temporal ensemble，需要同时让 `n_action_steps=1`：

```bash
./deploy_robot4_act_dinov3.py --n-action-steps 1 --temporal-ensemble-coeff 0.01
```

如果 `conda` 不在 `PATH` 里：

```bash
./deploy_robot4_act_dinov3.py --conda-bin /path/to/miniconda3/bin/conda
```

如果 ROS2 setup 路径不同：

```bash
./deploy_robot4_ros2_auto_cmd_bridge.py --ros-setup /opt/ros/humble/setup.bash
```

如果机器人上的 topic 名不同，可以覆盖对应 topic：

```bash
./deploy_robot4_ros2_auto_cmd_bridge.py \
  --head-cam-topic /zeno/h1/sensor/head_cam/image/compressed \
  --left-arm-cam-topic /zeno/h1/sensor/left_arm_cam/image/compressed \
  --right-arm-cam-topic /zeno/h1/sensor/right_arm_cam/image/compressed \
  --odom-topic /zeno/h1/sensor/odom_raw
```

## 6. 日志

现在 worker 和 bridge 都直接输出到各自终端。需要保存日志时，用 shell 重定向即可：

```bash
./deploy_robot4_act_dinov3.py 2>&1 | tee robot4_worker.log
./deploy_robot4_ros2_auto_cmd_bridge.py 2>&1 | tee robot4_bridge.log
```

主要看：

```text
worker: 模型加载、归一化来源、action_stats
bridge: ROS2 topic、观测是否 stale、推理延迟、控制发布状态
```
