# SuperMimic-H1_2

[English](README.md) | **中文** | [日本語](README.ja.md)

论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*
（[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）在 **Isaac Lab** 上、用 **Unitree H1-2** 人形机器人（27 自由度，去掉手）的**非官方**复现。

SuperMimic 从仿真数据里学一个可微的机器人世界模型，再在世界模型里做短视野想象、把梯度一路反传（BPTT）来训练动作跟踪策略，不用 PPO。
本仓库以 Isaac Lab 扩展的形式实现了完整流程。论文没有公开代码，这里的实现依据论文、两个公开的 G1 复现，以及论文没写明时我们自己的选择（见[与论文的差异](#与论文的差异)）。

> **状态**：代码已完成，想象侧的解析头和仿真逐项一致（见[一致性检查](#3-一致性检查)）。H1-2 的完整训练结果尚未发布。

## 目录

- [功能](#功能)
- [任务](#任务)
- [环境要求](#环境要求)
- [安装](#安装)
- [使用](#使用)
- [Docker](#docker)
- [目录结构](#目录结构)
- [与论文的差异](#与论文的差异)
- [结果](#结果)
- [致谢](#致谢)
- [许可](#许可)

## 功能

- **世界模型 + BPTT 训练器**（Algorithm A.1）：动力学 MLP（3 × 2048，输出积分成下一状态）、学习的终止头、8 步开环训练损失（Eq. 1）；
  视野 H = 32 的想象 rollout、确定性策略、拟合 GAE（λ = 0.95）目标的 critic，以及由 Isaac Lab 采集的 replay buffer。
- **解析头**：观测、8 项跟踪 / 正则奖励和终止特征都通过可微的 URDF 正运动学从预测状态算出，梯度能从奖励一路传回策略。
- **多 clip 动作库**，按回报做重置采样（每 1 秒一个 bin）。
- **Appendix F 评测**：每条 clip 从首帧整条播一遍，无噪声、无域随机化。
- **可视化**：半透明的参考动作"影子"，以及只平移跟随、鼠标仍可自由旋转缩放的相机。
- **数据管线**：LAFAN1 重定向到 H1-2（40 条，2.45 小时，与论文 Fig. 4a 同一套 clip），转成 50 Hz。
- 基于官方 Isaac Lab 镜像的 **Docker** 配置。

## 任务

`import supermimic_h1_2.tasks` 会注册以下 gym 任务：

| 任务 | 策略观测（每帧维数 × 10 帧历史） | 目标（每帧维数 × 10 帧） | 说明 |
|---|---|---|---|
| `Tracking-Flat-H1_2-SuperMimic-v0` | 87：关节位置（含编码器零偏）/ 速度、IMU 角速度、投影重力、上一动作 | 64：未来参考帧 | 论文观测（Table A.1）。没有基座位置 / 线速度，不需要状态估计。 |
| `Tracking-Flat-H1_2-SuperMimic-Anchor-v0` | 同 v0 | 64 + 9 维锚点相对位姿 | 偏离论文（SuperMimic-Reproduced 的实验 e3）。 |
| `Tracking-Flat-H1_2-SuperMimic-v1` | 87 + 6：参考根姿态相对机器人的 6D 表示 | 69：轨迹式目标 | v0 加上 supermimic_vibed 的改进（见[与论文的差异](#与论文的差异)）。 |
| `…-Eval-v0`、`…-Anchor-Eval-v0`、`…-Eval-v1` | | | Appendix F 评测环境。`eval_benchmark.py` / `play.py` 按 checkpoint 自动选择。 |

所有变体的机器人、奖励（Table A.2）、终止条件和域随机化（Table A.3）完全相同。

## 环境要求

| | 测试过的版本 |
|---|---|
| 系统 | Ubuntu（Linux） |
| Isaac Lab | 2.3.2 |
| Isaac Sim | 5.1.0 |
| PyTorch | 2.7.0 + CUDA 12.8（Isaac Sim 自带） |
| Python | 3.11 |
| GPU | 完整配置（4096 个环境、想象批 4096）在服务器单卡上训练；8 GB 的笔记本显卡（RTX 5060）只能跑冒烟测试配置。 |

默认开启 `torch.compile`，策略更新约快 4.5 倍；用 `--compile none` 关闭。

## 安装

1. **安装 Isaac Lab 2.3.2**：按[官方文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装并激活它的 Python 环境。

2. **克隆本仓库并安装扩展**（必须用 editable 模式：默认路径按仓库位置推算）：

   ```bash
   git clone <本仓库地址> SuperMimic-H1_2
   cd SuperMimic-H1_2
   python -m pip install -e source/supermimic_h1_2
   ```

3. **下载机器人模型和动作数据**（约 120 MB，不需要 Isaac）：

   ```bash
   python scripts/download_assets.py
   ```

   H1-2 的 USD 从 [unitreerobotics/unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model) 下载到 `assets/unitree_model/`，
   40 条 H1-2 CSV 从 [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 下载到 `datasets/lafan1_csv/h1_2/`。

4. **把 clip 转成 50 Hz 的 npz**（在 Isaac Lab 里逐帧回放，RTX 5060 笔记本显卡上约 5 分钟）：

   ```bash
   python scripts/csv_to_npz.py --headless
   ```

   输出在 `datasets/lafan1_h1_2_npz/*.npz`。

路径可以用环境变量覆盖：`SUPERMIMIC_UNITREE_MODEL_DIR`、`SUPERMIMIC_CSV_DIR`、`SUPERMIMIC_MOTION_DIR`、`SUPERMIMIC_LOG_ROOT`
（见 `source/supermimic_h1_2/supermimic_h1_2/paths.py`）。

## 使用

以下命令都在仓库根目录执行。

### 1. 训练

```bash
# 论文配置（LAFAN1、4096 个环境、想象批 4096、H = 32、UTD 0.4、32 万次迭代）
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v0 --run_name lafan1

# v1 配方
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v1 --run_name lafan1_v1

# 小显存冒烟测试
python scripts/supermimic/train.py --headless --num_envs 64 --small_nets --policy_batch 256 --wm_batch 256 \
    --total_iterations 600 --include walk1

# 续训（日志接着写进同一个目录）
python scripts/supermimic/train.py --headless --task <同一个任务> --resume logs/supermimic/<run>/ckpt_latest.pt
```

日志在 `logs/supermimic/<日期>_<run_name>/`：`config.json`、`tb/` 下的 TensorBoard 记录、每 4000 次迭代一个 checkpoint（`ckpt_XXXXXXX.pt`）和 `ckpt_latest.pt`。
用 `--logger wandb --log_project_name <名字>` 记到 Weights & Biases；`--max_hours` 到时间自动停。

常用参数：`--policy_batch`（显存不够时降到 2048）、`--wm_batch`、`--horizon`、`--replay_capacity`、`--motion_dir`、`--include <子串> …`。

多卡数据并行已实现（`python -m torch.distributed.run --nproc_per_node=N scripts/supermimic/train.py --headless --distributed`），但**没有测试过**。

### 2. 评测（Appendix F）

```bash
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>/ckpt_0040000.pt
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>   # 评测该 run 的全部 checkpoint
```

每条 clip 从首帧播一遍。**归一化回报** = Σ 每步奖励 / (0.1 × 步数)，1.0 为完美跟踪；触发训练用的终止条件（根高度 / 倾斜）即为该 clip **失败**。
逐 clip 结果写到 `<run>/eval/<ckpt>.json`，每个 checkpoint 追加一行到 `<run>/eval/summary.csv`。
不加 `--headless` 时会带参考影子渲染评测过程。

### 3. 一致性检查

检查想象里用的解析头和 Isaac Lab 是否逐项一致（全部 28 个 link 的 FK、策略观测、每一项奖励、目标窗口）。改了机器人、观测或奖励之后都要跑一次：

```bash
python scripts/supermimic/check_consistency.py --headless --num_envs 32 --steps 300 [--task <任务>]
```

正常结果（v0、v1 实测）：FK 位置误差 < 1e-5 m，速度误差 < 1e-4，观测误差约 1e-6，各项奖励误差约 1e-7。

### 4. 播放

```bash
python scripts/supermimic/play.py --ckpt logs/supermimic/<run>                  # 该 run 最新的 checkpoint
python scripts/supermimic/play.py --ckpt <ckpt.pt> --num_envs 4 --include walk --real_time
```

蓝色半透明的机器人是参考动作。相机跟随一个机器人，但鼠标可以自由旋转、缩放。按键：`[` / `]` 换跟随的环境，`L` 开关跟随。

## Docker

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
cd docker
docker compose build
docker compose run --rm prepare-data                    # 下载 + 转换（只需一次）
GPU=0 TASK=Tracking-Flat-H1_2-SuperMimic-v1 RUN_NAME=lafan1_v1 docker compose up -d train
GPU=0 CKPT=logs/supermimic/<run> docker compose run --rm eval
```

`logs/`、`datasets/`、`assets/unitree_model/`、`source/`、`scripts/` 都从宿主机挂载，改代码不用重建镜像。
设 `LOGGER=wandb WANDB_API_KEY=…` 记到 Weights & Biases，设 `RESUME=<ckpt>` 续训。

## 目录结构

```
SuperMimic-H1_2/
├── assets/h1_2_description/        可微 FK 用的 H1-2 URDF（来自 unitree_ros）
├── docker/                         Dockerfile + docker-compose.yml
├── scripts/
│   ├── download_assets.py          从 HuggingFace 下载机器人 USD 和 LAFAN1 H1-2 CSV
│   ├── csv_to_npz.py               CSV（30 fps）→ npz（50 Hz），在 Isaac Lab 里回放
│   └── supermimic/                 train.py、eval_benchmark.py、play.py、check_consistency.py
└── source/supermimic_h1_2/supermimic_h1_2/
    ├── paths.py                    默认路径（可用环境变量覆盖）
    ├── robots.py                   H1-2 机器人配置
    ├── tasks/                      gym 注册 + 环境配置
    └── mdp/
        ├── commands.py             多 clip 动作命令、按回报的重置采样
        ├── actions.py、events.py、observations.py、rewards.py、terminations.py
        ├── kinematics.py           可微 URDF 正运动学（torch）
        ├── motion_library.py       动作库、参考帧、目标窗口
        ├── state.py                世界模型的状态表示
        ├── world_model.py          动力学 + 终止头
        ├── heads.py                解析观测 / 奖励 / 终止头
        ├── imagination.py          想象 rollout、BPTT、critic
        ├── policy.py、replay.py、trainer.py、distributed.py
        ├── benchmark.py            Appendix F 评测
        └── viewer.py               参考影子 + 跟随相机
```

## 与论文的差异

论文没给数值、或我们有意偏离的地方：

| 项目 | 本仓库 |
|---|---|
| 机器人 | Unitree H1-2（去掉手，27 关节），不是 G1；执行器取自 unitree_rl_lab（隐式 PD）。 |
| 动作缩放 α | 0.25（论文没给）。 |
| 物理引擎 | PhysX（Isaac Lab）。 |
| 参考速度 | 用 URDF FK 算的 link 原点速度；npz 里存的 `body_lin_vel_w` 是质心速度，不使用。 |
| 奖励重锚 | 用当前这一步的机器人锚点（BeyondMimic 滞后一步）。 |
| replay 的终止状态 | `SuperMimicRLEnv` 加了重置前钩子，在重置覆盖之前取 s<sub>T+1</sub>（Isaac Lab 的 RecorderManager 在 4096 个环境时太慢）。 |
| 策略 | 确定性，与论文一致。 |
| `-Anchor-v0` | 目标里加参考锚点相对机器人的位姿（9 维），论文没有。 |
| `-v1` | v0 + 观测里加参考根姿态、轨迹式目标、开头 500 次更新只训世界模型、想象起点不越过 clip 末尾、解析软终止、critic 输入加跟踪误差。没有采用 supermimic_vibed 的：运行均值归一化、bf16、BeyondMimic 低增益执行器、Newton 物理。所有 v1 行为都在开关后面，默认值等于 v0。 |

## 结果

待补充。作为参考，论文报告 G1 在 LAFAN1 上用一张 H100 训练 1.2 小时达到归一化回报 0.82；两个 G1 复现的数字见下方链接。

## 致谢

- *SuperMimic*（[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）—— 方法本身。
- [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（Yuxi Liu，MIT）—— mjlab / G1 复现，本仓库逐模块移植自它。
- [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)（Chong Zhang，MIT）—— v1 配方的来源。
- [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) —— 跟踪奖励和终止条件。
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)、[unitree_ros](https://github.com/unitreerobotics/unitree_ros)、
  [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model) —— 机器人配置和模型。
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 和
  [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) —— 动作数据。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)。

## 许可

代码以 [MIT 许可证](LICENSE)发布。第三方代码、随仓库附带的 URDF 和外部数据适用各自的许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。动作数据和机器人 USD 不随仓库分发。
