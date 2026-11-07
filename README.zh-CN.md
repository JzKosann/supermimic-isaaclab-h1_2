# SuperMimic-H1_2

[English](README.md) | **中文** | [日本語](README.ja.md)

论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*
（[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）在 **Isaac Lab** 上、用 **Unitree H1-2** 人形机器人（27 自由度，去掉手）的**非官方**复现。

SuperMimic 从仿真数据里学一个可微的机器人世界模型，再在世界模型里做短视野想象、把梯度一路反传（BPTT）来训练动作跟踪策略，不用 PPO。
本仓库以 Isaac Lab 扩展的形式实现了完整流程。论文没有公开代码，这里的实现依据论文、两个公开的 G1 复现，以及论文没写明时我们自己的选择（见[与论文的差异](#与论文的差异)）。

> **状态**：代码已完成，想象侧的解析头和仿真逐项一致（见[一致性检查](#3-一致性检查)）。H1-2 的完整训练结果尚未发布。
>
> **更新（2026-11）：v2 配方。** 默认任务改为 `Tracking-Flat-H1_2-SuperMimic-v2`。它保留 v1 的观测和目标，加上输入归一化、
> 策略 EMA，以及一组训练稳定性和 clip 末尾处理的修正（见 [v2 配方](#v2-配方)）。这些改动来自我们更长的内部训练
> （250 小时的 BONES-SEED 动作库、地形实验、为真机 H1-2 部署做的 LAFAN1 微调）：v1 在约 1 万次更新后发散，v2 没有。
> 用本仓库从头训练 LAFAN1 v2 的结果尚未发布；v0 和 v1 的行为与之前完全相同（评测结果逐位一致）。

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
- **v2 训练修正**：运行均值输入归一化、策略 EMA、按论文尺度生效的梯度裁剪、估值上限、吸收态和按存活加权的 critic、
  无偏的重置采样估计、训练覆盖 clip 末尾，以及和部署端关节限位保护一致的关节目标截断。
- **微调**：从同一任务的任意 checkpoint 起步（`--init_from`），比如只在一部分 clip 上继续训练。
- **Appendix F 评测**：每条 clip 从首帧整条播一遍，无噪声、无域随机化；可选**末尾保持模式**，clip 播完后参考停在末帧，统计摔倒和漂移。
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
| `Tracking-Flat-H1_2-SuperMimic-v2` | 同 v1 | 同 v1 | **默认。** v1 加归一化、EMA 和稳定性修正（见 [v2 配方](#v2-配方)）。策略接口和 v1 相同。 |
| `…-Eval-v0`、`…-Anchor-Eval-v0`、`…-Eval-v1`、`…-Eval-v2` | | | Appendix F 评测环境。`eval_benchmark.py` / `play.py` 按 checkpoint 自动选择。 |

所有变体的机器人、跟踪奖励（Table A.2）、终止条件和域随机化（Table A.3）完全相同。v2 另外在关节限位项里惩罚超出截断范围的关节目标。

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
# v2 配方（默认任务；LAFAN1、4096 个环境、想象批 4096、H = 32、UTD 0.4、32 万次迭代）
python scripts/supermimic/train.py --headless --run_name lafan1_v2

# 论文配置 / v1 配方
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v0 --run_name lafan1
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v1 --run_name lafan1_v1

# 小显存冒烟测试（--wm_only_updates 20 让策略更新在这次运行里就开始）
python scripts/supermimic/train.py --headless --num_envs 64 --small_nets --policy_batch 256 --wm_batch 256 \
    --total_iterations 400 --wm_only_updates 20 --include walk1

# 续训（日志接着写进同一个目录）
python scripts/supermimic/train.py --headless --task <同一个任务> --resume logs/supermimic/<run>/ckpt_latest.pt

# 在一部分 clip 上微调某个 checkpoint（新建 run 目录；优化器和采样器重新开始，
# v1 / v2 会先只训 500 次世界模型）
python scripts/supermimic/train.py --headless --task <同一个任务> --init_from logs/supermimic/<run>/ckpt_latest.pt \
    --include walk run --total_iterations 20000 --run_name walk_run_ft
```

日志在 `logs/supermimic/<日期>_<run_name>/`：`config.json`、`tb/` 下的 TensorBoard 记录、每 4000 次迭代一个 checkpoint（`ckpt_XXXXXXX.pt`）和 `ckpt_latest.pt`。
用 `--logger wandb --log_project_name <名字>` 记到 Weights & Biases；`--max_hours` 到时间自动停。

常用参数：`--policy_batch`（显存不够时降到 2048）、`--wm_batch`、`--horizon`、`--replay_capacity`、`--wm_only_updates`、`--motion_dir`、`--include <子串> …`。

多卡数据并行：`python -m torch.distributed.run --nproc_per_node=N scripts/supermimic/train.py --headless --distributed`。
v0 / v1 的学习率乘 √N（同论文），v2 不乘。本仓库没有测试过多卡训练；同一套代码在我们内部训练里跑过 2–4 张 A100。

值得盯的训练指标（TensorBoard / wandb）：`policy/critic_over_bound_frac`（critic 估值超过 r<sub>max</sub>/(1−γ) = 10 的比例，
真实价值不可能超过 10；我们见到的每一次发散之前它都在上升）、`policy/*_peak`（一个日志区间里动作 / 状态 / 梯度的最大值）、
`sampler/last_bin_share`（落在 clip 最后 1 秒的重置比例，应接近 clip 数 / bin 数）和 `policy/nonfinite_skips`。

### 2. 评测（Appendix F）

```bash
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>/ckpt_0040000.pt
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>   # 评测该 run 的全部 checkpoint
```

每条 clip 从首帧播一遍。**归一化回报** = Σ 每步奖励 / (0.1 × 步数)，1.0 为完美跟踪；触发训练用的终止条件（根高度 / 倾斜）即为该 clip **失败**。
逐 clip 结果写到 `<run>/eval/<ckpt>.json`，每个 checkpoint 追加一行到 `<run>/eval/summary.csv`。
不加 `--headless` 时会带参考影子渲染评测过程。

选项：

- `--ema`：评测 v2 checkpoint 里存的 EMA 策略（通常比原版好；v2 建议都用它）。
- `--hold_s 8`：每条 clip 播完后参考停在末帧再保持 8 秒，和真机上参考播完就停住一样。另外报告保持阶段的摔倒、
  clip 结束时骨盆的水平漂移，以及该站着不动时机器人走开了多远；这时 `failed` 只算 clip 内的摔倒。
- `--only 0040000,latest`、`--every N`、`--skip_done` 选择 run 目录里的 checkpoint；`--tag` 给结果文件名加后缀。

### 3. 一致性检查

检查想象里用的解析头和 Isaac Lab 是否逐项一致（全部 28 个 link 的 FK、策略观测、每一项奖励、目标窗口）。改了机器人、观测或奖励之后都要跑一次：

```bash
python scripts/supermimic/check_consistency.py --headless --num_envs 32 --steps 300 [--task <任务>]
```

正常结果（v0、v1、v2 实测）：FK 位置误差 < 1e-5 m，速度误差 < 1e-4，观测误差约 1e-6，各项奖励误差约 1e-7。
v2 还会比较 Isaac Lab 里实际执行的截断后动作和想象里用的动作（应为 0）。

### 4. 播放

```bash
python scripts/supermimic/play.py --ckpt logs/supermimic/<run>                  # 该 run 最新的 checkpoint
python scripts/supermimic/play.py --ckpt <ckpt.pt> --num_envs 4 --include walk --real_time
python scripts/supermimic/play.py --ckpt <v2 ckpt.pt> --ema                     # v2 checkpoint 的 EMA 策略
```

蓝色半透明的机器人是参考动作。相机跟随一个机器人，但鼠标可以自由旋转、缩放。按键：`[` / `]` 换跟随的环境，`L` 开关跟随。

## Docker

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
cd docker
docker compose build
docker compose run --rm prepare-data                    # 下载 + 转换（只需一次）
GPU=0 RUN_NAME=lafan1_v2 docker compose up -d train     # TASK 默认是 Tracking-Flat-H1_2-SuperMimic-v2
GPU=0 CKPT=logs/supermimic/<run> EXTRA_ARGS="--ema" docker compose run --rm eval
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
        ├── commands.py             多 clip 动作命令、按回报的重置采样、clip 末尾冻结 / 保持
        ├── bin_estimator.py        重置采样用的每段回报估计（v0/v1 用 "ema"，v2 用 "count"）
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
| `-v2` | v1 + [v2 配方](#v2-配方)里列的改动。论文没有目标 critic，也没有估值上界；v2 仍然不用目标 critic，但给估值封顶，并让论文的裁剪阈值按论文的损失尺度生效。 |

### v2 配方

v2 保留 v1 的观测和目标，所以 v1 和 v2 的策略接口相同。每一项都是默认关闭的开关（v0 / v1 不变），由
`trainer.apply_recipe("v2")` 和 `H1_2SuperMimicV2EnvCfg` 打开。

| 改动 | 做什么 | 为什么 |
|---|---|---|
| 输入归一化 | 策略的观测 / 目标和 critic 的全部输入都过运行均值 / 方差（2·10<sup>8</sup> 个样本后冻结），同 Uni-SuperMimic。 | 不归一化时 critic 看到的是米、弧度原值。我们的 v1 在 250 小时动作库上约 1 万次更新后发散；加上这一项（连同下面两项，所以不是单因素消融）后同样的数据训了 5 万轮没有发散。 |
| 学习率、EMA | 三个网络学习率都是 4·10<sup>-4</sup>，多卡不乘 √N，余弦衰减 62.5 万次迭代；策略 EMA 0.995，存为 `policy_ema`，评测时用（`--ema`）。 | Uni-SuperMimic 的设置。我们的测试里 EMA 策略在 94 % 的留出 clip 上比原版好。 |
| 按论文尺度裁剪梯度 | 损失是 −J/H，而论文的裁剪（全局范数 1.0、逐元素 0.1）是针对 −J 的，所以两个阈值都除以 H。 | 按 −J/H 直接用论文阈值时，裁剪几乎从不生效。 |
| 估值上限 | 自举目标和 J 里的 V<sub>H</sub> 都夹到 r<sub>max</sub>/(1−γ) = 10 以下（同 realmimic）。 | 每步奖励最多 0.1，真实价值不可能超过 10。critic 没有目标网络、从自己自举，而 J 约 70 % 的权重在 γ<sup>H</sup>V<sub>H</sub> 上，高估的价值会被策略追逐、越涨越高；我们发散的地形训练里，记录到的平均估值有 55–61 % 超过 10。带上限的三条训练都稳定到最后（1.4 万轮），不带上限的四条全部发散。 |
| 吸收态、按存活加权 critic、跳过非有限梯度 | 存活权重低于 10<sup>-3</sup> 后，想象轨迹冻在死亡处（状态切断梯度、奖励为 0）；critic 损失按 sg(c<sub>t</sub>) 加权；梯度出现 NaN / Inf 的策略步 / critic 小批跳过；仿真里执行的动作夹在 ±20。 | 已经"死掉"的想象轨迹会被世界模型外推到数据之外很远：动作超过 100，单次更新的 critic 损失高达 10<sup>23</sup>。 |
| 无偏重置采样（`bin_estimator="count"`） | 在 clip 末尾或超时结束的回合，用最后几步的奖励补上延续价值；没访问过的段落按最差算；最初几次访问用累计平均。 | 原来的估计会把每条 clip 最后 1 秒的回报截断，看起来总是"最难"：在我们的大动作库里，重置分布中"难"的那部分有 88–97 % 落在 clip 最后 1 秒，稀有 clip 则长期看起来很容易。realmimic 也报告了同样的偏差。 |
| 训练覆盖 clip 末尾（`freeze_clip_end`） | clip 末尾之后参考停住、速度为 0：目标窗口里越过末尾的帧速度特征置零，想象起点可以到最后一帧并滚过末尾。 | 否则 clip 末尾从来练不到。我们的一次长时间微调里，新增失败约一半发生在 clip 最后 3 帧；部署前的 sim2sim 测试里策略在 clip 结束后立刻摔倒（本仓库可用 `--hold_s` 复现）。 |
| 关节目标截断（`guard_margin=0.02`） | 关节目标夹到 URDF 限位 − 0.02 rad：在执行噪声之后截（世界模型学到的就是截过的动作），想象里同样截；策略目标超出范围的部分加进关节限位惩罚。 | 部署端的保护本来就会截。没有这一项训练的策略，在一次 sim2sim 部署测试里 25–77 % 的步数关节目标（主要是踝 roll）顶在保护限位上；带截断微调后，测试 clip 上降到 1.8 %。 |

关节目标截断让策略的动作空间和 v1 略有不同。v1 的 checkpoint 可以在 `-Eval-v2` 里评测（接口相同），但它训练时没有截断。

## 结果

待补充。作为参考，论文报告 G1 在 LAFAN1 上用一张 H100 训练 1.2 小时达到归一化回报 0.82；两个 G1 复现的数字见下方链接。

## 致谢

- *SuperMimic*（[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）—— 方法本身。
- [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（Yuxi Liu，MIT）—— mjlab / G1 复现，本仓库逐模块移植自它。
- [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)（Chong Zhang，MIT）—— v1 配方的来源。
- [Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic)（Apache-2.0）—— v2 配方的输入归一化、学习率和策略 EMA。
- [realmimic](https://github.com/kevinhenryzz/realmimic)（Kevin Henry，Apache-2.0）—— v2 配方的无偏重置采样估计、clip 末尾冻结和估值上限。
- [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) —— 跟踪奖励和终止条件。
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)、[unitree_ros](https://github.com/unitreerobotics/unitree_ros)、
  [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model) —— 机器人配置和模型。
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 和
  [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) —— 动作数据。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)。

## 许可

代码以 [MIT 许可证](LICENSE)发布。第三方代码、随仓库附带的 URDF 和外部数据适用各自的许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。动作数据和机器人 USD 不随仓库分发。
