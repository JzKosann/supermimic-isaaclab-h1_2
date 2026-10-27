# SuperMimic-H1_2

**English** | [中文](README.zh-CN.md) | [日本語](README.ja.md)

An **unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*
([arXiv:2610.28480](https://arxiv.org/abs/2610.28480)) on **Isaac Lab** with the **Unitree H1-2** humanoid (27 DoF, hands removed).

SuperMimic learns a differentiable world model of the robot from simulator rollouts and trains a motion-tracking
policy by back-propagating through short imagined rollouts (BPTT), instead of PPO. This repository implements the
full pipeline as an Isaac Lab extension. No official code has been released; everything here follows the paper,
two public G1 reproductions, and our own choices where the paper is silent (listed [below](#differences-from-the-paper)).

> **Status.** The code is complete and the analytic heads match the simulator term by term (see
> [Consistency check](#3-consistency-check)). Full-length H1-2 training results have not been published yet.

## Contents

- [Features](#features)
- [Tasks](#tasks)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Repository layout](#repository-layout)
- [Differences from the paper](#differences-from-the-paper)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- **World model + BPTT trainer** (Algorithm A.1): a dynamics MLP (3 × 2048) whose output is integrated into the
  next state, a learned termination head, 8-step open-loop training loss (Eq. 1); imagined rollouts with horizon
  H = 32, a deterministic policy, a critic fitted to GAE(λ = 0.95) targets, and a replay buffer filled from Isaac Lab.
- **Analytic heads**: observations, the eight tracking / regularisation rewards and termination features are
  computed from the predicted state through a differentiable URDF forward-kinematics module, so gradients flow
  from the reward back into the policy.
- **Multi-clip motion library** with return-based reset sampling (1 s bins).
- **Appendix-F benchmark**: every clip played once from its first frame, no noise and no domain randomisation.
- **Viewer**: a translucent "ghost" of the reference motion, and a follow camera that keeps mouse orbit/zoom.
- **Data pipeline**: LAFAN1 retargeted to H1-2 (40 clips, 2.45 h — the same clip set as the paper's Fig. 4a),
  converted to 50 Hz.
- **Docker** setup based on the official Isaac Lab image.

## Tasks

Importing `supermimic_h1_2.tasks` registers these gym ids:

| Task | Policy observation (per frame × 10) | Goal (per frame × 10) | Notes |
|---|---|---|---|
| `Tracking-Flat-H1_2-SuperMimic-v0` | 87: joint pos (with encoder bias) / vel, IMU angular velocity, projected gravity, last action | 64: future reference frames | Paper observation (Table A.1). No base position / linear velocity, so no state estimator is needed. |
| `Tracking-Flat-H1_2-SuperMimic-Anchor-v0` | same as v0 | 64 + 9 anchor-relative pose | Deviates from the paper (experiment e3 of SuperMimic-Reproduced). |
| `Tracking-Flat-H1_2-SuperMimic-v1` | 87 + 6: reference root orientation relative to the robot (6D) | 69: trajectory-style goal | v0 plus the recipe of supermimic_vibed (see [Differences](#differences-from-the-paper)). |
| `…-Eval-v0`, `…-Anchor-Eval-v0`, `…-Eval-v1` | | | Appendix-F evaluation environments. `eval_benchmark.py` / `play.py` pick the right one from the checkpoint. |

All variants share the robot, rewards (Table A.2), terminations and domain randomisation (Table A.3).

## Requirements

| | Tested with |
|---|---|
| OS | Ubuntu (Linux) |
| Isaac Lab | 2.3.2 |
| Isaac Sim | 5.1.0 |
| PyTorch | 2.7.0 + CUDA 12.8 (the one shipped with Isaac Sim) |
| Python | 3.11 |
| GPU | Full configuration (4096 envs, imagination batch 4096): trained on a single server GPU. An 8 GB laptop GPU (RTX 5060) only runs the smoke-test configuration. |

`torch.compile` is used by default; it makes policy updates about 4.5× faster. Disable it with `--compile none`.

## Installation

1. **Install Isaac Lab 2.3.2** following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   and activate its Python environment.

2. **Clone this repository and install the extension** (editable mode is required: default paths are resolved
   relative to the repository):

   ```bash
   git clone <this-repo-url> SuperMimic-H1_2
   cd SuperMimic-H1_2
   python -m pip install -e source/supermimic_h1_2
   ```

3. **Download the robot model and the motion data** (≈ 120 MB, no Isaac needed):

   ```bash
   python scripts/download_assets.py
   ```

   This fetches the H1-2 USD from [unitreerobotics/unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model)
   into `assets/unitree_model/` and the 40 H1-2 CSV clips from
   [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
   into `datasets/lafan1_csv/h1_2/`.

4. **Convert the clips to 50 Hz npz** (replays each clip in Isaac Lab; ~5 min on an RTX 5060 laptop GPU):

   ```bash
   python scripts/csv_to_npz.py --headless
   ```

   Output: `datasets/lafan1_h1_2_npz/*.npz`.

Paths can be overridden with environment variables: `SUPERMIMIC_UNITREE_MODEL_DIR`, `SUPERMIMIC_CSV_DIR`,
`SUPERMIMIC_MOTION_DIR`, `SUPERMIMIC_LOG_ROOT` (see `source/supermimic_h1_2/supermimic_h1_2/paths.py`).

## Usage

All commands are run from the repository root.

### 1. Train

```bash
# Paper configuration (LAFAN1, 4096 envs, imagination batch 4096, H = 32, UTD 0.4, 320k iterations)
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v0 --run_name lafan1

# v1 recipe
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v1 --run_name lafan1_v1

# Smoke test on a small GPU
python scripts/supermimic/train.py --headless --num_envs 64 --small_nets --policy_batch 256 --wm_batch 256 \
    --total_iterations 600 --include walk1

# Resume (logs continue in the same directory)
python scripts/supermimic/train.py --headless --task <same task> --resume logs/supermimic/<run>/ckpt_latest.pt
```

Logs go to `logs/supermimic/<date>_<run_name>/`: `config.json`, TensorBoard events in `tb/`, a checkpoint every
4000 iterations (`ckpt_XXXXXXX.pt`) and `ckpt_latest.pt`. Use `--logger wandb --log_project_name <name>` for
Weights & Biases. `--max_hours` stops training at a wall-clock budget.

Useful flags: `--policy_batch` (lower to 2048 if you run out of memory), `--wm_batch`, `--horizon`,
`--replay_capacity`, `--motion_dir`, `--include <substring> …`.

Multi-GPU data parallelism is implemented (`python -m torch.distributed.run --nproc_per_node=N scripts/supermimic/train.py --headless --distributed`)
but **has not been tested**.

### 2. Evaluate (Appendix F)

```bash
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>/ckpt_0040000.pt
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>   # every checkpoint in the run
```

Each clip is played once from its first frame. **Normalised return** = Σ per-step reward / (0.1 × steps), where 1.0
is perfect tracking; a clip **fails** if a training termination (root height / tilt) fires. Per-clip results go to
`<run>/eval/<ckpt>.json` and one line per checkpoint is appended to `<run>/eval/summary.csv`.
Without `--headless` the evaluation is rendered with the reference ghost.

### 3. Consistency check

Checks that the analytic heads used in imagination match Isaac Lab term by term (FK of all 28 links, the policy
observation, all reward terms, the goal window). Run it after any change to the robot, observations or rewards:

```bash
python scripts/supermimic/check_consistency.py --headless --num_envs 32 --steps 300 [--task <task>]
```

Expected (measured with v0 and v1): FK position errors < 1e-5 m, velocity errors < 1e-4, observation errors ~1e-6, reward-term errors ~1e-7.

### 4. Play

```bash
python scripts/supermimic/play.py --ckpt logs/supermimic/<run>                  # latest checkpoint of the run
python scripts/supermimic/play.py --ckpt <ckpt.pt> --num_envs 4 --include walk --real_time
```

The blue translucent robot is the reference motion. The camera follows one robot but can be freely orbited and
zoomed with the mouse. Keys: `[` / `]` switch the followed env, `L` toggles following.

## Docker

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
cd docker
docker compose build
docker compose run --rm prepare-data                    # download + convert (once)
GPU=0 TASK=Tracking-Flat-H1_2-SuperMimic-v1 RUN_NAME=lafan1_v1 docker compose up -d train
GPU=0 CKPT=logs/supermimic/<run> docker compose run --rm eval
```

`logs/`, `datasets/`, `assets/unitree_model/`, `source/` and `scripts/` are mounted from the host, so code changes
need no rebuild. Set `LOGGER=wandb WANDB_API_KEY=…` for Weights & Biases, `RESUME=<ckpt>` to resume.

## Repository layout

```
SuperMimic-H1_2/
├── assets/h1_2_description/        H1-2 URDF used by the differentiable FK (from unitree_ros)
├── docker/                         Dockerfile + docker-compose.yml
├── scripts/
│   ├── download_assets.py          robot USD + LAFAN1 H1-2 CSV from HuggingFace
│   ├── csv_to_npz.py               CSV (30 fps) → npz (50 Hz) via Isaac Lab replay
│   └── supermimic/                 train.py, eval_benchmark.py, play.py, check_consistency.py
└── source/supermimic_h1_2/supermimic_h1_2/
    ├── paths.py                    default paths (overridable by env vars)
    ├── robots.py                   H1-2 articulation config
    ├── tasks/                      gym registration + environment configs
    └── mdp/
        ├── commands.py             multi-clip motion command, return-based reset sampling
        ├── actions.py, events.py, observations.py, rewards.py, terminations.py
        ├── kinematics.py           differentiable URDF FK (torch)
        ├── motion_library.py       clip storage, reference frames, goal windows
        ├── state.py                world-model state layout
        ├── world_model.py          dynamics + termination head
        ├── heads.py                analytic observation / reward / termination heads
        ├── imagination.py          imagined rollouts, BPTT, critic
        ├── policy.py, replay.py, trainer.py, distributed.py
        ├── benchmark.py            Appendix-F evaluation
        └── viewer.py               reference ghost + follow camera
```

## Differences from the paper

Choices where the paper gives no value, or where we deliberately deviate:

| Item | This repository |
|---|---|
| Robot | Unitree H1-2 without hands (27 joints) instead of G1; actuators from unitree_rl_lab (implicit PD). |
| Action scale α | 0.25 (not given in the paper). |
| Physics | PhysX (Isaac Lab). |
| Reference velocities | Link-origin velocities from URDF FK; the `body_lin_vel_w` stored in the npz is the centre-of-mass velocity and is not used. |
| Reward re-anchoring | Uses the robot anchor at the current step (BeyondMimic lags by one step). |
| Terminal states for the replay | `SuperMimicRLEnv` adds a pre-reset hook that reads s<sub>T+1</sub> before the reset overwrites it (Isaac Lab's RecorderManager is too slow at 4096 envs). |
| Policy | Deterministic, as in the paper. |
| `-Anchor-v0` | Adds the reference anchor pose relative to the robot (9 dims) to the goal. Not in the paper. |
| `-v1` | v0 + reference root orientation in the observation, trajectory-style goal, 500 world-model-only updates at the start, imagination starts never cross the clip end, analytic soft termination, tracking error as critic input. Not adopted from supermimic_vibed: running-mean normalisation, bf16, BeyondMimic low-gain actuators, Newton physics. All v1 behaviour is behind switches whose defaults equal v0. |

## Results

Coming soon. For reference, the paper reports a normalised return of 0.82 on LAFAN1 after 1.2 h on one H100
(G1); see the two G1 reproductions below for their numbers.

## Acknowledgements

- *SuperMimic* ([arXiv:2610.28480](https://arxiv.org/abs/2610.28480)) — the method.
- [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) (Yuxi Liu, MIT) — the mjlab / G1
  reproduction this code was ported from module by module.
- [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Chong Zhang, MIT) — the source of the v1 recipe.
- [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) — tracking rewards and
  terminations.
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab), [unitree_ros](https://github.com/unitreerobotics/unitree_ros),
  [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model) — robot configuration and models.
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and the
  [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) — motion data.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## License

The code is released under the [MIT License](LICENSE). Third-party code, the bundled URDF and the external data are
covered by their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The motion data and the robot
USD are not redistributed.
