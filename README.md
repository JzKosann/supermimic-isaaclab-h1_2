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
>
> **Update (2026-11): v2 recipe.** `Tracking-Flat-H1_2-SuperMimic-v2` is now the default task. It keeps the v1
> observation and goal and adds input normalisation, a policy EMA and a set of training-stability and clip-end fixes
> (see [v2](#v2-recipe)). These changes come from our longer private runs (a 250 h BONES-SEED library, terrain
> experiments, and LAFAN1 fine-tuning for deployment on a real H1-2), where v1 diverged after ~10k updates and v2 did
> not. A from-scratch LAFAN1 run of v2 with this repository has not been published yet; v0 and v1 behave exactly as
> before (same evaluation results, bit for bit).

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
- **v2 training fixes**: running-mean input normalisation, policy EMA, paper-scale gradient clipping, a value cap,
  absorbing states and a survival-weighted critic, an unbiased reset-sampling estimate, training through the clip end,
  and a joint-target guard matching a deployment-side joint-limit guard.
- **Fine-tuning** from any checkpoint of the same task (`--init_from`), e.g. on a subset of clips.
- **Appendix-F benchmark**: every clip played once from its first frame, no noise and no domain randomisation;
  optional **hold mode** that keeps the last reference frame after each clip and measures falls and drift.
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
| `Tracking-Flat-H1_2-SuperMimic-v2` | same as v1 | same as v1 | **Default.** v1 plus normalisation, EMA and stability fixes (see [v2](#v2-recipe)). Same policy interface as v1. |
| `…-Eval-v0`, `…-Anchor-Eval-v0`, `…-Eval-v1`, `…-Eval-v2` | | | Appendix-F evaluation environments. `eval_benchmark.py` / `play.py` pick the right one from the checkpoint. |

All variants share the robot, the tracking rewards (Table A.2), terminations and domain randomisation (Table A.3).
v2 additionally penalises joint targets beyond its guard in the joint-limit term.

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
# v2 recipe (default task; LAFAN1, 4096 envs, imagination batch 4096, H = 32, UTD 0.4, 320k iterations)
python scripts/supermimic/train.py --headless --run_name lafan1_v2

# Paper configuration / v1 recipe
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v0 --run_name lafan1
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v1 --run_name lafan1_v1

# Smoke test on a small GPU (--wm_only_updates 20 so that policy updates start within the run)
python scripts/supermimic/train.py --headless --num_envs 64 --small_nets --policy_batch 256 --wm_batch 256 \
    --total_iterations 400 --wm_only_updates 20 --include walk1

# Resume (logs continue in the same directory)
python scripts/supermimic/train.py --headless --task <same task> --resume logs/supermimic/<run>/ckpt_latest.pt

# Fine-tune a checkpoint on a subset of clips (new run directory; optimizers and sampler start fresh,
# v1 / v2 first re-fit only the world model for 500 updates)
python scripts/supermimic/train.py --headless --task <same task> --init_from logs/supermimic/<run>/ckpt_latest.pt \
    --include walk run --total_iterations 20000 --run_name walk_run_ft
```

Logs go to `logs/supermimic/<date>_<run_name>/`: `config.json`, TensorBoard events in `tb/`, a checkpoint every
4000 iterations (`ckpt_XXXXXXX.pt`) and `ckpt_latest.pt`. Use `--logger wandb --log_project_name <name>` for
Weights & Biases. `--max_hours` stops training at a wall-clock budget.

Useful flags: `--policy_batch` (lower to 2048 if you run out of memory), `--wm_batch`, `--horizon`,
`--replay_capacity`, `--wm_only_updates`, `--motion_dir`, `--include <substring> …`.

Multi-GPU data parallelism: `python -m torch.distributed.run --nproc_per_node=N scripts/supermimic/train.py --headless --distributed`.
v0 / v1 scale the learning rate by √N (as in the paper); v2 does not. Multi-GPU training with this repository has not been tested;
the same code path ran on 2–4 A100s in our private runs.

Training diagnostics worth watching (TensorBoard / wandb): `policy/critic_over_bound_frac` (share of critic values above
r<sub>max</sub>/(1−γ) = 10, which no true value can exceed; a rising value preceded every divergence we saw),
`policy/*_peak` (largest action / state / gradient within a log interval), `sampler/last_bin_share` (share of resets in
the last second of a clip; should stay near #clips / #bins) and `policy/nonfinite_skips`.

### 2. Evaluate (Appendix F)

```bash
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>/ckpt_0040000.pt
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>   # every checkpoint in the run
```

Each clip is played once from its first frame. **Normalised return** = Σ per-step reward / (0.1 × steps), where 1.0
is perfect tracking; a clip **fails** if a training termination (root height / tilt) fires. Per-clip results go to
`<run>/eval/<ckpt>.json` and one line per checkpoint is appended to `<run>/eval/summary.csv`.
Without `--headless` the evaluation is rendered with the reference ghost.

Options:

- `--ema` evaluates the EMA policy stored in v2 checkpoints (usually better than the raw policy; use it for v2).
- `--hold_s 8` keeps the last reference frame for 8 s after each clip, as a robot does when the reference simply stops.
  It additionally reports falls during the hold, the horizontal pelvis drift at the clip end, and how far the robot
  moves while it should stand still. `failed` then counts only falls inside the clip.
- `--only 0040000,latest`, `--every N`, `--skip_done` select checkpoints of a run directory; `--tag` adds a suffix to the
  result names.

### 3. Consistency check

Checks that the analytic heads used in imagination match Isaac Lab term by term (FK of all 28 links, the policy
observation, all reward terms, the goal window). Run it after any change to the robot, observations or rewards:

```bash
python scripts/supermimic/check_consistency.py --headless --num_envs 32 --steps 300 [--task <task>]
```

Expected (measured with v0, v1 and v2): FK position errors < 1e-5 m, velocity errors < 1e-4, observation errors ~1e-6, reward-term errors ~1e-7.
For v2 the check also compares the clamped action executed in Isaac Lab with the one used in imagination (expected: 0).

### 4. Play

```bash
python scripts/supermimic/play.py --ckpt logs/supermimic/<run>                  # latest checkpoint of the run
python scripts/supermimic/play.py --ckpt <ckpt.pt> --num_envs 4 --include walk --real_time
python scripts/supermimic/play.py --ckpt <v2 ckpt.pt> --ema                     # the EMA policy of a v2 checkpoint
```

The blue translucent robot is the reference motion. The camera follows one robot but can be freely orbited and
zoomed with the mouse. Keys: `[` / `]` switch the followed env, `L` toggles following.

## Docker

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
cd docker
docker compose build
docker compose run --rm prepare-data                    # download + convert (once)
GPU=0 RUN_NAME=lafan1_v2 docker compose up -d train            # TASK defaults to Tracking-Flat-H1_2-SuperMimic-v2
GPU=0 CKPT=logs/supermimic/<run> EXTRA_ARGS="--ema" docker compose run --rm eval
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
        ├── commands.py             multi-clip motion command, return-based reset sampling, clip-end freeze / hold
        ├── bin_estimator.py        per-bin return estimates for reset sampling ("ema" v0/v1, "count" v2)
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
| `-v2` | v1 + the changes listed in [v2 recipe](#v2-recipe). The paper specifies no target critic and no value bound; v2 still has no target critic but caps the value, and uses the paper's clipping thresholds on the paper's loss scale. |

### v2 recipe

v2 keeps the v1 observation and goal, so v1 and v2 policies have the same interface. Every item is a switch whose
default is off (v0 / v1 unchanged); `trainer.apply_recipe("v2")` and `H1_2SuperMimicV2EnvCfg` turn them on.

| Change | What it does | Why |
|---|---|---|
| Input normalisation | Policy observations / goals and all critic inputs go through a running mean / variance (frozen after 2·10<sup>8</sup> samples), as in Uni-SuperMimic. | Without it the critic sees raw metres and radians. Our v1 run on a 250 h library diverged after ~10k updates; the same data trained 50k iterations without divergence after this change (together with the next two rows, so not a single-factor ablation). |
| Learning rate, EMA | 4·10<sup>-4</sup> for all three networks, no √#GPUs scaling, cosine decay over 625k iterations; policy EMA 0.995, stored as `policy_ema` and used for evaluation (`--ema`). | Uni-SuperMimic's settings. The EMA policy was better than the raw one on 94 % of held-out clips in our runs. |
| Paper-scale gradient clipping | The loss is −J/H, but the paper's clipping (global norm 1.0, element-wise 0.1) is meant for −J; both thresholds are divided by H. | With −J/H the clip almost never acts. |
| Value cap | V is clamped to r<sub>max</sub>/(1−γ) = 10 in the bootstrap targets and in V<sub>H</sub> of J (as in realmimic). | Each step earns at most 0.1, so no true value exceeds 10. The critic is bootstrapped from itself without a target network and J puts ~70 % of its weight on γ<sup>H</sup>V<sub>H</sub>, so an over-estimated value is chased by the policy and grows further; in our divergent terrain runs 55–61 % of the logged mean values were above 10. The three runs with the cap stayed stable to the end (14k iterations); the four without it all diverged. |
| Absorbing state, survival-weighted critic, non-finite skip | Once the survival weight drops below 10<sup>-3</sup>, the imagined trajectory is frozen where it died (state detached, reward 0); the critic loss is weighted by sg(c<sub>t</sub>); policy steps / critic minibatches with NaN / Inf gradients are skipped; the executed action is clipped to ±20 in the simulator. | Imagined trajectories that had already died were extrapolated by the world model far outside its data: actions above 100 and single-update critic losses up to 10<sup>23</sup>. |
| Unbiased reset sampling (`bin_estimator="count"`) | Episodes that end at the clip end or by time-out get a continuation value from their last steps; unvisited bins count as worst; a cumulative mean for the first visits. | With the original estimate the return of the last second of every clip is cut off, so it always looks hardest: in our large libraries 88–97 % of the "hard" part of the reset distribution fell into last seconds, and rare clips kept looking easy. The same bias is reported by realmimic. |
| Training through the clip end (`freeze_clip_end`) | Past the clip end the reference stops with zero velocity: goal frames past the end get zero velocity features, and imagination may start up to the last frame and roll past it. | Otherwise the clip end is never trained. In our runs about half of the new failures of a long fine-tune were in the last 3 frames of a clip, and policies fell right after a clip ended in sim-to-sim deployment tests (reproducible here with `--hold_s`). |
| Joint-target guard (`guard_margin=0.02`) | Joint targets are clamped to the URDF limits − 0.02 rad, after the execution noise (so the world model learns the clamped action) and in imagination; the overshoot of the policy's targets beyond the guard is added to the joint-limit penalty. | A deployment-side guard clamps the targets anyway. A policy trained without it had its targets (mostly ankle roll) at the guard in 25–77 % of the steps of a sim-to-sim deployment test; after fine-tuning with the guard, 1.8 % on a test clip. |

The joint-target guard changes the policy's action space slightly relative to v1. A v1 checkpoint can be evaluated in
`-Eval-v2` (same interface) but it was not trained with the guard.

## Results

Coming soon. For reference, the paper reports a normalised return of 0.82 on LAFAN1 after 1.2 h on one H100
(G1); see the two G1 reproductions below for their numbers.

## Acknowledgements

- *SuperMimic* ([arXiv:2610.28480](https://arxiv.org/abs/2610.28480)) — the method.
- [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) (Yuxi Liu, MIT) — the mjlab / G1
  reproduction this code was ported from module by module.
- [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Chong Zhang, MIT) — the source of the v1 recipe.
- [Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) (Apache-2.0) — input normalisation, learning rate and
  policy EMA of the v2 recipe.
- [realmimic](https://github.com/kevinhenryzz/realmimic) (Kevin Henry, Apache-2.0) — the unbiased reset-sampling
  estimate, the clip-end freeze and the value cap of the v2 recipe.
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
