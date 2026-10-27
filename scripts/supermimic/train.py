"""SuperMimic training (world model + BPTT, no rsl_rl). Environments: tasks/h1_2_env_cfg.py; algorithm: mdp/.

Usage (from the repository root):
    single GPU (paper Fig. 4a configuration: 40 LAFAN1 clips, 4096 envs, imagination batch 4096, H=32, UTD 0.4, 320k iterations):
        python scripts/supermimic/train.py --headless --run_name lafan1
    with anchor-relative pose observation (experiment e3 of SuperMimic-Reproduced, deviates from the paper): --task Tracking-Flat-H1_2-SuperMimic-Anchor-v0
    v1 (v0 + the improvements of supermimic_vibed, see tasks/h1_2_env_cfg.py / mdp/trainer.py): --task Tracking-Flat-H1_2-SuperMimic-v1
    multi-GPU (one Isaac instance + 4096 envs per GPU, gradient all-reduce, learning rate x sqrt(#GPUs); untested):
        python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 scripts/supermimic/train.py --headless --distributed
    resume: --resume logs/supermimic/<run>/ckpt_latest.pt (logs continue in the same directory)
    smoke test (8 GB GPU): --num_envs 64 --small_nets --policy_batch 256 --wm_batch 256 --total_iterations 600 --include walk1
Logs in logs/supermimic/<date>_<run_name>/: config.json, tb/, ckpt_XXXXXXX.pt (every 4000 iterations), ckpt_latest.pt.
Evaluation: scripts/supermimic/eval_benchmark.py --ckpt <checkpoint or run directory>.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SuperMimic (world model + BPTT) on H1-2.")
parser.add_argument("--task", type=str, default="Tracking-Flat-H1_2-SuperMimic-v0")
parser.add_argument("--num_envs", type=int, default=4096, help="envs per GPU (paper: 4096)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run_name", type=str, default="lafan1")
parser.add_argument("--motion_dir", type=str, default=None, help="override the motion library directory (default: datasets/lafan1_h1_2_npz)")
parser.add_argument("--include", type=str, nargs="*", default=None, help="only use clips whose file name contains one of these substrings")
parser.add_argument("--total_iterations", type=int, default=None, help="number of simulator steps (paper: 320000)")
parser.add_argument("--max_hours", type=float, default=None, help="stop after this many hours (saves ckpt_latest)")
parser.add_argument("--policy_batch", type=int, default=None, help="imagination rollout batch size (paper: 4096; use 2048 if out of memory)")
parser.add_argument("--wm_batch", type=int, default=None, help="world-model windows per batch (paper: 2048)")
parser.add_argument("--horizon", type=int, default=None, help="imagination horizon H (paper: 32)")
parser.add_argument("--replay_capacity", type=int, default=None, help="replay steps per env (paper: 1000)")
parser.add_argument("--compile", type=str, default="default", help="torch.compile mode; none disables it")
parser.add_argument("--small_nets", action="store_true", help="small networks, for smoke tests only")
parser.add_argument("--checkpoint_interval", type=int, default=None)
parser.add_argument("--log_interval", type=int, default=None)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
parser.add_argument("--log_project_name", type=str, default="supermimic-h1_2")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import supermimic_h1_2.tasks  # noqa: F401  (registers the gym tasks)
from supermimic_h1_2.mdp.distributed import finalize, init_distributed
from supermimic_h1_2.paths import LOG_ROOT
from supermimic_h1_2.mdp.trainer import (
    Logger,
    TrainCfg,
    Trainer,
    apply_recipe,
    save_config,
)

torch.backends.cuda.matmul.allow_tf32 = True  # paper: the world model runs in TF32
torch.backends.cudnn.allow_tf32 = True

def make_cfg() -> TrainCfg:
    cfg = TrainCfg(seed=args_cli.seed)
    if args_cli.small_nets:
        cfg.world_model.hidden = (256, 256)
        cfg.policy.encoder_hidden = (256, 128)
        cfg.policy.decoder_hidden = (256, 128)
    for value, obj, attr in (
        (args_cli.total_iterations, cfg.schedule, "total_iterations"),
        (args_cli.max_hours, cfg.schedule, "max_hours"),
        (args_cli.checkpoint_interval, cfg.schedule, "checkpoint_interval"),
        (args_cli.log_interval, cfg.schedule, "log_interval"),
        (args_cli.policy_batch, cfg.policy_trainer, "batch_size"),
        (args_cli.wm_batch, cfg.wm_trainer, "batch_size"),
        (args_cli.horizon, cfg.policy_trainer, "horizon"),
        (args_cli.replay_capacity, cfg, "replay_capacity"),
    ):
        if value is not None:
            setattr(obj, attr, value)
    cfg.wm_trainer.compile = args_cli.compile
    cfg.policy_trainer.compile = args_cli.compile
    return cfg


def main():
    local_rank = app_launcher.local_rank if args_cli.distributed else 0
    rank, world = init_distributed(local_rank) if args_cli.distributed else (0, 1)
    device = f"cuda:{local_rank}"
    cfg = make_cfg()
    torch.manual_seed(cfg.seed + 1000 * rank)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = cfg.seed + 1000 * rank
    apply_recipe(cfg, env_cfg.recipe)
    if args_cli.motion_dir:
        env_cfg.commands.motion.motion_dir = args_cli.motion_dir
    if args_cli.include:
        env_cfg.commands.motion.include = args_cli.include
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if args_cli.resume:
        log_dir = Path(args_cli.resume).resolve().parent
    else:
        log_dir = LOG_ROOT / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}"
    logger = None
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        extra = {"task": args_cli.task, "num_envs_per_rank": args_cli.num_envs, "world_size": world, "argv": vars(args_cli)}
        logger = Logger(log_dir, args_cli.logger == "wandb", args_cli.log_project_name, log_dir.name, extra)
        print(f"[INFO] logging to {log_dir}")

    trainer = Trainer(env, cfg, log_dir, rank, logger, task=args_cli.task)
    if rank == 0:
        save_config(cfg, log_dir, {"task": args_cli.task, "recipe": env.cfg.recipe, "anchor_obs": bool(env.cfg.anchor_obs), "world_size": world})
    if args_cli.resume:
        trainer.load(args_cli.resume)
    try:
        trainer.run()
    finally:
        if logger is not None:
            logger.close()
        env.close()
        finalize()


if __name__ == "__main__":
    main()
    simulation_app.close()
