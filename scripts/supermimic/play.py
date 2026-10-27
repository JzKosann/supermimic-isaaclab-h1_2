"""Watch a SuperMimic checkpoint with a GUI: the evaluation environment (no noise, no domain randomisation, every clip played from its first frame) loops over random clips.
View: the translucent blue ghost is the reference motion; the camera can be orbited freely and only translates to follow one env ([ / ] switch env, L toggles following).
SuperMimic does not use rsl_rl (different policy class), so Isaac Lab's rsl_rl play.py cannot play it.

Usage (from the repository root):
    python scripts/supermimic/play.py --ckpt logs/supermimic/<run>/ckpt_0040000.pt
    python scripts/supermimic/play.py --ckpt logs/supermimic/<run> --num_envs 4 --include walk    # directory = latest checkpoint
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a SuperMimic checkpoint.")
parser.add_argument("--ckpt", "--checkpoint", type=str, required=True, help="checkpoint file, or run directory (uses the latest)")
parser.add_argument("--num_envs", type=int, default=None, help="number of envs played at once (default: 40)")
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--include", type=str, nargs="*", default=None, help="only play clips whose file name contains one of these substrings")
parser.add_argument("--follow_env", type=int, default=0, help="env the camera follows first")
parser.add_argument("--real_time", "--real-time", action="store_true", help="play at real-time speed (50 Hz)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import supermimic_h1_2.tasks  # noqa: F401  (registers the gym tasks)
from supermimic_h1_2.mdp.benchmark import eval_task_for
from supermimic_h1_2.mdp.policy import Policy, PolicyCfg
from supermimic_h1_2.mdp.viewer import FollowCamera


def main():
    path = Path(args_cli.ckpt)
    if path.is_dir():
        path = path / "ckpt_latest.pt" if (path / "ckpt_latest.pt").exists() else sorted(path.glob("ckpt_[0-9]*.pt"))[-1]
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    ck = torch.load(path, map_location=device, weights_only=False)
    task = eval_task_for(ck)
    print(f"[INFO] {path} (iteration {ck['iteration']}) | env {task}")
    env_cfg = parse_env_cfg(task, device=device, num_envs=args_cli.num_envs)
    if args_cli.motion_dir:
        env_cfg.commands.motion.motion_dir = args_cli.motion_dir
    if args_cli.include:
        env_cfg.commands.motion.include = args_cli.include
    env_cfg.enable_play_viz()
    env = gym.make(task, cfg=env_cfg).unwrapped
    policy = Policy(PolicyCfg(**ck["config"]["policy"])).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    obs, _ = env.reset()
    cam = FollowCamera(env, args_cli.follow_env)
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            obs, _, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
        i = cam.index
        if bool(term[i] | tout[i]):
            print(f"[play] env {i}: {'terminated (deviation / fall)' if bool(term[i]) else 'clip finished'}", flush=True)
        cam.update()
        if args_cli.real_time:
            time.sleep(max(env.step_dt - (time.time() - t0), 0.0))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
