"""SuperMimic checkpoint evaluation (paper Appendix F): every clip is played once, deterministically, from its first frame; no noise, no domain randomisation.

Usage (from the repository root):
    python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>/ckpt_0024000.pt
    python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>      # every ckpt_XXXXXXX.pt in the directory
Results go to <run>/eval/<ckpt>.json (per clip), and one line is appended to <run>/eval/summary.csv.
The evaluation env is chosen from the training task stored in the checkpoint (X-vN -> X-Eval-vN); early v0 checkpoints without a task get
-Eval-v0 / -Anchor-Eval-v0 depending on whether they use anchor observations. Paper Fig. 4a: 0.82 on LAFAN1 after 1.2 h.
Without --headless the view shows the translucent blue reference ghost and the camera can be orbited freely (follow camera: mdp/viewer.py); use play.py to only watch.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SuperMimic Appendix-F benchmark.")
parser.add_argument("--ckpt", type=str, required=True, help="checkpoint file, or run directory (evaluates every ckpt_XXXXXXX.pt in it)")
parser.add_argument("--num_envs", type=int, default=40, help="number of clips played in parallel (LAFAN1 has 40)")
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--include", type=str, nargs="*", default=None)
parser.add_argument("--every", type=int, default=1, help="directory mode: evaluate every n-th checkpoint")
parser.add_argument("--skip_done", action="store_true", help="directory mode: skip checkpoints that already have an eval json")
parser.add_argument("--follow_env", type=int, default=0, help="with a GUI: env the camera follows first")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pathlib import Path

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import supermimic_h1_2.tasks  # noqa: F401  (registers the gym tasks)
from supermimic_h1_2.mdp.benchmark import (
    eval_task_for,
    run_benchmark,
    save_results,
    summarize,
)
from supermimic_h1_2.mdp.policy import Policy, PolicyCfg
from supermimic_h1_2.mdp.viewer import FollowCamera


def main():
    target = Path(args_cli.ckpt)
    ckpts = sorted(target.glob("ckpt_[0-9]*.pt"))[:: args_cli.every] if target.is_dir() else [target]
    run_dir = target if target.is_dir() else target.parent
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    if args_cli.skip_done:
        ckpts = [c for c in ckpts if not (out_dir / f"{c.stem}.json").exists()]
    if not ckpts:
        print("[INFO] nothing to evaluate")
        return
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    task = eval_task_for(torch.load(ckpts[0], map_location="cpu", weights_only=False))
    print(f"[INFO] eval env {task}")
    env_cfg = parse_env_cfg(task, device=device, num_envs=args_cli.num_envs)
    if args_cli.motion_dir:
        env_cfg.commands.motion.motion_dir = args_cli.motion_dir
    if args_cli.include:
        env_cfg.commands.motion.include = args_cli.include
    if not args_cli.headless:
        env_cfg.enable_play_viz()
    env = gym.make(task, cfg=env_cfg).unwrapped
    cam = None if args_cli.headless else FollowCamera(env, args_cli.follow_env)

    summary_csv = out_dir / "summary.csv"
    if not summary_csv.exists():
        summary_csv.write_text("ckpt,iteration,clips,failed,mean_norm_return,step_weighted_norm_return\n")
    for path in ckpts:
        ck = torch.load(path, map_location=device, weights_only=False)
        policy = Policy(PolicyCfg(**ck["config"]["policy"])).to(device)
        policy.load_state_dict(ck["policy"])
        policy.eval()
        print(f"[INFO] {path.name} (iteration {ck['iteration']})", flush=True)
        results = run_benchmark(env, policy, on_step=None if cam is None else cam.update)
        s = summarize(results)
        save_results(results, out_dir / f"{path.stem}.json")
        with summary_csv.open("a") as f:
            f.write(f"{path.name},{ck['iteration']},{s['clips']},{s['failed']},{s['mean_norm_return']:.4f},{s['step_weighted_norm_return']:.4f}\n")
        terms = " ".join(f"{k}={v:.4f}" for k, v in s["terms"].items())
        print(
            f"  clips {s['clips']} | failed {s['failed']} | mean {s['mean_norm_return']:.4f} | "
            f"step-weighted {s['step_weighted_norm_return']:.4f}\n  terms/step: {terms}",
            flush=True,
        )
        for r in results:
            if r.failed:
                print(f"    FAIL {r.name}: step {r.fail_step}/{r.frames}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
