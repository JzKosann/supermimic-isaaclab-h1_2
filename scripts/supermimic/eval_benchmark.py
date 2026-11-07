"""SuperMimic checkpoint evaluation (paper Appendix F): every clip is played once, deterministically, from its first frame; no noise, no domain randomisation.

Usage (from the repository root):
    python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>/ckpt_0024000.pt
    python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>      # every ckpt_XXXXXXX.pt in the directory
    python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run> --only 0040000,latest --ema --hold_s 8
Results go to <run>/eval/<ckpt><suffix>.json (per clip), and one line is appended to <run>/eval/summary.csv.
--ema evaluates the EMA policy stored by v2 checkpoints (suffix _ema); --hold_s holds the last reference frame for that many seconds
after each clip (suffix _hold<s>) and reports falls during the hold, the drift at the clip end and how far the robot moves while holding.
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
parser.add_argument("--only", type=str, default=None, help="directory mode: only these (comma-separated iterations or latest, e.g. 0040000,latest)")
parser.add_argument("--skip_done", action="store_true", help="directory mode: skip checkpoints that already have an eval json")
parser.add_argument("--ema", action="store_true", help="evaluate the EMA policy of a v2 checkpoint instead of the raw one")
parser.add_argument("--hold_s", type=float, default=0.0, help="hold the last reference frame this many seconds after each clip (0 = end at the clip end)")
parser.add_argument("--tag", type=str, default="", help="extra suffix of the result file names (e.g. when changing --motion_dir)")
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
    if target.is_dir() and args_cli.only:
        only = args_cli.only.split(",")
        ckpts = [c for c in ckpts if any(c.stem.endswith(o) for o in only if o != "latest")]
        if "latest" in only and (target / "ckpt_latest.pt").exists():
            ckpts.append(target / "ckpt_latest.pt")
    run_dir = target if target.is_dir() else target.parent
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    suffix = ("_ema" if args_cli.ema else "") + args_cli.tag + (f"_hold{args_cli.hold_s:g}" if args_cli.hold_s > 0 else "")
    if args_cli.skip_done:
        ckpts = [c for c in ckpts if not (out_dir / f"{c.stem}{suffix}.json").exists()]
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
    if args_cli.hold_s > 0:
        env_cfg.commands.motion.end_hold_steps = round(args_cli.hold_s / (env_cfg.sim.dt * env_cfg.decimation))
    if not args_cli.headless:
        env_cfg.enable_play_viz()
    env = gym.make(task, cfg=env_cfg).unwrapped
    cam = None if args_cli.headless else FollowCamera(env, args_cli.follow_env)

    summary_csv = out_dir / "summary.csv"
    if not summary_csv.exists():
        summary_csv.write_text("ckpt,iteration,clips,failed,mean_norm_return,step_weighted_norm_return\n")
    for path in ckpts:
        ck = torch.load(path, map_location=device, weights_only=False)
        if args_cli.ema and "policy_ema" not in ck:
            raise ValueError(f"{path} has no EMA policy (only v2 checkpoints store one); drop --ema")
        policy = Policy(PolicyCfg(**ck["config"]["policy"])).to(device)
        policy.load_state_dict(ck["policy_ema" if args_cli.ema else "policy"])
        policy.eval()
        print(f"[INFO] {path.name}{suffix} (iteration {ck['iteration']})", flush=True)
        results = run_benchmark(env, policy, on_step=None if cam is None else cam.update)
        s = summarize(results)
        save_results(results, out_dir / f"{path.stem}{suffix}.json")
        with summary_csv.open("a") as f:
            f.write(f"{path.name}{suffix},{ck['iteration']},{s['clips']},{s['failed']},{s['mean_norm_return']:.4f},{s['step_weighted_norm_return']:.4f}\n")
        terms = " ".join(f"{k}={v:.4f}" for k, v in s["terms"].items())
        print(
            f"  clips {s['clips']} | failed {s['failed']} | mean {s['mean_norm_return']:.4f} | "
            f"step-weighted {s['step_weighted_norm_return']:.4f}\n  terms/step: {terms}",
            flush=True,
        )
        if "hold" in s:
            h = s["hold"]
            print(
                f"  hold {args_cli.hold_s:g}s: clips ended {h['clips_ended']} | fell during hold {h['hold_failed']} | "
                f"drift at clip end median {h['end_err_xy_median']:.2f} / p90 {h['end_err_xy_p90']:.2f} / max {h['end_err_xy_max']:.2f} m | "
                f"moved during hold median {h['hold_drift_median']:.2f} / p90 {h['hold_drift_p90']:.2f} / max {h['hold_drift_max']:.2f} m "
                f"(> 0.5 m: {h['hold_drift_over_0p5']})",
                flush=True,
            )
            for r in results:
                if r.hold_failed:
                    print(f"    HOLD-FAIL {r.name}: {r.steps - r.frames} steps after the clip end")
        for r in results:
            if r.failed:
                print(f"    FAIL {r.name}: step {r.fail_step}/{r.frames}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
