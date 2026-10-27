"""Consistency check: do the analytic heads used in imagination match the Isaac simulation term by term (the gate of SuperMimic-Reproduced plan D4; rerun after any environment change).

Runs random actions for a number of steps in the training environment (with domain randomisation) and compares, for every env that did not reset:
    1. FK(state) vs Isaac body_link_pos/quat/lin_vel/ang_vel (all 28 links)
    2. analytic observation analytic_observation(s, a, xi) vs the newest frame of the policy observation group (observation noise off)
    3. analytic reward heads.reward(s', a, a_prev, ref) vs the Isaac reward manager, term by term
    4. goal window vs the goal observation group
Usage: python scripts/supermimic/check_consistency.py --headless --num_envs 32 --steps 300
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SuperMimic analytic heads vs Isaac.")
parser.add_argument("--task", type=str, default="Tracking-Flat-H1_2-SuperMimic-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--action_std", type=float, default=0.5)
parser.add_argument("--include", type=str, nargs="*", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import supermimic_h1_2.tasks  # noqa: F401  (registers the gym tasks)
from supermimic_h1_2.mdp.events import get_sim_context
from supermimic_h1_2.mdp.heads import (
    AnalyticHeads,
    analytic_observation,
    anchor_relative_obs,
    obs_term_dims,
    reference_at,
    root_ori_relative_obs,
)
from supermimic_h1_2.mdp.rewards import REWARD_TERM_NAMES
from supermimic_h1_2.mdp.rotations import quat_apply, quat_error_magnitude
from supermimic_h1_2.mdp.state import StateIO

torch.backends.cuda.matmul.allow_tf32 = False


def newest_frame(obs: torch.Tensor, term_dims, history: int) -> torch.Tensor:
    out, start = [], 0
    for d in term_dims:
        out.append(obs[:, start + (history - 1) * d : start + history * d])
        start += history * d
    return torch.cat(out, dim=-1)


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.goal.enable_corruption = False
    if args_cli.include:
        env_cfg.commands.motion.include = args_cli.include
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    nj = robot.num_joints
    heads = AnalyticHeads(
        cmd.fk, cmd.body_names, cmd.cfg.anchor_body_name,
        robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(),
    )  # fmt: skip
    state_io = StateIO(env)
    ctx = get_sim_context(env)
    all_links = cmd.fk.link_indices(robot.body_names)
    ref_ori = getattr(env_cfg.observations.policy, "ref_root_ori", None) is not None  # v1
    term_dims = obs_term_dims(nj, ref_ori)
    history = env_cfg.observations.policy.history_length

    worst: dict[str, float] = {}

    def upd(key: str, v: torch.Tensor):
        if v.numel():
            worst[key] = max(worst.get(key, 0.0), float(v.max()))

    obs, _ = env.reset()
    n_checked = 0
    t_step = 0.0
    for _ in range(args_cli.steps):
        frame_t = cmd.frame.clone()
        a_prev = env.action_manager.action.clone()
        a = args_cli.action_std * torch.randn(env.num_envs, nj, device=device)
        t0 = time.perf_counter()
        obs, rew, term, tout, _ = env.step(a)
        torch.cuda.synchronize()
        t_step += time.perf_counter() - t0
        keep = ~(term | tout)
        if not bool(keep.any()):
            continue
        s = state_io.read()[keep]
        d = robot.data
        # 1. FK
        p, q, qj = s[:, 0:3], s[:, 3:7], s[:, 7 : 7 + nj]
        out = cmd.fk(p, q, qj, s[:, 7 + nj : 10 + nj], quat_apply(q, s[:, 10 + nj : 13 + nj]), s[:, 13 + nj :])
        origins = env.scene.env_origins[keep][:, None, :]
        upd("fk_pos [m]", (out["pos"][:, all_links] + origins - d.body_link_pos_w[keep]).norm(dim=-1))
        upd("fk_quat [rad]", quat_error_magnitude(out["quat"][:, all_links], d.body_link_quat_w[keep]))
        upd("fk_lin_vel [m/s]", (out["lin_vel"][:, all_links] - d.body_link_lin_vel_w[keep]).norm(dim=-1))
        upd("fk_ang_vel [rad/s]", (out["ang_vel"][:, all_links] - d.body_link_ang_vel_w[keep]).norm(dim=-1))
        # 2. observation
        o_ana = analytic_observation(s, a[keep], ctx.read()[keep], robot.data.default_joint_pos[0], nj)
        if ref_ori:  # the target frame at observation time = the updated cmd.frame
            o_ana = torch.cat([o_ana, root_ori_relative_obs(s[:, 3:7], lib.ref(cmd.frame[keep]).root_quat_w)], dim=-1)
        upd("obs", (o_ana - newest_frame(obs["policy"][keep], term_dims, history)).abs())
        # 3. rewards
        terms = heads.reward(s, a[keep], a_prev[keep], reference_at(lib, frame_t[keep]))
        env_terms = env.reward_manager._step_reward[keep] * env.step_dt
        for i, k in enumerate(REWARD_TERM_NAMES):
            upd(f"reward/{k}", (terms[k] - env_terms[:, i]).abs())
        upd("reward/total", (terms["total"] - rew[keep]).abs())
        # 4. goal (-Anchor variants also compare the anchor-relative pose: robot anchor from FK in imagination + reference anchor of the current target frame)
        upd("goal", (lib.goal_window(cmd.frame[keep], cmd.cfg.future_frames) - obs["goal"][keep][:, : cmd.cfg.future_frames * lib.goal_dim]).abs())
        if env.cfg.anchor_obs:
            ai = heads.anchor_index
            b = heads.robot_bodies(s)
            ref = reference_at(lib, cmd.frame[keep])
            rel = anchor_relative_obs(b["pos"][:, ai], b["quat"][:, ai], ref.body_pos_w[:, ai], ref.body_quat_w[:, ai])
            upd("goal/anchor", (rel - obs["goal"][keep][:, -9:]).abs())
        n_checked += int(keep.sum())

    print(f"\n[check] {n_checked} env-steps compared, {args_cli.steps} steps, {t_step / args_cli.steps * 1000:.1f} ms/step (envs {env.num_envs})")
    print(f"[check] reward per-step scale: max 0.1; xi friction range seen {float(ctx.friction.min()):.2f}..{float(ctx.friction.max()):.2f}")
    for k, v in worst.items():
        print(f"  max |diff| {k:24s} {v:.3e}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
