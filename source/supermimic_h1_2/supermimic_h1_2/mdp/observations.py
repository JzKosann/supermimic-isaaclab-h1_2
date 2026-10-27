"""Observation terms (paper Table A.1 + goal g_t).

policy group (10-frame history, flattened term-major, same layout as heads.flatten_history_term_major on the imagination side):
    joint_pos (with encoder bias) - q0, joint_vel, base_ang_vel, projected_gravity, last_action (clean action)
The last four use the functions of the same name in isaaclab.envs.mdp. v1 appends ref_root_ori (6D reference root orientation relative to the robot root, also with a 10-frame history).
goal group (no history): K=10 future reference frames; optionally + a 9-d BeyondMimic-style anchor-relative pose (deviates from the paper, needs state estimation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .events import get_sim_context

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_pos_biased_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos + get_sim_context(env, asset_cfg.name).encoder_bias - robot.data.default_joint_pos


def goal_future_frames(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).goal


def ref_root_ori(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """v1: 6D reference root orientation relative to the robot root (pelvis version of BeyondMimic's motion_anchor_ori_b; needs only the IMU orientation)."""
    return env.command_manager.get_term(command_name).ref_root_ori_obs()


def goal_anchor_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).anchor_relative_obs()[:, :3]


def goal_anchor_ori(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).anchor_relative_obs()[:, 3:]
