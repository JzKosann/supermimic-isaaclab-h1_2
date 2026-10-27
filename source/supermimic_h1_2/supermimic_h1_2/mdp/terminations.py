"""Paper terminations: root (pelvis) height deviates from the reference by > 0.25 m, or tilt difference > 0.8; reaching the clip end is a time-out (not a failure).
No termination on end-effector deviation (unlike BeyondMimic)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .rotations import projected_gravity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_termination_features(robot_root_pos, robot_root_quat, ref_root_pos, ref_root_quat) -> torch.Tensor:
    """[height error, tilt error] (N, 2); tilt = difference of the gravity z-component in the reference root frame and the robot root frame. The imagination-side termination head uses the same function."""
    tilt = projected_gravity(ref_root_quat)[..., 2] - projected_gravity(robot_root_quat)[..., 2]
    height = ref_root_pos[..., 2] - robot_root_pos[..., 2]
    return torch.stack([height, tilt], dim=-1)


def _features(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    return root_termination_features(cmd.robot_root_pos_w, cmd.robot_root_quat_w, cmd.ref_root_pos_w, cmd.ref_root_quat_w)


def bad_root_height(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    return _features(env, command_name)[:, 0].abs() > threshold


def bad_root_tilt(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    return _features(env, command_name)[:, 1].abs() > threshold


def clip_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).clip_end.clone()
