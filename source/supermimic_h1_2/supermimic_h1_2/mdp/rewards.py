"""Motion-tracking rewards of paper Table A.2: the simulator side (reward manager) and the imagination side (analytic reward head of the world model) share this math.

- Body positions / orientations: reference bodies are first "re-anchored" to the robot anchor (torso_link), aligning only the xy translation and yaw (z, roll, pitch keep the reference values),
  the same definition as BeyondMimic's update_relative_body_poses. Difference: BeyondMimic re-anchors with the previous step's robot
  anchor when the command updates (the reward lags one step); here the current robot anchor is used when the reward is computed, exactly as in imagination from the predicted state.
- Root position / orientation: the anchor (torso_link) is compared in the world frame; body linear / angular velocities are compared in the world frame (link-origin velocities).
- No self-collision term; joint limit -2 (BeyondMimic: -10).
All terms are multiplied by dt = 0.02 s (the Isaac reward manager multiplies weight * dt itself; imagination does it in weighted_terms); at most 0.1 per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .rotations import quat_apply, quat_box_minus, quat_conjugate, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass(frozen=True)
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori)


REWARD = TrackingRewardCfg()
REWARD_TERM_NAMES = (
    "root_pos",
    "root_ori",
    "body_pos",
    "body_ori",
    "body_lin_vel",
    "body_ang_vel",
    "action_rate",
    "joint_limit",
)


def reanchor_reference(ref_pos, ref_quat, ref_anchor_pos, ref_anchor_quat, robot_anchor_pos, robot_anchor_quat):
    """Transforms the reference bodies (N,B,.) into the frame that shares xy and yaw with the robot anchor."""
    nb = ref_pos.shape[-2]
    delta_pos = torch.cat([robot_anchor_pos[..., :2], ref_anchor_pos[..., 2:3]], dim=-1)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_conjugate(ref_anchor_quat)))
    delta_ori_b = delta_ori.unsqueeze(-2).expand(delta_ori.shape[:-1] + (nb, 4))
    rel_quat = quat_mul(delta_ori_b, ref_quat)
    rel_pos = delta_pos.unsqueeze(-2) + quat_apply(delta_ori_b, ref_pos - ref_anchor_pos.unsqueeze(-2))
    return rel_pos, rel_quat


def _sq(v: torch.Tensor) -> torch.Tensor:
    return (v * v).sum(-1)


def tracking_sq_errors(
    robot_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    robot_lin_vel: torch.Tensor,
    robot_ang_vel: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_quat: torch.Tensor,
    ref_lin_vel: torch.Tensor,
    ref_ang_vel: torch.Tensor,
    anchor_index: int,
) -> dict[str, torch.Tensor]:
    """Squared errors of the six tracking terms (N,); inputs are world-frame link quantities of the tracked bodies (N,B,.)."""
    ra_pos, ra_quat = robot_pos[..., anchor_index, :], robot_quat[..., anchor_index, :]
    ma_pos, ma_quat = ref_pos[..., anchor_index, :], ref_quat[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_pos, ref_quat, ma_pos, ma_quat, ra_pos, ra_quat)
    return {
        "root_pos": _sq(ma_pos - ra_pos),
        "root_ori": _sq(quat_box_minus(ma_quat, ra_quat)),
        "body_pos": _sq(rel_pos - robot_pos).mean(-1),
        "body_ori": _sq(quat_box_minus(rel_quat, robot_quat)).mean(-1),
        "body_lin_vel": _sq(ref_lin_vel - robot_lin_vel).mean(-1),
        "body_ang_vel": _sq(ref_ang_vel - robot_ang_vel).mean(-1),
    }


def joint_limit_violation(joint_pos: torch.Tensor, soft_limits: torch.Tensor) -> torch.Tensor:
    """Same as isaaclab mdp.joint_pos_limits: sum of the violation beyond the soft limits. soft_limits (...,NJ,2)."""
    return ((soft_limits[..., 0] - joint_pos).clamp(min=0.0) + (joint_pos - soft_limits[..., 1]).clamp(min=0.0)).sum(-1)


def weighted_terms(
    errors: dict[str, torch.Tensor],
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,
    cfg: TrackingRewardCfg = REWARD,
) -> dict[str, torch.Tensor]:
    """For imagination: terms (N,) already multiplied by weight and dt, plus 'total'; identical term by term to the Isaac reward manager."""
    dt = cfg.dt
    t = {
        "root_pos": dt * cfg.w_root_pos * torch.exp(-errors["root_pos"] / cfg.std_root_pos**2),
        "root_ori": dt * cfg.w_root_ori * torch.exp(-errors["root_ori"] / cfg.std_root_ori**2),
        "body_pos": dt * cfg.w_body_pos * torch.exp(-errors["body_pos"] / cfg.std_body_pos**2),
        "body_ori": dt * cfg.w_body_ori * torch.exp(-errors["body_ori"] / cfg.std_body_ori**2),
        "body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-errors["body_lin_vel"] / cfg.std_body_lin_vel**2),
        "body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-errors["body_ang_vel"] / cfg.std_body_ang_vel**2),
        "action_rate": dt * cfg.w_action_rate * _sq(action - prev_action),
        "joint_limit": dt * cfg.w_joint_limit * joint_limit_violation(joint_pos, soft_limits),
    }
    total = t["root_pos"]
    for k in REWARD_TERM_NAMES[1:]:
        total = total + t[k]
    t["total"] = total
    return t


# ---------------------------------------------------------------------- simulator-side reward terms (Isaac multiplies weight * dt itself)


def _kernel(env: ManagerBasedRLEnv, command_name: str, key: str, std: float) -> torch.Tensor:
    return torch.exp(-env.command_manager.get_term(command_name).tracking_errors()[key] / std**2)


def motion_root_pos(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return _kernel(env, command_name, "root_pos", std)


def motion_root_ori(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return _kernel(env, command_name, "root_ori", std)


def motion_body_pos(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return _kernel(env, command_name, "body_pos", std)


def motion_body_ori(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return _kernel(env, command_name, "body_ori", std)


def motion_body_lin_vel(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return _kernel(env, command_name, "body_lin_vel", std)


def motion_body_ang_vel(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return _kernel(env, command_name, "body_ang_vel", std)
