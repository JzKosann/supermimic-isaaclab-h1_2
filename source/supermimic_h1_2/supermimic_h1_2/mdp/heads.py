"""Analytic heads of the world model (SuperMimic-Reproduced plan D4): observations, rewards and termination features, all differentiable functions of the predicted state.

The reward head computes the poses / velocities of the tracked bodies from the state through kinematics.UrdfKinematics, then calls the same formulas in rewards.py as the simulator;
the observation head returns the noise-free observation identical term by term to the policy observation group; termination features use the same function as terminations.py.
Root positions in the state are relative to the env origin and reference quantities exclude the env origin, consistently on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import UrdfKinematics
from .motion_library import MotionLibrary
from .rewards import REWARD, TrackingRewardCfg, reanchor_reference, tracking_sq_errors, weighted_terms
from .rotations import (
    heading_yaw,
    projected_gravity,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    rot6d_from_quat,
    wrap_to_pi,
    yaw_quat,
)
from .state import split_state
from .terminations import root_termination_features

ANCHOR_OBS_DIM = 9


def anchor_relative_obs(robot_anchor_pos, robot_anchor_quat, ref_anchor_pos, ref_anchor_quat) -> torch.Tensor:
    """[position 3, rotation 6D 6] of the reference anchor in the robot anchor frame (= BeyondMimic motion_anchor_pos_b / motion_anchor_ori_b)."""
    pos = quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
    rel = quat_mul(quat_conjugate(robot_anchor_quat), ref_anchor_quat)
    return torch.cat([pos, rot6d_from_quat(rel)], dim=-1)


def root_ori_relative_obs(robot_root_quat: torch.Tensor, ref_root_quat: torch.Tensor) -> torch.Tensor:
    """v1 observation: 6D of the reference root (pelvis) orientation in the robot root frame (R_robot^T R_ref). Needs only the IMU orientation, no position estimate."""
    return rot6d_from_quat(quat_mul(quat_conjugate(robot_root_quat), ref_root_quat))


def obs_term_dims(nj: int, ref_ori_obs: bool = False) -> tuple[int, ...]:
    """Dimensions of the policy observation terms, in the order of h1_2_env_cfg.PolicyCfg: joint_pos, joint_vel, base_ang_vel, projected_gravity,
    actions (v1 adds ref_root_ori 6)."""
    return (nj, nj, 3, 3, nj) + ((6,) if ref_ori_obs else ())


@dataclass
class ReferenceFrame:
    """Reference quantities of a batch of target frames (without env origin)."""

    body_pos_w: torch.Tensor  # (N,B,3)
    body_quat_w: torch.Tensor  # (N,B,4)
    body_lin_vel_w: torch.Tensor  # (N,B,3)
    body_ang_vel_w: torch.Tensor  # (N,B,3)
    root_pos_w: torch.Tensor  # (N,3)
    root_quat_w: torch.Tensor  # (N,4)
    joint_pos: torch.Tensor | None = None  # (N,NJ), used by the v1 critic features


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w, r.joint_pos)


def analytic_observation(s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, default_joint_pos: torch.Tensor, nj: int) -> torch.Tensor:
    """Noise-free policy observation (single frame). xi[..., 1:1+NJ] is the encoder bias: the observed joint position is q + b - q0."""
    _, q, qj, _, w_b, qd = split_state(s, nj)
    return torch.cat([qj + xi[..., 1 : 1 + nj] - default_joint_pos, qd, w_b, projected_gravity(q), prev_action], dim=-1)


def flatten_history_term_major(hist: torch.Tensor, term_dims: tuple[int, ...]) -> torch.Tensor:
    """(H, N, D) old->new -> (N, H*D), the term-major layout of the Isaac observation manager [A_t0..A_tH-1, B_t0.., ...]."""
    H, N, _ = hist.shape
    chunks = []
    start = 0
    for d in term_dims:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


class AnalyticHeads:
    def __init__(
        self,
        fk: UrdfKinematics,
        body_names: list[str],
        anchor_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg = REWARD,
    ):
        self.fk = fk
        self.nj = len(fk.joint_names)
        self.tracked_idx = fk.link_indices(body_names)
        self.anchor_index = list(body_names).index(anchor_body_name)
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        self.reward_cfg = reward_cfg

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        p, q, qj, v, w_b, qd = split_state(s, self.nj)
        out = self.fk(p, q, qj, v, quat_apply(q, w_b), qd)
        i = self.tracked_idx
        return {k: out[k][..., i, :] for k in ("pos", "quat", "lin_vel", "ang_vel")}

    def reward(self, s_next, action, prev_action, ref: ReferenceFrame, bodies=None) -> dict[str, torch.Tensor]:
        """Reward r_{t+1} for reaching s_next with target frame ref: weighted (weight and dt) terms + 'total'."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        err = tracking_sq_errors(
            b["pos"], b["quat"], b["lin_vel"], b["ang_vel"],
            ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, self.anchor_index,
        )  # fmt: skip
        joint_pos = s_next[..., 7 : 7 + self.nj]
        return weighted_terms(err, action, prev_action, joint_pos, self.soft_limits, self.reward_cfg)

    @staticmethod
    def termination_features(s: torch.Tensor, ref_root_pos: torch.Tensor, ref_root_quat: torch.Tensor) -> torch.Tensor:
        return root_termination_features(s[..., 0:3], s[..., 3:7], ref_root_pos, ref_root_quat)

    @staticmethod
    def soft_termination(s, ref_root_pos, ref_root_quat, height_thr=0.25, tilt_thr=0.8, tz=0.02, tt=0.04) -> torch.Tensor:
        """v1: analytic termination probability; the hard thresholds |e_z|>0.25, |e_tilt|>0.8 are softened with a sigmoid (as in vibed mdp.soft_termination),
        replacing the learned termination head."""
        f = root_termination_features(s[..., 0:3], s[..., 3:7], ref_root_pos, ref_root_quat)
        alive = (1 - torch.sigmoid((f[..., 0].abs() - height_thr) / tz)) * (1 - torch.sigmoid((f[..., 1].abs() - tilt_thr) / tt))
        return 1 - alive

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Root-drift features for the critic only (plan D10): root position error in the reference heading frame 3 + yaw error 1."""
        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw.unsqueeze(-1)], dim=-1)

    def critic_tracking_features(self, s: torch.Tensor, ref: ReferenceFrame, bodies=None) -> torch.Tensor:
        """Extra v1 critic input (as in vibed mdp.critic_features): reference joints - joints NJ, re-anchored reference bodies - robot bodies
        (robot heading frame) 3B, reference root orientation relative to the robot root 6D. ref is the reference frame belonging to this state."""
        b = bodies if bodies is not None else self.robot_bodies(s)
        a = self.anchor_index
        rel_pos, _ = reanchor_reference(
            ref.body_pos_w, ref.body_quat_w, ref.body_pos_w[..., a, :], ref.body_quat_w[..., a, :], b["pos"][..., a, :], b["quat"][..., a, :]
        )
        q = s[..., 3:7]
        hq = yaw_quat(q).unsqueeze(-2).expand(b["pos"].shape[:-1] + (4,))
        e_body = quat_apply_inverse(hq, rel_pos - b["pos"]).flatten(-2)
        return torch.cat([ref.joint_pos - s[..., 7 : 7 + self.nj], e_body, root_ori_relative_obs(q, ref.root_quat_w)], dim=-1)
