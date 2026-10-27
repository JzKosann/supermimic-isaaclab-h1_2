"""Policy (motion encoder + shared decoder) and critic (paper Table A.4).

Deterministic policy a = dec(enc_motion(g), o_hist): enc MLP(2048,1024,512,256) -> 32-d z (bounded by tanh),
dec MLP(2048,1024,512,256) -> action, last layer orthogonal init with gain 0.01 (initial actions ~0, targets ~default pose).
critic V(canonical s, g, xi, e_root) MLP(256,128); e_root are root-drift features (plan D10, not visible to the policy).
Observations / goals are scaled by fixed constants (joint velocity /5, angular velocity /2); no running-mean normalisation (the same constants in imagination and simulation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .world_model import MLP


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = 32
    history: int = 10
    # per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions(, v1 ref_root_ori)
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0, 1.0)
    # per-frame goal scales (divide): joint_pos NJ, joint_vel NJ, gravity 3, root angular velocity 3, root linear velocity 3, root height 1
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    # per-frame scales of the v1 trajectory goal (divide): displacement 3, orientation 6D, root linear velocity 3, root angular velocity 3, joint_pos NJ, joint_vel NJ
    goal_traj_scale: tuple[float, ...] = (1.0, 1.0, 2.0, 2.0, 1.0, 5.0)
    decoder_last_gain: float = 0.01
    # set from the environment (filled in by Trainer)
    num_joints: int = 27
    future_frames: int = 10
    anchor_obs: bool = False
    ref_ori_obs: bool = False  # v1
    goal_mode: str = "paper"  # paper(v0) | traj(v1)

    @property
    def obs_term_dims(self) -> tuple[int, ...]:
        nj = self.num_joints
        return (nj, nj, 3, 3, nj) + ((6,) if self.ref_ori_obs else ())

    @property
    def obs_dim(self) -> int:
        return self.history * sum(self.obs_term_dims)

    @property
    def goal_frame_dims(self) -> tuple[int, ...]:
        nj = self.num_joints
        return (3, 6, 3, 3, nj, nj) if self.goal_mode == "traj" else (nj, nj, 3, 3, 3, 1)

    @property
    def goal_frame_scale(self) -> tuple[float, ...]:
        return self.goal_traj_scale if self.goal_mode == "traj" else self.goal_term_scale

    @property
    def goal_dim(self) -> int:
        return self.future_frames * sum(self.goal_frame_dims) + (9 if self.anchor_obs else 0)

    @property
    def obs_noise_half(self) -> tuple[float, ...]:
        """Half-widths of the uniform noise of Table A.1, in the order of the policy observation group (v1 reference orientation +-0.05, as in vibed)."""
        nj = self.num_joints
        return (0.01,) * nj + (0.5,) * nj + (0.2,) * 3 + (0.05,) * 3 + (0.0,) * nj + ((0.05,) * 6 if self.ref_ori_obs else ())


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)


def _obs_scale(cfg: PolicyCfg) -> torch.Tensor:
    return torch.cat([torch.full((cfg.history * d,), 1.0 / s) for d, s in zip(cfg.obs_term_dims, cfg.obs_term_scale)])


def _goal_scale(cfg: PolicyCfg) -> torch.Tensor:
    per_frame = torch.cat([torch.full((d,), 1.0 / s) for d, s in zip(cfg.goal_frame_dims, cfg.goal_frame_scale)])
    return torch.cat([per_frame.repeat(cfg.future_frames), torch.ones(9 if cfg.anchor_obs else 0)])


class Policy(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = MLP(cfg.goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.decoder = MLP(cfg.latent_dim + cfg.obs_dim, cfg.decoder_hidden, cfg.num_joints, last_gain=cfg.decoder_last_gain)
        self.register_buffer("obs_scale", _obs_scale(cfg))
        self.register_buffer("goal_scale", _goal_scale(cfg))

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.encoder(goal * self.goal_scale))
        return self.decoder(torch.cat([z, obs_hist * self.obs_scale], dim=-1))


class Critic(nn.Module):
    """extra: v0 has 4 root-drift features; v1 additionally has tracking errors (heads.critic_tracking_features)."""

    def __init__(self, cfg: CriticCfg, pcfg: PolicyCfg, feat_dim: int, xi_dim: int, extra_dim: int = 4):
        super().__init__()
        self.mlp = MLP(feat_dim + pcfg.goal_dim + xi_dim + extra_dim, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale(pcfg))

    def forward(self, feat, goal, xi_norm, extra) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, extra], dim=-1)).squeeze(-1)
