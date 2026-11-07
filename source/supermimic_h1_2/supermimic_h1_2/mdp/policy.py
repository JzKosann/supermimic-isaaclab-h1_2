"""Policy (motion encoder + shared decoder) and critic (paper Table A.4).

Deterministic policy a = dec(enc_motion(g), o_hist): enc MLP(2048,1024,512,256) -> 32-d z (bounded by tanh),
dec MLP(2048,1024,512,256) -> action, last layer orthogonal init with gain 0.01 (initial actions ~0, targets ~default pose).
critic V(canonical s, g, xi, e_root) MLP(256,128); e_root are root-drift features (plan D10, not visible to the policy).
v0 / v1: observations / goals are scaled by fixed constants (joint velocity /5, angular velocity /2), the same constants in imagination and simulation.
v2 (input_norm, following Uni-SuperMimic): policy observations / goals and all critic inputs go through a RunningNorm instead,
frozen after 2e8 samples. Policy statistics are updated from the real (noisy) simulator observations before each collection step,
critic statistics from the imagined critic inputs before each critic fit. Without it the critic sees raw metres / radians, which
was the main suspect when our v1 run on a 250 h dataset diverged; v2 trained 50k iterations on the same data without divergence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
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
    input_norm: bool = False  # v2: RunningNorm on observations / goals instead of the fixed scales above
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
    input_norm: bool = False  # v2: one RunningNorm over the concatenated critic input (replaces the goal scales)


class RunningNorm(nn.Module):
    """Per-dimension running mean / variance, output (x - mean) / sqrt(var + eps^2) (as Uni-SuperMimic networks.RunningNorm).
    Frozen once count >= max_count. With several GPUs the batch statistics of all ranks are all-gathered and merged in rank order,
    so every rank holds bit-identical statistics (and freezes at the same update)."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if bool(self.count >= self.max_count):
            return
        x = x.reshape(-1, x.shape[-1]).float()
        var, mean = torch.var_mean(x, dim=0, unbiased=False)
        packed = torch.cat([torch.full((1,), float(x.shape[0]), device=x.device), mean, var])
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            parts = [torch.empty_like(packed) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, packed)
        else:
            parts = [packed]
        d = self.mean.numel()
        for p in parts:  # Chan et al. merge: (count, mean, var) + (n, m, v)
            n, m, v = p[0].double(), p[1 : 1 + d], p[1 + d :]
            tot = self.count + n
            w_new, w_old = (n / tot).float(), (self.count / tot).float()
            delta = m - self.mean
            self.var.copy_(self.var * w_old + v * w_new + delta.square() * (w_old * w_new))
            self.mean.add_(delta * w_new)
            self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)


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
        # v2: statistics per dimension of the flattened 10-frame history (Uni shares one per-frame statistic across frames; same effect)
        self.obs_norm = RunningNorm(cfg.obs_dim) if cfg.input_norm else None
        self.goal_norm = RunningNorm(cfg.goal_dim) if cfg.input_norm else None

    @torch.no_grad()
    def update_norm(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> None:
        """v2: update the statistics from the real simulator observations (with sensor noise) and goals; no-op without input_norm."""
        if self.obs_norm is not None:
            self.obs_norm.update(obs_hist)
            self.goal_norm.update(goal)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.obs_norm is not None:
            z = torch.tanh(self.encoder(self.goal_norm(goal)))
            return self.decoder(torch.cat([z, self.obs_norm(obs_hist)], dim=-1))
        z = torch.tanh(self.encoder(goal * self.goal_scale))
        return self.decoder(torch.cat([z, obs_hist * self.obs_scale], dim=-1))


class Critic(nn.Module):
    """extra: v0 has 4 root-drift features; v1 additionally has tracking errors (heads.critic_tracking_features)."""

    def __init__(self, cfg: CriticCfg, pcfg: PolicyCfg, feat_dim: int, xi_dim: int, extra_dim: int = 4):
        super().__init__()
        in_dim = feat_dim + pcfg.goal_dim + xi_dim + extra_dim
        self.mlp = MLP(in_dim, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale(pcfg))
        self.norm = RunningNorm(in_dim) if cfg.input_norm else None

    @torch.no_grad()
    def update_norm(self, feat, goal, xi_norm, extra) -> None:
        """v2: update the statistics from the critic inputs of an imagined rollout (all H+1 steps); no-op without input_norm."""
        if self.norm is not None:
            self.norm.update(torch.cat([feat, goal, xi_norm, extra], dim=-1))

    def forward(self, feat, goal, xi_norm, extra) -> torch.Tensor:
        if self.norm is not None:
            return self.mlp(self.norm(torch.cat([feat, goal, xi_norm, extra], dim=-1))).squeeze(-1)
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, extra], dim=-1)).squeeze(-1)
