"""Imagines H=32 steps in the frozen world model, optimises the policy by BPTT and fits the critic to GAE targets (paper Sec. 3.3, Eq. 2/3, Table A.5).

Each step: the model gives the noise-free observation o_t, sensor noise eta_t (Table A.1) is resampled before it enters the 10-frame history, a_t = pi(o_t + eta_t, g_t),
s_{t+1} = f_w(s_t, a_t, xi), r_{t+1} comes from the analytic reward head, d_{t+1} from the termination head, soft survival weight c_{t+1} = c_t (1 - d_{t+1}).
J = E[sum gamma^t c_t r_{t+1} + gamma^H c_H V(s_H)], first-order gradient w.r.t. theta (back-propagated through everything, including the gradient of c w.r.t. d and of V w.r.t. s_H).
critic: GAE(lambda=0.95) targets are computed once per rollout and then fixed, 5 epochs x 4 minibatches, all inputs detached.
When imagination runs past the clip end the target stays at the last frame (plan D15).

v1 (following supermimic_vibed; three switches in PolicyTrainerCfg, off by default = v0):
    start_within_clip         imagination starts only from states whose "start frame + H" does not cross the last frame of the clip, so nothing is learned on a frozen last-frame target
    analytic_termination      d uses analytic soft thresholds (heads.soft_termination) instead of the learned termination head
    critic_tracking_features  extra critic input: tracking errors (heads.critic_tracking_features), measured against the state's own reference frame
The reference-orientation observation term (PolicyCfg.ref_ori_obs) and the trajectory goal (goal_mode="traj") follow the environment and are not switched here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .distributed import allreduce_grads
from .heads import (
    ReferenceFrame,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
    reference_at,
    root_ori_relative_obs,
)
from .motion_library import MotionLibrary
from .policy import Critic, Policy
from .replay import ReplayBuffer, StartBatch
from .rewards import REWARD_TERM_NAMES
from .world_model import WorldModel, maybe_compile

ANCHOR_NOISE_HALF = (0.25,) * 3 + (0.05,) * 6


class ImagStep:
    """One imagination step (dynamics + FK reward + termination probability + next noise-free observation + anchor pose [+ v1 critic features]) packed into one function,
    so that torch.compile can fuse the hundreds of small per-link FK kernels. Reference quantities are passed as plain tensors so the compiled graph has fixed shapes."""

    def __init__(
        self,
        wm: WorldModel,
        default_joint_pos: torch.Tensor,
        compile_mode: str | None,
        analytic_termination: bool = False,
        critic_tracking: bool = False,
    ):
        self.wm = wm
        self.q0 = default_joint_pos
        self.analytic_termination = analytic_termination
        self.critic_tracking = critic_tracking
        self._fn = maybe_compile(self._step, compile_mode)

    def _step(self, s, a, a_prev, xi, ref_pos, ref_quat, ref_lin, ref_ang, ref_root_pos, ref_root_quat, ref_joint_pos):
        wm = self.wm
        heads = wm.heads
        ref = ReferenceFrame(ref_pos, ref_quat, ref_lin, ref_ang, ref_root_pos, ref_root_quat, ref_joint_pos)
        s_next = wm.step(s, a, xi)
        bodies = heads.robot_bodies(s_next)
        terms = heads.reward(s_next, a, a_prev, ref, bodies)
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0)
        if self.analytic_termination:
            d = heads.soft_termination(s_next, ref_root_pos, ref_root_quat)
        else:
            d = torch.sigmoid(wm.termination_logits(s_next, ref_root_pos, ref_root_quat))
        obs_clean = analytic_observation(s_next, a, xi, self.q0, heads.nj)
        ai = heads.anchor_index
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)
        # the reference frame belonging to s_next is this step's target frame ref
        crit = (
            torch.cat([heads.root_error_features(s_next, ref), heads.critic_tracking_features(s_next, ref, bodies)], dim=-1)
            if self.critic_tracking
            else None
        )
        return s_next, terms["total"], term_vec, d, obs_clean, anchor_pose, crit

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame):
        return self._fn(
            s, a, a_prev, xi, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w,
            ref.root_pos_w, ref.root_quat_w, ref.joint_pos,
        )  # fmt: skip


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, S)
    rewards: torch.Tensor  # (H, n) r_{t+1}
    dhat: torch.Tensor  # (H+1, n) predicted termination probabilities (d_0 = 0)
    survival: torch.Tensor  # (H+1, n) c_t
    goals: torch.Tensor  # (H+1, n, G)
    critic_extra: torch.Tensor  # (H+1, n, E) extra critic input (v0: 4 root-drift features)
    actions: torch.Tensor  # (H, n, A)
    reward_terms: torch.Tensor  # (8,) mean of each term over the rollout (diagnostics)


def imagine(wm: WorldModel, policy: Policy, lib: MotionLibrary, start: StartBatch, horizon: int, step_fn: ImagStep, obs_noise: bool = True) -> ImagRollout:
    pcfg = policy.cfg
    heads = wm.heads
    dev = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    term_dims = pcfg.obs_term_dims
    noise_half = torch.tensor(pcfg.obs_noise_half, device=dev)
    anchor_half = torch.tensor(ANCHOR_NOISE_HALF, device=dev)

    def add_noise(o, half):
        return o + (torch.rand_like(o) * 2 - 1) * half if obs_noise else o

    # target frames of the whole rollout (frame + t, clamped to the last frame of the clip), fetched at once
    last = lib.last_frame_of(start.frame)
    frames = torch.minimum(start.frame[None, :] + torch.arange(horizon + 1, device=dev)[:, None], last[None, :])
    ref_all = reference_at(lib, frames.reshape(-1))

    def ref_at(t: int) -> ReferenceFrame:
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(*(getattr(ref_all, k)[sl] for k in ReferenceFrame.__dataclass_fields__))

    def make_goal(t: int, ref: ReferenceFrame, anchor_pose):
        g = lib.goal_window(frames[t], pcfg.future_frames)
        if not pcfg.anchor_obs:
            return g
        ai = heads.anchor_index
        rel = anchor_relative_obs(anchor_pose[:, 0:3], anchor_pose[:, 3:7], ref.body_pos_w[:, ai], ref.body_quat_w[:, ai])
        return torch.cat([g, add_noise(rel, anchor_half)], dim=-1)

    def observe(s, a_prev, frame_quat):
        o = analytic_observation(s, a_prev, start.xi, q0, heads.nj)
        if pcfg.ref_ori_obs:  # v1: reference root orientation of the target frame at observation time, relative to the robot root
            o = torch.cat([o, root_ori_relative_obs(s[..., 3:7], frame_quat)], dim=-1)
        return add_noise(o, noise_half)

    hist = [
        observe(start.hist_states[i], start.hist_actions[i], lib.ref(start.hist_frames[i]).root_quat_w)
        for i in range(start.hist_states.shape[0])
    ]
    s, a_prev = start.state, start.prev_action
    c = torch.ones(n, device=dev)
    ref = ref_at(0)
    anchor_pose = None
    if pcfg.anchor_obs:
        b0 = heads.robot_bodies(s)
        ai = heads.anchor_index
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    states, rewards, actions = [s], [], []
    dhats, survivals = [torch.zeros(n, device=dev)], [c]
    goals = [make_goal(0, ref, anchor_pose)]
    if step_fn.critic_tracking:  # the start state's own reference frame = the frame before the target frame (clamped to the clip start)
        base = torch.maximum(start.frame - 1, lib.clip_start[lib.frame_clip[start.frame]])
        ref_own = reference_at(lib, base)
        extras = [torch.cat([heads.root_error_features(s, ref_own), heads.critic_tracking_features(s, ref_own)], dim=-1)]
    else:
        extras = [heads.root_error_features(s, ref)]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=dev)
    for t in range(horizon):
        a = policy(flatten_history_term_major(torch.stack(hist, 0), term_dims), goals[-1])
        s, r, term_vec, d, obs_clean, anchor_pose, crit = step_fn(s, a, a_prev, start.xi, ref)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)
        ref = ref_at(t + 1)
        if pcfg.ref_ori_obs:
            obs_clean = torch.cat([obs_clean, root_ori_relative_obs(s[..., 3:7], ref.root_quat_w)], dim=-1)
        hist = hist[1:] + [add_noise(obs_clean, noise_half)]
        a_prev = a
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(make_goal(t + 1, ref, anchor_pose))
        extras.append(crit if crit is not None else heads.root_error_features(s, ref))
    return ImagRollout(
        states=torch.stack(states),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        critic_extra=torch.stack(extras),
        actions=torch.stack(actions),
        reward_terms=term_sum / horizon,
    )


def policy_objective(ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """Eq. 2: J = mean_n[sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H]."""
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    J = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        J = J + gamma**H * ro.survival[-1] * terminal_values
    return J.mean()


@torch.no_grad()
def gae_targets(rewards, dhat_next, values, gamma: float, lam: float) -> torch.Tensor:
    """Eq. 3: R_t = r_{t+1} + gamma (1 - d_{t+1}) [(1 - lambda) V_{t+1} + lambda R_{t+1}], R_H = V_H. Returns (H, n)."""
    H = rewards.shape[0]
    R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        out[t] = R
    return out


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    # v1 switches (see the module docstring), off by default = v0
    start_within_clip: bool = False
    analytic_termination: bool = False
    critic_tracking_features: bool = False


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        self.step_fn = ImagStep(
            wm, lib.default_joint_pos, cfg.compile, cfg.analytic_termination, cfg.critic_tracking_features
        )

    def set_lr(self, lr: float, critic_lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = critic_lr

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        cfg = self.cfg
        extra = None
        if cfg.start_within_clip:
            extra = self.lib.last_frame_of(buffer.frame) - buffer.frame >= cfg.horizon
        start = buffer.sample_starts(cfg.batch_size, self.policy.cfg.history, extra)
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(self.wm, self.policy, self.lib, start, cfg.horizon, self.step_fn, cfg.obs_noise)
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        v_H = self.critic(feat[-1], ro.goals[-1], xi_norm, ro.critic_extra[-1]) if cfg.use_critic else None
        J = policy_objective(ro, v_H, cfg.gamma)
        self.opt.zero_grad(set_to_none=True)
        (-J / cfg.horizon).backward()
        allreduce_grads(self.policy)
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        m = {
            "imag_objective": J.detach(),
            "imag_return_undisc": (ro.survival[:-1] * ro.rewards).sum(0).mean().detach(),
            "imag_survival_H": ro.survival[-1].mean().detach(),
            "imag_dhat_mean": ro.dhat[1:].mean().detach(),
            "policy_grad_norm": gn.detach(),
            "action_abs_mean": ro.actions.abs().mean().detach(),
        }
        for k, v in zip(REWARD_TERM_NAMES, ro.reward_terms):
            m[f"imag_r/{k}"] = v

        # ---- critic: GAE targets of this rollout (fixed)
        if cfg.use_critic:
            with torch.no_grad():
                X_feat = feat.detach()
                X_goal = ro.goals.detach()
                X_er = ro.critic_extra.detach()
                X_xi = xi_norm[None].expand(H1, n, -1)
                flat = lambda x: x.reshape(H1 * n, -1)  # noqa: E731
                values = self.critic(flat(X_feat), flat(X_goal), flat(X_xi), flat(X_er)).reshape(H1, n)
                targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)
                X = [x[:-1].reshape(-1, x.shape[-1]) for x in (X_feat, X_goal, X_xi, X_er)]
                Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            closs = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    closs = torch.nn.functional.mse_loss(self.critic(*(x[idx] for x in X)), Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
            m["critic_loss"] = closs.detach()
            m["critic_value_mean"] = values.mean()
            m["critic_target_mean"] = targets.mean()
        return {k: float(v) for k, v in m.items()}
