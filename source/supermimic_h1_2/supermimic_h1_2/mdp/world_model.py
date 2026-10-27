"""World model (paper Sec. 3.2): learned dynamics + learned termination head + analytic observation / reward heads (heads.py), and the Eq. 1 fit.

Dynamics s_{t+1} = integrate(s_t, f_w(canonical(s_t), a_t, xi)), f_w is an MLP(2048, 2048, 2048) (Table A.4) with SiLU,
last layer zero-initialised. The termination head takes [root height error, tilt error] and their absolute values, MLP(64, 64) -> logit, trained with BCE.
Eq. 1: start from a recorded state, roll out k=8 steps autoregressively with the recorded executed actions; loss = normalised state error (summed over dims) + BCE, back-propagated through all 8 steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import AnalyticHeads
from .replay import ReplayBuffer, WindowBatch
from .motion_library import MotionLibrary
from .state import StateSpec, StateSpecCfg
from .distributed import allreduce_grads


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, zero_last: bool = False, last_gain: float | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8


class TerminationHead(nn.Module):
    def __init__(self, hidden: tuple[int, ...], height_thr: float, tilt_thr: float):
        super().__init__()
        self.mlp = MLP(4, hidden, 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        nj = heads.nj
        self.spec = StateSpec(cfg.state_spec, nj, device)
        self.dynamics = MLP(self.spec.feat_dim + nj + self.spec.xi_dim, cfg.hidden, self.spec.delta_dim, zero_last=True)
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads  # analytic heads, no parameters
        self.to(device)

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        delta = self.dynamics(torch.cat([self.spec.canonical(s), action, self.spec.normalize_xi(xi)], dim=-1))
        return self.spec.integrate(s, delta)

    def termination_logits(self, s_next: torch.Tensor, ref_root_pos: torch.Tensor, ref_root_quat: torch.Tensor) -> torch.Tensor:
        return self.term_head(self.heads.termination_features(s_next, ref_root_pos, ref_root_quat))

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat):
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        logits = self.termination_logits(s_next, ref_root_pos, ref_root_quat)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, e * e, bce, logits

    def enable_compile(self, mode: str | None) -> None:
        self._loss_step_fn = maybe_compile(self._loss_step, mode)

    def window_loss(self, states, a_exec, xi, done, ref_root_pos, ref_root_quat):
        """states (k+1,n,S), a_exec (k,n,A), xi (n,X), done (k,n) are the termination labels of states 1..k,
        ref_root_* (k,n,.) is the target frame of the j-th transition (used for the termination features of state j+1)."""
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = 0.0
        acc = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        nse_first = nse_last = None
        for j in range(k):
            s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], ref_root_pos[j], ref_root_quat[j])
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                nse_first = nse if j == 0 else nse_first
                nse_last = nse
                acc = acc + ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        # keep metrics on the GPU (no float()) to avoid a CPU sync per minibatch
        metrics = {
            "state_loss": (total_state / k).detach(),
            "term_bce": (total_term / k).detach(),
            "nse_1step": nse_first,
            "nse_kstep": nse_last,
            "term_acc": acc / k,
        }
        return loss, metrics

    @torch.no_grad()
    def rollout_error(self, states, a_exec, xi) -> torch.Tensor:
        """Open-loop k-step normalised squared error (k,)."""
        s = states[0]
        out = []
        for j in range(a_exec.shape[0]):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)


def maybe_compile(fn, mode: str | None):
    if not mode or mode == "none":
        return fn
    return torch.compile(fn, mode=None if mode == "default" else mode, dynamic=False)


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    compile: str | None = "default"


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm, self.lib, self.cfg = wm, lib, cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        r = self.lib.ref(w.frame.reshape(-1))
        return self.wm.window_loss(
            w.states, w.a_exec, w.xi, w.done, r.root_pos_w.reshape(k, n, 3), r.root_quat_w.reshape(k, n, 4)
        )

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            loss, m = self.loss_on(w)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(self.wm)
            m["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
            self.opt.step()
            m["loss"] = loss.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on the most recently collected transitions (as in paper Fig. 3c: "measured online on policy rollouts")."""
        w = buffer.sample_windows(n, k, recent=True)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
