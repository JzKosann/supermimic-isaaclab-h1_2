"""GPU ring replay buffer ([env, time] layout) and the collector on the Isaac side.

Each row stores the state s_t at the start of a control step and the following transition: clean action a_t, executed action a_exec_t, xi_t, reward r_{t+1},
target frame, and the termination label done of s_{t+1}. At the end of an episode an extra terminal row stores s_{T+1} and its termination label (no transition starts
from it), so that a k-step window can use it as its last target state and the termination head gets positive samples. Windows and imagination starts never cross episodes.
1000 steps per env (paper Table A.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WindowBatch:
    states: torch.Tensor  # (k+1, n, S)
    a_exec: torch.Tensor  # (k, n, A)
    xi: torch.Tensor  # (n, X)
    done: torch.Tensor  # (k, n) termination labels of states 1..k
    frame: torch.Tensor  # (k, n) target frame of each transition


@dataclass
class StartBatch:
    state: torch.Tensor  # (n, S)
    xi: torch.Tensor  # (n, X)
    frame: torch.Tensor  # (n,)
    prev_action: torch.Tensor  # (n, A) clean action of the previous step (0 at the episode start)
    hist_states: torch.Tensor  # (hist, n, S) s_{t-hist+1..t}; rows before the episode start are back-filled with the first row of the episode
    hist_actions: torch.Tensor  # (hist, n, A) previous-step actions aligned with hist_states
    hist_frames: torch.Tensor  # (hist, n) target frames aligned with hist_states (for the v1 reference-orientation observation)


class ReplayBuffer:
    def __init__(self, num_envs: int, capacity: int, state_dim: int, act_dim: int, xi_dim: int, device):
        self.E, self.T = num_envs, capacity
        self.device = torch.device(device)
        f = lambda *shape: torch.zeros(num_envs, capacity, *shape, device=self.device)  # noqa: E731
        self.s = f(state_dim)
        self.a = f(act_dim)
        self.a_exec = f(act_dim)
        self.xi = f(xi_dim)
        self.r = f()
        self.done = f()
        self.frame = torch.zeros(num_envs, capacity, dtype=torch.long, device=self.device)
        self.ep_id = torch.full((num_envs, capacity), -1, dtype=torch.long, device=self.device)
        self.terminal = torch.zeros(num_envs, capacity, dtype=torch.bool, device=self.device)
        self.ep_start = torch.zeros(num_envs, capacity, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros(num_envs, capacity, dtype=torch.bool, device=self.device)
        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.ep_counter = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def add(self, env_ids, s, a, a_exec, xi, r, done, frame, terminal, ep_start) -> None:
        p = self.ptr[env_ids]
        self.s[env_ids, p] = s
        self.a[env_ids, p] = a
        self.a_exec[env_ids, p] = a_exec
        self.xi[env_ids, p] = xi
        self.r[env_ids, p] = r
        self.done[env_ids, p] = done
        self.frame[env_ids, p] = frame
        self.ep_id[env_ids, p] = self.ep_counter[env_ids]
        self.terminal[env_ids, p] = terminal
        self.ep_start[env_ids, p] = ep_start
        self.valid[env_ids, p] = True
        self.ptr[env_ids] = (p + 1) % self.T

    @torch.no_grad()
    def end_episode(self, env_ids) -> None:
        self.ep_counter[env_ids] += 1

    @property
    def size(self) -> int:
        return int(self.valid.sum())

    def _rolled(self, x: torch.Tensor, j: int) -> torch.Tensor:
        return torch.roll(x, shifts=-j, dims=1) if j else x

    def _dist_to_ptr(self) -> torch.Tensor:
        """(E,T) distance of each row behind the write pointer: 0 = the most recently written row."""
        t = torch.arange(self.T, device=self.device)[None, :]
        return (self.ptr[:, None] - t - 1) % self.T

    @torch.no_grad()
    def window_mask(self, k: int) -> torch.Tensor:
        """Rows t..t+k are valid and in the same episode, and t..t+k-1 are not terminal rows; a window cannot wrap past the write pointer."""
        ok = self.valid & ~self.terminal
        for j in range(1, k + 1):
            v = self._rolled(self.valid, j) & (self._rolled(self.ep_id, j) == self.ep_id)
            if j < k:
                v &= ~self._rolled(self.terminal, j)
            ok &= v
        return ok & (self._dist_to_ptr() >= k)

    @torch.no_grad()
    def _sample(self, mask: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = mask.reshape(-1).float()
        if not bool(flat.any()):
            raise RuntimeError("replay buffer has no sampleable rows yet")
        idx = torch.multinomial(flat, n, replacement=True)
        return idx // self.T, idx % self.T

    @torch.no_grad()
    def sample_windows(self, n: int, k: int, recent: bool = False) -> WindowBatch:
        mask = self.window_mask(k)
        if recent:  # only windows written in the last 2k steps (online error)
            mask &= self._dist_to_ptr() < 2 * k + 1
        e, t = self._sample(mask, n)
        tt = (t[None, :] + torch.arange(k + 1, device=self.device)[:, None]) % self.T
        ee = e[None, :].expand(k + 1, n)
        return WindowBatch(
            states=self.s[ee, tt],
            a_exec=self.a_exec[ee[:-1], tt[:-1]],
            xi=self.xi[e, t],
            done=self.done[ee[1:], tt[1:]],
            frame=self.frame[ee[:-1], tt[:-1]],
        )

    @torch.no_grad()
    def sample_starts(self, n: int, hist: int, extra_mask: torch.Tensor | None = None) -> StartBatch:
        """extra_mask (E,T): additional start conditions (v1: start frame + H does not cross the last frame of the clip)."""
        mask = self.valid & ~self.terminal & (self._dist_to_ptr() < self.T - hist - 1)  # history rows must not be being overwritten
        if extra_mask is not None:
            mask &= extra_mask
        e, t = self._sample(mask, n)
        tt = (t[None, :] + (torch.arange(hist, device=self.device) - (hist - 1))[:, None]) % self.T  # (hist, n)
        ee = e[None, :].expand(hist, n)
        same_ep = (self.ep_id[ee, tt] == self.ep_id[e, t][None, :]) & self.valid[ee, tt]
        # rows before the episode start are back-filled with the first row of the episode (same back-fill semantics as Isaac's CircularBuffer)
        first_ok = same_ep.float().argmax(dim=0)
        tt_fixed = torch.where(same_ep, tt, tt[first_ok, torch.arange(n, device=self.device)][None, :].expand(hist, n))
        tprev = (tt_fixed - 1) % self.T
        prev_ok = (self.ep_id[ee, tprev] == self.ep_id[e, t][None, :]) & self.valid[ee, tprev] & ~self.ep_start[ee, tt_fixed]
        hist_actions = torch.where(prev_ok[..., None], self.a[ee, tprev], torch.zeros_like(self.a[ee, tprev]))
        return StartBatch(
            state=self.s[e, t],
            xi=self.xi[e, t],
            frame=self.frame[e, t],
            prev_action=hist_actions[-1],
            hist_states=self.s[ee, tt_fixed],
            hist_actions=hist_actions,
            hist_frames=self.frame[ee, tt_fixed],
        )


class ReplayCollector:
    """Writes Isaac transitions into the buffer.

    h1_2_env_cfg.SuperMimicRLEnv calls on_pre_reset at the start of _reset_idx (the robot is still in the terminal state s_{T+1} and the action is still
    this step's), which writes the last transition and the terminal row; after_step is called after env.step returns and writes transitions for envs that did not reset,
    then remembers the new state of every env (reset envs are already placed at the start of the new episode) as s_t of the next row.
    """

    def __init__(self, env, buffer: ReplayBuffer, state_io, ctx, command, action_term):
        self.env, self.buffer, self.state_io, self.ctx = env, buffer, state_io, ctx
        self.cmd, self.action_term = command, action_term
        n, dev = env.num_envs, env.device
        self.prev_s = torch.zeros(n, buffer.s.shape[-1], device=dev)
        self.prev_xi = torch.zeros(n, buffer.xi.shape[-1], device=dev)
        self.prev_frame = torch.zeros(n, dtype=torch.long, device=dev)
        self.ep_start = torch.ones(n, dtype=torch.bool, device=dev)
        self.has_prev = torch.zeros(n, dtype=torch.bool, device=dev)

    def start(self) -> None:
        """Call once after env.reset()."""
        self._remember()
        self.ep_start[:] = True

    def _remember(self) -> None:
        self.prev_s = self.state_io.read()
        self.prev_xi = self.ctx.read()
        self.prev_frame = self.cmd.frame.clone()
        self.has_prev[:] = True

    def _write_rows(self, ids: torch.Tensor, done_next: torch.Tensor) -> None:
        self.buffer.add(
            ids,
            s=self.prev_s[ids],
            a=self.env.action_manager.action[ids],
            a_exec=self.action_term.executed_raw[ids],
            xi=self.prev_xi[ids],
            r=self.env.reward_buf[ids],
            done=done_next,
            frame=self.prev_frame[ids],
            terminal=torch.zeros(len(ids), dtype=torch.bool, device=self.env.device),
            ep_start=self.ep_start[ids],
        )
        self.ep_start[ids] = False

    @torch.no_grad()
    def on_pre_reset(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device)
        ids = env_ids[self.has_prev[env_ids]]
        if ids.numel() == 0:
            return
        terminated = self.env.reset_terminated[ids].float()
        self._write_rows(ids, terminated)
        n = ids.numel()
        zeros_a = torch.zeros(n, self.buffer.a.shape[-1], device=self.env.device)
        self.buffer.add(
            ids,
            s=self.state_io.read()[ids],
            a=zeros_a,
            a_exec=zeros_a,
            xi=self.prev_xi[ids],
            r=torch.zeros(n, device=self.env.device),
            done=terminated,
            frame=self.cmd.frame[ids],
            terminal=torch.ones(n, dtype=torch.bool, device=self.env.device),
            ep_start=torch.zeros(n, dtype=torch.bool, device=self.env.device),
        )
        self.buffer.end_episode(ids)
        self.has_prev[ids] = False

    @torch.no_grad()
    def after_step(self) -> None:
        done = self.env.reset_buf
        ids = (~done & self.has_prev).nonzero().squeeze(-1)
        if ids.numel() > 0:
            self._write_rows(ids, torch.zeros(ids.numel(), device=self.env.device))
        self._remember()
        self.ep_start |= done
