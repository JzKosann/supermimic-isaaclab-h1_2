"""Joint position action with execution noise and encoder bias (paper Sec. 3.1, SuperMimic-Reproduced plan D8/D9).

- The clean policy action a_t stays in raw_actions (action_manager.action); the last_action observation and the action_rate penalty use it;
- execution adds noise eps ~ N(0, 0.1^2) (only to widen the action distribution the world model is fitted on); executed_raw = a_t + eps goes into the replay buffer;
- PD target = executed_raw * scale + q0 - encoder bias (the robot "believes" it is at q + b, so reaching t requires commanding t - b).

v2 additions (both off by default, i.e. v0 / v1 behave as before):
- action_clip: |executed_raw| <= action_clip (action units). Bounds what an exploding policy can feed into the replay buffer; 20 in v2.
- guard_margin: joint targets are clamped to [hard joint limit + margin, hard joint limit - margin] (rad), as a deployment-side
  joint-limit guard would do (on H1-2: URDF limits - 0.02 rad). The clamp is applied after the execution noise, so the world model
  learns from the clamped action; imagination applies the same bounds (heads.AnalyticHeads.exec_action), and the overshoot of the
  clean action beyond the bounds is penalised in the joint-limit reward (rewards.joint_limit_with_guard) so the clamped part still
  has a gradient.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

from .events import get_sim_context
from .rewards import target_overshoot

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if self._joint_ids != slice(None):
            raise ValueError("NoisyJointPositionAction expects all joints in articulation order")
        self.ctx = get_sim_context(env, cfg.asset_name)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        # guard bounds in action units (NJ, 2), from the nominal default pose (before any startup randomisation); None = no guard
        self.action_bounds: torch.Tensor | None = None
        if cfg.guard_margin is not None:
            limits = self._asset.data.default_joint_pos_limits[0]
            q0 = self._offset[0]
            scale = self._scale[0] if isinstance(self._scale, torch.Tensor) else self._scale
            lo, hi = limits[:, 0] + cfg.guard_margin, limits[:, 1] - cfg.guard_margin
            self.action_bounds = torch.stack([(lo - q0) / scale, (hi - q0) / scale], dim=-1)

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    def target_overshoot(self) -> torch.Tensor:
        """(N,) how far the joint targets of the clean policy action (no execution noise) go beyond the guard (rad); 0 without a guard."""
        if self.action_bounds is None:
            return torch.zeros(self.num_envs, device=self.device)
        return target_overshoot(self._raw_actions, self.action_bounds, self._scale)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.cfg.noise_std > 0:
            self._executed_raw = self._raw_actions + self.cfg.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        if self.cfg.action_clip > 0:
            self._executed_raw = self._executed_raw.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        if self.action_bounds is not None:
            self._executed_raw = self._executed_raw.clamp(self.action_bounds[:, 0], self.action_bounds[:, 1])
        self._processed_actions = self._executed_raw * self._scale + self._offset - self.ctx.encoder_bias

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0


@configclass
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = NoisyJointPositionAction
    noise_std: float = 0.1
    action_clip: float = 0.0  # bound on |executed action| in action units; 0 = off (v2: 20)
    guard_margin: float | None = None  # joint-target guard: hard joint limits shrunk by this many rad; None = off (v2: 0.02)
