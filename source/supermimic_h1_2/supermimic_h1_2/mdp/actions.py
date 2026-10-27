"""Joint position action with execution noise and encoder bias (paper Sec. 3.1, SuperMimic-Reproduced plan D8/D9).

- The clean policy action a_t stays in raw_actions (action_manager.action); the last_action observation and the action_rate penalty use it;
- execution adds noise eps ~ N(0, 0.1^2) (only to widen the action distribution the world model is fitted on); executed_raw = a_t + eps goes into the replay buffer;
- PD target = executed_raw * scale + q0 - encoder bias (the robot "believes" it is at q + b, so reaching t requires commanding t - b).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

from .events import get_sim_context

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

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.cfg.noise_std > 0:
            self._executed_raw = self._raw_actions + self.cfg.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        self._processed_actions = self._executed_raw * self._scale + self._offset - self.ctx.encoder_bias

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0


@configclass
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = NoisyJointPositionAction
    noise_std: float = 0.1
