"""Sim context xi and the domain randomisation of paper Table A.3.

xi (one per env, 31-d for H1-2) = [foot friction 1, joint encoder bias NJ, torso CoM offset 3]; input to the world model and the critic.
- Foot friction U[0.3, 1.2], resampled every episode (mode="reset"), one value shared by both feet, applied only to the collision shapes of ankle_roll_link;
  ground friction is 1.0 with combine mode multiply, so the effective friction is this value.
- Encoder bias U[-0.01, 0.01] rad, resampled every episode; applied to both the observation (q + b) and the execution (target - b), i.e. a real zero-offset error
  (SuperMimic-Reproduced plan D8 E1: randomize_joint_default_pos cannot be used, it only changes one side).
- Torso CoM offset x U[+-0.025], y/z U[+-0.05] m, sampled once at startup (changing the PhysX CoM goes through the CPU, too slow per episode).
- No pushes (paper: a deterministic world model cannot capture abrupt state changes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SimContext:
    def __init__(self, num_envs: int, num_joints: int, device):
        self.friction = torch.ones(num_envs, 1, device=device)
        self.encoder_bias = torch.zeros(num_envs, num_joints, device=device)
        self.com_offset = torch.zeros(num_envs, 3, device=device)

    def read(self) -> torch.Tensor:
        return torch.cat([self.friction, self.encoder_bias, self.com_offset], dim=-1).clone()


def get_sim_context(env: ManagerBasedEnv, asset_name: str = "robot") -> SimContext:
    ctx = getattr(env, "_supermimic_ctx", None)
    if ctx is None:
        robot = env.scene[asset_name]
        ctx = SimContext(env.num_envs, robot.num_joints, env.device)
        env._supermimic_ctx = ctx
    return ctx


class randomize_foot_friction(ManagerTermBase):
    """Foot friction: one value per env, written to all collision shapes of the selected bodies (static = dynamic friction)."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        view = self.asset.root_physx_view
        # number of shapes per body (parsed the same way as isaaclab randomize_rigid_body_material)
        shapes_per_body = [self.asset._physics_sim_view.create_rigid_body_view(p).max_shapes for p in view.link_paths[0]]
        if sum(shapes_per_body) != view.max_shapes:
            raise ValueError(f"failed to parse shapes per body: {sum(shapes_per_body)} != {view.max_shapes}")
        body_ids = asset_cfg.body_ids if asset_cfg.body_ids != slice(None) else range(self.asset.num_bodies)
        self.shape_ids = torch.tensor(
            [sum(shapes_per_body[:b]) + k for b in body_ids for k in range(shapes_per_body[b])], dtype=torch.long
        )
        self.materials = view.get_material_properties().clone()  # (N, S, 3) on CPU: static friction, dynamic friction, restitution

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_range: tuple[float, float],
    ):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        ids_cpu = env_ids.cpu()
        mu = torch.empty(len(ids_cpu)).uniform_(*friction_range)
        self.materials[ids_cpu[:, None], self.shape_ids[None, :], 0] = mu[:, None]
        self.materials[ids_cpu[:, None], self.shape_ids[None, :], 1] = mu[:, None]
        self.asset.root_physx_view.set_material_properties(self.materials, ids_cpu)
        get_sim_context(env, asset_cfg.name).friction[env_ids, 0] = mu.to(env.device)


def randomize_encoder_bias(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, bias_range: tuple[float, float]
):
    ctx = get_sim_context(env, asset_cfg.name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ctx.encoder_bias[env_ids] = torch.empty(len(env_ids), ctx.encoder_bias.shape[1], device=env.device).uniform_(*bias_range)


def randomize_body_com(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, com_range: dict[str, tuple[float, float]]
):
    """Adds a random offset to the CoM of one body and records it in xi (like isaaclab randomize_rigid_body_com; single body only)."""
    asset = env.scene[asset_cfg.name]
    ids_cpu = torch.arange(env.num_envs) if env_ids is None else env_ids.cpu()
    lo = torch.tensor([com_range.get(k, (0.0, 0.0))[0] for k in "xyz"])
    hi = torch.tensor([com_range.get(k, (0.0, 0.0))[1] for k in "xyz"])
    offset = lo + (hi - lo) * torch.rand(len(ids_cpu), 3)
    body_id = asset_cfg.body_ids[0]
    coms = asset.root_physx_view.get_coms().clone()
    coms[ids_cpu, body_id, :3] += offset
    asset.root_physx_view.set_coms(coms, ids_cpu)
    get_sim_context(env, asset_cfg.name).com_offset[ids_cpu.to(env.device)] = offset.to(env.device)
