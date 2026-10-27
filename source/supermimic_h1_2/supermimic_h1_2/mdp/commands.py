"""Multi-clip motion command: future-frame goals, return-based reset sampling (paper D.1), clip-end time-out.

Frame convention as in BeyondMimic: at reset frame = f0 and the robot is placed at reference f0; compute() then advances frame to f0+1, the goal window
in the observation starts at frame, the next state s_{t+1} is scored (reward / termination) against frame, then frame advances again. When the clip reaches its last frame it stays there,
clip_end is set and the next step ends the episode as a time-out (no playback across clips, no teleport).

Return-based reset sampling (SuperMimic-Reproduced plan D14): every clip is split into 1 s bins; at the end of an episode the discounted return-to-go of each step is computed backwards
and folded into an EMA of its bin (alpha=0.05, initial value G_max=0.1/(1-gamma)=10); reset distribution = 50% uniform + 40% softmax(-25 G_b/G_max)
+ 10% historical softmax weights (EMA alpha=0.01). For episodes truncated by the 10 s time-out, steps less than 100 steps before the truncation are not added to the EMA.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from .heads import anchor_relative_obs, root_ori_relative_obs
from .kinematics import H1_2_URDF, UrdfKinematics
from .motion_library import MotionLibrary, discover_npz
from .rewards import tracking_sq_errors
from .rotations import quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


class MultiClipMotionCommand(CommandTerm):
    cfg: MultiClipMotionCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: MultiClipMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_names = list(cfg.body_names)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.anchor_index = self.body_names.index(cfg.anchor_body_name)
        self.ghost: Articulation | None = env.scene.articulations.get("ghost")  # only present with enable_play_viz
        self.robot_root_index = self.robot.body_names.index(cfg.root_body_name)
        if self.robot_root_index != 0:
            raise ValueError("root body must be the articulation root link")

        self.fk = UrdfKinematics(self.robot.joint_names, cfg.urdf_path, cfg.root_body_name, device=self.device)
        files = discover_npz(cfg.motion_dir, cfg.exclude)
        if cfg.include:
            files = [f for f in files if any(k in f for k in cfg.include)]
        self.library = MotionLibrary(
            files, self.fk, self.body_names, self.robot.data.default_joint_pos[0], self.device, cfg.bin_frames,
            goal_mode=cfg.goal_mode,
        )  # fmt: skip
        print(f"[INFO] SuperMimic {self.library.summary()}")
        lib = self.library

        n = self.num_envs
        self.frame = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.forced_frame = torch.full((n,), -1, dtype=torch.long, device=self.device)  # for evaluation: the next reset starts exactly at this frame
        self._errors_stamp = -1
        self._errors: dict[str, torch.Tensor] = {}

        # return-based reset sampling
        self.bin_return = torch.full((lib.num_bins,), cfg.g_max, device=self.device)
        self.bin_memory = torch.full((lib.num_bins,), 1.0 / lib.num_bins, device=self.device)
        self.max_ep_steps = min(int(math.ceil(env.cfg.episode_length_s / env.step_dt)) + 2, cfg.max_logged_steps)
        self.ep_rewards = torch.zeros(n, self.max_ep_steps, device=self.device)
        self.ep_bins = torch.zeros(n, self.max_ep_steps, dtype=torch.long, device=self.device)
        self.ep_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self._gamma_pow = cfg.gamma ** torch.arange(self.max_ep_steps, device=self.device, dtype=torch.float64)

        for k in ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_joint_pos", "error_joint_vel", "sampling_entropy"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ reference (with env origin)

    @property
    def command(self) -> torch.Tensor:
        return self.goal

    @property
    def goal(self) -> torch.Tensor:
        return self.library.goal_window(self.frame, self.cfg.future_frames)

    @property
    def ref_body_pos_w(self) -> torch.Tensor:
        return self.library.body_pos_w[self.frame] + self._env.scene.env_origins[:, None, :]

    @property
    def ref_body_quat_w(self) -> torch.Tensor:
        return self.library.body_quat_w[self.frame]

    @property
    def ref_root_pos_w(self) -> torch.Tensor:
        return self.library.ref(self.frame).root_pos_w + self._env.scene.env_origins

    @property
    def ref_root_quat_w(self) -> torch.Tensor:
        return self.library.ref(self.frame).root_quat_w

    # ------------------------------------------------------------------ robot (link-origin quantities)

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_link_pos_w

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.root_link_quat_w

    def tracking_errors(self) -> dict[str, torch.Tensor]:
        """Six squared tracking errors of the current state against the current target frame; computed once per step (the reward manager calls the 6 terms in a row)."""
        stamp = int(self._env.common_step_counter)
        if stamp != self._errors_stamp:
            d = self.robot.data
            b = self.body_indexes
            self._errors = tracking_sq_errors(
                d.body_link_pos_w[:, b],
                d.body_link_quat_w[:, b],
                d.body_link_lin_vel_w[:, b],
                d.body_link_ang_vel_w[:, b],
                self.ref_body_pos_w,
                self.ref_body_quat_w,
                self.library.body_lin_vel_w[self.frame],
                self.library.body_ang_vel_w[self.frame],
                self.anchor_index,
            )
            self._errors_stamp = stamp
        return self._errors

    def anchor_relative_obs(self) -> torch.Tensor:
        """[position 3, rotation 6D 6] of the reference anchor in the robot anchor frame (BeyondMimic motion_anchor_pos_b/ori_b)."""
        a = self.body_indexes[self.anchor_index]
        return anchor_relative_obs(
            self.robot.data.body_link_pos_w[:, a],
            self.robot.data.body_link_quat_w[:, a],
            self.ref_body_pos_w[:, self.anchor_index],
            self.ref_body_quat_w[:, self.anchor_index],
        )

    def ref_root_ori_obs(self) -> torch.Tensor:
        """v1 observation: 6D of the reference root orientation of the current target frame in the robot root frame."""
        return root_ori_relative_obs(self.robot.data.root_link_quat_w, self.ref_root_quat_w)

    # ------------------------------------------------------------------ metrics

    def _update_metrics(self):
        d = self.robot.data
        a = self.anchor_index
        r = self.library.ref(self.frame)
        self.metrics["error_anchor_pos"] = (self.ref_body_pos_w[:, a] - self.robot_body_pos_w[:, a]).norm(dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.ref_body_quat_w[:, a], self.robot_body_quat_w[:, a])
        self.metrics["error_body_pos"] = (self.ref_body_pos_w - self.robot_body_pos_w).norm(dim=-1).mean(-1)
        self.metrics["error_joint_pos"] = (r.joint_pos - d.joint_pos).norm(dim=-1)
        self.metrics["error_joint_vel"] = (r.joint_vel - d.joint_vel).norm(dim=-1)

    # ------------------------------------------------------------------ return bookkeeping of the sampler

    def _log_step_rewards(self, env_mask: torch.Tensor | None) -> None:
        reward = getattr(self._env, "reward_buf", None)
        if reward is None:
            return
        ids = torch.arange(self.num_envs, device=self.device) if env_mask is None else env_mask.nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        pos = self.ep_len[ids].clamp_max(self.max_ep_steps - 1)
        self.ep_rewards[ids, pos] = reward[ids]
        self.ep_bins[ids, pos] = self.library.bin_of_frame(self.frame[ids])
        self.ep_len[ids] = torch.clamp(self.ep_len[ids] + 1, max=self.max_ep_steps)

    def _finalize_episodes(self, env_ids: torch.Tensor) -> None:
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        if int(lens.sum()) == 0:
            return
        T = int(lens.max())
        r = self.ep_rewards[env_ids, :T].double()
        bins = self.ep_bins[env_ids, :T]
        tmask = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        gp = self._gamma_pow[:T]
        rtg = torch.flip(torch.cumsum(torch.flip(r * tmask * gp[None, :], [1]), 1), [1]) / gp[None, :]
        tm = getattr(self._env, "termination_manager", None)
        terminated = tm.terminated[env_ids] if tm is not None else torch.ones_like(lens, dtype=torch.bool)
        truncated = ~terminated & ~self.clip_end[env_ids]
        remaining = lens[:, None] - torch.arange(T, device=self.device)[None, :]
        tmask = tmask & ~(truncated[:, None] & (remaining < cfg.min_remaining_steps))
        valid = tmask.reshape(-1)
        b = bins.reshape(-1)[valid]
        g = rtg.reshape(-1)[valid].float()
        if b.numel() == 0:
            return
        sums = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, g)
        cnts = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, torch.ones_like(g))
        touched = cnts > 0
        self.bin_return[touched] = (1 - cfg.ema_alpha) * self.bin_return[touched] + cfg.ema_alpha * sums[touched] / cnts[touched]

    def _sample_frames(self, n: int) -> torch.Tensor:
        lib, cfg = self.library, self.cfg
        if cfg.sampling_mode == "start":
            return lib.clip_start[torch.randint(0, lib.num_clips, (n,), device=self.device)]
        if cfg.sampling_mode == "uniform":
            f = torch.randint(0, lib.num_frames, (n,), device=self.device)
            return torch.maximum(torch.minimum(f, lib.last_frame_of(f) - 2), lib.clip_start[lib.frame_clip[f]])
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return / cfg.g_max, dim=0)
        self.bin_memory = (1 - cfg.memory_alpha) * self.bin_memory + cfg.memory_alpha * hard
        p = cfg.uniform_ratio / lib.num_bins + cfg.hard_ratio * hard + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        p = p / p.sum()
        b = torch.multinomial(p, n, replacement=True)
        f = lib.bin_start[b] + (torch.rand(n, device=self.device) * (lib.bin_end[b] - lib.bin_start[b]).float()).long()
        f = torch.maximum(torch.minimum(f, lib.last_frame_of(f) - 2), lib.clip_start[lib.frame_clip[f]])
        self.metrics["sampling_entropy"][:] = -(p * (p + 1e-12).log()).sum() / math.log(max(lib.num_bins, 2))
        return f

    # ------------------------------------------------------------------ reset / update

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        # credit the reward of the last step of the episode (already in reward_buf) to the current frame, then fold the episode into the bin returns
        self._log_step_rewards(mask)
        self._finalize_episodes(env_ids)
        self.ep_len[env_ids] = 0

        forced = self.forced_frame[env_ids]
        self.frame[env_ids] = torch.where(forced >= 0, forced, self._sample_frames(len(env_ids)))
        self.clip_end[env_ids] = False

        r = self.library.ref(self.frame[env_ids])
        n = len(env_ids)
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        pr = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(pr[:, 0], pr[:, 1], (n, 6), device=self.device)
        root_pos = r.root_pos_w + self._env.scene.env_origins[env_ids] + rs[:, 0:3]
        root_quat = quat_mul(quat_from_euler_xyz(rs[:, 3], rs[:, 4], rs[:, 5]), r.root_quat_w)
        vr = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        vs = sample_uniform(vr[:, 0], vr[:, 1], (n, 6), device=self.device)
        root_vel = torch.cat([r.root_lin_vel_w + vs[:, :3], r.root_ang_vel_w + vs[:, 3:]], dim=-1)
        joint_pos = r.joint_pos.clone()
        lo, hi = self.cfg.joint_position_range
        if hi > lo:
            joint_pos += sample_uniform(lo, hi, joint_pos.shape, device=self.device)
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])

        self.robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, r.joint_vel.clone(), env_ids=env_ids)
        if self.ghost is not None:
            self._write_ghost(env_ids)

    def _update_command(self):
        # envs that did not reset at this step: the reward of this step was earned against the current frame, credit it (reset envs were already credited in _resample_command)
        reset_buf = getattr(self._env, "reset_buf", None)
        self._log_step_rewards(None if reset_buf is None else ~reset_buf)
        last = self.library.last_frame_of(self.frame)
        self.frame += 1
        over = self.frame > last
        self.clip_end |= over
        self.frame = torch.where(over, last, self.frame)
        if self.ghost is not None:
            self._write_ghost(None)

    def _write_ghost(self, env_ids: torch.Tensor | None):
        """Places the reference ghost at the current reference frame (without reset perturbations); velocities zeroed, PD target = reference joint angles, so it stays put during physics substeps."""
        ids = slice(None) if env_ids is None else env_ids
        r = self.library.ref(self.frame[ids])
        pose = torch.cat([r.root_pos_w + self._env.scene.env_origins[ids], r.root_quat_w], dim=-1)
        self.ghost.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.ghost.write_root_link_velocity_to_sim(torch.zeros_like(pose[:, :6]), env_ids=env_ids)
        self.ghost.write_joint_state_to_sim(r.joint_pos, torch.zeros_like(r.joint_pos), env_ids=env_ids)
        self.ghost.set_joint_position_target(r.joint_pos, env_ids=env_ids)

    # ------------------------------------------------------------------ visualisation (for play: reference bodies drawn as spheres)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "ref_visualizer"):
                self.ref_visualizer = VisualizationMarkers(self.cfg.ref_visualizer_cfg)
            self.ref_visualizer.set_visibility(True)
        elif hasattr(self, "ref_visualizer"):
            self.ref_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.ref_visualizer.visualize(translations=self.ref_body_pos_w.reshape(-1, 3))


@configclass
class MultiClipMotionCommandCfg(CommandTermCfg):
    class_type: type = MultiClipMotionCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    asset_name: str = "robot"
    motion_dir: str = MISSING
    include: list[str] | None = None  # only clips whose file name contains any of these substrings (None = all)
    exclude: list[str] | None = None
    urdf_path: str = H1_2_URDF
    anchor_body_name: str = "torso_link"
    root_body_name: str = "pelvis"
    body_names: list[str] = MISSING
    future_frames: int = 10
    goal_mode: str = "paper"  # paper (v0) | traj (v1), see motion_library.py
    bin_frames: int = 50
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: str = "return"  # return | uniform | start
    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0
    g_max: float = 10.0
    gamma: float = 0.99
    ema_alpha: float = 0.05
    memory_alpha: float = 0.01
    min_remaining_steps: int = 100
    max_logged_steps: int = 1024
    ref_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/supermimic_ref",
        markers={"sphere": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)))},
    )
