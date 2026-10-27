"""Multi-clip motion library: all clips are concatenated along one global frame axis on the GPU and gathered by frame index.

Each frame stores the reference state (same layout as the world-model state, see state.py):
    [root_pos 3, root_quat 4, joint_pos NJ, root_lin_vel_w 3, root_ang_vel_w 3, joint_vel NJ]
root is the pelvis (body 0 of the npz); root_lin_vel_w is the pelvis link-origin velocity (central difference of positions; the
body_lin_vel_w stored in the npz is Isaac's centre-of-mass velocity, a different quantity, and is not used). Reference poses / velocities of the tracked bodies are computed by kinematics.UrdfKinematics
from the reference state (link-origin quantities), the same FK used in imagination for the robot bodies of predicted states, so simulator and imagination rewards are measured the same way.

Goal features (the paper does not specify the per-frame content, U5), two kinds:
- goal_mode="paper" (v0, SuperMimic-Reproduced plan D5): 2*NJ+10 dims per frame (64 for H1-2)
    [joint_pos - q0 (NJ), joint_vel (NJ), reference root projected gravity 3, reference root angular velocity (root frame) 3,
     reference root linear velocity (reference heading frame) 3, reference root height 1]
- goal_mode="traj" (v1, following motion.goal of supermimic_vibed): 2*NJ+15 dims per frame (69 for H1-2)
    [displacement of the future root relative to the root of the "current reference frame" (current reference heading frame) 3, future root orientation (current reference heading frame) 6D,
     root linear velocity (root frame) 3, root angular velocity (root frame) 3, joint_pos - q0 (NJ), joint_vel (NJ)]
  "current reference frame" = the frame before the first frame of the window (the reference frame the robot is at now, clamped to the clip start); this tells where the reference goes next and how much it turns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .kinematics import UrdfKinematics
from .rotations import projected_gravity, quat_apply_inverse, quat_conjugate, quat_mul, rot6d_from_quat, yaw_quat


@dataclass
class ClipInfo:
    name: str
    start: int
    length: int


@dataclass
class RefState:
    root_pos_w: torch.Tensor  # (N,3)  without env origin
    root_quat_w: torch.Tensor  # (N,4)
    joint_pos: torch.Tensor  # (N,NJ)
    root_lin_vel_w: torch.Tensor  # (N,3)
    root_ang_vel_w: torch.Tensor  # (N,3) world frame
    joint_vel: torch.Tensor  # (N,NJ)


@dataclass
class RefBodies:
    pos_w: torch.Tensor  # (N,B,3)
    quat_w: torch.Tensor  # (N,B,4)
    lin_vel_w: torch.Tensor  # (N,B,3)
    ang_vel_w: torch.Tensor  # (N,B,3)


def goal_features(ref: RefState, default_joint_pos: torch.Tensor) -> torch.Tensor:
    q = ref.root_quat_w
    ang_b = quat_apply_inverse(q, ref.root_ang_vel_w)
    lin_h = quat_apply_inverse(yaw_quat(q), ref.root_lin_vel_w)
    return torch.cat(
        [ref.joint_pos - default_joint_pos, ref.joint_vel, projected_gravity(q), ang_b, lin_h, ref.root_pos_w[..., 2:3]],
        dim=-1,
    )


def discover_npz(motion_dir: str, exclude: list[str] | None = None) -> list[str]:
    files = sorted(os.path.join(motion_dir, f) for f in os.listdir(motion_dir) if f.endswith(".npz"))
    if exclude:
        files = [f for f in files if not any(e in os.path.basename(f) for e in exclude)]
    return files


class MotionLibrary:
    def __init__(
        self,
        npz_files: list[str],
        fk: UrdfKinematics,
        body_names: list[str],
        default_joint_pos: torch.Tensor,
        device: str | torch.device,
        bin_frames: int = 50,
        fps: float = 50.0,
        goal_mode: str = "paper",
    ):
        if not npz_files:
            raise ValueError("MotionLibrary needs at least one clip")
        self.device = torch.device(device)
        self.body_names = list(body_names)
        self.joint_names = list(fk.joint_names)
        self.nj = len(self.joint_names)
        self.default_joint_pos = default_joint_pos.detach().to(self.device, torch.float32).reshape(-1)
        self.bin_frames = int(bin_frames)

        clips: list[ClipInfo] = []
        chunks: list[np.ndarray] = []
        start = 0
        for f in npz_files:
            with np.load(f) as d:
                clip_fps = float(np.asarray(d["fps"]).reshape(-1)[0])
                if abs(clip_fps - fps) > 1e-6:
                    raise ValueError(f"{f}: fps {clip_fps} != {fps}")
                jn = [str(x) for x in d["joint_names"]]
                bn = [str(x) for x in d["body_names"]]
                jidx = np.array([jn.index(n) for n in self.joint_names])
                r = bn.index("pelvis")
                root_pos = d["body_pos_w"][:, r].astype(np.float64)
                n = root_pos.shape[0]
                root_vel = np.gradient(root_pos, 1.0 / fps, axis=0) if n > 1 else np.zeros_like(root_pos)
                chunks.append(
                    np.concatenate(
                        [
                            root_pos,
                            d["body_quat_w"][:, r],
                            d["joint_pos"][:, jidx],
                            root_vel,
                            d["body_ang_vel_w"][:, r],
                            d["joint_vel"][:, jidx],
                        ],
                        axis=1,
                    ).astype(np.float32)
                )
            clips.append(ClipInfo(os.path.splitext(os.path.basename(f))[0], start, n))
            start += n
        self.clips = clips
        self.num_clips = len(clips)
        self.num_frames = start
        self._ref = torch.as_tensor(np.concatenate(chunks, 0), device=self.device)
        self.state_dim = self._ref.shape[1]

        # reference bodies (the B tracked links), run through FK in chunks to avoid using too much memory at once
        tracked = fk.link_indices(self.body_names).to(self.device)
        pos, quat, lin, ang = [], [], [], []
        with torch.no_grad():
            for s in range(0, self.num_frames, 65536):
                r = self.ref(torch.arange(s, min(s + 65536, self.num_frames), device=self.device))
                out = fk(r.root_pos_w, r.root_quat_w, r.joint_pos, r.root_lin_vel_w, r.root_ang_vel_w, r.joint_vel)
                pos.append(out["pos"][:, tracked])
                quat.append(out["quat"][:, tracked])
                lin.append(out["lin_vel"][:, tracked])
                ang.append(out["ang_vel"][:, tracked])
        self.body_pos_w = torch.cat(pos)
        self.body_quat_w = torch.cat(quat)
        self.body_lin_vel_w = torch.cat(lin)
        self.body_ang_vel_w = torch.cat(ang)
        self.goal_mode = goal_mode
        all_ref = self.ref(torch.arange(self.num_frames, device=self.device))
        if goal_mode == "paper":
            self.goal_feat = goal_features(all_ref, self.default_joint_pos)
            self.goal_dim = self.goal_feat.shape[1]
        elif goal_mode == "traj":
            q = all_ref.root_quat_w
            # precompute the parts that depend only on the future frame itself; displacement / orientation relative to the current reference frame are computed in goal_window
            self.goal_feat = torch.cat(
                [
                    quat_apply_inverse(q, all_ref.root_lin_vel_w),
                    quat_apply_inverse(q, all_ref.root_ang_vel_w),
                    all_ref.joint_pos - self.default_joint_pos,
                    all_ref.joint_vel,
                ],
                dim=-1,
            )
            self.goal_dim = 9 + self.goal_feat.shape[1]
        else:
            raise ValueError(f"unknown goal_mode {goal_mode}")

        self.clip_start = torch.tensor([c.start for c in clips], dtype=torch.long, device=self.device)
        self.clip_len = torch.tensor([c.length for c in clips], dtype=torch.long, device=self.device)
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), self.clip_len)
        # split each clip into bins of bin_frames (1 s), numbered globally across the library (for return-based reset sampling)
        bins_per_clip = (self.clip_len + self.bin_frames - 1) // self.bin_frames
        self.clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.num_bins = int(bins_per_clip.sum().item())
        bin_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), bins_per_clip)
        local_bin = torch.arange(self.num_bins, device=self.device) - self.clip_bin_offset[bin_clip]
        self.bin_start = self.clip_start[bin_clip] + local_bin * self.bin_frames
        self.bin_end = torch.minimum(self.bin_start + self.bin_frames, self.clip_start[bin_clip] + self.clip_len[bin_clip])

    # ------------------------------------------------------------------ per-frame access

    def _split(self, r: torch.Tensor) -> RefState:
        nj = self.nj
        return RefState(r[..., 0:3], r[..., 3:7], r[..., 7 : 7 + nj], r[..., 7 + nj : 10 + nj], r[..., 10 + nj : 13 + nj], r[..., 13 + nj :])

    def ref(self, frame: torch.Tensor) -> RefState:
        return self._split(self._ref[frame])

    def bodies(self, frame: torch.Tensor) -> RefBodies:
        return RefBodies(self.body_pos_w[frame], self.body_quat_w[frame], self.body_lin_vel_w[frame], self.body_ang_vel_w[frame])

    def state(self, frame: torch.Tensor) -> torch.Tensor:
        """Reference frame in the world-model state layout (root angular velocity converted to the root frame, see state.py)."""
        r = self.ref(frame)
        w_b = quat_apply_inverse(r.root_quat_w, r.root_ang_vel_w)
        return torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos, r.root_lin_vel_w, w_b, r.joint_vel], dim=-1)

    # ------------------------------------------------------------------ queries

    def bin_of_frame(self, frame: torch.Tensor) -> torch.Tensor:
        clip = self.frame_clip[frame]
        return self.clip_bin_offset[clip] + (frame - self.clip_start[clip]) // self.bin_frames

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return self.clip_last[self.frame_clip[frame]]

    def goal_window(self, frame: torch.Tensor, k: int) -> torch.Tensor:
        """(N, k*goal_dim): goal features of frames frame..frame+k-1; past the clip end the last frame is repeated."""
        last = self.last_frame_of(frame)
        idx = torch.minimum(frame[:, None] + torch.arange(k, device=self.device)[None, :], last[:, None])
        if self.goal_mode == "paper":
            return self.goal_feat[idx].reshape(frame.shape[0], k * self.goal_dim)
        base = torch.maximum(frame - 1, self.clip_start[self.frame_clip[frame]])
        hq = yaw_quat(self._ref[base, 3:7])[:, None, :].expand(-1, k, -1)
        dp = quat_apply_inverse(hq, self._ref[idx, 0:3] - self._ref[base, None, 0:3])
        rot = rot6d_from_quat(quat_mul(quat_conjugate(hq), self._ref[idx, 3:7]))
        return torch.cat([dp, rot, self.goal_feat[idx]], dim=-1).reshape(frame.shape[0], k * self.goal_dim)

    def summary(self) -> str:
        hours = self.num_frames / 50.0 / 3600.0
        mem = sum(t.numel() * t.element_size() for t in (self._ref, self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w, self.goal_feat)) / 1e9
        return (
            f"MotionLibrary: {self.num_clips} clips, {self.num_frames} frames ({hours:.3f} h @50Hz), "
            f"{self.num_bins} bins, goal {self.goal_dim}-d/frame, {mem:.2f} GB on {self.device}"
        )
