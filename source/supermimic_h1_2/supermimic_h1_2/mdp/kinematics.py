"""Differentiable forward kinematics (torch) of H1-2, parsed directly from the URDF.

SuperMimic-Reproduced (kinematics/g1_fk.py) builds the tree from a MuJoCo model; the Isaac USD used here was converted from
unitree_ros/robots/h1_2_description/h1_2_handless.urdf (shipped in assets/h1_2_description/) (see unitree_model/H1-2/h1_2_handless/config.yaml,
merge_fixed_joints=True), so parsing the same URDF gives exactly Isaac's link frames, without depending on mujoco (not in the Isaac Lab image).

Given the root (pelvis) pose / velocity and joint positions / velocities (Isaac joint order), returns the world-frame pose and velocity of every link origin:
    pos / quat                pose of the link frame origin          (= Isaac body_link_pos_w / body_link_quat_w)
    lin_vel / ang_vel         linear / angular velocity of the origin (= Isaac body_link_lin_vel_w / body_link_ang_vel_w)
Note that Isaac's body_lin_vel_w is the centre-of-mass velocity, not this quantity. URDF joint axes pass through the child link origin, so velocity propagation along the tree is just
    w_c = w_p + R_c axis qd        v_c = v_p + w_p × (p_c - p_p)
with no axis-anchor offset term as in MuJoCo. Only leaf links whose descendants are all fixed joints (logo/imu/camera/lidar, merged into
torso_link in the USD) are skipped.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import torch

from ..paths import H1_2_URDF
from .rotations import quat_apply, quat_from_angle_axis, quat_from_rpy, quat_mul, quat_normalize



@dataclass
class _Link:
    name: str
    parent: int  # index of the parent link in this list, -1 for the root
    origin_pos: tuple[float, float, float]
    origin_quat: tuple[float, float, float, float]
    axis: tuple[float, float, float]
    joint: int  # index in joint_names (Isaac order); -1 for fixed joints


class UrdfKinematics:
    """Batched differentiable FK for a single-tree URDF with revolute / fixed joints only."""

    def __init__(
        self,
        joint_names: list[str],
        urdf_path: str = H1_2_URDF,
        root_link: str = "pelvis",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.joint_names = list(joint_names)
        root = ET.parse(urdf_path).getroot()
        children: dict[str, list[ET.Element]] = {}
        for j in root.findall("joint"):
            children.setdefault(j.find("parent").get("link"), []).append(j)

        def has_movable_descendant(link: str) -> bool:
            for j in children.get(link, []):
                if j.get("type") != "fixed" or has_movable_descendant(j.find("child").get("link")):
                    return True
            return False

        links = [_Link(root_link, -1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1)]
        queue = [0]
        while queue:  # BFS, so a parent link always comes before its children
            p = queue.pop(0)
            for j in children.get(links[p].name, []):
                child = j.find("child").get("link")
                jtype = j.get("type")
                if jtype == "fixed" and not has_movable_descendant(child):
                    continue
                if jtype not in ("revolute", "continuous", "fixed"):
                    raise ValueError(f"unsupported joint type {jtype} ({j.get('name')})")
                o = j.find("origin")
                xyz = tuple(float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split())
                rpy = tuple(float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split())
                a = j.find("axis")
                axis = tuple(float(x) for x in (a.get("xyz") if a is not None else "1 0 0").split())
                jidx = -1
                if jtype != "fixed":
                    if j.get("name") not in self.joint_names:
                        raise ValueError(f"URDF joint {j.get('name')} not in joint_names")
                    jidx = self.joint_names.index(j.get("name"))
                links.append(_Link(child, p, xyz, quat_from_rpy(*rpy), axis, jidx))
                queue.append(len(links) - 1)
        self.links = links
        self.link_names = [l.name for l in links]
        movable = sorted(l.joint for l in links if l.joint >= 0)
        if movable != list(range(len(self.joint_names))):
            raise ValueError("URDF movable joints do not match joint_names one-to-one")

        t = lambda x: torch.tensor(x, dtype=dtype, device=self.device)  # noqa: E731
        self._origin_pos = t([l.origin_pos for l in links])
        self._origin_quat = t([l.origin_quat for l in links])
        axes = torch.tensor([l.axis for l in links], dtype=dtype)
        self._axis = (axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(self.device)

    @property
    def num_links(self) -> int:
        return len(self.links)

    def link_index(self, name: str) -> int:
        return self.link_names.index(name)

    def link_indices(self, names) -> torch.Tensor:
        return torch.tensor([self.link_names.index(n) for n in names], dtype=torch.long, device=self.device)

    def forward(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_lin_vel_w: torch.Tensor | None = None,
        root_ang_vel_w: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """root_* (...,3/4), joint_* (...,NJ); velocities are computed only if all three velocity inputs are given. Returns (...,L,.) in link_names order."""
        batch = root_pos.shape[:-1]
        compute_vel = root_lin_vel_w is not None
        L = self.num_links
        pos: list = [None] * L
        quat: list = [None] * L
        lin: list = [None] * L
        ang: list = [None] * L
        pos[0] = root_pos
        quat[0] = quat_normalize(root_quat)
        if compute_vel:
            lin[0] = root_lin_vel_w
            ang[0] = root_ang_vel_w
        for i in range(1, L):
            l = self.links[i]
            p = l.parent
            qc = quat_mul(quat[p], self._origin_quat[i].expand(batch + (4,)))
            pc = pos[p] + quat_apply(quat[p], self._origin_pos[i].expand(batch + (3,)))
            if l.joint >= 0:
                axis = self._axis[i].expand(batch + (3,))
                qc = quat_mul(qc, quat_from_angle_axis(joint_pos[..., l.joint], axis))
                if compute_vel:
                    ang[i] = ang[p] + quat_apply(qc, axis) * joint_vel[..., l.joint : l.joint + 1]
            elif compute_vel:
                ang[i] = ang[p]
            if compute_vel:
                lin[i] = lin[p] + torch.cross(ang[p], pc - pos[p], dim=-1)
            pos[i] = pc
            quat[i] = qc
        out = {"pos": torch.stack(pos, dim=-2), "quat": torch.stack(quat, dim=-2)}
        if compute_vel:
            out["lin_vel"] = torch.stack(lin, dim=-2)
            out["ang_vel"] = torch.stack(ang, dim=-2)
        return out

    __call__ = forward
