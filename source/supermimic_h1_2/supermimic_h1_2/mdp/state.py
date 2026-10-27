"""State representation of the world model (SuperMimic-Reproduced plan D1-D3, parameterised for the 27 joints of H1-2).

Simulator state s (67-d for H1-2, root position relative to its env origin):
    [root_pos 3, root_quat wxyz 4, joint_pos NJ, root_lin_vel_w 3, root_ang_vel_b 3, joint_vel NJ]
root is the pelvis link origin; root_lin_vel_w is the link-origin linear velocity (Isaac root_link_lin_vel_w); the angular velocity is in the root frame.

Network input (heading-invariant, 2*NJ+13 dims):
    [root_z 1, rot6d of the heading-removed root orientation 6, root linear velocity in the heading frame 3, root angular velocity in the root frame 3, joint_pos NJ, joint_vel NJ]
Network output delta (2*NJ+12 dims) and integration (dt = 0.02 s):
    dv_h 3, dw_b 3, dqd NJ          velocity increments (heading frame / root frame)
    dp_h 3, dth_b 3, dq NJ          position corrections on top of explicit integration
    v' = v + R_h dv_h    w' = w + dw_b    qd' = qd + dqd
    p' = p + dt v' + R_h dp_h    q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
A zero output = constant-velocity extrapolation; the last layer of the dynamics MLP is zero-initialised, so training starts from this physical prior.
Error normalisation (D3): each component of the error vector [dp 3, rotvec 3, dv 3, dw 3, dq NJ, dqd NJ] is divided by a nominal scale sigma.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rotations import (
    heading_yaw,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_conjugate,
    quat_from_rotvec,
    quat_from_yaw,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)


def state_dim(nj: int) -> int:
    return 13 + 2 * nj


def split_state(s: torch.Tensor, nj: int):
    """-> root_pos, root_quat, joint_pos, root_lin_vel_w, root_ang_vel_b, joint_vel"""
    return s[..., 0:3], s[..., 3:7], s[..., 7 : 7 + nj], s[..., 7 + nj : 10 + nj], s[..., 10 + nj : 13 + nj], s[..., 13 + nj : 13 + 2 * nj]


class StateIO:
    """Reads the full (N, 13+2NJ) state from Isaac."""

    def __init__(self, env, asset_name: str = "robot"):
        self.robot = env.scene[asset_name]
        self.origins = env.scene.env_origins

    def read(self) -> torch.Tensor:
        d = self.robot.data
        q = d.root_link_quat_w
        return torch.cat(
            [
                d.root_link_pos_w - self.origins,
                q,
                d.joint_pos,
                d.root_link_lin_vel_w,
                quat_apply_inverse(q, d.root_link_ang_vel_w),
                d.joint_vel,
            ],
            dim=-1,
        ).clone()


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # nominal error scales (D3): [root position m, root rotation vector rad, root linear velocity m/s, root angular velocity rad/s, joint position rad, joint velocity rad/s]
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # output scales of the dynamics head: [dv_h, dw_b, dqd, dp_h, dth_b, dq]
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # input feature scales (divide): [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel]
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    # xi normalisation (offset, scale): foot friction 1, encoder bias NJ, torso CoM offset 3 (expanded to NJ in StateSpec)
    xi_friction: tuple[float, float] = (0.75, 0.45)
    xi_bias_scale: float = 0.01
    xi_com_scale: float = 0.05


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, nj: int, device, dtype=torch.float32):
        self.cfg = cfg
        self.nj = nj
        self.dt = cfg.dt
        self.state_dim = state_dim(nj)
        self.feat_dim = 13 + 2 * nj
        self.delta_dim = 12 + 2 * nj
        self.xi_dim = 4 + nj

        def rep(scales, sizes):
            return torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)]).to(device=device, dtype=dtype)

        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, nj, nj))
        self.out_scale = rep(cfg.out_scale, (3, 3, nj, 3, 3, nj))
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, nj, nj))
        self.xi_offset = rep((cfg.xi_friction[0], 0.0, 0.0), (1, nj, 3))
        self.xi_scale = rep((cfg.xi_friction[1], cfg.xi_bias_scale, cfg.xi_com_scale), (1, nj, 3))

    def split(self, s: torch.Tensor):
        return split_state(s, self.nj)

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = quat_from_yaw(heading_yaw(q))
        q_local = quat_canonical(quat_mul(quat_conjugate(hq), q))
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), quat_apply_inverse(hq, v), w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        nj = self.nj
        d = delta * self.out_scale
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6 : 6 + nj]
        dp_h, dth_b, dq = d[..., 6 + nj : 9 + nj], d[..., 9 + nj : 12 + nj], d[..., 12 + nj : 12 + 2 * nj]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = quat_from_yaw(heading_yaw(q))
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        p_new = p + self.dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(self.dt * w_new + dth_b)))
        qj_new = qj + self.dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        rot_err = rotvec_from_quat(quat_mul(quat_conjugate(quat_normalize(q2)), quat_normalize(q1)))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean of the per-dimension normalised squared errors (the reported "prediction error")."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)
