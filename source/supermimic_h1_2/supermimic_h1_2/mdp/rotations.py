"""Quaternion utilities (wxyz, pure torch, differentiable, arbitrary leading batch dims).

isaaclab.utils.math is not used: its functions are @torch.jit.script, and the imagination rollout is torch.compile'd as a whole; ScriptFunctions
cause graph breaks in dynamo. The simulator side (rewards / observations / terminations) and imagination share this implementation, so both use the same numerics.
Products and rotations are element-wise (no 3x3 matmul), so results are not affected by TF32.
"""

from __future__ import annotations

import math

import torch

_EPS = 1.0e-8


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """w >= 0 (removes the double cover)."""
    return torch.where(q[..., :1] < 0, -q, q)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotates vector v by unit quaternion q."""
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """angle (...,), axis (...,3) unit vector."""
    half = 0.5 * angle.unsqueeze(-1)
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """Exponential map: rotation vector -> unit quaternion. Uses the Taylor expansion of sin(a/2)/a near zero so the gradient stays finite."""
    angle_sq = (v * v).sum(-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1e-30))
    small = angle < 1e-4
    k = torch.where(small, 0.5 - angle_sq / 48.0, torch.sin(0.5 * angle) / angle)
    w = torch.where(small, 1.0 - angle_sq / 8.0, torch.cos(0.5 * angle))
    return torch.cat([w, v * k], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Logarithm map: unit quaternion -> rotation vector, angle in [0, pi]."""
    q = quat_canonical(q)
    w = q[..., :1]
    xyz = q[..., 1:]
    s = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, w)
    small = s < 1e-6
    k = torch.where(small, 2.0 + angle * angle / 12.0, angle / torch.where(small, torch.ones_like(s), s))
    return xyz * k


def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Rotation vector of q1 relative to q2, log(q1 * q2^-1) (= isaaclab quat_box_minus)."""
    return rotvec_from_quat(quat_mul(q1, quat_conjugate(q2)))


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return quat_box_minus(q1, q2).norm(dim=-1)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, flattened (...,6)."""
    return matrix_from_quat(q)[..., :, :2].reshape(q.shape[:-1] + (6,))


def heading_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion keeping only the heading (yaw)."""
    return quat_from_yaw(heading_yaw(q))


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF rpy (fixed axes x->y->z, i.e. R = Rz Ry Rx) -> wxyz (Python floats, for URDF parsing)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
    """Gravity direction (0,0,-1) expressed in the body frame."""
    g = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    g[..., 2] = -1.0
    return quat_apply_inverse(q, g)


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
