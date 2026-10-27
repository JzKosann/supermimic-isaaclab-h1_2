"""Reproduction of SuperMimic (arXiv:2610.28480) on Isaac Lab / H1-2: motion tracking with a differentiable world model + short-horizon BPTT.

Ported module by module from SuperMimic-Reproduced (an independent mjlab / G1 reproduction) to Isaac Lab; dimensions parameterised for H1-2 (27 joints).
Environment side (exported by this package, used by tasks/h1_2_env_cfg.py):
    commands.py      multi-clip motion command, return-based reset sampling
    actions.py       joint position action with execution noise + encoder bias
    events.py        sim context xi and domain randomisation (friction / encoder bias / torso CoM)
    observations.py  policy / goal observation terms
    rewards.py       Table A.2 rewards (shared by the simulator and imagination)
    terminations.py  root height / tilt terminations, clip-end time-out
Training side (used by scripts/supermimic/train.py):
    kinematics.py    differentiable URDF FK          motion_library.py   motion library
    state.py         state representation            world_model.py      dynamics + termination head + Eq. 1
    heads.py         analytic obs / reward heads     imagination.py      imagined rollouts + BPTT + critic
    policy.py        policy / critic                 replay.py           replay buffer + collector
    trainer.py       Algorithm A.1                   benchmark.py        Appendix-F evaluation
    viewer.py        follow camera for rendered runs (play / evaluation without --headless)
"""

from .actions import NoisyJointPositionAction, NoisyJointPositionActionCfg
from .commands import MultiClipMotionCommand, MultiClipMotionCommandCfg
from .events import get_sim_context, randomize_body_com, randomize_encoder_bias, randomize_foot_friction
from .observations import goal_anchor_ori, goal_anchor_pos, goal_future_frames, joint_pos_biased_rel, ref_root_ori
from .rewards import (
    REWARD,
    motion_body_ang_vel,
    motion_body_lin_vel,
    motion_body_ori,
    motion_body_pos,
    motion_root_ori,
    motion_root_pos,
)
from .terminations import bad_root_height, bad_root_tilt, clip_end
