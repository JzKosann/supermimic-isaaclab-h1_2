"""Isaac Lab config of the Unitree H1-2 without hands (27 joints).

Actuators, default pose and rigid-body / articulation solver settings are taken from unitree_rl_lab
(https://github.com/unitreerobotics/unitree_rl_lab, source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .paths import H1_2_USD

H1_2_ACTION_SCALE = 0.25  # joint target = default pose + 0.25 * policy output (the paper gives no alpha; common value for H1-2)

UNITREE_H1_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=H1_2_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "left_hip_yaw_joint": 0,
            "left_hip_roll_joint": 0,
            "left_hip_pitch_joint": -0.16,
            "left_knee_joint": 0.36,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0,
            "right_hip_roll_joint": 0,
            "right_hip_pitch_joint": -0.16,
            "right_knee_joint": 0.36,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "torso_joint": 0,
            "left_shoulder_pitch_joint": 0.4,
            "left_shoulder_roll_joint": 0,
            "left_shoulder_yaw_joint": 0,
            "left_elbow_joint": 0.3,
            "left_wrist_pitch_joint": 0,  # unitree_rl_lab gives no default for the wrists; use 0
            "left_wrist_yaw_joint": 0,
            "left_wrist_roll_joint": 0,
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": 0,
            "right_shoulder_yaw_joint": 0,
            "right_elbow_joint": 0.3,
            "right_wrist_pitch_joint": 0,
            "right_wrist_yaw_joint": 0,
            "right_wrist_roll_joint": 0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "GO2HV-1": IdealPDActuatorCfg(
            joint_names_expr=[".*ankle.*", ".*_shoulder_pitch_.*", ".*_shoulder_roll_.*"],
            effort_limit=40,
            velocity_limit=9,
            stiffness={".*ankle.*": 40.0, ".*_shoulder_.*": 100.0},
            damping=2.0,
            armature=0.01,
        ),
        "GO2HV-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_.*", ".*_elbow_.*"],
            effort_limit=18,
            velocity_limit=20,
            stiffness=50,
            damping=2.0,
            armature=0.01,
        ),
        "M107-24-1": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_.*"],
            effort_limit=300.0,
            velocity_limit=14.0,
            stiffness=200.0,
            damping=4.0,
            armature=0.01,
        ),
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", "torso_joint"],
            effort_limit=200,
            velocity_limit=23.0,
            stiffness={".*_hip_.*": 150.0, "torso_joint": 300.0},
            damping={".*_hip_.*": 2.0, "torso_joint": 6.0},
            armature=0.01,
        ),
        "N5020-16": IdealPDActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            effort_limit=25.0,
            velocity_limit=37.0,
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)
