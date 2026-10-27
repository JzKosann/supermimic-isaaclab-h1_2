"""SuperMimic reproduction environments (arXiv:2610.28480; ported item by item from SuperMimic-Reproduced (mjlab / G1) to Isaac Lab / H1-2).
gym ids: Tracking-Flat-H1_2-SuperMimic-v0 (paper observation), -Anchor-v0 (+ anchor-relative pose, deviates from the paper), -Eval-v0 / -Anchor-Eval-v0 (Appendix-F evaluation),
-v1 / -Eval-v1 (see the v1 notes at the end of this file).
Training does not use rsl_rl: scripts/supermimic/train.py (world model + BPTT); the algorithm and the environment terms live in supermimic_h1_2/mdp/.

Robot UNITREE_H1_2_CFG (supermimic_h1_2/robots.py), action scale 0.25, 50 Hz control:
- Observation (Table A.1): joint positions (with encoder bias) / velocities, IMU angular velocity, projected gravity, last action, 10-frame history, 870 dims in total;
  no anchor pose and no base linear velocity (no state estimation needed for deployment). Goal g_t = features of the next 10 reference frames, 640 dims (separate goal group).
- Rewards (Table A.2): BeyondMimic's six tracking terms + action smoothness -0.1 + joint limits -2, no self-collision; at most 0.1 per step.
- Terminations: pelvis height deviation > 0.25 m or tilt difference > 0.8; clip end or 10 s count as time-outs.
- Domain randomisation (Table A.3): foot friction U[0.3,1.2] and encoder bias +-0.01 rad resampled every episode, torso CoM offset sampled at startup; no pushes.
- Actions: N(0, 0.1^2) noise is added at execution; observations and the smoothness term use the clean action.
- Data: datasets/lafan1_h1_2_npz (LAFAN1, 40 clips, 2.45 h; H1-2 retargeting of the same clip set as paper Fig. 4a).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from supermimic_h1_2 import mdp as sm
from supermimic_h1_2.mdp.viewer import spawn_ghost
from supermimic_h1_2.paths import MOTION_DIR
from supermimic_h1_2.robots import H1_2_ACTION_SCALE, UNITREE_H1_2_CFG

# the 14 tracked bodies (same set as BeyondMimic); entry 0 must be the root pelvis
H1_2_TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
# random range of the root velocity at reset (BeyondMimic)
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
R = sm.REWARD

def ghost_cfg(robot: ArticulationCfg) -> ArticulationCfg:
    """Reference ghost for viewing (enable_play_viz): same USD as the robot, translucent, no collisions, no gravity; the motion command poses it at the reference frame every step."""
    return robot.replace(
        prim_path="{ENV_REGEX_NS}/Ghost",
        spawn=robot.spawn.replace(
            func=spawn_ghost,
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=robot.spawn.rigid_props.replace(disable_gravity=True),
            articulation_props=robot.spawn.articulation_props.replace(enabled_self_collisions=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 1.0), opacity=0.35),
        ),
    )


class SuperMimicRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv + a pre-reset hook: the replay collector needs the terminal state s_{T+1} before the reset overwrites it.
    (Isaac's RecorderManager exports HDF5 per env in Python on every reset, which is too slow at 4096 envs, so it is not used.)"""

    pre_reset_hook = None

    def _reset_idx(self, env_ids):
        if self.pre_reset_hook is not None:
            self.pre_reset_hook(env_ids)
        super()._reset_idx(env_ids)


@configclass
class SuperMimicSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = UNITREE_H1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )


@configclass
class CommandsCfg:
    motion = sm.MultiClipMotionCommandCfg(
        motion_dir=MOTION_DIR,
        body_names=H1_2_TRACKED_BODIES,
        anchor_body_name="torso_link",
        root_body_name="pelvis",
        pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    joint_pos = sm.NoisyJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=H1_2_ACTION_SCALE, use_default_offset=True, noise_std=0.1
    )


@configclass
class PolicyCfg(ObsGroup):
    """The order must match analytic_observation in mdp/heads.py."""

    joint_pos = ObsTerm(func=sm.joint_pos_biased_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
    base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    actions = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 10
        self.flatten_history_dim = True


@configclass
class PolicyV1Cfg(PolicyCfg):
    """v1: appends the 6D reference root orientation relative to the robot root (also with a 10-frame history); needs only the IMU orientation, no position estimate."""

    ref_root_ori = ObsTerm(func=sm.ref_root_ori, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))


@configclass
class GoalCfg(ObsGroup):
    future_frames = ObsTerm(func=sm.goal_future_frames, params={"command_name": "motion"})
    anchor_pos = None  # only in the -Anchor variants (see H1_2SuperMimicEnvCfg.enable_anchor_obs)
    anchor_ori = None

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()
    goal: GoalCfg = GoalCfg()


@configclass
class EventCfg:
    torso_com = EventTerm(
        func=sm.randomize_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    encoder_bias = EventTerm(
        func=sm.randomize_encoder_bias,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    )
    foot_friction = EventTerm(
        func=sm.randomize_foot_friction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES), "friction_range": (0.3, 1.2)},
    )


@configclass
class RewardsCfg:
    motion_root_pos = RewTerm(func=sm.motion_root_pos, weight=R.w_root_pos, params={"command_name": "motion", "std": R.std_root_pos})
    motion_root_ori = RewTerm(func=sm.motion_root_ori, weight=R.w_root_ori, params={"command_name": "motion", "std": R.std_root_ori})
    motion_body_pos = RewTerm(func=sm.motion_body_pos, weight=R.w_body_pos, params={"command_name": "motion", "std": R.std_body_pos})
    motion_body_ori = RewTerm(func=sm.motion_body_ori, weight=R.w_body_ori, params={"command_name": "motion", "std": R.std_body_ori})
    motion_body_lin_vel = RewTerm(
        func=sm.motion_body_lin_vel, weight=R.w_body_lin_vel, params={"command_name": "motion", "std": R.std_body_lin_vel}
    )
    motion_body_ang_vel = RewTerm(
        func=sm.motion_body_ang_vel, weight=R.w_body_ang_vel, params={"command_name": "motion", "std": R.std_body_ang_vel}
    )
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=R.w_action_rate)
    joint_limit = RewTerm(
        func=base_mdp.joint_pos_limits, weight=R.w_joint_limit, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    clip_end = DoneTerm(func=sm.clip_end, params={"command_name": "motion"}, time_out=True)
    root_height = DoneTerm(func=sm.bad_root_height, params={"command_name": "motion", "threshold": 0.25})
    root_tilt = DoneTerm(func=sm.bad_root_tilt, params={"command_name": "motion", "threshold": 0.8})


@configclass
class H1_2SuperMimicEnvCfg(ManagerBasedRLEnvCfg):
    scene: SuperMimicSceneCfg = SuperMimicSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    anchor_obs: bool = False
    recipe: str = "v0"  # training recipe, see apply_recipe in mdp/trainer.py

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

    def enable_anchor_obs(self) -> None:
        """Appends the reference anchor pose in the robot anchor frame to the goal group (9 dims, BeyondMimic motion_anchor_pos_b/ori_b)."""
        self.anchor_obs = True
        self.observations.goal.anchor_pos = ObsTerm(
            func=sm.goal_anchor_pos, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        self.observations.goal.anchor_ori = ObsTerm(
            func=sm.goal_anchor_ori, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )

    def enable_play_viz(self) -> None:
        """For viewing with a GUI (play / evaluation without --headless): adds the reference ghost; the viewer uses origin_type world and sm.FollowCamera only translates to follow
        (asset_root resets the camera every frame, so the mouse cannot orbit). The ghost has no collisions and is not part of observations / rewards."""
        self.scene.ghost = ghost_cfg(self.scene.robot)
        self.sim.render.enable_translucency = True  # the default render mode disables translucency, so a ghost with opacity < 1 would be invisible
        self.viewer.origin_type = "world"

    def make_eval(self) -> None:
        """Appendix F: every clip played whole from its first frame; no observation / action / reset noise, no domain randomisation (foot friction 1.0)."""
        self.episode_length_s = 1.0e5  # the clip end decides
        self.observations.policy.enable_corruption = False
        self.observations.goal.enable_corruption = False
        self.actions.joint_pos.noise_std = 0.0
        m = self.commands.motion
        m.pose_range = {}
        m.velocity_range = {}
        m.joint_position_range = (0.0, 0.0)
        m.sampling_mode = "start"
        self.events.torso_com = None
        self.events.encoder_bias = None
        self.events.foot_friction = EventTerm(
            func=sm.randomize_foot_friction,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES), "friction_range": (1.0, 1.0)},
        )


@configclass
class H1_2SuperMimicAnchorEnvCfg(H1_2SuperMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.enable_anchor_obs()


@configclass
class H1_2SuperMimicEvalEnvCfg(H1_2SuperMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 40
        self.make_eval()


@configclass
class H1_2SuperMimicAnchorEvalEnvCfg(H1_2SuperMimicEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.enable_anchor_obs()


# v1: v0 + the parts of supermimic_vibed (github.com/zita-ch/supermimic_vibed, Isaac Lab 3 / G1, 0.829 on LAFAN1 after 1.47 h, no failures on all 40 clips) that worked better than ours.
# Robot, rewards, termination thresholds, domain randomisation and data are the same as v0; v0 and v1 differ only in:
#   observation  + 6D reference root orientation relative to the robot root (PolicyV1Cfg): proprioception alone gives no yaw feedback, so the heading drifts over a clip (82 deg on average in SuperMimic-Reproduced's diagnosis)
#                and all world-frame velocity rewards are lost; vibed reached 0.82 by adding only the orientation, not the position
#   goal         trajectory style (goal_mode="traj"): displacement / orientation of the next 10 frames relative to the current reference frame + root velocities + joints (the paper does not specify the per-frame content)
#   training     500 world-model-only updates at the start; imagination starts never cross the clip end; analytic soft termination instead of the termination head; tracking errors in the critic
#                (these four are in trainer.apply_recipe / imagination.PolicyTrainerCfg)
# Not adopted: running-mean normalisation, bf16, BeyondMimic low-gain actuators (no trustworthy armature values for H1-2), Newton physics.
@configclass
class H1_2SuperMimicV1EnvCfg(H1_2SuperMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.recipe = "v1"
        self.observations.policy = PolicyV1Cfg()
        self.commands.motion.goal_mode = "traj"


@configclass
class H1_2SuperMimicV1EvalEnvCfg(H1_2SuperMimicV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 40
        self.make_eval()
