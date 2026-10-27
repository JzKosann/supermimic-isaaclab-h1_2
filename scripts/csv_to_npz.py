"""LAFAN1 H1-2 retargeted CSV (30 fps) -> motion-library npz (50 Hz). Each frame is written into the robot in Isaac Lab and read back (FK replay).

Each CSV row: root position xyz, root quaternion xyzw, 27 joint angles (order in CSV_JOINT_NAMES, i.e. the H1_2 order of the dataset README).
Root position / joints are linearly interpolated and the root orientation slerped to 50 Hz; velocities by central differences.

Usage (from the repository root, after scripts/download_assets.py):
    python scripts/csv_to_npz.py --headless
    python scripts/csv_to_npz.py --headless --csv_dir <CSV directory> --output_dir <output directory>

One <name>.npz per CSV:
    fps            : [50]
    joint_pos      : (T, 27)    Isaac Lab joint order
    joint_vel      : (T, 27)
    body_pos_w     : (T, B, 3)
    body_quat_w    : (T, B, 4)  wxyz
    body_lin_vel_w : (T, B, 3)
    body_ang_vel_w : (T, B, 3)
    joint_names    : Isaac Lab joint order
    body_names     : Isaac Lab body order
Training only uses the root (pelvis) pose / angular velocity and the joint positions / velocities; all other body quantities are recomputed by the motion library with URDF FK.
"""

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from supermimic_h1_2.paths import H1_2_USD, LAFAN1_CSV_DIR, MOTION_DIR

parser = argparse.ArgumentParser(description="LAFAN1 H1-2 CSV → NPZ (Isaac Lab FK)")
parser.add_argument("--csv_dir", default=str(LAFAN1_CSV_DIR), help="CSV directory (default: datasets/lafan1_csv/h1_2)")
parser.add_argument("--output_dir", default=MOTION_DIR, help="output directory (default: datasets/lafan1_h1_2_npz)")
parser.add_argument("--input_fps", type=int, default=30)
parser.add_argument("--output_fps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

# kinematic replay only: gravity has no effect (the state is written every frame) and the joints apply no torque
REPLAY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=H1_2_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05)),
    actuators={
        "all": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0, effort_limit=500.0, velocity_limit=100.0),
    },
)

# joint order from CSV column 7 on (H1_2 in "The Order of Configuration" of the dataset README)
CSV_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))
    robot: ArticulationCfg = REPLAY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    """Loads one CSV, interpolates it to output_fps and computes velocities."""

    def __init__(self, csv_path: str, input_fps: int, output_fps: int, device: str):
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self._load(csv_path)
        self._interpolate()
        self._compute_velocities()

    def _load(self, path: str):
        data = torch.from_numpy(np.loadtxt(path, delimiter=",")).float().to(self.device)
        self.root_pos_input = data[:, 0:3]
        self.root_rot_input = data[:, 3:7][:, [3, 0, 1, 2]]  # xyzw → wxyz
        self.dof_pos_input = data[:, 7:]
        self.T_input = data.shape[0]
        self.duration = (self.T_input - 1) * self.input_dt

    def _interpolate(self):
        t_out = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.T_output = t_out.shape[0]
        phase = t_out / self.duration
        i0 = (phase * (self.T_input - 1)).floor().long()
        i1 = torch.clamp(i0 + 1, max=self.T_input - 1)
        blend = phase * (self.T_input - 1) - i0
        w = blend.unsqueeze(1)
        self.root_pos = self.root_pos_input[i0] * (1 - w) + self.root_pos_input[i1] * w
        self.dof_pos = self.dof_pos_input[i0] * (1 - w) + self.dof_pos_input[i1] * w
        self.root_rot = torch.stack(
            [quat_slerp(self.root_rot_input[a], self.root_rot_input[b], c) for a, b, c in zip(i0, i1, blend)]
        )

    def _compute_velocities(self):
        dt = self.output_dt
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=dt, dim=0)[0]
        self.dof_vel = torch.gradient(self.dof_pos, spacing=dt, dim=0)[0]
        q_rel = quat_mul(self.root_rot[2:], quat_conjugate(self.root_rot[:-2]))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        self.root_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_frame(self, idx):
        s = slice(idx, idx + 1)
        return self.root_pos[s], self.root_rot[s], self.root_lin_vel[s], self.root_ang_vel[s], self.dof_pos[s], self.dof_vel[s]


def apply_state(robot, scene, frame, csv_to_isaac):
    root_pos, root_rot, root_lv, root_av, dof_pos, dof_vel = frame
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = root_pos
    root_state[:, :2] += scene.env_origins[:, :2]
    root_state[:, 3:7] = root_rot
    root_state[:, 7:10] = root_lv
    root_state[:, 10:13] = root_av
    robot.write_root_state_to_sim(root_state)
    jp = robot.data.default_joint_pos.clone()
    jv = robot.data.default_joint_vel.clone()
    jp[:, csv_to_isaac] = dof_pos
    jv[:, csv_to_isaac] = dof_vel
    robot.write_joint_state_to_sim(jp, jv)


def run(sim: SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    csv_to_isaac = robot.find_joints(CSV_JOINT_NAMES, preserve_order=True)[0]
    joint_names, body_names = list(robot.joint_names), list(robot.body_names)
    print(f"[INFO] joints={len(joint_names)}, bodies={len(body_names)}")

    csv_files = sorted(Path(args_cli.csv_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no CSV in {args_cli.csv_dir}; run scripts/download_assets.py first")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(csv_files)} files → {output_dir}")

    ok = 0
    for fi, csv_path in enumerate(csv_files):
        print(f"[{fi + 1:3d}/{len(csv_files)}] {csv_path.name} ...", end=" ", flush=True)
        motion = MotionLoader(str(csv_path), args_cli.input_fps, args_cli.output_fps, sim.device)
        T = motion.T_output
        log = {
            "joint_pos": np.empty((T, len(joint_names)), dtype=np.float32),
            "joint_vel": np.empty((T, len(joint_names)), dtype=np.float32),
            "body_pos_w": np.empty((T, len(body_names), 3), dtype=np.float32),
            "body_quat_w": np.empty((T, len(body_names), 4), dtype=np.float32),
            "body_lin_vel_w": np.empty((T, len(body_names), 3), dtype=np.float32),
            "body_ang_vel_w": np.empty((T, len(body_names), 3), dtype=np.float32),
        }
        apply_state(robot, scene, motion.get_frame(0), csv_to_isaac)  # place the robot at the first frame before recording
        sim.render()
        scene.update(sim.get_physics_dt())
        for t in range(T):
            apply_state(robot, scene, motion.get_frame(t), csv_to_isaac)
            sim.render()
            scene.update(sim.get_physics_dt())
            d = robot.data
            log["joint_pos"][t] = d.joint_pos[0].cpu().numpy()
            log["joint_vel"][t] = d.joint_vel[0].cpu().numpy()
            log["body_pos_w"][t] = d.body_pos_w[0].cpu().numpy()
            log["body_quat_w"][t] = d.body_quat_w[0].cpu().numpy()
            log["body_lin_vel_w"][t] = d.body_lin_vel_w[0].cpu().numpy()
            log["body_ang_vel_w"][t] = d.body_ang_vel_w[0].cpu().numpy()
        if not simulation_app.is_running():
            break
        out_path = output_dir / (csv_path.stem + ".npz")
        np.savez(
            str(out_path),
            fps=np.array([args_cli.output_fps], dtype=np.float32),
            joint_names=np.array(joint_names),
            body_names=np.array(body_names),
            **log,
        )
        print(f"OK  T={T}", flush=True)
        ok += 1
    print(f"\nDone: {ok}/{len(csv_files)}", flush=True)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run(sim, scene)
    # without clearing the SimulationContext first, simulation_app.close() hangs busy-looping (observed on Isaac Sim 5.1)
    sim.clear_all_callbacks()
    sim.clear_instance()


if __name__ == "__main__":
    main()
    simulation_app.close()
