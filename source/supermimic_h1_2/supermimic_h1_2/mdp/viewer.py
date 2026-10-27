"""Viewing SuperMimic with a GUI (scripts/supermimic/play.py, eval_benchmark.py without --headless): spawning the reference ghost + camera control.

Isaac Lab's viewer.origin_type="asset_root" resets the camera to a fixed offset every frame, so the mouse cannot orbit; here origin_type="world"
(env_cfg.enable_play_viz) is used and each step only translates the camera position and target by the displacement of the followed robot root (height low-pass filtered); mouse orbit / zoom work as usual.
Keys: [ / ] switch the followed env, L toggles following. The clip name is printed when the followed env switches clips.
"""

from __future__ import annotations

import weakref

import numpy as np
from pxr import Usd, UsdGeom

from isaaclab.sim.schemas import modify_collision_properties
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import bind_visual_material, clone, make_uninstanceable


@clone
def spawn_ghost(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn function of the reference ghost. The visuals/collisions of the Unitree USD are instanceable, so collision_props / visual_material
    cannot be applied directly (collisions would remain and push the real robot away): spawn as a normal USD, un-instance, then disable collisions and bind a translucent material.
    The material is bound to every Xform / mesh: the visuals prims of the original USD carry strongerThanDescendants bindings, which override a binding on only the root or only the meshes."""
    prim = _spawn_from_usd_file(
        prim_path, cfg.usd_path, cfg.replace(collision_props=None, visual_material=None), translation, orientation
    )
    make_uninstanceable(prim_path)
    modify_collision_properties(prim_path, cfg.collision_props)
    material_path = f"{prim_path}/{cfg.visual_material_path}"
    cfg.visual_material.func(material_path, cfg.visual_material)
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Imageable):
            bind_visual_material(p.GetPath(), material_path)
    return prim


class FollowCamera:
    def __init__(self, env, env_index: int = 0, offset=(2.5, 2.5, 1.2)):
        import carb

        self.env = env
        self.robot = env.scene["robot"]
        cmd = env.command_manager.get_term("motion") if "motion" in env.command_manager.active_terms else None
        self.cmd = cmd if hasattr(cmd, "library") else None  # only the SuperMimic motion command can report clip names
        self.index = env_index % env.num_envs
        # follow only with a GUI, or when recording headless video (--enable_cameras, off-screen viewport); pure headless has no viewport, only clip names are printed.
        # has_rtx_sensors() cannot be used: it is True only if the scene has camera sensors, and False when recording video
        self.follow = env.sim.has_gui() or env.sim.render_mode.value >= env.sim.RenderMode.PARTIAL_RENDERING.value
        self._clip = -1
        self._prev = self._root()
        self._z = self._prev[2]  # low-pass filtered height
        env.sim.set_camera_view(eye=self._prev + np.asarray(offset), target=self._prev + np.array([0.0, 0.0, -0.2]))
        try:
            import omni.appwindow

            window = omni.appwindow.get_default_app_window()
        except ImportError:  # no window extension when rendering headless
            window = None
        if window is not None:
            self._input = carb.input.acquire_input_interface()
            self._sub = self._input.subscribe_to_keyboard_events(
                window.get_keyboard(), lambda event, *args, obj=weakref.proxy(self): obj._on_key(event)
            )
            print("[viewer] mouse: orbit / zoom freely; [ / ] switch the followed env, L toggles following", flush=True)

    def _root(self) -> np.ndarray:
        return self.robot.data.root_pos_w[self.index].cpu().numpy().astype(float)

    def _on_key(self, event) -> bool:
        import carb

        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        name = event.input.name
        if name in ("LEFT_BRACKET", "RIGHT_BRACKET"):
            self.index = (self.index + (1 if name == "RIGHT_BRACKET" else -1)) % self.env.num_envs
            self._clip = -1
        elif name == "L":
            self.follow = not self.follow
            print(f"[viewer] follow {'on' if self.follow else 'off'}", flush=True)
        return True

    def update(self) -> None:
        """Call after every env.step. When switching env, the camera is translated by the displacement between the two robots, keeping the current view angle.
        Height is low-pass filtered (time constant ~20 steps): pelvis bobbing while walking is filtered out, height changes on stairs / slopes are followed; teleports from resets / env switches (>1 m in one step) are jumped over directly."""
        p = self._root()
        jump = np.linalg.norm(p[:2] - self._prev[:2]) > 1.0
        z = p[2] if jump else self._z + 0.05 * (p[2] - self._z)
        d = np.array([p[0] - self._prev[0], p[1] - self._prev[1], z - self._z])
        self._prev, self._z = p, z
        if self.follow and np.any(d):
            from omni.kit.viewport.utility import get_active_viewport
            from omni.kit.viewport.utility.camera_state import ViewportCameraState

            cs = ViewportCameraState("/OmniverseKit_Persp", get_active_viewport())
            self.env.sim.set_camera_view(eye=np.array(cs.position_world) + d, target=np.array(cs.target_world) + d)
        if self.cmd is None:
            return
        lib = self.cmd.library
        clip = int(lib.frame_clip[self.cmd.frame[self.index]])
        if clip != self._clip:
            self._clip = clip
            print(f"[viewer] env {self.index}: {lib.clips[clip].name} ({lib.clips[clip].length} frames)", flush=True)
