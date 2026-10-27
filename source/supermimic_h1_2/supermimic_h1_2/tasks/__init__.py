"""Registers the SuperMimic gym tasks on import. Training / evaluation scripts live in scripts/supermimic/ (no rsl_rl).

    Tracking-Flat-H1_2-SuperMimic-v0              paper observation (Table A.1)
    Tracking-Flat-H1_2-SuperMimic-Anchor-v0       + reference anchor pose relative to the robot (deviates from the paper)
    Tracking-Flat-H1_2-SuperMimic-v1              v0 + the improvements of supermimic_vibed (see the end of h1_2_env_cfg.py)
    ...-Eval-vN                                   matching Appendix-F evaluation envs (eval_benchmark.py / play.py pick one from the checkpoint)
"""

import gymnasium as gym

from . import h1_2_env_cfg

_TASKS = {
    "Tracking-Flat-H1_2-SuperMimic-v0": h1_2_env_cfg.H1_2SuperMimicEnvCfg,
    "Tracking-Flat-H1_2-SuperMimic-Anchor-v0": h1_2_env_cfg.H1_2SuperMimicAnchorEnvCfg,
    "Tracking-Flat-H1_2-SuperMimic-Eval-v0": h1_2_env_cfg.H1_2SuperMimicEvalEnvCfg,
    "Tracking-Flat-H1_2-SuperMimic-Anchor-Eval-v0": h1_2_env_cfg.H1_2SuperMimicAnchorEvalEnvCfg,
    "Tracking-Flat-H1_2-SuperMimic-v1": h1_2_env_cfg.H1_2SuperMimicV1EnvCfg,
    "Tracking-Flat-H1_2-SuperMimic-Eval-v1": h1_2_env_cfg.H1_2SuperMimicV1EvalEnvCfg,
}

for _id, _cfg in _TASKS.items():
    gym.register(
        id=_id,
        entry_point=f"{__name__}.h1_2_env_cfg:SuperMimicRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cfg},
    )
