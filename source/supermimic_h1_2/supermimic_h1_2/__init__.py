"""Unofficial reproduction of SuperMimic (arXiv:2610.28480) on Isaac Lab / Unitree H1-2: whole-body motion tracking with a differentiable world model + short-horizon BPTT.

    supermimic_h1_2.tasks   registers the gym tasks on import (import it after AppLauncher has started)
    supermimic_h1_2.mdp     environment terms and the training algorithm
    supermimic_h1_2.robots  H1-2 articulation config
    supermimic_h1_2.paths   default paths (overridable by environment variables)
"""

__version__ = "0.1.0"
