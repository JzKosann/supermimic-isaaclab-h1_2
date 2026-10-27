"""SuperMimic training loop (paper Algorithm A.1): one iteration = every env takes one control step (collection, with execution noise);
"updates" are accumulated at UTD 0.4: one update = 10 world-model gradient steps (2048 8-step windows per batch) + one policy BPTT step + critic fitting.
320k iterations in total (simulator steps, reading A of plan D12, ~128k updates); cosine learning-rate decay down to 0.03x at 250k/320k, constant afterwards.
No rsl_rl: the policy is a deterministic MLP and its gradient is the first-order back-propagation through the world model, not a likelihood-ratio estimate.

Recipes (chosen by the recipe field of the env cfg, see apply_recipe):
    v0  the paper + the choices of SuperMimic-Reproduced
    v1  v0 + the parts of supermimic_vibed that worked better: 500 world-model-only updates at the start, imagination starts never cross
        the clip end, analytic soft termination, tracking errors in the critic; the reference-orientation observation and the trajectory goal live in the env cfg (tasks/h1_2_env_cfg.py)
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .distributed import allreduce_sum, broadcast_module, world_size
from .events import get_sim_context
from .heads import AnalyticHeads
from .imagination import PolicyTrainer, PolicyTrainerCfg
from .policy import Critic, CriticCfg, Policy, PolicyCfg
from .replay import ReplayBuffer, ReplayCollector
from .rewards import REWARD
from .state import StateIO
from .world_model import WMTrainerCfg, WorldModel, WorldModelCfg, WorldModelTrainer


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000
    updates_per_env_step: float = 0.4
    warmup_iterations: int = 40  # fill k-step windows before the first update
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50
    checkpoint_interval: int = 4000
    max_hours: float | None = None
    wm_only_updates: int = 0  # number of initial updates that train only the world model (v1: 500, as vibed's wm_only_iters)


@dataclass
class TrainCfg:
    seed: int = 0
    replay_capacity: int = 1000
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)


def apply_recipe(cfg: TrainCfg, recipe: str) -> None:
    if recipe == "v0":
        return
    if recipe != "v1":
        raise ValueError(f"unknown recipe {recipe}")
    cfg.schedule.wm_only_updates = 500
    cfg.policy_trainer.start_within_clip = True
    cfg.policy_trainer.analytic_termination = True
    cfg.policy_trainer.critic_tracking_features = True


def cosine_lr(step: int, total: int, base: float, final_ratio: float, decay_fraction: float) -> float:
    decay = max(1, int(total * decay_fraction))
    if step >= decay:
        return base * final_ratio
    return base * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * step / decay)))


class Logger:
    """Always writes TensorBoard; with --logger wandb also writes to wandb. Instantiated on rank 0 only."""

    def __init__(self, log_dir: Path, use_wandb: bool, project: str, run_name: str, config: dict):
        from torch.utils.tensorboard import SummaryWriter

        self.tb = SummaryWriter(str(log_dir / "tb"))
        self.wandb = None
        if use_wandb:
            import wandb

            entity = os.environ.get("WANDB_USERNAME") or None  # same wandb usage as rsl_rl
            wandb.init(project=project, entity=entity, name=run_name, config=config, dir=str(log_dir))
            self.wandb = wandb

    def log(self, scalars: dict[str, float], step: int) -> None:
        for k, v in scalars.items():
            self.tb.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log(scalars, step=step)

    def close(self) -> None:
        self.tb.flush()
        if self.wandb is not None:
            self.wandb.finish()


def build_networks(cfg: TrainCfg, env, device):
    """Builds the world model / policy / critic with dimensions filled in from the environment (shared by train.py and evaluation)."""
    cmd = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    cfg.policy.num_joints = robot.num_joints
    cfg.policy.future_frames = cmd.cfg.future_frames
    cfg.policy.anchor_obs = bool(env.cfg.anchor_obs)
    cfg.policy.ref_ori_obs = getattr(env.cfg.observations.policy, "ref_root_ori", None) is not None
    cfg.policy.goal_mode = cmd.cfg.goal_mode
    heads = AnalyticHeads(
        cmd.fk, cmd.body_names, cmd.cfg.anchor_body_name,
        robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(),
    )  # fmt: skip
    wm = WorldModel(cfg.world_model, heads, device)
    policy = Policy(cfg.policy).to(device)
    extra_dim = 4 + (heads.nj + 3 * len(cmd.body_names) + 6 if cfg.policy_trainer.critic_tracking_features else 0)
    critic = Critic(cfg.critic, cfg.policy, wm.spec.feat_dim, wm.spec.xi_dim, extra_dim).to(device)
    obs_dims = env.observation_manager.group_obs_dim
    if obs_dims["policy"] != (cfg.policy.obs_dim,) or obs_dims["goal"] != (cfg.policy.goal_dim,):
        raise ValueError(f"env obs dims {obs_dims} != policy cfg ({cfg.policy.obs_dim}, {cfg.policy.goal_dim})")
    return heads, wm, policy, critic


class Trainer:
    def __init__(self, env, cfg: TrainCfg, log_dir: Path, rank: int = 0, logger: Logger | None = None, task: str = ""):
        self.env, self.cfg, self.log_dir, self.rank, self.logger = env, cfg, log_dir, rank, logger
        self.task = task
        self.is_main = rank == 0
        self.world = world_size()
        self.device = env.device
        if self.world > 1:  # paper: 2 GPUs -> learning rate x sqrt(2)
            for c in (cfg.wm_trainer, cfg.policy_trainer):
                c.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        self.cmd = env.command_manager.get_term("motion")
        self.lib = self.cmd.library
        self.heads, self.wm, self.policy, self.critic = build_networks(cfg, env, self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_module(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)

        spec = self.wm.spec
        nj = self.heads.nj
        self.buffer = ReplayBuffer(env.num_envs, cfg.replay_capacity, spec.state_dim, nj, spec.xi_dim, self.device)
        self.collector = ReplayCollector(
            env, self.buffer, StateIO(env), get_sim_context(env), self.cmd, env.action_manager.get_term("joint_pos")
        )
        env.pre_reset_hook = self.collector.on_pre_reset
        if self.is_main:
            n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6  # noqa: E731
            replay_gb = sum(t.numel() * t.element_size() for t in vars(self.buffer).values() if isinstance(t, torch.Tensor)) / 1e9
            print(
                f"[INFO] SuperMimic: world {self.world}, envs/rank {env.num_envs}, state {spec.state_dim}, xi {spec.xi_dim}, "
                f"obs {cfg.policy.obs_dim}, goal {cfg.policy.goal_dim} | params WM {n(self.wm):.2f}M, policy {n(self.policy):.2f}M, "
                f"critic {n(self.critic):.3f}M | replay {replay_gb:.2f} GB"
            )
        self.iteration = 0
        self.num_updates = 0
        self._utd_acc = 0.0
        self._last_lr = cfg.wm_trainer.lr

    # ------------------------------------------------------------------ checkpoint

    def save(self, name: str) -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.cmd.bin_return,
                "bin_memory": self.cmd.bin_memory,
                "config": asdict(self.cfg),
                "task": self.task,
                "recipe": self.env.cfg.recipe,
                "joint_names": list(self.env.scene["robot"].joint_names),
            },
            self.log_dir / name,
        )

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        if ck["bin_return"].shape == self.cmd.bin_return.shape:
            self.cmd.bin_return.copy_(ck["bin_return"])
            self.cmd.bin_memory.copy_(ck["bin_memory"])
        self.iteration = int(ck["iteration"])
        self.num_updates = int(ck["num_updates"])
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    def run(self) -> None:
        cfg, sch, env = self.cfg, self.cfg.schedule, self.env
        obs, _ = env.reset()
        self.collector.start()
        ep_sum = torch.zeros(env.num_envs, device=self.device)
        ep_len = torch.zeros(env.num_envs, device=self.device)
        fin = [0.0, 0.0, 0.0]  # sum of normalised returns, number of episodes, sum of lengths (this log interval)
        agg: dict[str, float] = {}  # sums of each metric within this log interval
        agg_n: dict[str, int] = {}  # and their counts (no policy metrics during warm-up, so each metric is counted separately)
        env_log: dict[str, float] = {}
        env_log_n = 0
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        t_start = last_log = time.time()
        it0 = self.iteration
        base_lr = cfg.wm_trainer.lr
        pol_ratio = cfg.policy_trainer.lr / base_lr
        critic_ratio = cfg.policy_trainer.critic_lr / base_lr
        max_r = REWARD.max_step_reward

        while self.iteration < sch.total_iterations:
            # ---- (1) collect one control step
            t0 = time.perf_counter()
            with torch.no_grad():
                action = self.policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, extras = env.step(action)
            self.collector.after_step()
            ep_sum += rew
            ep_len += 1
            done = term | tout
            if bool(done.any()):
                ids = done.nonzero().squeeze(-1)
                fin[0] += float((ep_sum[ids] / (max_r * ep_len[ids].clamp_min(1))).sum())
                fin[1] += float(ids.numel())
                fin[2] += float(ep_len[ids].sum())
                ep_sum[ids] = 0.0
                ep_len[ids] = 0.0
                for k, v in extras.get("log", {}).items():
                    env_log[k] = env_log.get(k, 0.0) + float(v)
                env_log_n += 1
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---- (2)+(3) update the world model and the policy according to the UTD
            if self.iteration >= it0 + sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(self.iteration, sch.total_iterations, base_lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * pol_ratio, lr * critic_ratio)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    m_pi = {}
                    if self.num_updates >= sch.wm_only_updates:
                        t0 = time.perf_counter()
                        m_pi = self.policy_trainer.update(self.buffer)
                        torch.cuda.synchronize()
                        timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in [(f"wm/{k}", v) for k, v in m_wm.items()] + [(f"policy/{k}", v) for k, v in m_pi.items()]:
                        agg[k] = agg.get(k, 0.0) + v
                        agg_n[k] = agg_n.get(k, 0) + 1
                    self._last_lr = lr

            # ---- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                fin = allreduce_sum(fin, self.device)
                if self.is_main:
                    self._log(fin, agg, agg_n, env_log, env_log_n, timers, t_start, last_log)
                last_log = time.time()
                fin = [0.0, 0.0, 0.0]
                agg, agg_n, env_log, env_log_n = {}, {}, {}, 0
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = allreduce_sum([float((time.time() - t_start) / 3600 > sch.max_hours)], self.device)[0]
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")

    def _log(self, fin, agg, agg_n, env_log, env_log_n, timers, t_start, last_log) -> None:
        sch = self.cfg.schedule
        it = self.iteration
        elapsed = time.time() - t_start
        s: dict[str, float] = {k: v / agg_n[k] for k, v in agg.items()}
        if agg:
            s["lr"] = self._last_lr
        tr = fin[0] / fin[1] if fin[1] > 0 else float("nan")
        if fin[1] > 0:
            s["train/episode_norm_return"] = tr
            s["train/episode_len"] = fin[2] / fin[1]
        s.update({k: v / max(env_log_n, 1) for k, v in env_log.items()})
        err = None
        if self.buffer.size > 4096:
            try:
                err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                s["wm/online_nse_1step"] = float(err[0])
                s["wm/online_nse_8step"] = float(err[-1])
            except RuntimeError:
                err = None
        for k, v in timers.items():
            s[f"time/{k}_s_per_iter"] = v / sch.log_interval
        s["time/hours"] = elapsed / 3600
        s["time/updates"] = self.num_updates
        s["sampler/bin_return_mean"] = float(self.cmd.bin_return.mean())
        if self.logger is not None:
            self.logger.log(s, it)
        dt = max(time.time() - last_log, 1e-6)
        print(
            f"it {it} | {elapsed / 3600:.2f} h | {sch.log_interval / dt:.1f} it/s | upd {self.num_updates} | "
            f"train_ret {tr:.3f} (n={int(fin[1])}) | wm nse1 {s.get('wm/nse_1step', float('nan')):.4f} "
            f"nse8(online) {float(err[-1]) if err is not None else float('nan'):.4f} | "
            f"imag_ret {s.get('policy/imag_return_undisc', float('nan')):.3f} surv {s.get('policy/imag_survival_H', float('nan')):.3f} | "
            f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s",
            flush=True,
        )


def save_config(cfg: TrainCfg, log_dir: Path, extra: dict) -> None:
    (log_dir / "config.json").write_text(json.dumps({**extra, **asdict(cfg)}, indent=2, default=str))
