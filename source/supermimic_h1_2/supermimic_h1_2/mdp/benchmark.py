"""Whole-clip evaluation of paper Appendix F: every clip is played once, deterministically, from its first frame; no observation / action / reset noise and no domain randomisation (foot friction 1.0).
Failure = a training termination (root height / tilt) fires. Normalised return = sum of per-step rewards / (0.1 x steps); 1.0 is perfect tracking.
Uses Tracking-Flat-H1_2-SuperMimic-Eval-v0 (h1_2_env_cfg.H1_2SuperMimicEvalEnvCfg) or the matching Eval variant.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .rewards import REWARD


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if the clip did not fail
    terms: dict  # per-step mean of each reward term (already multiplied by weight and dt)


def eval_task_for(ck: dict) -> str:
    """Training task X-vN stored in the checkpoint -> X-Eval-vN; early v0 checkpoints without a task are mapped by whether they use anchor observations."""
    if ck.get("task"):
        base, version = ck["task"].rsplit("-", 1)
        return f"{base}-Eval-{version}"
    if ck["config"]["policy"]["anchor_obs"]:
        return "Tracking-Flat-H1_2-SuperMimic-Anchor-Eval-v0"
    return "Tracking-Flat-H1_2-SuperMimic-Eval-v0"


@torch.no_grad()
def run_benchmark(env, policy, verbose: bool = True, on_step=None) -> list[ClipResult]:
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    n_env, n_clips = env.num_envs, lib.num_clips
    dev = env.device
    # queue: env e plays clips e, e+N, e+2N, ...; forced_frame is "the frame the next reset starts from"
    current = torch.arange(n_env, device=dev)
    nxt = current + n_env
    cmd.forced_frame[:] = lib.clip_start[current.clamp(max=n_clips - 1)]
    obs, _ = env.reset()
    cmd.forced_frame[:] = lib.clip_start[nxt.clamp(max=n_clips - 1)]
    active = current < n_clips
    ret = torch.zeros(n_env, device=dev)
    steps = torch.zeros(n_env, dtype=torch.long, device=dev)
    names = list(env.reward_manager.active_terms)
    term_sum = torch.zeros(n_env, len(names), device=dev)
    results: dict[int, ClipResult] = {}
    while len(results) < n_clips:
        obs, rew, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
        if on_step is not None:
            on_step()
        ret += rew
        term_sum += env.reward_manager._step_reward * env.step_dt
        steps += 1
        done = (term | tout) & active
        if not bool(done.any()):
            continue
        ids = done.nonzero().squeeze(-1)
        for e in ids.tolist():
            ci = int(current[e])
            st = int(steps[e])
            results[ci] = ClipResult(
                name=lib.clips[ci].name,
                frames=lib.clips[ci].length,
                steps=st,
                norm_return=float(ret[e]) / (REWARD.max_step_reward * max(st, 1)),
                failed=bool(term[e]),
                fail_step=st if bool(term[e]) else -1,
                terms={k: float(term_sum[e, i]) / max(st, 1) for i, k in enumerate(names)},
            )
            if verbose and (len(results) % 20 == 0 or len(results) == n_clips):
                print(f"  evaluated {len(results)}/{n_clips}", flush=True)
        current[ids] = nxt[ids]
        nxt[ids] += n_env
        active[ids] = current[ids] < n_clips
        cmd.forced_frame[ids] = lib.clip_start[nxt[ids].clamp(max=n_clips - 1)]
        ret[ids] = 0.0
        term_sum[ids] = 0.0
        steps[ids] = 0
    return [results[i] for i in range(n_clips)]


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    tot = sum(r.steps for r in results)
    out = {
        "clips": n,
        "failed": sum(r.failed for r in results),
        "mean_norm_return": sum(r.norm_return for r in results) / max(n, 1),
        "step_weighted_norm_return": sum(r.norm_return * r.steps for r in results) / max(tot, 1),
    }
    if results:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: Path) -> None:
    path.write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
