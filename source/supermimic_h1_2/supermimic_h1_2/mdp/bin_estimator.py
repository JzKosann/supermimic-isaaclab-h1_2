"""Per-bin (1 s) return estimates for return-based reset sampling. Pure torch, no Isaac dependency.

The original estimate ("ema", SuperMimic-Reproduced plan D14, used by v0 / v1) has two biases:
    1. an episode that plays to the clip end has its return cut off there, so the last 1 s of every clip always has the lowest
       return and looks "hardest" -- with softmax sampling most resets end up in the last second of a clip;
    2. it starts from the maximum and moves 5 % per visit, so rarely visited bins keep looking "easy".
The same two biases are reported in realmimic's README (github.com/kevinhenryzz/realmimic, Apache-2.0); "count" follows the
algorithm described there:
    - episodes that did not end in a failure (clip end, time-out) get a continuation value from the reward rate of their last steps;
    - bins never visited count as the worst (0);
    - the first 1 / ema_alpha visits use a cumulative mean, then it turns into an EMA.
"""

from __future__ import annotations

import torch


def initial_return(estimator: str, g_max: float) -> float:
    """Initial return of an unvisited bin: the worst for "count", the best for "ema"."""
    return 0.0 if estimator == "count" else g_max


def episode_returns_to_go(
    r: torch.Tensor,
    lens: torch.Tensor,
    terminated: torch.Tensor,
    clip_end: torch.Tensor,
    gamma_pow: torch.Tensor,
    gamma: float,
    g_max: float,
    estimator: str,
    bootstrap_steps: int = 50,
    min_remaining_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """r (n,T) per-step rewards, lens (n,) episode lengths -> discounted return-to-go of each step (n,T) and the valid-step mask (n,T)."""
    T = r.shape[1]
    steps = torch.arange(T, device=r.device)[None, :]
    tmask = steps < lens[:, None]
    r = r * tmask
    rtg = torch.flip(torch.cumsum(torch.flip(r * gamma_pow[None, :T], [1]), 1), [1]) / gamma_pow[None, :T]
    remaining = lens[:, None] - steps
    if estimator == "count":
        cut = ~terminated
        if cut.any():
            tail = tmask & (steps >= lens[:, None] - bootstrap_steps)
            rate = (r * tail).sum(1) / tail.sum(1).clamp_min(1)
            v = (rate / (1.0 - gamma)).clamp(0.0, g_max)
            rtg = rtg + cut[:, None] * (gamma ** remaining.clamp_min(0).double()) * v[:, None]
    elif estimator == "ema":
        # time-out truncation (not the clip end): the last steps have no future, leave them out
        truncated = ~terminated & ~clip_end
        tmask = tmask & ~(truncated[:, None] & (remaining < min_remaining_steps))
    else:
        raise ValueError(f"bin_estimator must be count or ema, got {estimator!r}")
    return rtg, tmask


def fold_into_bins(
    bin_return: torch.Tensor,
    bin_count: torch.Tensor,
    bins: torch.Tensor,
    rtg: torch.Tensor,
    tmask: torch.Tensor,
    estimator: str,
    ema_alpha: float,
    g_max: float,
) -> None:
    """Averages the per-step returns of a batch of finished episodes per bin and writes them into the return table (in place)."""
    valid = tmask.reshape(-1)
    b = bins.reshape(-1)[valid]
    g = rtg.reshape(-1)[valid].float()
    if b.numel() == 0:
        return
    sums = torch.zeros_like(bin_return).index_add_(0, b, g)
    cnts = torch.zeros_like(bin_return).index_add_(0, b, torch.ones_like(g))
    touched = cnts > 0
    mean = sums[touched] / cnts[touched]
    if estimator == "count":
        mean = mean.clamp(0.0, g_max)
        alpha = (1.0 / (bin_count[touched] + 1.0)).clamp_min(ema_alpha)
        bin_count[touched] += 1.0
    else:
        alpha = ema_alpha
    bin_return[touched] = (1 - alpha) * bin_return[touched] + alpha * mean
