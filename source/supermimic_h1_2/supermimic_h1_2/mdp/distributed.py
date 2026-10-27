"""Data parallelism (SuperMimic-Reproduced plan D13): one Isaac instance per GPU, each with its own num_envs environments and replay buffer;
every rank keeps its own copy of the networks and all-reduces (averages) gradients explicitly after backward (no DDP wrapper, so torch.compile'd functions are unaffected).
Learning rate x sqrt(world_size) (paper: 2 GPUs x sqrt(2)). Only tensor collectives are used (on Blackwell,
broadcast_object_list caused an illegal memory access).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed(local_rank: int) -> tuple[int, int]:
    """Initialise NCCL when launched by torchrun; returns (rank, world_size)."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return 0, 1
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size()


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


@torch.no_grad()
def allreduce_grads(module: torch.nn.Module) -> None:
    if world_size() == 1:
        return
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size()
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n


@torch.no_grad()
def broadcast_module(module: torch.nn.Module, src: int = 0) -> None:
    if world_size() == 1:
        return
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=src)


def allreduce_sum(values: list[float], device) -> list[float]:
    if world_size() == 1:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def finalize() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
