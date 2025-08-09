"""Distributed training utilities using PyTorch DDP.

This module centralises setup and teardown logic for distributed
training.  It automatically initialises the process group, assigns the
correct CUDA device to each rank and provides helpers to wrap models in
``DistributedDataParallel``.  The functions are intentionally lightweight
so they can be reused across CLI entry points and unit tests.
"""
from __future__ import annotations

import os
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed(devices: Iterable[int] | None = None, backend: str = "nccl") -> Tuple[int, int]:
    """Initialise torch distributed processing.

    Parameters
    ----------
    devices:
        Optional iterable of GPU ids to use.  When ``None`` the function
        relies on ``torchrun``/``torch.distributed`` environment
        variables.
    backend:
        Backend to use for ``init_process_group``.

    Returns
    -------
    Tuple[int, int]
        ``(rank, world_size)`` of the current process.
    """
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    if devices is not None and len(devices) > 0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(d) for d in devices))
        world_size = len(devices)
        rank = int(os.environ.get("RANK", 0))
    else:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


def get_device(rank: int) -> torch.device:
    """Return device for ``rank`` respecting available CUDA devices."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def wrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap ``model`` in :class:`~torch.nn.parallel.DistributedDataParallel` if needed."""
    if dist.is_initialized():
        device = get_device(dist.get_rank())
        model.to(device)
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    return model


def cleanup() -> None:
    """Destroy the process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return ``True`` if this rank is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0
