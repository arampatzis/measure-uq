"""
Utilities for the trainer module.

This module provides functions for creating optimizers and schedulers for
training models.
"""

from typing import Any

import torch
from torch import nn, optim

from measure_uq.utilities import DeviceLikeType


def is_model_fully_on_device(
    model: torch.nn.Module,
    device: DeviceLikeType,
) -> bool:
    """
    Check if all parameters and buffers of a model are on the specified device.

    Parameters
    ----------
    model : nn.Module
        The model to check.
    device : DeviceLikeType
        The device to check against.

    Returns
    -------
    bool
        True if all parameters and buffers are on the given device, False otherwise.
    """
    target_device = torch.device(device)
    for t in list(model.parameters()) + list(model.buffers()):
        if t.device != target_device:
            return False
    return True


def are_tensors_on_device(
    tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    device: DeviceLikeType,
) -> bool:
    """
    Check if a tensor or a collection of tensors are on the specified device.

    Parameters
    ----------
    tensors : torch.Tensor or list/tuple of torch.Tensor
        Tensor or collection of tensors to check.
    device : DeviceLikeType
        Device to check against.

    Returns
    -------
    bool
        True if all tensors are on the given device, False otherwise.
    """
    target_device = torch.device(device)

    if isinstance(tensors, torch.Tensor):
        return tensors.device == target_device

    return all(t.device == target_device for t in tensors)


def get_optimizer(
    otype: str,
    model: nn.Module,
    learning_rate: float,
) -> optim.Optimizer:
    """
    Create an optimizer based on the given type and parameters.

    Parameters
    ----------
    otype : str
        The type of optimizer to create.
    model : nn.Module
        The model whose parameters should be optimized.
    learning_rate : float
        The initial learning rate for the optimizer.

    Returns
    -------
    optim.Optimizer
        The created optimizer.
    """
    match otype.lower():
        case "adam":
            return optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )
        case _:
            print("Unknown optimizer type, using Adam.")
            return optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )


def get_scheduler(
    stype: str,
    optimizer: optim.Optimizer,
    **kwargs: Any,
) -> optim.lr_scheduler.LRScheduler:
    """
    Create a learning rate scheduler based on the given type and parameters.

    Parameters
    ----------
    stype : str
        The type of scheduler to create. Can be 'step', 'multi', or None.
    optimizer : optim.Optimizer
        The optimizer whose learning rate will be scheduled.
    **kwargs
        Additional keyword arguments for the scheduler.

    Returns
    -------
    optim.lr_scheduler.LRScheduler
        The created learning rate scheduler.

    Notes
    -----
    If an unknown scheduler type is provided, a default StepLR scheduler
    with `step_size=1_000_000` and `gamma=1` is used.
    """
    match stype:
        case None:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1_000_000,
                gamma=1,
            )
        case "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                **kwargs,
            )
        case "multi":
            return torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer,
                **kwargs,
            )
        case _:
            print(
                "Unknown scheduler type."
                "Using step scheduler with step_size=1_000_000 and gamma=1.",
            )
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1_000_000,
                gamma=1,
            )
