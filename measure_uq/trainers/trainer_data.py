"""
Provides a class for storing and managing data associated with a trainer.

The :class:`TrainerData` class is used by the :class:`Trainer` to store and
manage data associated with the trainer. This includes the physics-informed
neural network (PINN) model, optimizer, and learning rate scheduler,
as well as the partial differential equation (PDE) that the model should
satisfy.

The :class:`TrainerData` class also stores the current iteration number and
loss values for the training and testing phases.

"""
from dataclasses import dataclass, field

import torch

from measure_uq.pde import PDE
from measure_uq.utilities import LossContainer


@dataclass(kw_only=True)
class TrainerData:
    """
    Stores and manages data associated with a trainer.

    Parameters
    ----------
    pde : PDE
        The partial differential equation (PDE) that the model should satisfy.
    iterations : int
        The current iteration number.
    model : torch.nn.Module
        The physics-informed neural network (PINN) model.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters.
    scheduler : torch.optim.lr_scheduler._LRScheduler | None, optional
        The learning rate scheduler used to update the learning rate.
        Defaults to None.
    """

    pde: PDE

    iterations: int

    model: torch.nn.Module

    optimizer: torch.optim.Optimizer

    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

    iteration: int = field(init=False, repr=True, default=0)

    test_every: int | None = 100

    losses_train: LossContainer = field(
        init=False,
        repr=True,
        default_factory=LossContainer,
    )

    losses_test: LossContainer = field(
        init=False,
        repr=True,
        default_factory=LossContainer,
    )
