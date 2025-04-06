"""
Trainer data module.

This module provides the data structures for storing and managing training-related data.

The module includes:
- TrainerData class for storing model, optimizer, scheduler, and training progress

Classes
-------
TrainerData
    A dataclass for storing and managing data associated with a trainer.
"""

from dataclasses import dataclass, field

import torch

from measure_uq.models import ModelWithCombinedInput
from measure_uq.pde import PDE
from measure_uq.utilities import SparseDynamicArray


@dataclass(kw_only=True)
class TrainerData:
    """
    A class for storing and managing data associated with a trainer.

    This class is used by the Trainer to store and manage data associated with the
    training process. It includes the physics-informed neural network (PINN) model,
    optimizer, and learning rate scheduler, as well as the partial differential
    equation (PDE) that the model should satisfy.

    The class also stores the current iteration number and loss values for the
    training and testing phases using SparseDynamicArray for memory efficiency.

    Parameters
    ----------
    pde : PDE
        The partial differential equation (PDE) that the model should satisfy.
    iterations : int
        The number of iterations to train the model.
    model : ModelWithCombinedInput
        The physics-informed neural network (PINN) model.
    optimizer : torch.optim.Optimizer
        The optimizer for training the model.
    scheduler : torch.optim.lr_scheduler._LRScheduler | None, optional
        The learning rate scheduler, by default None.
    test_every : int, optional
        The number of iterations between testing the model, by default 100.

    Attributes
    ----------
    iteration : int
        The current iteration number, initialized to 0.
    losses_train : SparseDynamicArray
        The loss values for the training phase, stored in a sparse array.
    losses_test : SparseDynamicArray
        The loss values for the testing phase, stored in a sparse array.
    """

    pde: PDE

    iterations: int

    model: ModelWithCombinedInput

    optimizer: torch.optim.Optimizer

    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

    iteration: int = field(init=False, repr=True, default=0)

    test_every: int = 100

    losses_train: SparseDynamicArray = field(
        init=False,
        repr=True,
        default_factory=lambda: SparseDynamicArray(shape=1000, dtype=float),
    )

    losses_test: SparseDynamicArray = field(
        init=False,
        repr=True,
        default_factory=lambda: SparseDynamicArray(shape=1000, dtype=float),
    )
