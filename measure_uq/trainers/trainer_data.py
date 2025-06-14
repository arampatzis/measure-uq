"""
Trainer data module.

This module provides the data structures for storing and managing training-related data.

This module provides:
    - TrainerData class for storing and managing training data, including:
        - Model, optimizer and scheduler storage
        - Training and testing loss tracking
        - Iteration counting and test frequency management
        - PDE problem definition storage
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LRScheduler

from measure_uq.models import ModelWithCombinedInput
from measure_uq.pde import PDE
from measure_uq.utilities import DeviceLikeType, SparseDynamicArray


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

    Attributes
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
    iteration : int
        The current iteration number, automatically created by the Trainer.
    losses_train : SparseDynamicArray
        The loss values for the training phase, automatically created by the
        TrainerData.
    losses_test : SparseDynamicArray
        The loss values for the testing phase, automatically created by the TrainerData.
    """

    pde: PDE

    iterations: int

    model: ModelWithCombinedInput

    optimizer: torch.optim.Optimizer

    scheduler: LRScheduler | None = None

    test_every: int = 100

    device: DeviceLikeType = "cpu"

    iteration: int = field(init=False, repr=True, default=0)

    save_path: str | Path = field(
        repr=True,
        default="data/best_model.pickle",
    )

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

    best_test_loss: float = field(init=False, repr=True, default=float("inf"))

    def __post_init__(self) -> None:
        """Post-initialization method to move the model and optimizer to the device."""
        self.model.to(self.device)
