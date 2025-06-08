"""
Defines classes for implementing stopping criteria in training.

This module provides:
    - Stopper class for defining a stopping criterion
    - StopperList class for managing a list of stoppers
    - TrainingLossStopper class for stopping training based on the improvement of
    training loss
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from measure_uq.trainers.trainer_data import TrainerData


@dataclass(kw_only=True)
class Stopper(ABC):
    """
    Abstract base class for defining a stopping criterion.

    Attributes
    ----------
    trainer_data : TrainerData
        The trainer data instance containing the training losses and other relevant
        information.
    """

    @abstractmethod
    def should_stop(self, trainer_data: TrainerData) -> bool:
        """
        Determine whether training should be stopped.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.

        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class Stoppers:
    """
    A list of stoppers responsible for determining when to halt training.

    Attributes
    ----------
    stoppers : list of Stopper
        A list containing instances of Stopper.
    """

    trainer_data: TrainerData

    stoppers: list[Stopper]

    def should_stop(self) -> bool:
        """
        Check if any stopper in the list signals to stop training.

        Returns
        -------
        bool
            True if any stopper indicates to stop, False otherwise.
        """
        return any(stopper.should_stop(self.trainer_data) for stopper in self.stoppers)


@dataclass(kw_only=True)
class TrainingLossStopper(Stopper):
    """
    Stopper that halts training based on the improvement of training loss.

    Attributes
    ----------
    patience : int
        Number of epochs to wait for an improvement in loss before stopping.
    delta : float
        Minimum change in the monitored loss to qualify as an improvement.
    best_loss : float or None
        The best observed loss value.
    counter : int
        Counts the number of epochs with no improvement.
    """

    patience: int = 5
    delta: float = 0.0
    best_loss: float | None = None
    counter: int = 0

    def should_stop(self, trainer_data: TrainerData) -> bool:
        """
        Determine whether training should be stopped based on the training loss.

        This method checks if the training loss has improved by at least `delta` within
        the last `patience` iterations. If there is no sufficient improvement for
        `patience` consecutive iterations, it signals to stop training.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing training loss history and other information.

        Returns
        -------
        bool
            True if training should be stopped due to lack of improvement in loss,
            False if training should continue.

        Notes
        -----
        The method handles different loss value types (list, tuple, numpy array, scalar)
        by converting them to float. The best loss value and counter are updated
        internally to track progress.
        """
        loss_value = trainer_data.losses_train.v[-1]

        if isinstance(loss_value, list | tuple):
            loss = float(loss_value[0])
        elif isinstance(loss_value, np.ndarray):
            loss = float(loss_value.item())
        else:
            loss = float(loss_value)

        if self.best_loss is None or loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("TrainingLossStopper: early stop triggered")
                return True
        return False
