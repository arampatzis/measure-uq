"""
Defines classes for implementing stopping criteria in training.

Classes:
--------
Stopper : Abstract base class for defining a stopping criterion.
    Provides an interface to set the trainer and check if training should stop.

StopperList :
    A list of Stopper instances, responsible for collectively determining when to halt
    training.

TrainingLossStopper :
    A specific Stopper implementation that halts training based on the improvement of
    training loss.

Notes
-----
This module relies on the `Trainer` class from the `measure_uq.trainer` module.

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

    Parameters
    ----------
    trainer_data : TrainerData
        The trainer data instance containing the training losses and other relevant
        information.

    Methods
    -------
    should_stop() : bool
        Determine whether training should be stopped.

    Returns
    -------
        bool
            True if training should be stopped, False otherwise.
    """

    trainer_data: TrainerData

    @abstractmethod
    def should_stop(self) -> bool:
        """
        Determine whether training should be stopped.

        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class StopperList:
    """
    A list of stoppers responsible for determining when to halt training.

    Attributes
    ----------
    stoppers : list of Stopper
        A list containing instances of Stopper.
    """

    stoppers: list[Stopper]

    def should_stop(self) -> bool:
        """
        Check if any stopper in the list signals to stop training.

        Returns
        -------
        bool
            True if any stopper indicates to stop, False otherwise.
        """
        return any(stopper.should_stop() for stopper in self.stoppers)


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

    def should_stop(self) -> bool:
        """
        Determine whether training should be stopped based on the training loss.

        Parameters
        ----------
        trainer : measure_uq.trainer.Trainer
            The trainer instance containing the training loss history.

        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """
        loss_value = self.trainer_data.losses_train.v[-1]

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
                print(self.trainer_data.losses_train.v[-self.patience :])
                return True
        return False
