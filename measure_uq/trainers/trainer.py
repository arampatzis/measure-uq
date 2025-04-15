"""
Trainer module.

This module provides the Trainer class for managing the training process of models
in the measure-uq package.

The Trainer class handles the training loop, including callbacks for monitoring and
logging, as well as stopping criteria to determine when training should end.

Classes
-------
Trainer
    A class to manage the training process with callbacks and stopping criteria.
"""

import pickle
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Self

import torch

from measure_uq.callbacks import Callback, CallbackList
from measure_uq.gradients import clear
from measure_uq.stoppers import Stopper, StopperList
from measure_uq.trainers.trainer_data import TrainerData

# ruff: noqa: S301


@dataclass(kw_only=True)
class Trainer:
    """
    A dataclass to manage the training process.

    This class initializes the training configuration using provided trainer data
    and optional callbacks and stoppers. The internal attributes `_callbacks` and
    `_stoppers` are set up during initialization and are not directly exposed.

    Parameters
    ----------
    trainer_data : TrainerData
        The data required for training, including model, optimizer, and training
        configurations.
    callbacks : list[Callback] or None, optional
        A list of callback objects to handle events during training. If None,
        no callbacks are used. Default is None.
    stoppers : list[Stopper] or None, optional
        A list of stopper objects to handle stopping criteria during training.
        If None, no stoppers are used. Default is None.

    Attributes
    ----------
    trainer_data : TrainerData
        The provided training data used during the training process.
    _callbacks : CallbackList
        An internal list of callbacks initialized during object creation.
    _stoppers : StopperList
        An internal list of stoppers initialized during object creation.
    """

    trainer_data: TrainerData

    _callbacks: CallbackList = field(init=False)

    callbacks: InitVar[list[Callback] | None] = None

    _stoppers: StopperList = field(init=False)

    stoppers: InitVar[list[Stopper] | None] = None

    def __post_init__(
        self,
        callbacks: list[Callback] | None = None,
        stoppers: list[Stopper] | None = None,
    ) -> None:
        """
        Initialize the trainer after construction.

        Parameters
        ----------
        callbacks : list[Callback] | None, optional
            A list of callbacks to handle events during training. If None, an empty list
            is used.
        stoppers : list[Stopper] | None, optional
            A list of stoppers to handle stopping criteria during training. If None,
            an empty list is used.

        Notes
        -----
        This method:
        - Initializes the callback list and sets up each callback
        - Initializes the stopper list
        - Both callbacks and stoppers are stored in internal attributes
        (_callbacks and _stoppers)
        """
        if callbacks is None:
            callbacks = []
        self._callbacks = CallbackList(callbacks=callbacks)
        self._callbacks.init()

        if stoppers is None:
            stoppers = []
        self._stoppers = StopperList(stoppers=stoppers)

    def train(self) -> None:
        """
        Train the model using the provided PDE, optimizer, and callbacks.

        This method performs the training loop, which includes resampling conditions,
        performing optimization steps, testing on training and test data, and invoking
        callbacks and stoppers at appropriate stages.

        Notes
        -----
        - The training loop continues until the specified number of iterations is
          reached or a stopping criterion is met.
        - The method invokes callbacks at the beginning and end of training, as well as
          at the beginning and end of each iteration.
        - The method also handles resampling conditions and updating the learning rate
          scheduler if provided.

        Returns
        -------
        None
        """
        self._callbacks.on_train_begin()

        while self.trainer_data.iteration < self.trainer_data.iterations:
            self._callbacks.on_iteration_begin()

            self.trainer_data.pde.resample_conditions(self.trainer_data.iteration)

            self.trainer_data.optimizer.step(self.closure)  # type: ignore[arg-type]

            if self.trainer_data.scheduler is not None:
                self.trainer_data.scheduler.step()

            self.test_on_train()

            self.test_on_test()

            self._callbacks.on_iteration_end()

            if self._stoppers.should_stop():
                break

            self.trainer_data.iteration += 1

        self._callbacks.on_train_end()

    def closure(self) -> torch.Tensor:
        """
        Compute the loss for the closure function.

        This method is used by optimizers that require a closure function. It clears
        the cached autograd state, zeroes the gradients, sets the model to training
        mode, computes the training loss, and performs backpropagation.

        Returns
        -------
        torch.Tensor
            The computed training loss.
        """
        clear()  # Important: clear cached autograd state (e.g., Jacobians)

        self.trainer_data.optimizer.zero_grad()
        self.trainer_data.model.train()

        loss = self.trainer_data.pde.loss_train_for_closure(
            self.trainer_data.model,
        )
        loss.backward()  # type: ignore[no-untyped-call]

        return loss

    def test_on_train(self) -> None:
        """Evaluate the model on the training dataset and record the loss.

        This method sets the model to evaluation mode, computes the training loss
        for the current iteration, and stores the loss value in the trainer data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.trainer_data.model.eval()
        loss = self.trainer_data.pde.loss_train(
            self.trainer_data.model,
            self.trainer_data.iteration,
        )

        self.trainer_data.losses_train[self.trainer_data.iteration] = loss.item()

    def test_on_test(self) -> None:
        """Evaluate the model on the test dataset and record the loss.

        This method sets the model to evaluation mode, computes the test loss
        for the current iteration if the iteration is a multiple of the test
        interval, and stores the loss value in the trainer data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (
            self.trainer_data.iteration % self.trainer_data.test_every == 0
            and self.trainer_data.iteration > 0
        ):
            self.trainer_data.model.eval()
            loss = self.trainer_data.pde.loss_test(
                self.trainer_data.model,
                self.trainer_data.iteration,
            )

            self.trainer_data.losses_test[self.trainer_data.iteration] = loss.item()

    def save(self, filename: str | Path = "trainer.pickle") -> None:
        """
        Save the trainer to a file using pickling.

        Parameters
        ----------
        filename : str | Path
            The name of the file to save the trainer to, by default "trainer.pickle".
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str | Path) -> Self:
        """
        Load the trainer data from a file.

        Parameters
        ----------
        filename : str | Path
            The name of the file from which the trainer data will be loaded.

        Returns
        -------
        Self
            An instance of the class with the loaded trainer data.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
