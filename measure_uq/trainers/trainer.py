"""
Trainer module.

This module provides the Trainer class for managing the training process of models
in the measure-uq package.

The Trainer class handles the training loop, including callbacks for monitoring and
logging, as well as stopping criteria to determine when training should end.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import torch

from measure_uq.callbacks import Callbacks
from measure_uq.gradients import clear
from measure_uq.stoppers import Stoppers
from measure_uq.trainers.trainer_data import TrainerData
from measure_uq.trainers.utilities import is_model_fully_on_device
from measure_uq.utilities import KeyController

# ruff: noqa: S301


@dataclass(kw_only=True)
class Trainer:
    """
    A class to manage the training process of models.

    This class handles the training loop, including optimization steps, callbacks for
    monitoring and logging, and stopping criteria. It provides methods for training,
    evaluation, saving and loading model states.

    Attributes
    ----------
    trainer_data : TrainerData
        Contains all training-related data including the model, optimizer, scheduler,
        loss history, and other configuration.
    callbacks : Callbacks
        A collection of callbacks that are triggered at specific points during training
        to perform monitoring, logging, or other auxiliary tasks.
    stoppers : Stoppers
        A collection of stopping criteria that determine when training should end based
        on various conditions.

    Notes
    -----
    The training process can be controlled interactively using keyboard controls:
    - Space bar to pause/resume training
    - 'x' key to stop training

    The class uses a closure-based optimization approach compatible with optimizers
    that require it, such as LBFGS.
    """

    trainer_data: TrainerData

    callbacks: Callbacks | None = None

    stoppers: Stoppers | None = None

    def __post_init__(self) -> None:
        """
        Initialize the trainer after construction.

        This method is automatically called after the trainer is instantiated.
        It initializes the callbacks by calling their init() methods.
        """
        self.callbacks = self.callbacks or Callbacks(
            trainer_data=self.trainer_data,
            callbacks=[],
        )

        self.callbacks.init()

        stoppers = self.stoppers or Stoppers(
            trainer_data=self.trainer_data,
            stoppers=[],
        )
        self.stoppers = stoppers

        device = self.trainer_data.device

        if not is_model_fully_on_device(self.trainer_data.model, device):
            raise ValueError(f"Model is not fully on device {device}")

        if not self.trainer_data.pde.conditions_train.is_on_device(device):
            raise ValueError(f"Train conditions are not on device {device}")

        if not self.trainer_data.pde.conditions_test.is_on_device(device):
            raise ValueError(f"Test conditions are not on device {device}")

        if not self.trainer_data.pde.parameters_train.is_on_device(device):
            raise ValueError(f"Train parameters are not on device {device}")

        if not self.trainer_data.pde.parameters_test.is_on_device(device):
            raise ValueError(f"Test parameters are not on device {device}")

    @property
    def safe_callbacks(self) -> Callbacks:
        """
        Get the callbacks property with type checking.

        Returns
        -------
        Callbacks
            The callbacks property.
        """
        assert self.callbacks is not None
        return self.callbacks

    @property
    def safe_stoppers(self) -> Stoppers:
        """
        Get the stoppers property with type checking.

        Returns
        -------
        Stoppers
            The stoppers property.
        """
        assert self.stoppers is not None
        return self.stoppers

    def one_train_step(self) -> None:
        """
        Perform a single training step.

        This method executes one iteration of the training loop, which includes:
        - Triggering the beginning of iteration callbacks.
        - Resampling conditions for the PDE based on the current iteration.
        - Performing an optimization step using the defined optimizer.
        - Testing the model on the training data.
        - Updating the learning rate scheduler.
        - Testing the model on the test data.
        - Triggering the end of iteration callbacks.

        Notes
        -----
        The method assumes that the optimizer has a `step` method that accepts a
        closure.
        """
        self.safe_callbacks.on_iteration_begin()

        self.trainer_data.pde.resample_conditions(self.trainer_data.iteration)

        self.trainer_data.optimizer.step(self.closure)  # type: ignore[arg-type]

        self.test_on_train()
        self.step_scheduler()
        self.test_on_test()

        self.safe_callbacks.on_iteration_end()

    def train(self) -> None:
        """
        Train the model using the specified training loop.

        This method manages the training process, including interactive controls
        for pausing, resuming, and stopping the training loop. It iterates over
        the training steps, checks for stopping criteria, and handles callbacks
        at the beginning and end of training.

        Interactive Controls:
        - Press space to pause/resume the training loop.
        - Press 'x' to stop the training loop.

        Raises
        ------
        Exception
            If an error occurs during the training process, it will be caught
            and handled to ensure proper cleanup.

        Notes
        -----
        - The method uses a `KeyController` for handling interactive controls.
        - It triggers callbacks at the start and end of training, as well as
          at each iteration.
        - The training loop continues until the specified number of iterations
          is reached or a stopping criterion is met.
        """
        controller = KeyController(
            use_toggle=True,
            key_bindings={"toggle": " ", "stop": "x"},
        )

        self.safe_callbacks.on_train_begin()

        try:
            while self.trainer_data.iteration < self.trainer_data.iterations:
                controller.check_pause()

                if controller.stop_requested:
                    print("Stop requested. Exiting training loop.")
                    break

                self.one_train_step()

                if self.safe_stoppers.should_stop():
                    print("Stopping criteria met.")
                    break

                self.trainer_data.iteration += 1

        finally:
            controller.close()
            self.safe_callbacks.on_train_end()

            if plt.get_fignums():
                plt.close("all")

            print("Training finished and cleaned up.")

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

    def step_scheduler(self) -> None:
        """
        Update the learning rate scheduler.

        This method steps the learning rate scheduler if it is defined in the
        trainer data. It is typically called after an optimization step to
        adjust the learning rate according to the scheduler's policy.

        Returns
        -------
        None
            This method does not return anything.
        """
        if self.trainer_data.scheduler is None:
            return
        if isinstance(
            self.trainer_data.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            self.trainer_data.scheduler.step(self.trainer_data.losses_train.v[-1])  # type: ignore[arg-type]
        else:
            self.trainer_data.scheduler.step()

    def test_on_train(self) -> None:
        """
        Evaluate the model on the training dataset and record the loss.

        This method sets the model to evaluation mode, computes the training loss
        for the current iteration, and stores the loss value in the trainer data.

        Returns
        -------
        None
            This method does not return anything.
        """
        self.trainer_data.model.eval()
        loss = self.trainer_data.pde.loss_train(
            self.trainer_data.model,
            self.trainer_data.iteration,
        )

        self.trainer_data.losses_train[self.trainer_data.iteration] = loss.item()

    def test_on_test(self) -> None:
        """
        Evaluate the model on the test dataset and record the loss.

        This method sets the model to evaluation mode, computes the test loss
        for the current iteration if the iteration is a multiple of the test
        interval, and stores the loss value in the trainer data.

        Returns
        -------
        None
            This method does not return anything.
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
