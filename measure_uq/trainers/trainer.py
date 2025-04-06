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

from dataclasses import InitVar, dataclass, field

from measure_uq.callbacks import Callback, CallbackList
from measure_uq.gradients import clear
from measure_uq.stoppers import Stopper, StopperList
from measure_uq.trainers.trainer_data import TrainerData


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
        Train the model.

        Notes
        -----
        This method implements the main training loop:

        1. Calls 'on_train_begin' callback.

        2. For each iteration:

           - Updates model parameters using optimizer.
           - Updates learning rate if scheduler is provided.
           - Checks stopping criteria.

        3. Calls 'on_train_end' callback when training completes.

        The training loop continues until either:

        - The maximum number of iterations is reached.
        - A stopper indicates training should stop.
        """

        self._callbacks.on_train_begin()

        while self.trainer_data.iteration < self.trainer_data.iterations:
            self.trainer_data.optimizer.step(self.closure)

            if self.trainer_data.scheduler is not None:
                self.trainer_data.scheduler.step()

            if self._stoppers.should_stop():
                break

        self._callbacks.on_train_end()

    def closure(self) -> float:
        """
        Closure function to be passed to the optimizer.

        Returns
        -------
        float
            The computed loss value.

        Notes
        -----
        This function performs the following steps:
        - Calls 'on_iteration_begin' callback.
        - Zeros gradients and sets model to training mode.
        - Computes training loss and its gradients.
        - Performs test step if needed.
        - Updates iteration counter and clears gradients.
        - Calls 'on_iteration_end' callback.
        - Returns the loss value.
        """


        self._callbacks.on_iteration_begin()

        self.trainer_data.optimizer.zero_grad()
        self.trainer_data.model.train()
        loss = self.trainer_data.pde.loss_train(
            self.trainer_data.model,
            self.trainer_data.iteration,
        )
        loss.backward()  # type: ignore[no-untyped-call]

        self.trainer_data.losses_train[self.trainer_data.iteration] = loss.item()
        if (
            self.trainer_data.iteration % self.trainer_data.test_every == 0
            and self.trainer_data.iteration > 0
        ):
            self.test_step()

        self.trainer_data.iteration += 1
        clear()

        self._callbacks.on_iteration_end()

        return loss.item()

    def test_step(self) -> None:
        """
        Perform one testing step.

        Notes
        -----
        This method:
        1. Sets the model to evaluation mode
        2. Computes the loss on testing data
        3. Stores the test loss in the losses_test array

        It is called every `test_every` iterations during training.
        """
        self.trainer_data.model.eval()
        loss = self.trainer_data.pde.loss_test(
            self.trainer_data.model,
            self.trainer_data.iteration,
        )

        self.trainer_data.losses_test[self.trainer_data.iteration] = loss.item()
