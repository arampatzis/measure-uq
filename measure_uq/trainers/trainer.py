"""
Defines a `Trainer` class for managing the training process of a machine
learning model using callbacks and stoppers.

Classes
-------
Trainer
    A class for handling the training process with customizable callbacks and stopping
    criteria.
"""
from dataclasses import InitVar, dataclass, field

from measure_uq.callbacks import Callback, CallbackList
from measure_uq.gradients import clear
from measure_uq.stoppers import Stopper, StopperList
from measure_uq.trainers.trainer_data import TrainerData


@dataclass(kw_only=True)
class Trainer:
    """
    A dataclass to manage the training process, including callbacks and stopping
    criteria.

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
        callbacks: list[Callback] | None,
        stoppers: list[Stopper] | None,
    ):
        """
        Initializes the trainer after construction.

        Sets the `_callbacks` attribute and sets the trainer for each callback.

        Args:
        ----
        callbacks: list[Callback] | None
            A list of _callbacks. If None, an empty list is used.
        stoppers : list[Stopper] | None
            A list of user-defined stoppers, by default None.
        """
        if callbacks is None:
            callbacks = []
        self._callbacks = CallbackList(callbacks=callbacks)
        self._callbacks.init()

        if stoppers is None:
            stoppers = []
        self._stoppers = StopperList(stoppers=stoppers)

    def train(self):
        """
        Train the model.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method starts the training of the model. The training loop is
        controlled by the `iterations` attribute. The training is done by
        repeatedly calling the optimizer's `step` method with the model's
        closure. The closure is a function that computes the loss, and zeros
        the gradients of the model's parameters. The loss is then computed,
        and the gradients are computed using the loss and the model's
        parameters. The optimizer then uses the gradients to update the
        model's parameters.

        The training loop is interrupted every `test_every` iterations to
        test the model. The test step is done by calling the `test_step`
        method.

        After the training loop has finished, the `on_train_end` callback is
        called.
        """
        self._callbacks.on_train_begin()

        while self.trainer_data.iteration < self.trainer_data.iterations:
            self.trainer_data.optimizer.step(self.closure)

            if self.trainer_data.scheduler is not None:
                self.trainer_data.scheduler.step()

            if self._stoppers.should_stop():
                break

        self._callbacks.on_train_end()

    def closure(self):
        """
        The closure function to be passed to the optimizer.

        This function is the one that the optimizer will call to compute the
        loss. It is called at every iteration of the training loop. The
        function first calls `on_iteration_begin` on the _callbacks, then
        zeros the gradients of the model's parameters, computes the loss,
        and computes the gradients of the loss with respect to the model's
        parameters. It then calls `on_iteration_end` on the _callbacks, and
        finally returns the loss.

        Parameters
        ----------
        None

        Returns
        -------
        loss : Tensor
            The computed loss.

        Notes
        -----
        This function is used to compute the loss, and to compute the
        gradients of the loss with respect to the model's parameters. The
        gradients are then used by the optimizer to update the model's
        parameters.
        """
        self._callbacks.on_iteration_begin()

        self.trainer_data.optimizer.zero_grad()
        self.trainer_data.model.train()
        loss = self.trainer_data.pde.loss_train(
            self.trainer_data.model,
            self.trainer_data.iteration,
        )
        loss.backward()

        self.trainer_data.losses_train(self.trainer_data.iteration, loss.item())

        if (
            self.trainer_data.iteration % self.trainer_data.test_every == 0
            and self.trainer_data.iteration > 0
        ):
            self.test_step()

        self.trainer_data.iteration += 1
        clear()

        self._callbacks.on_iteration_end()

        return loss

    def test_step(self):
        """
        Perform one testing step.

        This method first sets the model in evaluation mode, then computes
        the loss on the testing data. The result is saved in the
        `losses_test` list.
        """
        self.trainer_data.model.eval()
        loss = self.trainer_data.pde.loss_test(
            self.trainer_data.model,
            self.trainer_data.iteration,
        )

        self.trainer_data.losses_test(
            self.trainer_data.iteration,
            loss.item(),
        )
