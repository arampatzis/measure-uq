"""
Defines the Trainer class for training a PINN. The Trainer
stores the model, optimizer, scheduler, and other parameters for training
a PINN. It provides a train method to train the model.
"""
from dataclasses import InitVar, dataclass, field

import torch
from torch import nn, optim

from measure_uq.callbacks import Callback, CallbackList
from measure_uq.gradients import clear
from measure_uq.pde import PDE
from measure_uq.stoppers import Stopper, StopperList
from measure_uq.utilities import LossContainer


def get_optimizer(otype: str, model: nn.Module, learning_rate: float):
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


def get_scheduler(stype: str, optimizer: optim.Optimizer, **kwargs):
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
    torch.optim.lr_scheduler._LRScheduler
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


@dataclass(kw_only=True)
class Trainer:
    """
    A dataclass to manage training configurations and workflows.

    This class serves as a central controller for managing the training process,
    including model updates, tracking losses, handling callbacks, and implementing
    stopping criteria.

    Attributes
    ----------
    pde : PDE
        The partial differential equation (PDE) to be solved during training.
    iterations : int
        The total number of training iterations to perform.
    model : nn.Module
        The neural network model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model parameters.
    scheduler : torch.optim.lr_scheduler._LRScheduler | None, optional
        A learning rate scheduler for adjusting the learning rate, by default None.
    iteration : int, optional
        The current iteration during training, by default 0.
    test_every : int | None, optional
        Frequency (in iterations) of testing during training, by default 100.
    losses_train : LossContainer
        Container to store training losses.
    losses_test : LossContainer
        Container to store testing losses.
    _callbacks : CallbackList
        Internal list of callbacks for handling events during training.
    callbacks : InitVar[list[Callback] | None], optional
        A list of user-defined callbacks, by default None.
    _stoppers : StopperList
        Internal list of stopping criteria for early termination.
    stoppers : InitVar[list[Stopper] | None], optional
        A list of user-defined stoppers, by default None.
    """

    pde: PDE

    iterations: int

    model: nn.Module

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
        self._callbacks.set_trainer(self)

        if stoppers is None:
            stoppers = []
        self._stoppers = StopperList(stoppers=stoppers)
        self._stoppers.set_trainer(self)

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

        while self.iteration < self.iterations:
            self.optimizer.step(self.closure)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.iteration % self.test_every == 0 and self.iteration > 0:
                self.test_step()

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

        self.optimizer.zero_grad()
        self.model.train()
        loss = self.pde.loss_train(self.model, self.iteration)
        loss.backward()

        self.losses_train(self.iteration, loss.item())

        self.iteration += 1
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
        self.model.eval()
        loss = self.pde.loss_test(self.model, self.iteration)

        self.losses_test(self.iteration, loss.item())
