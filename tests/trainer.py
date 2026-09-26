"""
Tests for the trainer module.

This module contains tests for the trainer module of measure-uq.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from measure_uq.models import PINN
from measure_uq.networks import FeedforwardBuilder
from measure_uq.pde import PDE, Condition, Conditions, Parameters
from measure_uq.stoppers import Stopper, Stoppers
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


@dataclass(kw_only=True)
class SwitchableNaNCondition(Condition):
    """A condition whose residual is the model output, or NaN when switched on."""

    nan: bool = False

    def sample_points(self) -> None:
        """Sample a fixed set of points."""
        self.points = torch.linspace(0, 1, 5)[:, None]

    def eval(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: ARG002
        """
        Return the output of the model, or NaN if `nan` is True.

        Parameters
        ----------
        x : Tensor
            The input of the model.
        y : Tensor
            The output of the model.

        Returns
        -------
        Tensor
            The residual.
        """
        return y * float("nan") if self.nan else y


@dataclass(kw_only=True)
class StopOnceAt(Stopper):
    """Stop training once, at a given iteration."""

    iteration: int
    fired: bool = False

    def should_stop(self, trainer_data: TrainerData) -> bool:
        """
        Stop the first time the given iteration is reached.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.

        Returns
        -------
        bool
            True the first time `iteration` is reached, False otherwise.
        """
        if not self.fired and trainer_data.iteration == self.iteration:
            self.fired = True
            return True
        return False


def test_nan_detected_immediately_after_resume() -> None:
    """
    Test that a NaN loss stops training at the iteration it occurs.

    A stopper ends training after the loss of iteration 2 is logged, so resuming
    logs iteration 2 a second time. The NaN check must still look at the loss
    just computed, not at the stale entry from the first run.
    """
    condition = SwitchableNaNCondition()
    conditions: list[Condition] = [condition]
    pde = PDE(
        conditions_train=Conditions(conditions=conditions),
        conditions_test=Conditions(conditions=conditions),
        parameters_train=Parameters(values=torch.tensor([[1.0]])),
        parameters_test=Parameters(values=torch.tensor([[1.0]])),
    )
    model = PINN(network_builder=FeedforwardBuilder([2, 4, 1]))
    trainer_data = TrainerData(
        pde=pde,
        iterations=10,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        test_every=10**9,  # never test, so no checkpoint is written
    )
    trainer = Trainer(
        trainer_data=trainer_data,
        stoppers=Stoppers(
            trainer_data=trainer_data,
            stoppers=[StopOnceAt(iteration=2)],
        ),
    )

    trainer.train()
    assert trainer_data.iteration == 2

    condition.nan = True
    trainer.train()

    assert trainer_data.iteration == 2
