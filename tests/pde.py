"""
Tests for the pde module.

This module contains tests for the pde module of measure-uq.
"""

# ruff: noqa: S301

import pickle
from dataclasses import dataclass

import torch
from torch import Tensor

from measure_uq.models import PINN
from measure_uq.networks import FeedforwardBuilder
from measure_uq.pde import PDE, Condition, Conditions, Parameters


@dataclass(kw_only=True)
class IdentityCondition(Condition):
    """A condition whose residual is the output of the model."""

    def sample_points(self) -> None:
        """Sample a fixed set of points."""
        self.points = torch.linspace(0, 1, 5)[:, None]

    def eval(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: ARG002
        """
        Return the output of the model as the residual.

        Parameters
        ----------
        x : Tensor
            The input of the model.
        y : Tensor
            The output of the model.

        Returns
        -------
        Tensor
            The output of the model.
        """
        return y


def make_shared_pde() -> PDE:
    """
    Create a PDE whose training and test conditions share the same objects.

    Returns
    -------
    PDE
        The PDE, with different training and test parameters.
    """
    conditions: list[Condition] = [IdentityCondition(), IdentityCondition()]

    return PDE(
        conditions_train=Conditions(conditions=conditions),
        conditions_test=Conditions(conditions=conditions),
        parameters_train=Parameters(values=torch.tensor([[1.0], [2.0]])),
        parameters_test=Parameters(values=torch.tensor([[3.0], [4.0], [5.0]])),
    )


def test_loss_test_does_not_write_into_train_log() -> None:
    """
    Test that test losses are stored separately from training losses.

    When the training and test conditions share the same objects, evaluating
    the test loss must not add entries to the training log `loss` of each
    condition.
    """
    pde = make_shared_pde()
    model = PINN(network_builder=FeedforwardBuilder([2, 4, 1]))

    for iteration in range(4):
        pde.loss_train(model, iteration)
        if iteration % 2 == 0:
            pde.loss_test(model, iteration)

    for condition in pde.conditions_train:
        assert list(condition.loss.i) == [0, 1, 2, 3]
        assert list(condition.loss_test.i) == [0, 2]

    expected = pde.conditions_test.l2_loss(model, pde.parameters_test.values)
    for i, condition in enumerate(pde.conditions_test):
        assert torch.isclose(
            torch.tensor(condition.loss_test[-1], dtype=torch.float32),
            expected[i],
        )


def test_condition_unpickle_without_loss_test() -> None:
    """Test that conditions pickled before `loss_test` existed can be loaded."""
    condition = IdentityCondition()
    condition.loss[0] = 1.0
    del condition.loss_test

    restored = pickle.loads(pickle.dumps(condition))

    assert len(restored.loss) == 1
    assert len(restored.loss_test) == 0
    restored.loss_test[5] = 2.0
    assert restored.loss_test(5) == 2.0
