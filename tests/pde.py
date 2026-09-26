"""
Tests for the pde module.

This module contains tests for the pde module of measure-uq.
"""

# ruff: noqa: S301

import pickle
from dataclasses import dataclass

import chaospy
import torch
from torch import Tensor

from measure_uq.models import PINN, PINN_PCE
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


def make_shared_pde(loss_weights: list[float] | None = None) -> PDE:
    """
    Create a PDE whose training and test conditions share the same objects.

    Parameters
    ----------
    loss_weights : list[float] | None, optional
        The weights for each condition's loss, by default None.

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
        loss_weights=loss_weights,
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


def test_loss_keeps_model_dtype() -> None:
    """
    Test that the loss is computed in the dtype of the model output.

    PINN_PCE produces float64 outputs; the loss must not be rounded to float32.
    """
    expansion = chaospy.generate_expansion(
        3,
        chaospy.J(chaospy.Uniform(0, 1)),
        normed=True,
    )
    model = PINN_PCE(
        network_builder=FeedforwardBuilder([1, 8, len(expansion)]),
        expansion=expansion,
    )
    pde = make_shared_pde()
    parameters = pde.parameters_train.values

    loss = pde.loss_train(model, 0)

    expected = torch.stack(
        [
            torch.mean(torch.linalg.vector_norm(c(model, parameters), dim=1) ** 2)
            for c in pde.conditions_train
        ],
    ).sum()
    assert loss.dtype == torch.float64
    assert torch.allclose(loss, expected, rtol=1e-14, atol=0.0)


def test_integer_loss_weights() -> None:
    """Test that integer loss weights are accepted and applied."""
    model = PINN(network_builder=FeedforwardBuilder([2, 4, 1]))
    pde = make_shared_pde(loss_weights=[1, 10])

    loss = pde.loss_train(model, 0)

    res = pde.conditions_train.l2_loss(model, pde.parameters_train.values)
    assert torch.isclose(loss, res[0] + 10 * res[1])
