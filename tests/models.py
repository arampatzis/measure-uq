"""
Unit tests for the models module.

This module contains unit tests for the models module of measure-uq.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import tensor

from measure_uq.models import PINN_PCE
from measure_uq.networks import FeedforwardBuilder


class MockModule(torch.nn.Module):
    """
    A mock torch.nn.Module for testing.

    This class is used in unit tests to create mock instances of torch.nn.Module.
    It provides a way to check if the forward method of the module was called with
    the correct arguments.

    Parameters
    ----------
    return_value : torch.Tensor
        The value to be returned by the mock forward method.

    Attributes
    ----------
    _mock : unittest.mock.MagicMock
        The mock object for the forward method of the module.
    """

    def __init__(self, return_value: torch.Tensor) -> None:
        """
        Initialize the MockModule with a return value.

        Parameters
        ----------
        return_value : torch.Tensor
            The value to be returned by the mock forward method.
        """
        super().__init__()
        self._mock = MagicMock()
        self._mock.return_value = return_value

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Mock the forward method.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the forward method.
        **kwargs : Any
            Keyword arguments passed to the forward method.

        Returns
        -------
        torch.Tensor
            The mock return value.
        """
        return self._mock(*args, **kwargs)

    def assert_called_with(self, *args: Any, **kwargs: Any) -> None:
        """
        Assert that the forward method was called with the correct arguments.

        Parameters
        ----------
        *args : Any
            Positional arguments expected in the call.
        **kwargs : Any
            Keyword arguments expected in the call.
        """
        self._mock.assert_called_with(*args, **kwargs)


def test_pinn_pce() -> None:
    """
    Test the evaluation of the PINN_PCE model.

    Input/Output of the NN: [-1, 2, -3]
    Expansion:[...]
    """
    Nsp = 30
    Np = 1
    Ne = 3

    # Set the evaluation of the basis functions.
    # Assume one parameter, and an expansion of size one that maps `x` to `x`.
    expansion = MagicMock()
    expansion.return_value = np.tile(
        np.arange(
            1,
            Nsp + 1,
            dtype=np.float32,
        )[:, None],
        (1, Ne),
    ).T
    expansion.__len__.return_value = Ne

    model = PINN_PCE(
        network_builder=FeedforwardBuilder([1, 1, 1]),
        expansion=expansion,
    )

    # Set the output of each network to be equal to the input
    Nsx = 20
    x = -torch.arange(
        1,
        Nsx + 1,
        dtype=torch.float32,
    ).view(-1, 1)

    model.net = MockModule(x)

    # These values are irrelevant for this test
    p = torch.zeros((Nsp, Np), dtype=torch.float32)

    # Evaluate the model
    m = model(x, p).squeeze(-1)

    # The way we mocked the model evaluation, corresponds to the product of the
    # input with the single expansion
    mm = model.combine_input(x, tensor(expansion.return_value[0, :][:, None]))
    mm = Ne * torch.prod(mm, 1)

    assert torch.equal(m, mm)
