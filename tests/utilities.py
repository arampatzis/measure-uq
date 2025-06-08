"""
Tests for the utilities module.

This module contains tests for the utilities module of measure-uq.
"""

import chaospy
import numpy as np
import pytest
import torch

from measure_uq.utilities import SparseDynamicArray, torch_numpoly_call


def test_torch_numpoly_call() -> None:
    """
    Test the function torch_numpoly_call.

    Test the function torch_numpoly_call by comparing its output with the same
    computation done with the numpoly library.
    """
    joint = chaospy.J(
        chaospy.Uniform(1, 3),
        chaospy.Uniform(-2, 1),
    )

    expansion = chaospy.generate_expansion(
        5,
        joint,
        normed=True,
    )

    x = np.random.uniform(1, 3, (2, 10))

    e1 = expansion(*x)

    e2 = torch_numpoly_call(
        torch.tensor(expansion.exponents, dtype=torch.float64),
        torch.tensor(np.array(expansion.coefficients), dtype=torch.float64),
        torch.tensor(x.T, dtype=torch.float64),
    ).T

    assert np.allclose(e1, e2.detach().numpy(), atol=1e-12)


def test_sparse_dynamic_arrays() -> None:
    """Test the SparseDynamicArray class for dynamic sparse array operations."""
    x = SparseDynamicArray(shape=5, dtype=float)

    for i in range(0, 1001, 10):
        x[i] = 2 * i + 1

    assert np.isclose(x[-1], 2001, atol=1e-16)

    assert x.i[-1] == 1000

    assert x.v[-1] == x[-1]

    assert x[0] == x(0)

    assert x[1] == x(10)

    assert x[-1] == x(1000)

    with pytest.raises(IndexError):
        x(11)
