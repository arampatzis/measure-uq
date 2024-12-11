"""
Utilities module

This module provides utility functions and classes for use in the measure-uq package.

"""
from dataclasses import dataclass, field

import torch

INT_INF = (1 << 32) - 1


@dataclass(kw_only=True)
class LossContainer:
    """
    A dataclass for storing loss values and their associated iteration indices.

    Attributes
    ----------
    values : list
        A list of loss values.
    index : list
        A list of iteration indices corresponding to the loss values.
    """

    values: list = field(default_factory=list)

    index: list = field(default_factory=list)

    def __call__(self, i: int, y: float):
        """
        Store a loss value and its associated iteration index.

        Parameters
        ----------
        i : int
            The iteration index.
        y : float
            The loss value.
        """
        self.values.append(y)
        self.index.append(i)


def cartesian_product_of_rows(*tensors):
    """
    Compute the Cartesian product of the rows of multiple 2D tensors.

    Parameters
    ----------
    *tensors : torch.Tensor
        A variable number of 2D tensors.

    Returns
    -------
    torch.Tensor
        A tensor containing all combinations of rows from the input tensors.
        Each row is a concatenation of rows from the input tensors.
    """
    assert len(tensors) > 0

    # Start with the first tensor
    result = tensors[0]

    for tensor in tensors[1:]:
        # Get the number of rows in the current result and the new tensor
        rows_result = result.shape[0]
        rows_tensor = tensor.shape[0]

        # Repeat the rows of the current result
        repeated_result = result.repeat_interleave(rows_tensor, dim=0)

        # Repeat the rows of the new tensor
        repeated_tensor = tensor.repeat(rows_result, 1)

        # Concatenate along the last dimension
        result = torch.cat([repeated_result, repeated_tensor], dim=1)

    return result


def extend_vector_tensor(
    x: torch.Tensor,
    n: int,
    default_value: float = 0,
) -> torch.Tensor:
    """
    Extend a given tensor to a specified size, filling with a default value
    if necessary.

    Parameters
    ----------
    x : Tensor
        The tensor to extend.
    n : int
        The desired length of the output tensor.
    default_value : int or float, optional
        The default value to fill the tensor with when extending its length.
        Defaults to 0.

    Returns
    -------
    Tensor
        The extended tensor, with length `N`.
    """
    nx = x.shape[0]

    if n <= 0:
        raise ValueError("`n` must be positive.")

    if nx == 0:
        return torch.full((n,), default_value)

    if nx == n:
        return x

    if nx < n:
        z = torch.empty(n)
        z[:n] = x
        z[n:] = x[-1]
        return z

    raise ValueError("The size of `x` cannot be greater than the size of `y`.")
