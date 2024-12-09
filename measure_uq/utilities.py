"""
Utilities module

This module provides utility functions and classes for use in the measure-uq package.

"""
from dataclasses import dataclass, field

import torch


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
