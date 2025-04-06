"""
Provide functions for computing gradients.

The functions in this module are used for computing gradients of the loss
function of a model with respect to the parameters of the model.

The functions in this module are a modification of the `gradients` module,
copied from `deepxde`. Licensed under the GNU Lesser General Public License
(LGPL) 2.1. See the LICENSE.LGPL file in the root directory for details.
Original source: https://github.com/lululxvi/deepxde
"""

__all__ = ["hessian", "jacobian"]

import torch

from . import gradients_reverse


def jacobian(
    ys: torch.Tensor,
    xs: torch.Tensor,
    i: int | None = None,
    j: int | None = None,
) -> torch.Tensor:
    """Compute the Jacobian matrix of a tensor with respect to the input tensor.

    Compute `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
    J as J[i, j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and j = 0, ..., dim_x - 1.

    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i, j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Parameters
    ----------
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        i (int or None): i-th row. If i is ``None``, returns the j-th column
            J[:, j].
        j (int or None): j-th column. If j is ``None``, returns the i-th row
            J[i, :], i.e., the gradient of y_i. i and j cannot be both ``None``,
            unless J has only one element, which is returned.

    Returns
    -------
        (i, j)th entry J[i, j], i-th row J[i, :], or j-th column J[:, j].
    """
    return gradients_reverse.jacobian(ys, xs, i=i, j=j)


def hessian(
    ys: torch.Tensor,
    xs: torch.Tensor,
    component: int = 0,
    i: int = 0,
    j: int = 0,
) -> torch.Tensor:
    """Compute the Hessian matrix of a tensor with respect to the input tensor.

    Compute Hessian matrix H as
    H[i, j] = d^2y / dx_i dx_j, where i,j = 0, ..., dim_x - 1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i, j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Parameters
    ----------
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        component: `ys[:, component]` is used as y to compute the Hessian.
        i (int): i-th row.
        j (int): j-th column.

    Returns
    -------
        H[`i`, `j`].
    """
    return gradients_reverse.hessian(ys, xs, component=component, i=i, j=j)


def clear() -> None:
    """Clear cached Jacobians and Hessians."""
    gradients_reverse.clear()
