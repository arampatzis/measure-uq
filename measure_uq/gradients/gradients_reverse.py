"""
Compute gradients using reverse-mode autodiff.

This module provides functions for computing gradients, Jacobians and Hessians
using reverse-mode automatic differentiation.

The functions in this module are used for computing gradients of the loss
function of a model with respect to the parameters of the model.

The functions in this module are a modification of the `gradients` module,
copied from `deepxde`. Licensed under the GNU Lesser General Public License
(LGPL) 2.1. See the LICENSE.LGPL file in the root directory for details.
Original source: https://github.com/lululxvi/deepxde
"""

__all__ = ["hessian", "jacobian"]

import torch

from .jacobian import Jacobian, Jacobians


class JacobianReverse(Jacobian):
    def __init__(self, ys, xs):
        self.ys = ys
        self.xs = xs

        self.dim_y = ys.shape[1]
        self.dim_x = xs.shape[1]

        self.J = {}

    def __call__(self, i=None, j=None):
        """
        Returns (`i`, `j`)th entry J[`i`, `j`].

        - If `i` is ``None``, returns the jth column J[:, `j`].
        - If `j` is ``None``, returns the ith row J[`i`, :], i.e., the gradient of y_i.
        - `i` and `j` cannot be both ``None``.
        """
        if i is None and j is None:
            if self.dim_x > 1 or self.dim_y > 1:
                raise ValueError("i and j cannot be both None.")
            i = 0
            j = 0
        if i is not None and not 0 <= i < self.dim_y:
            raise ValueError(f"i={i} is not valid.")
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError(f"j={j} is not valid.")

        # Compute a column is not supported in reverse mode, unless there is only one
        # output.
        if i is None:
            if self.dim_y > 1:
                raise NotImplementedError(
                    "Reverse-mode autodiff doesn't support computing a column.",
                )
            i = 0

        # Compute J[i, :]
        if i not in self.J:
            # retain_graph=True has memory leak?
            y = self.ys[:, i : i + 1] if self.dim_y > 1 else self.ys
            self.J[i] = torch.autograd.grad(
                y,
                self.xs,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )[0]

        if j is None or self.dim_x == 1:
            return self.J[i]

        # Compute J[i, j]
        if (i, j) not in self.J:
            self.J[i, j] = self.J[i][:, j : j + 1]

        return self.J[i, j]


_Jacobians = Jacobians(JacobianReverse)


def jacobian(ys, xs, i=None, j=None):
    """
    Compute the Jacobian matrix of the output tensor with respect to the input tensor.

    This function supports lazy evaluation, computing only the necessary parts
    of the Jacobian matrix. It reuses cached results to avoid redundant computations
    for previously computed output-input-(i, j) triplets.

    Args:
    ----
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        i (int or None): `i`th row. If `i` is `None`,
            returns the `j`th column J[:, `j`].
        j (int or None): `j`th column. If `j` is `None`,
            returns the `i`th row J[`i`, :]. `i` and `j` cannot be both `None`, unless
            J has only one element, which is returned.

    Returns:
    -------
        (`i`, `j`)th entry J[`i`, `j`], `i`th row J[`i`, :], or `j`th column J[:, `j`].
    """
    return _Jacobians(ys, xs, i=i, j=j)


class Hessian:
    """
    Compute the Hessian matrix of a given output component with respect to the input.

    This class is a wrapper of the `jacobian` function, but it is specialized to
    compute the Hessian matrix of a given component of the output tensor with
    respect to the input tensor. The Hessian matrix is computed using reverse-mode
    autodiff.

    The Hessian matrix is computed lazily, i.e., it only computes the necessary
    parts of the Hessian matrix when needed. For previously computed output-input
    pairs, it reuses cached results to avoid duplicate computations.
    """

    def __init__(self, ys, xs, component=0):
        dim_y = ys.shape[1]

        if component >= dim_y:
            raise ValueError(
                "The component of ys={} cannot be larger than the dimension={}.".format(
                    component,
                    dim_y,
                ),
            )

        # There is no duplicate computation of grad_y.
        grad_y = jacobian(ys, xs, i=component, j=None)
        self.H = JacobianReverse(grad_y, xs)

    def __call__(self, i=0, j=0):
        """Returns H[`i`, `j`]."""
        return self.H(j, i)


class Hessians:
    """
    Compute multiple Hessians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self):
        self.Hs = {}

    def __call__(self, ys, xs, component=0, i=0, j=0):
        """
        Compute the Hessian entry

            H[i, j]=d^2y / dx_i dx_j, where i,j = 0, ..., dim_x - 1,

        for given output and input tensors.

        This method reuses previously computed Hessian instances for the same
        (output, input, component) triplet to avoid redundant computations.

        Parameters
        ----------
        ys : Tensor
            Output tensor of shape (batch_size, dim_y).
        xs : Tensor
            Input tensor of shape (batch_size, dim_x).
        component : int, optional
            The component of `ys` to use for computing the Hessian, by default 0.
        i : int, optional
            The row index for the Hessian entry, by default 0.
        j : int, optional
            The column index for the Hessian entry, by default 0.

        Returns
        -------
        Tensor
            The Hessian entry H[i, j] for the specified component.
        """
        key = (ys, xs, component)

        if key not in self.Hs:
            self.Hs[key] = Hessian(ys, xs, component=component)
        return self.Hs[key](i, j)

    def clear(self):
        """Clear cached Hessians."""
        self.Hs = {}


_Hessians = Hessians()


def hessian(ys, xs, component=0, i=0, j=0):
    """
    Compute the Hessian matrix entry H[i, j] = d^2y / dx_i dx_j for the specified
    component of the output tensor, using reverse-mode autodiff.

    This function supports lazy evaluation, computing only the necessary parts
    of the Hessian matrix. It reuses cached results to avoid redundant computations
    for previously computed output-input-component triplets.

    Args:
    ----
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        component: `ys[:, component]` is used as y to compute the Hessian.
        i (int): `i`th row index of the Hessian matrix.
        j (int): `j`th column index of the Hessian matrix.

    Returns:
    -------
        Tensor: The Hessian entry H[i, j] for the specified component.
    """
    return _Hessians(ys, xs, component=component, i=i, j=j)


def clear():
    """Clear cached Jacobians and Hessians."""
    _Jacobians.clear()
    _Hessians.clear()
