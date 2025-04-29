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
    """
    Compute the Jacobian matrix of a tensor with respect to the input tensor.

    This class is a wrapper of the `jacobian` function, but it is specialized to
    compute the Jacobian matrix of a given component of the output tensor with
    respect to the input tensor. The Jacobian matrix is computed using reverse-mode
    autodiff.

    Parameters
    ----------
    ys : torch.Tensor
        Output tensor of shape (batch_size, dim_y).
    xs : torch.Tensor
        Input tensor of shape (batch_size, dim_x).

    Attributes
    ----------
    xs : torch.Tensor
        Input tensor of shape (batch_size, dim_x).
    ys : torch.Tensor
        Output tensor of shape (batch_size, dim_y).
    dim_y : int
        The dimension of the output tensor.
    dim_x : int
        The dimension of the input tensor.
    J : dict
        A dictionary of cached Jacobian matrices.
    """

    def __init__(self, ys: torch.Tensor, xs: torch.Tensor) -> None:
        """
        Initialize the JacobianReverse.

        Parameters
        ----------
        ys : torch.Tensor
            Output tensor of shape (batch_size, dim_y).
        xs : torch.Tensor
            Input tensor of shape (batch_size, dim_x).
        """
        self.ys = ys
        self.xs = xs

        self.dim_y = ys.shape[1]
        self.dim_x = xs.shape[1]

        self.J = {}

    def __call__(self, i: int | None = None, j: int | None = None) -> torch.Tensor:
        """
        Return the Jacobian entry J[i, j].

        - If `i` is ``None``, returns the jth column J[:, `j`].
        - If `j` is ``None``, returns the ith row J[`i`, :], i.e., the gradient of y_i.
        - `i` and `j` cannot be both ``None``.

        Parameters
        ----------
        i : int | None
            The index of the row to return.
        j : int | None
            The index of the column to return.

        Returns
        -------
        torch.Tensor
            The Jacobian entry J[i, j].
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


def jacobian(
    ys: torch.Tensor,
    xs: torch.Tensor,
    i: int | None = None,
    j: int | None = None,
) -> torch.Tensor:
    """
    Compute the Jacobian matrix of a tensor with respect to the input tensor.

    This function supports lazy evaluation, computing only the necessary parts
    of the Jacobian matrix. It reuses cached results to avoid redundant computations
    for previously computed output-input-(i, j) triplets.

    Parameters
    ----------
    ys : torch.Tensor
        Output tensor of shape (batch_size, dim_y).
    xs : torch.Tensor
        Input tensor of shape (batch_size, dim_x).
    i : int or None, optional
        Row index. If `i` is None, the function returns the `j`-th column J[:, j].
    j : int or None, optional
        Column index. If `j` is None, the function returns the `i`-th row J[i, :].
        `i` and `j` cannot both be None unless the Jacobian has only one element.

    Returns
    -------
    torch.Tensor
        The `(i, j)`-th entry J[i, j], the `i`-th row J[i, :], or the `j`-th column
        J[:, j], depending on the specified arguments.
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

    Parameters
    ----------
    ys : torch.Tensor
        Output tensor of shape (batch_size, dim_y).
    xs : torch.Tensor
        Input tensor of shape (batch_size, dim_x).
    component : int
        The component of `ys` to use for computing the Hessian.

    Attributes
    ----------
    H : torch.Tensor
        The Hessian matrix.
    """

    def __init__(self, ys: torch.Tensor, xs: torch.Tensor, component: int = 0) -> None:
        """
        Initialize the Hessian.

        Parameters
        ----------
        ys : torch.Tensor
            Output tensor of shape (batch_size, dim_y).
        xs : torch.Tensor
            Input tensor of shape (batch_size, dim_x).
        component : int
            The line of `ys` to use for computing the Hessian.
        """
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

    def __call__(self, i: int = 0, j: int = 0) -> torch.Tensor:
        """
        Return the Hessian entry H[i, j].

        Parameters
        ----------
        i : int
            The row index of the Hessian entry.
        j : int
            The column index of the Hessian entry.

        Returns
        -------
        torch.Tensor
            The Hessian entry H[i, j].
        """
        return self.H(j, i)


class Hessians:
    """
    Compute multiple Hessians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.

    Attributes
    ----------
    Hs : dict
        A dictionary of cached Hessian instances.
    """

    def __init__(self) -> None:
        """Initialize the Hessians."""
        self.Hs: dict[tuple[torch.Tensor, torch.Tensor, int], Hessian] = {}

    def __call__(
        self,
        ys: torch.Tensor,
        xs: torch.Tensor,
        component: int = 0,
        i: int = 0,
        j: int = 0,
    ) -> torch.Tensor:
        """
        Compute the Hessian at [i, j].

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

    def clear(self) -> None:
        """Clear cached Hessians."""
        self.Hs = {}


# Initialize the Hessians class.
_Hessians = Hessians()


def hessian(
    ys: torch.Tensor,
    xs: torch.Tensor,
    component: int = 0,
    i: int = 0,
    j: int = 0,
) -> torch.Tensor:
    """
    Compute the Hessian matrix entry H[i, j].

    Compute the Hessian matrix entry H[i, j] = d²y / dx_i dx_j for the specified
    component of the output tensor, using reverse-mode autodiff.

    This function supports lazy evaluation, computing only the necessary parts
    of the Hessian matrix. It reuses cached results to avoid redundant computations
    for previously computed output-input-component triplets.

    Parameters
    ----------
    ys : torch.Tensor
        Output tensor of shape (batch_size, dim_y).
    xs : torch.Tensor
        Input tensor of shape (batch_size, dim_x).
    component : int
        ``ys[:, component]`` is used as y to compute the Hessian.
    i : int
        Row index of the Hessian matrix.
    j : int
        Column index of the Hessian matrix.

    Returns
    -------
    torch.Tensor
        The Hessian entry H[i, j] for the specified component.
    """
    return _Hessians(ys, xs, component=component, i=i, j=j)


def clear() -> None:
    """Clear cached Jacobians and Hessians."""
    _Jacobians.clear()
    _Hessians.clear()
