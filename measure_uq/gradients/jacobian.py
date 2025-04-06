"""
Provide functions for computing Jacobian matrices.

Compute Jacobian matrix J as J[i, j] = dy_i / dx_j, where i = 0, ..., dim_y - 1
and j = 0, ..., dim_x - 1.

This module provides functions for computing Jacobian matrices using reverse-mode
automatic differentiation.

The functions in this module are used for computing gradients of the loss function
of a model with respect to the parameters of the model.

The functions in this module are a modification of the `jacobian` module, copied from
`deepxde`.
Licensed under the GNU Lesser General Public License (LGPL) 2.1.
See the LICENSE.LGPL file in the root directory for details.
Original source: https://github.com/lululxvi/deepxde

"""

from abc import ABC, abstractmethod

import torch


class Jacobian(ABC):
    """
    Abstract base class for computing Jacobian matrices.

    Compute `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
    J as J[i, j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and j = 0, ..., dim_x - 1.

    It is lazy evaluation, i.e., it only computes J[i, j] when needed.

    Parameters
    ----------
        ys: torch.Tensor
            Output Tensor of shape (batch_size, dim_y).
        xs: torch.Tensor
            Input Tensor of shape (batch_size, dim_x).
    """

    def __init__(self, ys: torch.Tensor, xs: torch.Tensor) -> None:
        """
        Initialize the Jacobian.

        Parameters
        ----------
        ys : torch.Tensor
            Output tensor of shape (batch_size, dim_y).
        xs : torch.Tensor
            Input tensor of shape (batch_size, dim_x).
        """
        self.ys = ys
        self.dim_y = ys.shape[1]
        self.dim_x = xs.shape[1]

        self.J: dict[int | tuple[int, int], torch.Tensor] = {}

    @abstractmethod
    def __call__(self, i: int | None = None, j: int | None = None) -> torch.Tensor:
        """
        Return (`i`, `j`)th entry J[`i`, `j`].

        Note that:
        - If `i` is ``None``, returns the jth column J[:, `j`].
        - If `j` is ``None``, returns the ith row J[`i`, :], i.e., the gradient of y_i.
        - `i` and `j` cannot be both ``None``.

        Add these checks to the __call__ method:
        if i is None and j is None:
            if self.dim_x > 1 or self.dim_y > 1:
                raise ValueError("i and j cannot be both None.")
            i = 0
            j = 0
        if i is not None and not 0 <= i < self.dim_y:
            raise ValueError(f"i={i} is not valid.")
        """
        raise NotImplementedError("This method should be implemented by the subclass.")


class Jacobians:
    """
    Compute multiple Jacobians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self, jacobian_class: type[Jacobian]):
        self.jacobian_class = jacobian_class
        self.Js: dict = {}

    def __call__(
        self,
        ys: torch.Tensor,
        xs: torch.Tensor,
        i: int | None = None,
        j: int | None = None,
    ) -> torch.Tensor:
        """
        Compute the Jacobian matrix.

        For backend tensorflow and pytorch, self.Js cannot be reused across iteration.
        For backend pytorch, we need to reset self.Js in each iteration to avoid
        memory leak.

        For backend pytorch, in each iteration, ys and xs are new tensors
        converted from np.ndarray, so self.Js will increase over iteration.

        Example
        -------
        mydict = {}

        def f(x):
            print(mydict)
            y = 1 * x
            print(hash(y), hash(x))
            mydict[(y, x)] = 1
            print(mydict)

        for i in range(2):
            x = np.random.random((3, 4))
            x = torch.from_numpy(x)
            x.requires_grad_()
            f(x)
        """
        key = (ys, xs)

        if key not in self.Js:
            self.Js[key] = self.jacobian_class(ys, xs)
        return self.Js[key](i, j)

    def clear(self) -> None:
        """Clear cached Jacobians."""
        self.Js = {}
