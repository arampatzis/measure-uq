"""Definition of the ode and its parameters"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.pde import Condition


def analytical_solution(t: float | np.ndarray, p: list | tuple) -> np.ndarray:
    """
    Compute the exact solution for the ODE dy/dt = p1 * y, y(0) = p2.

    Parameters
    ----------
    t : float or array_like
        Point(s) where the solution is evaluated.
    p : array_like
        Parameters [p1, p2] where:
        - p1 : float
            Coefficient in the ODE.
        - p2 : float
            Initial condition y(0).

    Returns
    -------
    y : ndarray
        The value of y(t).
    """
    return p[0] * torch.exp(p[1] * t)


@dataclass(kw_only=True)
class Condition1(Condition):
    """Represents the residual of the ODE."""

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        y : Tensor
            The solution tensor.

        Returns
        -------
        Tensor
            The result of the condition evaluation.
        """
        assert x[:, 2][:, None].shape == y.shape

        dy_dt = jacobian(y, x, j=0)
        f = x[:, 2][:, None] * y

        assert dy_dt.shape == f.shape

        return dy_dt - f


@dataclass(kw_only=True)
class Condition2(Condition):
    """Represents the initial condition of the ODE."""

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the initial condition by computing the difference between the
        derivative and the given value.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        y : Tensor
            The solution tensor.

        Returns
        -------
        Tensor
            The result of the condition evaluation.
        """
        assert y.shape == x[:, 1][:, None].shape

        return y - x[:, 1][:, None]


@dataclass
class CallbackLog(Callback):
    """A callback that prints the loss value at each iteration."""

    print_every: int = 100

    def on_iteration_end(self) -> None:
        """Prints the loss value at each iteration."""
        if (
            self.trainer_data.iteration % self.print_every == 0
            or self.trainer_data.iteration == self.trainer_data.iterations - 1
        ):
            print(
                f"{self.trainer_data.losses_train.i[-1]:10}:  "
                f"{self.trainer_data.losses_train.v[-1]:.5e}",
            )
