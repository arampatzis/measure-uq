"""Definition of the ode and its parameters."""

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.pde import Condition


def analytical_solution(t: float | np.ndarray, p: list | tuple) -> np.ndarray | None:
    """
    Compute the exact solution for the second-order ODE y'' = p1 * y.

    Parameters
    ----------
    t : float or array_like
        Point(s) where the solution is evaluated.
    p : array_like
        Parameters [p1, p2, p3] where p1 is the coefficient in the ODE, p2 is the
        boundary condition at t = 0 (y(0)), and p3 is the boundary condition at
        t = 1 (y(1)).

    Returns
    -------
    y : ndarray
        The value of y(t).
    """
    p1, p2, p3 = p

    # (linear solution)
    if np.isclose(p1, 0):
        return p2 + (p3 - p2) * t

    # (hyperbolic solution)
    if p1 > 0:
        sqrt_p1 = np.sqrt(p1)
        np.cosh(sqrt_p1)
        C1 = p2
        C2 = (p3 - p2 * np.cosh(sqrt_p1)) / np.sinh(sqrt_p1)
        return C1 * np.cosh(sqrt_p1 * t) + C2 * np.sinh(sqrt_p1 * t)

    # (trigonometric solution)
    if p1 < 0:
        sqrt_neg_p1 = np.sqrt(-p1)
        np.cos(sqrt_neg_p1)
        C1 = p2
        C2 = (p3 - p2 * np.cos(sqrt_neg_p1)) / np.sin(sqrt_neg_p1)
        return C1 * np.cos(sqrt_neg_p1 * t) + C2 * np.sin(sqrt_neg_p1 * t)
    return None


@dataclass(kw_only=True)
class Condition1(Condition):
    """Residual of the ODE."""

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual of the ODE.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        y : Tensor
            The solution tensor.

        Returns
        -------
        Tensor
            The value of the residual at the points.
        """
        assert x[:, 1][:, None].shape == y.shape

        dy_dt = jacobian(y, x, j=0)
        dy_dt = jacobian(dy_dt, x, j=0)
        f = x[:, 1][:, None] * y

        assert dy_dt.shape == f.shape

        return dy_dt - f


@dataclass(kw_only=True)
class Condition2(Condition):
    """Condition for the first boundary condition."""

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the condition at the given points.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        y : Tensor
            The solution tensor.

        Returns
        -------
        Tensor
            The value of the condition at the points.
        """
        assert y.shape == x[:, 2][:, None].shape

        return y - x[:, 2][:, None]


@dataclass(kw_only=True)
class Condition3(Condition):
    """Condition for the second boundary condition."""

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the condition at the given points.

        Parameters
        ----------
        y : Tensor
            Value of y at the points.
        x : Tensor
            Points.

        Returns
        -------
        Tensor
            The value of the condition at the points.
        """
        assert y.shape == x[:, 3][:, None].shape

        return y - x[:, 3][:, None]


@dataclass
class CallbackLog(Callback):
    """
    Callback class for logging the training progress.

    Parameters
    ----------
    print_every : int
        Frequency of logging in terms of number of iterations.
    """

    print_every: int = 100

    def on_iteration_end(self) -> None:
        """Log the training loss at specified intervals."""
        if (
            self.trainer_data.iteration % self.print_every == 0
            or self.trainer_data.iteration == self.trainer_data.iterations - 1
        ):
            print(
                f"{self.trainer_data.losses_train.i[-1]:10}:  "
                f"{self.trainer_data.losses_train.v[-1]:.5e}",
            )
