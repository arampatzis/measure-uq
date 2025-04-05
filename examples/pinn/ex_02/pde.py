"""Definition of the ode and its parameters"""

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.pde import Condition


def analytical_solution(t: float | np.ndarray, p: list | tuple) -> np.ndarray | None:
    """
    Compute the exact solution for the ODE y'' = p1 * y with initial conditions:
    y(0) = p2, y'(0) = p3.

    Parameters
    ----------
    t : float or array_like
        Point(s) where the solution is evaluated.
    p : array_like
        Parameters [p1, p2, p3], where:
        - p1 : float
            Coefficient in the ODE.
        - p2 : float
            Initial condition y(0).
        - p3 : float
            Initial condition y'(0).

    Returns
    -------
    y : ndarray
        The value of y(t).
    """
    p1, p2, p3 = p

    # (linear solution)
    if np.isclose(p1, 0):
        return p2 + p3 * t

    # (hyperbolic solution)
    if p1 > 0:
        sqrt_p1 = np.sqrt(p1)
        C1 = p2
        C2 = p3 / sqrt_p1
        return C1 * np.cosh(sqrt_p1 * t) + C2 * np.sinh(sqrt_p1 * t)

    # (trigonometric solution)
    if p1 < 0:
        sqrt_neg_p1 = np.sqrt(-p1)
        C1 = p2
        C2 = p3 / sqrt_neg_p1
        return C1 * np.cos(sqrt_neg_p1 * t) + C2 * np.sin(sqrt_neg_p1 * t)

    return None


@dataclass(kw_only=True)
class Condition1(Condition):
    """Residual of the ODE."""

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
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
        assert x[:, 1][:, None].shape == y.shape

        dy_dt = jacobian(y, x, j=0)
        dy_dt = jacobian(dy_dt, x, j=0)
        f = x[:, 1][:, None] * y

        assert dy_dt.shape == f.shape

        return dy_dt - f


@dataclass(kw_only=True)
class Condition2(Condition):
    """Initial condition y(0) = p2."""

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            The input tensor.
        y : Tensor
            The solution tensor.

        Returns
        -------
        Tensor
            The value of the condition at the point.
        """
        assert y.shape == x[:, 2][:, None].shape

        return y - x[:, 2][:, None]


@dataclass(kw_only=True)
class Condition3(Condition):
    """Initial condition y'(0) = p3."""

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            The point where the condition is evaluated.
        y : Tensor
            The solution of the ODE.

        Returns
        -------
        Tensor
            The value of the condition at the point.
        """
        assert y.shape == x[:, 3][:, None].shape

        dy_dt = jacobian(y, x, j=0)

        return dy_dt - x[:, 3][:, None]


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
