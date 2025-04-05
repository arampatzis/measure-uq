"""
Defines the ordinary differential equation (ODE) and its parameters, and provides
the analytical solution for the ODE. The ODE is given by:

.. math::
    y' = p1 * y
    y(0) = p2

The script includes the following components:
- Analytical solution function to compute the exact solution of the ODE.
- Condition1 class to represent the residual of the ODE.
- Necessary imports and dataclass definitions.

The analytical solution function takes time points and parameters as inputs and returns
the exact solution of the ODE at those points. The Condition1 class evaluates the
residual of the ODE given the input and output tensors.

This script is intended to be used as part of a Physics Informed Neural Network (PINN)
framework for solving ODEs.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters


def analytical_solution(
    t: float | Tensor,
    p: Tensor,
) -> Tensor:
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
class Condition1WithResampling(Condition1):
    """Represents the residual of the ODE."""

    N: int

    def sample_points(self) -> None:
        """Sample random points for the ODE residual evaluation."""
        print("Re-sample ODE variables for Condition1")

        self.points = torch.from_numpy(np.random.uniform(0, 1, (self.N, 1))).float()


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


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    """
    Parameters of the ODE sampled from a uniform distribution.

    Parameters
    ----------
    N : int
        Number of parameters to sample.

    Attributes
    ----------
    values : Tensor
        The sampled values of the parameters.
    """

    N: int

    def sample_values(self) -> None:
        """
        Sample random values for the parameters from a uniform distribution.

        This method assigns sampled values to the `values` attribute, ensuring
        they are ready for gradient computation.

        Attributes
        ----------
        values : Tensor
            The sampled values of the parameters, with shape (N, 2). The first
            column contains values sampled from a uniform distribution in the
            range [1, 3], and the second column contains values sampled from
            a uniform distribution in the range [-2, 1].
        """
        print("Re-sample ODE parameters")
        self.values = torch.cat(
            (
                torch.from_numpy(np.random.uniform(1, 3, (self.N, 1))).float(),
                torch.from_numpy(np.random.uniform(-2, 1, (self.N, 1))).float(),
            ),
            dim=1,
        )


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
