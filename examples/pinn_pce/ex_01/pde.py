"""
Defines the ordinary differential equation (ODE) and the parameters class for use in a
Physics Informed Neural Network with Polynomial Chaos Expansion (PINN-PCE):

.. math::
    y' = p1 * y
    y(0) = p2

The script includes the following components:
1. Definition of the ODE and its analytical solution.
2. Implementation of the Condition1 and Condition2 classes for sampling points
and evaluating the ODE.
3. Definition of the Parameters class for managing parameter values.
4. Utility functions for computing the Jacobian and handling callbacks.

The results are displayed using torch and numpy.
"""

# ruff: noqa: D101 D102 D103

from dataclasses import dataclass

import chaospy
import numpy as np
import torch
from torch import Tensor, tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters


def analytical_solution(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution of the ODE.

    Parameters
    ----------
    t : torch.Tensor
        Time points.
    p : torch.Tensor
        Parameters.

    Returns
    -------
    torch.Tensor
    """
    return p[0] * torch.exp(p[1] * t)


@dataclass(kw_only=True)
class Condition1(Condition):
    """
    Residual ot the ODE.

    Parameters
    ----------
    N : int
        Number of points to sample.
    """

    N: int

    def __post_init__(self) -> None:
        assert self.N > 0
        super().__post_init__()

    def sample_points(self) -> None:
        """Sample points for the ODE."""
        print("Sample ODE variables for Condition1")

        self.points = torch.from_numpy(
            np.random.uniform(0, 1, (self.N, 1)),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the ODE."""
        assert x[:, 2][:, None].shape == y.shape
        dy_dt = jacobian(y, x, j=0)

        f = x[:, 2][:, None] * y
        assert dy_dt.shape == f.shape

        return dy_dt - f


@dataclass(kw_only=True)
class Condition2(Condition):
    """Initial condition of the ODE."""

    def sample_points(self) -> None:
        """Sample points for the ODE."""
        print("Sample ODE variables for Condition2")
        self.points = tensor([[0.0]])

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the ODE."""
        assert y.shape == x[:, 1][:, None].shape
        return y - x[:, 1][:, None]


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    """
    Class representing random parameters for the ODE.

    Parameters
    ----------
    joint : chaospy.J
        Joint probability distribution of the parameters.
    N : int
        Number of samples to generate.

    Methods
    -------
    sample_values()
        Samples values for the ODE parameters based on the joint distribution.
    """

    joint: chaospy.J
    N: int

    def sample_values(self) -> None:
        """Sample values for the ODE parameters based on the joint distribution."""
        print("Sample ODE parameters")
        self.values = tensor(self.joint.sample(self.N).T).float()


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
                f"{len(self.trainer_data.losses_train):10}:  "
                f"{self.trainer_data.losses_train[-1]:.5e}",
            )
