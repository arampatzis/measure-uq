"""
Description of the heat equation on the line:

.. math::
    u_t - a / k^2 u_xx = 0
    u(0, x) = \sin(k x)
    u(t, 0) = 0
    u(t, \pi) = exp(-a t) \sin(\pi k)
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
from measure_uq.utilities import cartesian_product_of_rows


def analytical_solution(t, x, p):
    """Analytical solution of the heat equation on the line."""
    y = np.exp(-p[0] * t) * np.sin(p[1] * x)
    return y


@dataclass(kw_only=True)
class Residual(Condition):
    Nt: int    
    Nx: int

    def __post_init__(self):
        assert self.Nt > 0
        assert self.Nx > 0
        super().__post_init__()

    def sample_points(self):
        print("Re-sample PDE variables for Residual")

        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, 1, (self.Nt, 1))),
            tensor(np.random.uniform(0, np.pi, (self.Nx, 1))),
        ).float()
        
    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        
        assert x.shape[0] == y.shape[0]
        
        dy_dt = jacobian(y, x, j=0)
        dy_dx = jacobian(y, x, j=1)
        dy_d2 = jacobian(dy_dx, x, j=1)

        c = x[:, 2][:, None] / x[:, 3][:, None] ** 2
        f = c * dy_d2
        
        assert dy_dt.shape == f.shape

        return dy_dt - f


@dataclass(kw_only=True)
class InitialCondition(Condition):
    Nx: int
    
    def sample_points(self):
        print("Re-sample PDE variables for InitialCondition")
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.]]),
            tensor(np.random.uniform(0, np.pi, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape[0] == y.shape[0]
        
        kx = x[:, 3][:, None] * x[:, 1][:, None]
        
        return y - torch.sin(kx)


@dataclass(kw_only=True)
class BoundaryConditionLeft(Condition):
    Nt: int
    
    def sample_points(self):
        print("Re-sample PDE variables for BoundaryConsitionLeft")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, 1, (self.Nt, 1))),
            torch.tensor([[0.]]),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        return y


@dataclass(kw_only=True)
class BoundaryConditionRight(Condition):
    Nt: int
    
    def sample_points(self):
        print("Re-sample PDE variables for BoundaryConsitionRight")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, 1, (self.Nt, 1))),
            torch.tensor([[np.pi]]),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        at = x[:, 2][:, None] * x[:, 0][:, None]
        pix = np.pi*x[:, 1][:, None]
        
        return y - torch.exp(at) * torch.sin(pix)


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    joint: chaospy.J
    N: int

    def sample_values(self):
        print("Re-sample PDE parameters")
        self.values = tensor(self.joint.sample(self.N).T).float()


@dataclass
class CallbackLog(Callback):
    """A callback that prints the loss value at each iteration."""

    print_every: int = 100

    def on_iteration_end(self):
        """Prints the loss value at each iteration."""
        if (
            self.trainer_data.iteration % self.print_every == 0
            or self.trainer_data.iteration == self.trainer_data.iterations - 1
        ):
            print(
                f"{self.trainer_data.losses_train.index[-1]:10}:  "
                f"{self.trainer_data.losses_train.values[-1]:.5e}",
            )
