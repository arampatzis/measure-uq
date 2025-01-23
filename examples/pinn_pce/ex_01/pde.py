"""Definition of the ODE and the parameters class"""

# ruff: noqa: D101 D102 D103

from dataclasses import dataclass

import chaospy
import numpy as np
import torch
from torch import Tensor, tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters


def analytical_solution(t, p):
    return p[0] * torch.exp(p[1] * t)


@dataclass(kw_only=True)
class Condition1(Condition):
    N: int

    def __post_init__(self):
        assert self.N > 0
        super().__post_init__()

    def sample_points(self):
        print("Re-sample ODE variables for Condition1")

        self.points = torch.from_numpy(
            np.random.uniform(0, 1, (self.N, 1)),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        assert x[:, 2][:, None].shape == y.shape
        dy_dt = jacobian(y, x, j=0)

        f = x[:, 2][:, None] * y
        assert dy_dt.shape == f.shape

        return dy_dt - f


@dataclass(kw_only=True)
class Condition2(Condition):
    def sample_points(self):
        print("Re-sample ODE variables for Condition2")
        self.points = tensor([[0.0]])

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        assert y.shape == x[:, 1][:, None].shape
        return y - x[:, 1][:, None]


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    joint: chaospy.J
    N: int

    def sample_values(self):
        print("Re-sample ODE parameters")
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
