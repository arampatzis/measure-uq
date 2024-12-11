#!/usr/bin/env python3

"""
Solution of the ordinary differential equation (ODE):

.. math::
    y' = p2 * y
    y(0) = p1
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, optim, tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.models import PINN
from measure_uq.pde import PDE, Condition, Parameters
from measure_uq.plots import plot_losses
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def analytical_solution(t: float | np.ndarray, p: list | tuple):
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

    N: int

    def sample_points(self):
        """Sample random points for the ODE residual evaluation."""
        print("Re-sample ODE variables for Condition1")

        self.points = torch.from_numpy(np.random.uniform(0, 2, (self.N, 1))).float()

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Evaluate the residual.

        Parameters
        ----------
        y : Tensor
            The solution tensor.
        x : Tensor
            The input tensor.

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

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Evaluate the initial condition by computing the difference between the
        derivative and the given value.

        Parameters
        ----------
        y : Tensor
            The solution tensor.
        x : Tensor
            The input tensor.

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

    def sample_values(self):
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


def main():
    """
    Main function to set up and train the Physics Informed Neural Network (PINN)
    for solving the ODE.

    This function initializes the model, defines the conditions and parameters
    for training and testing, and trains the model using the specified optimizer
    and callbacks.
    """
    model = PINN([3, 20, 40, 60, 40, 20, 1])

    conditions_train = [
        Condition1(N=100),
        Condition2(points=tensor([[0.0]])),
    ]
    conditions_test = [
        Condition1(N=100),
        Condition2(points=tensor([[0.0]])),
    ]

    parameters_train = RandomParameters(N=20)
    parameters_test = RandomParameters(N=100)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_parameters_every=1000,
        resample_conditions_every=[
            100,
        ],
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=15000,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=50, history_size=10, lr=0.5),
        test_every=10,
    )

    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=[
            CallbackLog(trainer_data=trainer_data, print_every=100),
        ],
    )

    trainer.train()

    fig1, ax1 = plot_losses(trainer_data)

    plt.show()


if __name__ == "__main__":
    main()
