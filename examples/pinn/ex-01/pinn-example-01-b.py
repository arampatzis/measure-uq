#!/usr/bin/env python3

r"""
Solves the ordinary differential equation (ODE):

.. math::
    y' = p_2 y

with the initial condition:

.. math::
    y'(0) = p_1
"""

from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, optim, tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.models import PINN
from measure_uq.pde import PDE, Condition, Parameters
from measure_uq.plots import plot_losses, plot_ode_on_grid
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def analytical_solution(t: float | np.ndarray, p: list | tuple):
    """
    Computes the exact solution for the ODE dy/dt = p1 * y / p2, y'(0) = 0.

    Parameters
    ----------
    t : float or np.ndarray
        Point(s) where the solution is evaluated.
    p : list or tuple
        Parameters [p1, p2] where:
        - p1: Coefficient in the ODE.
        - p2: Coefficient in the ODE.

    Returns
    -------
    np.ndarray
        The value of y(t).
    """
    return p[0] * torch.exp(p[1] * t) / p[1]


@dataclass(kw_only=True)
class Condition1(Condition):
    """
    Represents the residual of the ODE.

    Parameters
    ----------
    points : Tensor
        The points at which to evaluate the condition.
    """

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
    """
    Represents the initial condition of the ODE.

    Parameters
    ----------
    points : Tensor
        The points at which to evaluate the condition.
    """

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Evaluate the boundary condition by computing the difference
        between the derivative and the given value.

        Parameters
        ----------
        y : Tensor
            The solution tensor.
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The result of the boundary condition evaluation.
        """
        assert y.shape == x[:, 1][:, None].shape

        dy_dt = jacobian(y, x, j=0)

        return dy_dt - x[:, 1][:, None]


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

    def on_iteration_end(self):
        """Log the training loss at specified intervals."""
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
    model = PINN([3, 20, 20, 1])

    conditions_train = [
        Condition1(points=torch.linspace(0, 2, 101).reshape(-1, 1)),
        Condition2(points=tensor([[0.0]])),
    ]
    conditions_test = deepcopy(conditions_train)

    parameters_train = Parameters(
        values=torch.cartesian_prod(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([-1.0, -1.5]),
        ),
    )
    parameters_test = deepcopy(parameters_train)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        loss_weights=torch.tensor([1.0, 1.0]),
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=3000,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=50, history_size=10, lr=0.5),
    )

    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=[CallbackLog(trainer_data=trainer_data, print_every=100)],
    )

    trainer.train()

    fig1, ax1 = plot_losses(trainer_data)

    fig2, ax2 = plot_ode_on_grid(
        model=model,
        t=torch.linspace(0, 3, 100)[:, None],
        parameters=parameters_test,
        analytical_solution=analytical_solution,
        approximate_solution=lambda t, p: model(model.combine_input(t, p[None, :])),
        figsize=(20, 10),
    )

    plt.show()


if __name__ == "__main__":
    main()
