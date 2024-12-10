#!/usr/bin/env python3
"""
Solves the ordinary differential equation (ODE):

.. math::
    y'' = p_1 y

with the initial conditions:

.. math::
    y(0) = p_2, y'(0) = p_3
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
from measure_uq.utilities import cartesian_product_of_rows


def analytical_solution(t: float | np.ndarray, p: list | tuple):
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

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y : Tensor
            The solution of the ODE.
        x : Tensor
            The points where the condition is evaluated.

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

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y : Tensor
            The solution of the ODE.
        x : Tensor
            The point where the condition is evaluated.

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

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y : Tensor
            The solution of the ODE.
        x : Tensor
            The point where the condition is evaluated.

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
    model = PINN([4, 40, 40, 1])

    conditions_train = [
        Condition1(points=torch.linspace(0, 2, 101).reshape(-1, 1)),
        Condition2(points=tensor([[0.0]])),
        Condition3(points=tensor([[0.0]])),
    ]
    conditions_test = deepcopy(conditions_train)

    parameters_train = Parameters(
        values=cartesian_product_of_rows(
            tensor(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                ],
            ),
            tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
            ),
        ),
    )
    parameters_test = deepcopy(parameters_train)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=3000,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=5000, history_size=10),
    )

    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=[CallbackLog(trainer_data=trainer_data, print_every=100)],
    )

    trainer.train()

    fig1, ax1 = plot_losses(trainer_data)

    fig2, ax2 = plot_ode_on_grid(
        model=model,
        t=torch.linspace(0, 2, 100)[:, None],
        parameters=parameters_test,
        analytical_solution=analytical_solution,
        approximate_solution=lambda t, p: model(model.combine_input(t, p[None, :])),
        figsize=(20, 10),
    )

    plt.show()


if __name__ == "__main__":
    main()
