#!/usr/bin/env python3

"""
Solution of the ordinary differential equation (ODE):

.. math::
    y' = p1 * y
    y(0) = p2
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
from measure_uq.stoppers import TrainingLossStopper
from measure_uq.trainer import Trainer


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


@dataclass
class CallbackLog(Callback):
    """A callback that prints the loss value at each iteration."""

    print_every: int = 100

    def on_iteration_end(self):
        """Prints the loss value at each iteration."""
        if (
            self.trainer.iteration % self.print_every == 0
            or self.trainer.iteration == self.trainer.iterations - 1
        ):
            print(
                f"{self.trainer.losses_train.index[-1]:10}:  "
                f"{self.trainer.losses_train.values[-1]:.5e}",
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
    )

    trainer = Trainer(
        pde=pde,
        iterations=10000,
        model=model,
        optimizer=optim.Adam(
            model.parameters(),
            lr=0.01,
            amsgrad=True,
        ),
        callbacks=[CallbackLog(print_every=100)],
        stoppers=[TrainingLossStopper(patience=100, delta=1e-9)],
    )

    trainer.train()

    fig1, ax1 = plot_losses(pde, trainer)

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
