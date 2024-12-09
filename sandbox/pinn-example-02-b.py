#!/usr/bin/env python3

r"""
Solves the second-order ODE:

.. math::
    y'' = p_1 \\cdot y

with the boundary conditions:

.. math::
    y(0) = p_2
    y(1) = p_3
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
from measure_uq.trainer import Trainer
from measure_uq.utilities import cartesian_product_of_rows


def analytical_solution(t: float | np.ndarray, p: list | tuple):
    """
    Compute the exact solution for the second-order ODE y'' = p1 * y.

    Parameters
    ----------
    t : float or array_like
        Point(s) where the solution is evaluated.
    p : array_like
        Parameters [p1, p2, p3] where:
        - p1 : float
            Coefficient in the ODE.
        - p2 : float
            Boundary condition at t = 0 (y(0)).
        - p3 : float
            Boundary condition at t = 1 (y(1)).

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

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y : Tensor
            Value of y at the points.
        x : Tensor
            Points.

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

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
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
        assert y.shape == x[:, 2][:, None].shape

        return y - x[:, 2][:, None]


@dataclass(kw_only=True)
class Condition3(Condition):
    """Condition for the second boundary condition."""

    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Evaluates the condition at the given points.

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
    """Callback for logging the loss values during training."""

    print_every: int = 100

    def on_iteration_end(self):
        """
        Logs the loss values during training.

        Parameters
        ----------
        trainer : Trainer
            Trainer object.
        """
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
    model = PINN([4, 40, 40, 1])

    conditions_train = [
        Condition1(points=torch.linspace(0, 2, 101).reshape(-1, 1)),
        Condition2(points=tensor([[0.0]])),
        Condition3(points=tensor([[1.0]])),
    ]
    conditions_test = deepcopy(conditions_train)

    parameters_train = Parameters(
        values=cartesian_product_of_rows(
            tensor(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                    [-1.0],
                    [-2.0],
                    [-3.0],
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
        loss_weights=torch.tensor([10.0, 1.0, 1.0]),
    )

    trainer = Trainer(
        pde=pde,
        iterations=400,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=50, history_size=10),
        callbacks=[CallbackLog(print_every=100)],
    )

    trainer.train()

    fig1, ax1 = plot_losses(pde, trainer)

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
