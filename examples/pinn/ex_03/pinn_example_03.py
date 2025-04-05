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

import matplotlib.pyplot as plt
import torch
from torch import optim, tensor

from examples.pinn.ex_03.pde import (
    CallbackLog,
    Condition1,
    Condition2,
    Condition3,
    analytical_solution,
)
from measure_uq.models import PINN
from measure_uq.pde import PDE, Conditions, Parameters
from measure_uq.plots import plot_losses, plot_ode_on_grid
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData
from measure_uq.utilities import cartesian_product_of_rows


def main() -> None:
    """
    Main function to set up and train the Physics Informed Neural Network (PINN)
    for solving the ODE.

    This function initializes the model, defines the conditions and parameters
    for training and testing, and trains the model using the specified optimizer
    and callbacks.
    """
    model = PINN([4, 40, 40, 1])

    conditions_train = Conditions(
        conditions=[
            Condition1(points=torch.linspace(0, 2, 101).reshape(-1, 1)),
            Condition2(points=tensor([[0.0]])),
            Condition3(points=tensor([[1.0]])),
        ],
    )
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
        loss_weights=torch.FloatTensor([1.0, 1.0, 1.0]),
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=1000,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=50, history_size=10),
        test_every=10,
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
        approximate_solution=lambda t, p: model(t, p[None, :])[1],
        figsize=(20, 10),
    )

    plt.show()


if __name__ == "__main__":
    main()
