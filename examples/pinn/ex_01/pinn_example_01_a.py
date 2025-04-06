#!/usr/bin/env python3

"""
Solution of the ordinary differential equation (ODE).

.. math::

    y' = p1 * y
    y(0) = p2

The script sets up and trains a Physics Informed Neural Network (PINN) to solve
the ODE. It initializes the model, defines the conditions and parameters for
training and testing, and trains the model using the specified optimizer and
callbacks.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from torch import optim, tensor

from examples.pinn.ex_01.pde import (
    CallbackLog,
    Condition1,
    Condition2,
    analytical_solution,
)
from measure_uq.models import PINN
from measure_uq.pde import PDE, Conditions, Parameters
from measure_uq.plots import plot_losses, plot_ode_on_grid
from measure_uq.stoppers import TrainingLossStopper
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def main() -> None:
    """
    Set up and train the Physics Informed Neural Network (PINN) for solving the ODE.

    This function initializes the model, defines the conditions and parameters for
    training and testing, and trains the model using the specified optimizer and
    callbacks.
    """
    model = PINN([3, 20, 20, 1])

    conditions_train = Conditions(
        conditions=[
            Condition1(points=torch.linspace(0, 2, 101).reshape(-1, 1)),
            Condition2(points=tensor([[0.0]])),
        ],
    )
    conditions_test = Conditions(
        conditions=[
            Condition1(points=torch.linspace(0, 2, 101).reshape(-1, 1)),
            Condition2(points=tensor([[0.0]])),
        ],
    )

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

    trainer_data = TrainerData(
        pde=pde,
        iterations=1000,
        model=model,
        optimizer=optim.Adam(
            model.parameters(),
            lr=0.01,
            amsgrad=True,
        ),
    )

    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=[
            CallbackLog(trainer_data=trainer_data, print_every=100),
        ],
        stoppers=[
            TrainingLossStopper(trainer_data=trainer_data, patience=100, delta=1e-9),
        ],
    )

    trainer.train()

    fig1, ax1 = plot_losses(trainer_data)

    fig2, ax2 = plot_ode_on_grid(
        model=model,
        t=torch.linspace(0, 3, 100)[:, None],
        parameters=parameters_test,
        analytical_solution=analytical_solution,
        approximate_solution=lambda t, p: model(t, p[None, :])[1],
        figsize=(20, 10),
    )

    plt.show()


if __name__ == "__main__":
    main()
