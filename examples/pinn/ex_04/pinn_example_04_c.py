#!/usr/bin/env python3
"""
Solves the ordinary differential equation (ODE) given by.

.. math::
    y' = p2 * y
    y(0) = p1

The script sets up and trains a Physics Informed Neural Network (PINN) to solve the ODE.
It performs the following steps:
1. Creates data files for training and testing.
2. Initializes the PINN model with the specified architecture.
3. Defines the conditions and parameters for training and testing.
4. Trains the model using the specified optimizer and callbacks.
5. Saves the trained model and PDE to files for later use.
6. Plots the training losses and displays them using matplotlib.

The results, including the model and the PDE, are saved to files for later use.
"""

import matplotlib.pyplot as plt
import torch
from torch import optim

from examples.pinn.ex_04.pde import (
    CallbackLog,
    Condition1,
    Condition2,
    RandomParameters,
)
from measure_uq.models import PINN
from measure_uq.pde import PDE, Conditions
from measure_uq.plots import plot_losses
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def make_data() -> None:
    """Create data files for training and testing."""
    torch.save(torch.linspace(0, 2, 101)[:, None], "p1_train.pt")
    torch.save(torch.tensor([[0.0]]), "p2_train.pt")

    torch.save(torch.linspace(0, 2, 201)[:, None], "p1_test.pt")
    torch.save(torch.tensor([[0.0]]), "p2_test.pt")


def main() -> None:
    """
    Set up and train the Physics Informed Neural Network (PINN) for solving the ODE.

    This function initializes the model, defines the conditions and parameters
    for training and testing, and trains the model using the specified optimizer
    and callbacks.
    """
    make_data()

    model = PINN([3, 20, 20, 20, 20, 1])

    conditions = [
        Condition1(points=torch.load("p1_train.pt")),
        Condition2(points=torch.load("p2_train.pt")),
    ]
    # Load training data
    conditions_train = Conditions(conditions=conditions)
    conditions_test = Conditions(conditions=conditions)

    parameters_train = RandomParameters(N=20)
    parameters_test = RandomParameters(N=100)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_parameters_every=5000,
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=2000,
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

    pde.save("pde-c.pickle")
    model.save("model-c.pt")

    plot_losses(trainer_data)

    plt.show()


if __name__ == "__main__":
    main()
