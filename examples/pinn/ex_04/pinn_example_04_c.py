#!/usr/bin/env python3

"""
Solution of the ordinary differential equation (ODE):

.. math::
    y' = p2 * y
    y(0) = p1
"""


import matplotlib.pyplot as plt
import torch
from torch import optim

from measure_uq.models import PINN
from measure_uq.pde import PDE
from measure_uq.plots import plot_losses
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData

from .pde import CallbackLog, Condition1, Condition2, RandomParameters


def make_data():
    """Create data files for training and testing."""
    torch.save(torch.linspace(0, 2, 101)[:, None], "p1_train.pt")
    torch.save(torch.tensor([[0.0]]), "p2_train.pt")

    torch.save(torch.linspace(0, 2, 201)[:, None], "p1_test.pt")
    torch.save(torch.tensor([[0.0]]), "p2_test.pt")


def main():
    """
    Main function to set up and train the Physics Informed Neural Network (PINN)
    for solving the ODE.

    This function initializes the model, defines the conditions and parameters
    for training and testing, and trains the model using the specified optimizer
    and callbacks.
    """
    make_data()

    model = PINN([3, 20, 20, 20, 20, 1])

    conditions_train = [
        Condition1(points=torch.load("p1_train.pt")),
        Condition2(points=torch.load("p2_train.pt")),
    ]
    conditions_test = [
        Condition1(points=torch.load("p1_test.pt")),
        Condition2(points=torch.load("p2_test.pt")),
    ]

    parameters_train = RandomParameters(N=20)
    parameters_test = RandomParameters(N=100)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_parameters_every=5000,
    )

    pde.save("pde-c.pickle")

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

    trainer.save("model-c.pt")

    fig1, ax1 = plot_losses(trainer_data)

    plt.show()


if __name__ == "__main__":
    main()
