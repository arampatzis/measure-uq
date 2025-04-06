#!/usr/bin/env python3
"""
Script for training a PINN model.

.. math::
    y' = p2 * y
    y(0) = p1

The script sets up and trains a Physics Informed Neural Network (PINN) to solve the ODE.
It initializes the model, defines the conditions and parameters for training and testing
and trains the model using the specified optimizer and callbacks. The results, including
the model and the PDE, are saved to files for later use. Additionally, the script plots
the training losses and displays them using matplotlib.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from torch import optim, tensor

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


def main() -> None:
    """
    Train a PINN model to solve an ODE.

    This function sets up and trains a PINN model to solve an ordinary
    differential equation. It demonstrates the basic workflow of training
    a PINN model with fixed points and parameters.
    """
    model = PINN([3, 20, 20, 20, 20, 1])

    conditions = [
        Condition1(points=torch.linspace(0, 2, 101).reshape(-1, 1)),
        Condition2(points=tensor([[0.0]])),
    ]
    conditions_train = Conditions(conditions=conditions)
    conditions_test = deepcopy(conditions_train)

    parameters_train = RandomParameters(N=20)
    parameters_test = RandomParameters(N=100)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_parameters_every=1000,
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=5000,
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

    pde.save("pde-a.pickle")
    model.save("model-a.pt")

    plot_losses(trainer_data)

    plt.show()


if __name__ == "__main__":
    main()
