#!/usr/bin/env python3
"""
Solve an ODE using a PINN-PCE model.

The ODE is defined as,

.. math::
    y' = p1 * y
    y(0) = p2

The script performs the following steps:
1. Defines the joint probability distribution for the parameters using chaospy.
2. Initializes the PINN_PCE model.
3. Sets up the conditions and parameters for training and testing.
4. Trains the model using the specified optimizer and callbacks.
5. Saves the trained model and PDE.
6. Plots the training losses.

The results are displayed using matplotlib.
"""

# ruff: noqa: D103

import chaospy
import matplotlib.pyplot as plt
import torch
from torch import optim

from examples.pinn_pce.ex_01.pde import (
    CallbackLog,
    Condition1,
    Condition2,
    RandomParameters,
)
from measure_uq.models import PINN_PCE
from measure_uq.pde import PDE, Conditions
from measure_uq.plots import plot_losses
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def main() -> None:
    """
    Set up and train a PINN-PCE model to solve an ODE.

    This function initializes the model, defines the conditions and parameters
    for training and testing, and trains the model using the specified optimizer
    and callbacks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    joint = chaospy.J(
        chaospy.Uniform(1, 3),
        chaospy.Uniform(-2, 1),
    )

    expansion = chaospy.generate_expansion(
        5,
        joint,
        normed=True,
    )

    device = "cuda:0"

    model = PINN_PCE(
        [1, 20, 20, 20, 20, 20, 20, len(expansion)],
        expansion,
    ).to(torch.device(device))

    conditions = [
        Condition1(N=100),
        Condition2(),
    ]

    conditions_train = Conditions(device=device, conditions=conditions)
    conditions_test = Conditions(device=device, conditions=conditions)

    parameters_train = RandomParameters(device=device, joint=joint, N=40)
    parameters_test = RandomParameters(device=device, joint=joint, N=40)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_conditions_every=(50,),
        resample_parameters_every=100,
        loss_weights=[1.0, 2.0],
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=500,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=5, history_size=5, lr=1),
        test_every=10,
    )

    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=[
            CallbackLog(trainer_data=trainer_data, print_every=5),
        ],
    )

    pde.save("data/pde.pickle")

    trainer.train()

    model.save("data/model.pt")

    fig1, ax1 = plot_losses(trainer_data)

    plt.show()


if __name__ == "__main__":
    main()
