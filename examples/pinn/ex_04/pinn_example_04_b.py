#!/usr/bin/env python3
"""
Script for training a PINN model with resampling.

Solution of the ordinary differential equation (ODE):

.. math::
    y' = p2 * y
    y(0) = p1
"""


import matplotlib.pyplot as plt
from torch import optim, tensor

from examples.pinn.ex_04.pde import (
    CallbackLog,
    Condition1WithResampling,
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
    Train a PINN model with resampling.

    This function sets up and trains a PINN model with resampling of points and
    parameters during training. It demonstrates how to use the resampling
    functionality to improve the model's performance.
    """
    model = PINN([3, 20, 40, 60, 40, 20, 1])

    conditions = [
        Condition1WithResampling(N=100),
        Condition2(points=tensor([[0.0]])),
    ]
    conditions_train = Conditions(conditions=conditions)
    conditions_test = Conditions(conditions=conditions)

    parameters_train = RandomParameters(N=20)
    parameters_test = RandomParameters(N=100)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_parameters_every=500,
        resample_conditions_every=[
            100,
        ],
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

    pde.save("pde-b.pickle")
    model.save("model-b.pt")

    plot_losses(trainer_data)

    plt.show()


if __name__ == "__main__":
    main()
