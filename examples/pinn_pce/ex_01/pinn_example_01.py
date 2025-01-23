#!/usr/bin/env python3

"""
Solution of the ordinary differential equation (ODE):

.. math::
    y' = p1 * y
    y(0) = p2
"""

# ruff: noqa: D103

from copy import deepcopy

import chaospy
import matplotlib.pyplot as plt
from torch import optim

from measure_uq.models import PINN_PCE
from measure_uq.pde import PDE
from measure_uq.plots import plot_losses
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData

from .pde import CallbackLog, Condition1, Condition2, RandomParameters


def main():
    joint = chaospy.J(
        chaospy.Uniform(1, 3),
        chaospy.Uniform(-2, 1),
    )

    expansion = chaospy.generate_expansion(
        5,
        joint,
        normed=True,
    )

    model = PINN_PCE(
        [1, 20, 20, 20, 20, 20, 20, len(expansion)],
        expansion,
    )

    conditions_train = [
        Condition1(N=100),
        Condition2(),
    ]
    conditions_test = deepcopy(conditions_train)

    parameters_train = RandomParameters(joint=joint, N=40)
    parameters_test = RandomParameters(joint=joint, N=40)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_conditions_every=(100,),
        resample_parameters_every=500,
        loss_weights=[1.0, 2.0],
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=5000,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=20, history_size=10, lr=0.1),
        test_every=10,
    )

    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=[
            CallbackLog(trainer_data=trainer_data, print_every=100),
        ],
    )

    pde.save("data/pde.pickle")

    trainer.train()

    model.save("data/model.pt")

    fig1, ax1 = plot_losses(trainer_data)

    plt.show()


if __name__ == "__main__":
    main()
