#!/usr/bin/env python3

"""
Solution of the heat equation on the line:

.. math::
    u_t - a / k^2 u_xx = 0
    u(0, x) = \sin(k x)
    u(t, 0) = 0
    u(t, \pi) = exp(-a t) \sin(\pi k)
"""

# ruff: noqa: D103

from copy import deepcopy

import chaospy
import matplotlib.pyplot as plt
from torch import optim
import torch

from measure_uq.models import PINN_PCE
from measure_uq.pde import PDE
from measure_uq.plots import plot_losses
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData

from chaospy import J
from pde import CallbackLog, RandomParameters
from pde import Residual, InitialCondition, BoundaryConditionLeft, BoundaryConditionRight


def main():
    
    torch.set_num_threads(12)
    torch.set_num_interop_threads(24)
    
    joint = J(
        chaospy.Uniform(1, 3),
        chaospy.Uniform(1, 3),
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
        Residual(Nt=20, Nx=50),
        InitialCondition(Nx=50),
        BoundaryConditionLeft(Nt=20),
        BoundaryConditionRight(Nt=20),
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
