#!/usr/bin/env python3
r"""Solution of the heat equation on the line.

.. math::
    u_t - a / k^2 u_xx = 0
    u(0, x) = \sin(k x)
    u(t, 0) = 0
    u(t, \pi) = exp(-a t) \sin(\pi k)
"""


import chaospy
import matplotlib.pyplot as plt
from chaospy import J
from torch import optim

from examples.pinn_pce.ex_02.pde import (
    BoundaryConditionLeft,
    BoundaryConditionRight,
    CallbackLog,
    InitialCondition,
    RandomParameters,
    Residual,
)
from measure_uq.models import PINN_PCE
from measure_uq.pde import PDE, Conditions
from measure_uq.plots import plot_losses
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def main() -> None:
    """Solve the heat equation on the line using the PINN-PCE method."""
    joint = J(
        chaospy.Uniform(1, 3),
        chaospy.Uniform(1, 3),
    )

    expansion = chaospy.generate_expansion(
        5,
        joint,
        normed=True,
    )

    device = "cuda:0"

    model = PINN_PCE(
        [2, 20, 20, 20, 20, 20, 20, len(expansion)],
        expansion,
    ).to(device)

    conditions = [
        Residual(Nt=10, Nx=20),
        InitialCondition(Nx=20),
        BoundaryConditionLeft(Nt=20),
        BoundaryConditionRight(Nt=20),
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
        resample_conditions_every=(100,),
        resample_parameters_every=500,
        loss_weights=[1.0, 2.0, 1.0, 1.0],
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=5000,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=5, history_size=5, lr=1),
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
