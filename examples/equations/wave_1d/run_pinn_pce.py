#!/usr/bin/env python3
r"""
Solution of the wave equation on the line using a PINN.

.. math::
    u_{tt} - a u_{xx} = 0, \quad (t,x) \in [0,T] \times [0,X]

    u(0, x) = \sin(k x)

    u_t(0, x) = 0, \quad x \in [0,X]

    u(t, 0) = 0, \quad t \in [0,T]

    u(t, X) = \sin(k X) \cos(\sqrt{a} k t), \quad t \in [0,T]
"""

import chaospy
import numpy as np
from chaospy import J
from torch import optim

from examples.equations.wave_1d.pde import (
    BoundaryConditionLeft,
    BoundaryConditionRight,
    InitialCondition,
    InitialVelocity,
    RandomParameters,
    Residual,
)
from measure_uq.callbacks import (
    CallbackLog,
    Callbacks,
    ModularPlotCallback,
)
from measure_uq.models import PINN_PCE
from measure_uq.networks import FeedforwardBuilder
from measure_uq.pde import PDE, Conditions
from measure_uq.plots import (
    ConditionLossPanel,
    TrainTestLossPanel,
)
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def train() -> None:
    """Train the PINN-PCE."""
    joint = J(
        chaospy.Uniform(1, 3),
        chaospy.Uniform(1, 3),
    )

    expansion = chaospy.generate_expansion(
        10,
        joint,
        normed=True,
    )

    device = "cuda:0"

    model = PINN_PCE(
        network_builder=FeedforwardBuilder(
            layer_sizes=[2, 64, 64, 64, 64, 64, len(expansion)],
            activation="tanh",
        ),
        expansion=expansion,
    )

    T = 2.0
    X = np.pi

    conditions = [
        Residual(X=X, T=T, Nt=20, Nx=50),
        InitialCondition(X=X, Nx=50),
        InitialVelocity(X=X, Nx=50),
        BoundaryConditionLeft(T=T, Nt=20),
        BoundaryConditionRight(X=X, T=T, Nt=20),
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
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=5000,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=10, history_size=10, lr=1),
        test_every=10,
        save_path="data/best_model_pinn_pce.pickle",
        device=device,
    )

    callbacks = Callbacks(
        trainer_data=trainer_data,
        callbacks=[
            CallbackLog(print_every=10),
            ModularPlotCallback(
                plot_every=100,
                panels=[
                    TrainTestLossPanel,
                    ConditionLossPanel,
                ],
            ),
        ],
    )

    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=callbacks,
    )

    trainer.train()

    pde.save("data/pde_pinn_pce.pickle")
    model.save("data/model_pinn_pce.pt")
    trainer.save("data/trainer_pinn_pce.pickle")


if __name__ == "__main__":
    train()
