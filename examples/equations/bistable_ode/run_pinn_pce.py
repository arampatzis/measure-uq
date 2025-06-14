#!/usr/bin/env python3
r"""
Bistable ODE Simulation.

This script simulates the bistable ODE:

.. math::

    \frac{dy}{dt} = r (y - 1) (2 - y) (y - 3), \quad y(0) = y_0

where :math:`r \sim \mathcal{U}(0.8, 1.2)` and :math:`y_0 \sim \mathcal{U}(0, 4)`.
"""

import chaospy
from chaospy import J
from torch import optim

from examples.equations.bistable_ode.ode import (
    InitialCondition,
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
        chaospy.Uniform(0, 4),
        chaospy.Uniform(0.8, 1.2),
    )

    expansion = chaospy.generate_expansion(
        10,
        joint,
        normed=True,
    )

    device = "cuda:0"

    model = PINN_PCE(
        network_builder=FeedforwardBuilder(
            layer_sizes=[1, 32, 32, 32, 32, 32, 32, len(expansion)],
            activation="tanh",
        ),
        expansion=expansion,
    )

    T = 8.0

    conditions = [
        Residual(T=T, Nt=100),
        InitialCondition(),
    ]
    conditions_train = Conditions(device=device, conditions=conditions)
    conditions_test = Conditions(device=device, conditions=conditions)

    parameters_train = RandomParameters(device=device, joint=joint, N=50)
    parameters_test = RandomParameters(device=device, joint=joint, N=50)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_conditions_every=(50,),
        resample_parameters_every=50,
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=1000,
        model=model,
        optimizer=optim.LBFGS(model.parameters(), max_iter=20, history_size=20, lr=1),
        test_every=10,
        device=device,
        save_path="data/best_model_pinn_pce.pickle",
    )

    callbacks = Callbacks(
        trainer_data=trainer_data,
        callbacks=[
            CallbackLog(print_every=10),
            ModularPlotCallback(
                plot_every=10,
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
