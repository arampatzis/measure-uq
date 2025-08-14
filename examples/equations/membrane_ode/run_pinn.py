#!/usr/bin/env python3
r"""
Passive membrane ODE with a PINN.

We solve

    tau dV/dt = -V + R I,   V(0) = V_rest

with random parameters V_rest ~ U(-70, -60), I ~ U(1, 3).
"""

import chaospy
from chaospy import J
from torch import optim

from examples.equations.membrane_ode.ode import (
    InitialCondition,
    RandomParameters,
    Residual,
)
from measure_uq.callbacks import (
    CallbackLog,
    Callbacks,
    ModularPlotCallback,
)
from measure_uq.models import PINN
from measure_uq.networks import FeedforwardBuilder
from measure_uq.pde import PDE, Conditions
from measure_uq.plots import (
    ConditionLossPanel,
    TrainTestLossPanel,
)
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def train() -> None:
    """Train the PINN for the membrane ODE."""
    joint = J(
        chaospy.Uniform(-70.0, -60.0),  # V_rest
        chaospy.Uniform(1.0, 3.0),      # I
    )

    device = "cpu"

    model = PINN(
        network_builder=FeedforwardBuilder(
            layer_sizes=[3, 32, 32, 32, 32, 1],
            activation="snake",
        ),
    )

    T = 50.0

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
        optimizer=optim.LBFGS(
            model.parameters(),
            max_iter=20,
            history_size=20,
            lr=1,
        ),
        test_every=10,
        device=device,
        save_path="data/best_model_pinn.pickle",
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

    pde.save("data/pde_pinn.pickle")
    model.save("data/model_pinn.pickle")
    trainer.save("data/trainer_pinn.pickle")


if __name__ == "__main__":
    train()


