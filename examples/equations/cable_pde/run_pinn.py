#!/usr/bin/env python3
r"""
Train a PINN on the cable equation with fixed parameters.

PDE residual: tau * V_t + V - lambda^2 * V_xx = 0

Conditions:
- Initial: V(0, x) = V_rest
- Left boundary (Dirichlet): V(t, 0) = V_inject
- Right boundary (Neumann): V_x(t, L) = 0
"""

from torch import optim

from examples.equations.cable_pde.pde import (
    BoundaryConditionLeft,
    BoundaryConditionRight,
    InitialCondition,
    ModelParameters,
    Residual,
)
from measure_uq.callbacks import CallbackLog, Callbacks, ModularPlotCallback
from measure_uq.models import PINN
from measure_uq.networks import FeedforwardBuilder
from measure_uq.pde import PDE, Conditions
from measure_uq.plots import ConditionLossPanel, TrainTestLossPanel
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def train() -> None:
    """Train the PINN on the cable equation."""

    device = "cpu"

    # Model with inputs (t, x) and parameters (tau, lambda, V_rest, V_inject)
    model = PINN(
        network_builder=FeedforwardBuilder(
            layer_sizes=[6, 64, 64, 64, 64, 1],
            activation="snake",
        ),
    )

    T_MAX = 100.0
    L_MAX = 3.0

    conditions = [
        Residual(Nt=256, Nx=128, T_max=T_MAX, L_max=L_MAX),
        InitialCondition(Nx=128, L_max=L_MAX),
        BoundaryConditionLeft(Nt=256, T_max=T_MAX),
        BoundaryConditionRight(Nt=256, T_max=T_MAX, L_max=L_MAX),
    ]

    conditions_train = Conditions(device=device, conditions=conditions)
    conditions_test = Conditions(device=device, conditions=conditions)

    # Fixed parameters packed into a 1x4 tensor; repeated internally by the framework
    parameters_train = ModelParameters()
    parameters_test = ModelParameters()

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_conditions_every=(50, ),
        resample_parameters_every=100,
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=1000,
        model=model,
        optimizer=optim.LBFGS(
            model.parameters(),
            max_iter=20,
            history_size=20,
            lr=1.0,
        ),
        test_every=10,
        device=device,
        save_path="data/best_model_cable_pinn.pickle",
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

    pde.save("data/pde_cable_pinn.pickle")
    model.save("data/model_cable_pinn.pickle")
    trainer.save("data/trainer_cable_pinn.pickle")


if __name__ == "__main__":
    train()


