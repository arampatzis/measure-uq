#!/usr/bin/env python3
r"""
Solution of the Lotka-Volterra system.

.. math::
    \frac{dx}{dt} =  a x - b x y
    \frac{dy}{dt} = -c y + d x y
"""

# ruff: noqa: ERA001

import click
import torch
from torch import optim

from examples.systems.lotka_volterra.ode import (
    InitialCondition,
    RandomParameters,
    Residual,
    analytical_solution,
)
from measure_uq.callbacks import (
    CallbackLog,
    Callbacks,
    ModularPlotCallback,
    call_callbacks,
)
from measure_uq.models import PINN
from measure_uq.pde import PDE, Conditions
from measure_uq.plots import (
    ConditionLossPanel,
    LossPanel,
    ResidualPanel,
    SolutionComparisonPanel,
)
from measure_uq.stoppers import Stoppers, TrainingLossStopper
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def train() -> None:
    """Train the PINN."""
    device = "cuda:0"

    model = PINN(
        [5, 128, 128, 128, 128, 2],
    ).to(device)

    conditions = [
        Residual(N=3000, T=10),
        InitialCondition(initial_values=torch.tensor([15.0, 10.0])),
    ]
    conditions_train = Conditions(device=device, conditions=conditions)
    conditions_test = Conditions(device=device, conditions=conditions)

    parameters_train = RandomParameters(device=device)
    parameters_test = RandomParameters(device=device)

    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        loss_weights=(1.0, 1.0),
    )

    # optimizer = optim.LBFGS(model.parameters(), max_iter=5, history_size=5, lr=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    # optimizer = optim.LBFGS(model.parameters(), max_iter=50, history_size=10, lr=1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
        threshold=1e-3,
        min_lr=1e-6,
    )

    trainer_data = TrainerData(
        pde=pde,
        iterations=2000,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        test_every=10,
    )

    callbacks = Callbacks(
        trainer_data=trainer_data,
        callbacks=[
            CallbackLog(print_every=10),
            ModularPlotCallback(
                plot_every=20,
                panels=[
                    LossPanel,
                    ConditionLossPanel,
                    ResidualPanel,
                    (
                        SolutionComparisonPanel,
                        {"analytical_solution": analytical_solution},
                    ),
                ],
            ),
        ],
    )

    stoppers = Stoppers(
        trainer_data=trainer_data,
        stoppers=[
            TrainingLossStopper(patience=50, delta=1e-5),
        ],
    )

    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=callbacks,
        stoppers=stoppers,
    )

    trainer.train()

    pde.save("data/lotka_volterra_pde.pickle")
    model.save("data/lotka_volterra_model.pickle")
    trainer.save("data/lotka_volterra_trainer.pickle")


def plot_loss() -> None:
    """Plot the solution of the ODE."""
    trainer = Trainer.load("data/lotka_volterra_trainer.pickle")
    trainer_data = trainer.trainer_data

    callbacks = Callbacks(
        trainer_data=trainer_data,
        callbacks=[
            CallbackLog(),
            ModularPlotCallback(
                plot_every=20,
                panels=[
                    LossPanel,
                    ConditionLossPanel,
                    ResidualPanel,
                    (
                        SolutionComparisonPanel,
                        {"analytical_solution": analytical_solution},
                    ),
                ],
            ),
        ],
    )

    call_callbacks(callbacks, method="on_iteration_end")


@click.command()
@click.option("--plot", is_flag=True, help="Run the plot function instead of training.")
def main(plot: bool) -> None:
    """
    Run the training or plotting.

    Parameters
    ----------
    plot : bool
        If True, run the plotting function.
    """
    if plot:
        plot_loss()
    else:
        train()


if __name__ == "__main__":
    main()
