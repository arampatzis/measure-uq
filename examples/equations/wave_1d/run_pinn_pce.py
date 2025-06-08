#!/usr/bin/env python3
r"""
Solution of the heat equation on the line using a PINN-PCE method.

.. math::
    u_t - a / k^2 u_xx = 0
    u(0, x) = \sin(k x)
    u(t, 0) = 0
    u(t, \pi) = exp(-a t) \sin(\pi k)
"""

import chaospy
import click
import matplotlib.pyplot as plt
import numpy as np
from chaospy import J
from torch import optim

from examples.equations.heat_1d.pde import (
    BoundaryConditionLeft,
    BoundaryConditionRight,
    InitialCondition,
    RandomParameters,
    Residual,
)
from examples.equations.heat_1d.plot import plot
from examples.equations.wave_1d.pde import InitialVelocity
from measure_uq.callbacks import (
    CallbackLog,
    Callbacks,
    ModularPlotCallback,
)
from measure_uq.models import PINN_PCE
from measure_uq.pde import PDE, Conditions
from measure_uq.plots import (
    ConditionLossPanel,
    TrainTestLossPanel,
    plot_losses,
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
        [2, 20, 20, 20, 20, 20, 20, len(expansion)],
        expansion,
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
        optimizer=optim.LBFGS(model.parameters(), max_iter=5, history_size=5, lr=1),
        test_every=10,
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


def plot_all() -> None:
    """Plot the solution of the PDE."""
    trainer = Trainer.load("data/trainer_pinn_pce.pickle")
    fig1, ax1 = plot_losses(trainer.trainer_data)

    fig2, ax2, anime = plot(
        model_path="data/model_pinn_pce.pt",
        pde_path="data/pde_pinn_pce.pickle",
        model_type=PINN_PCE,
    )

    plt.show()


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
        plot_all()
    else:
        train()


if __name__ == "__main__":
    main()
