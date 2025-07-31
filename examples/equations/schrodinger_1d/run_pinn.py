#!/usr/bin/env python3
r"""
Solution of the 1D time-dependent Schrödinger equation using a PINN.

.. math::
    i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \psi}{\partial x^2} + V(x) \psi
    \psi(0, x) = \psi_0(x)
    \psi(t, x_{min}) = 0
    \psi(t, x_{max}) = 0

Where:
- \psi(t, x) is the complex wave function
- \hbar is the reduced Planck constant
- m is the particle mass
- V(x) is the potential energy function
- \psi_0(x) is the initial wave function
"""

import chaospy
import numpy as np
from chaospy import J
from torch import optim

from examples.equations.schrodinger_1d.pde import (
    BoundaryConditionLeft,
    BoundaryConditionRight,
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
from measure_uq.stoppers import Stoppers, TrainingLossStopper
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def train() -> None:
    """Train the PINN for the 1D Schrödinger equation."""
    # Define joint distribution for parameters
    # Parameters: hbar, m, k0, sigma
    joint = J(
        chaospy.Normal(1.0, 0.1),    # hbar (reduced Planck constant)
        chaospy.Normal(1.0, 0.1),    # m (particle mass)
        chaospy.Uniform(3.0, 5.0),   # k0 (initial wave number)
        chaospy.Uniform(0.5, 1.0),   # sigma (width of the Gaussian wave packet)
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Define the neural network model
    # Note: Output size is 2 for real and imaginary parts
    model = PINN(
        network_builder=FeedforwardBuilder(
            layer_sizes=[6, 50, 50, 50, 50, 50, 2],
            activation="tanh",
        ),
    ).to(device)

    # Define domain parameters
    T = 2.0         # Maximum time
    X_min = -10.0   # Minimum spatial coordinate
    X_max = 10.0    # Maximum spatial coordinate

    # Define conditions
    conditions = [
        Residual(Nt=20, Nx=40, T=T, X_min=X_min, X_max=X_max),
        InitialCondition(Nx=40, X_min=X_min, X_max=X_max),
        BoundaryConditionLeft(Nt=20, T=T, X_min=X_min),
        BoundaryConditionRight(Nt=20, T=T, X_max=X_max),
    ]
    conditions_train = Conditions(device=device, conditions=conditions)
    conditions_test = Conditions(device=device, conditions=conditions)

    # Define parameters
    parameters_train = RandomParameters(device=device, joint=joint, N=40)
    parameters_test = RandomParameters(device=device, joint=joint, N=40)

    # Define PDE
    pde = PDE(
        conditions_train=conditions_train,
        conditions_test=conditions_test,
        parameters_train=parameters_train,
        parameters_test=parameters_test,
        resample_conditions_every=(50,),
        resample_parameters_every=100,
    )

    # Create directories for saving results if they don't exist
    import os
    os.makedirs("data", exist_ok=True)

    # Define trainer data
    trainer_data = TrainerData(
        pde=pde,
        iterations=5000,
        model=model,
        optimizer=optim.LBFGS(
            model.parameters(),
            max_iter=5,
            history_size=5,
            lr=1,
        ),
        test_every=40,
        save_path="data/best_model_schrodinger_pinn.pickle",
        device=device,
    )

    # Define callbacks
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

    # Define stoppers
    Stoppers(
        trainer_data=trainer_data,
        stoppers=[
            TrainingLossStopper(patience=50, delta=1e-5),
        ],
    )

    # Create and run trainer
    trainer = Trainer(
        trainer_data=trainer_data,
        callbacks=callbacks,
    )

    trainer.train()

    # Save results
    pde.save("data/pde_schrodinger_pinn.pickle")
    model.save("data/model_schrodinger_pinn.pickle")
    trainer.save("data/trainer_schrodinger_pinn.pickle")


if __name__ == "__main__":
    import torch
    train()