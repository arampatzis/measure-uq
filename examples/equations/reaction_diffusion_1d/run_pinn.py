#!/usr/bin/env python3
r"""
Solve a nonlinear 1D reaction-diffusion equation with random coefficients.

.. math::
    \frac{\partial u}{\partial t}
    - D \frac{\partial^2 u}{\partial x^2}
    + g(x) u^3 = f(x),
    \quad t \in [0, T],\ x \in [-1, 1]

with:

**Initial condition:**

.. math::
    u(0, x) = 0.5 \cos^2(\pi x)

**Boundary conditions:**

.. math::
    u(t, -1) = u(t, 1) = 0.5

**Reaction coefficient:**

.. math::
    g(x) = 0.2 + e^{r_1 x} \cos^2(r_2 x),\quad
    r_1 \sim \mathcal{U}(0.5, 1),\quad
    r_2 \sim \mathcal{U}(3, 4)

**Forcing term:**

.. math::
    f(x) = \exp\left( -\frac{(x - 0.25)^2}{2 k_1^2} \right) \sin^2(k_2 x),\quad
    k_1 \sim \mathcal{U}(0.2, 0.8),\quad
    k_2 \sim \mathcal{U}(1, 4)
"""

import chaospy
from chaospy import J
from torch import optim

from examples.equations.reaction_diffusion_1d.pde import (
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
from measure_uq.trainers.trainer import Trainer
from measure_uq.trainers.trainer_data import TrainerData


def train() -> None:
    """Train the PINN."""
    joint = J(
        chaospy.Uniform(0.5, 1),
        chaospy.Uniform(3, 4),
        chaospy.Uniform(0.2, 0.8),
        chaospy.Uniform(1, 4),
    )

    device = "cuda:0"

    model = PINN(
        network_builder=FeedforwardBuilder(
            layer_sizes=[6, 40, 40, 40, 40, 40, 40, 1],
            activation="snake",
        ),
    )

    T = 4.0

    conditions = [
        Residual(Nt=20, Nx=40, T=T),
        InitialCondition(Nx=40),
        BoundaryConditionLeft(Nt=20, T=T),
        BoundaryConditionRight(Nt=20, T=T),
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
        loss_weights=[5.0, 10.0, 1.0, 1.0],
    )
    pde.save("data/pde_pinn.pickle")

    trainer_data = TrainerData(
        pde=pde,
        iterations=5000,
        model=model,
        optimizer=optim.LBFGS(
            model.parameters(),
            max_iter=20,
            history_size=20,
            lr=1,
        ),
        test_every=40,
        save_path="data/best_model_pinn.pickle",
        device=device,
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
