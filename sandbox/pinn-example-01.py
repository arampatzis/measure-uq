#!/usr/bin/env python3
"""Script to test the PINN for solving the ODE dy/dt = alpha * y."""

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from box import Box


def ode(
    t: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> torch.Tensor:
    """
    Return the residual of the ODE dy/dt = alpha * y.

    Parameters
    ----------
    t : np.typing.ArrayLike
        Time points
    y : np.typing.ArrayLike
        Solution value(s)
    alpha : float
        Constant in the ODE

    Returns
    -------
    np.typing.ArrayLike
        The value(s) of dy/dt
    """
    return dde.grad.jacobian(y, t) - alpha * y


def solution(t: np.ndarray, y0: float, alpha: float) -> np.ndarray:
    """
    Compute the solution of the ODE dy/dt = alpha * y with initial condition y(0) = y0.

    Parameters
    ----------
    t : np.ndarray
        Time points where to evaluate the solution.
    y0 : float
        Initial condition at t=0.
    alpha : float
        Constant in the ODE.

    Returns
    -------
    np.ndarray
        Solution of the ODE at the given time points.
    """
    return y0 * np.exp(alpha * t)


class WandbLog(dde.callbacks.Callback):
    """
    Callback for logging training metrics to Weights & Biases.

    Attributes
    ----------
    every : int
        Frequency of logging in terms of training steps.

    Methods
    -------
    on_batch_end():
        Logs training loss and learning rate to Weights & Biases at specified intervals.
    on_train_end():
        Placeholder for actions to perform at the end of training.
    """

    def __init__(self, every: int):
        self.every = every
        super().__init__()

    def on_batch_end(self) -> None:
        """
        Log training loss and learning rate to Weights & Biases at specified intervals.

        Parameters
        ----------
        self : Callback
            The callback object.

        Returns
        -------
        None
        """
        if self.model.train_state.step % self.every == 0:
            print("Log in wandb...")
            _, loss = self.model._outputs_losses(
                True,
                self.model.train_state.X_train,
                self.model.train_state.y_train,
                self.model.train_state.train_aux_vars,
            )
            wandb.log(
                {
                    "iteration": self.model.train_state.step,
                    "train/loss": np.sum(loss),
                    "lr": self.model.opt.param_groups[-1]["lr"],
                },
            )


def main():
    """
    Train a neural network to solve a simple ordinary differential equation
    using DeepXDE. The ODE is defined as dy/dt = -2*y, with initial condition y(0) = 1.
    The solution is y(t) = exp(-2*t).

    The script takes no command-line arguments.

    The script will log the following metrics to Weights & Biases:

    - The training loss at each iteration.
    - The learning rate at each iteration.
    - A plot of the true solution and the predicted solution at the end of training.

    The script will also save the following artifacts to Weights & Biases:

    - The configuration of the model (hyperparameters).
    - The trained model itself.
    """
    config = Box(
        {
            "project_name": "deepxpde-for-simple-ode",
            "iterations": 5000,
            "optimizer": {
                "type": "adam",
                "lr": 0.1,
            },
            "ode": {
                "y0": 1,
                "Tf": 1,
                "alpha": -2,
            },
        },
    )

    wandb.init(
        project=config.project_name,
        config=config.to_dict(),
    )

    geom = dde.geometry.TimeDomain(0, config.ode.Tf)
    ic = dde.icbc.IC(
        geom,
        lambda _: config.ode.y0,
        lambda _, on_initial: on_initial,
        component=0,
    )
    pde = dde.data.PDE(
        geom,
        lambda t, y: ode(t, y, config.ode.alpha),
        ic,
        train_distribution="pseudo",
        num_domain=19,
        num_boundary=2,
        num_test=500,
    )

    layer_size = [1] + [50] * 3 + [1]
    net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")

    model = dde.Model(pde, net)
    model.compile(
        config.optimizer.type,
        lr=config.optimizer.lr,
        loss_weights=[0.01, 1],
        decay=("step", 1000, 0.2),
    )
    losshistory, train_state = model.train(
        iterations=config.iterations,
        callbacks=[
            WandbLog(100),
        ],
    )

    fig, ax = plt.subplots(1, 1)

    t = np.linspace(0, 1, 100).reshape(100, 1)
    y_true = solution(t, config.ode.y0, config.ode.alpha)
    ax.plot(t, y_true, color="black", label="exact solution")

    y_pred = model.predict(t)

    ax.plot(t, y_pred, color="orange", linestyle="dashed", label="x_pred")
    ax.legend()

    wandb.log({"plot": wandb.Image(fig)})


if __name__ == "__main__":
    main()
