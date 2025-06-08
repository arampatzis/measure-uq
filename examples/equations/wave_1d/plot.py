"""
Plot the solution of the 1D wave equation.

This module provides functionality to visualize solutions of the 1D wave equation,
including both the predicted solution from trained neural networks (PINN/PINN-PCE)
and analytical solutions. It creates animated plots showing the evolution of
the wave solution over time.

The main function `plot()` takes a trained model, PDE configuration, and model type
"""

# ruff: noqa: D103 ERA001

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.lines import Line2D
from torch import tensor

from examples.equations.wave_1d.pde import analytical_solution
from measure_uq.models import PINN, PINN_PCE
from measure_uq.pde import PDE
from measure_uq.utilities import KeyController, cartesian_product_of_rows, to_numpy

plt.rc("figure", figsize=[16, 9])

torch.set_printoptions(
    threshold=200_000,
    precision=4,
    linewidth=120,
)  # type: ignore[no-untyped-call]


def plot(
    model_path: str | Path,
    pde_path: str | Path,
    model_type: type[PINN | PINN_PCE],
) -> tuple[plt.Figure, plt.Axes, animation.FuncAnimation]:
    """
    Plot the solution of the PDE.

    Parameters
    ----------
    model_path : str | Path
        The path to the model.
    pde_path : str | Path
        The path to the PDE.
    model_type : type[PINN | PINN_PCE]
        The type of the model.

    Returns
    -------
    tuple[plt.Figure, plt.Axes, animation.FuncAnimation]
        The figure, axes, and animation.
    """
    model = model_type.load(model_path)
    pde = PDE.load(pde_path)

    T = getattr(pde.conditions_train.conditions[0], "T", 1.0)
    X = getattr(pde.conditions_train.conditions[0], "X", 1.0)

    t = tensor(np.linspace(0, T, 40)[:, None])
    x = tensor(np.linspace(0, X, 100)[:, None])

    parameters = pde.parameters_test.values.detach().cpu()

    tx = cartesian_product_of_rows(t, x).float()

    _, y = model.forward(tx, parameters)

    Nx = x.shape[0]
    Nt = t.shape[0]
    Np = parameters.shape[0]

    yy = np.array(
        [y[k::Np, :].reshape(Nt, Nx).detach().numpy().T for k in range(Np)],
    )

    solutions = np.array(
        [to_numpy(analytical_solution(t, x, p)).T for p in parameters],
    )

    mean_y = yy.mean(axis=0)
    std_y = yy.std(axis=0)

    mean_solution = solutions.mean(axis=0)
    std_solution = solutions.std(axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    axs[0].set_ylim(1.05 * yy.min(), 1.05 * yy.max())
    axs[1].set_ylim(
        1.05 * (mean_y.min() - std_y.max()), 1.05 * (mean_y.max() + std_y.max())
    )
    axs[2].set_ylim(1.05 * mean_solution.min(), 1.05 * mean_solution.max())
    axs[3].set_ylim(1.05 * std_solution.min(), 1.05 * std_solution.max())

    i = 0

    ax0_lines1 = axs[0].plot(x, solutions[:, :, i].T)
    ax0_lines2 = axs[0].plot(x, yy[:, :, i].T, "--")

    (ax1_mean,) = axs[1].plot(x, mean_y[:, i], ".")
    axs[1].fill_between(
        x.squeeze(),
        mean_y[:, i] - std_y[:, i],
        mean_y[:, i] + std_y[:, i],
        color="blue",
        alpha=0.3,
        label="Uncertainty band (1 std)",
    )

    (ax2_line1,) = axs[2].plot(x, mean_solution[:, i])
    (ax2_line2,) = axs[2].plot(x, mean_y[:, i], "--")

    (ax3_line1,) = axs[3].plot(x, std_solution[:, i])
    (ax3_line2,) = axs[3].plot(x, std_y[:, i], "--")

    for a in axs:
        a.grid()

    controller = KeyController(
        use_toggle=True,
        key_bindings={"toggle": " ", "stop": "x"},
    )

    def animate(
        i: int,
    ) -> tuple[Line2D, Line2D, Line2D, Line2D, Line2D, Line2D, Line2D, Line2D]:
        """
        Animate the solution of the PDE.

        Parameters
        ----------
        i : int
            The index of the frame.

        Returns
        -------
        tuple[Line2D, Line2D, Line2D, Line2D, Line2D, Line2D, Line2D]
            The lines of the animation.
        """
        controller.check_pause()
        if controller.stop_requested:
            print("Stopping animation...")
            controller.close()
            anime.event_source.stop()
            if plt.get_fignums():
                plt.close("all")
            return ()  # type: ignore[return-value]

        fig.suptitle(f"t = {t[i].item():.2f}")

        for k, line in enumerate(ax0_lines1):
            line.set_ydata(solutions[k, :, i].T)

        for k, line in enumerate(ax0_lines2):
            line.set_ydata(yy[k, :, i].T)

        ax1_mean.set_ydata(mean_y[:, i])

        axs[1].collections[0].remove()

        ax1_fill = axs[1].fill_between(
            x.squeeze(),
            mean_y[:, i] - std_y[:, i],
            mean_y[:, i] + std_y[:, i],
            color="blue",
            alpha=0.3,
            label="Uncertainty band (1 std)",
        )

        ax2_line1.set_ydata(mean_y[:, i])
        ax2_line2.set_ydata(mean_solution[:, i])
        ax3_line1.set_ydata(std_y[:, i])
        ax3_line2.set_ydata(std_solution[:, i])

        return (
            ax0_lines1,
            ax0_lines2,
            ax1_mean,
            ax1_fill,
            ax2_line1,
            ax2_line2,
            ax3_line1,
            ax3_line2,
        )

    anime = animation.FuncAnimation(
        fig,
        animate,
        interval=200,
        blit=False,
        frames=t.shape[0],
    )

    return fig, axs, anime
