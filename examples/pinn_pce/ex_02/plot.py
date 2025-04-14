#!/usr/bin/env python3
"""Solution of the ordinary differential equation (ODE).

.. math::
    y' = p1 * y
    y(0) = p2
"""

# ruff: noqa: D103 ERA001

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.lines import Line2D
from torch import tensor

from examples.pinn_pce.ex_02.pde import analytical_solution
from measure_uq.models import PINN, PINN_PCE
from measure_uq.pde import PDE
from measure_uq.utilities import cartesian_product_of_rows, to_numpy

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
    """Plot the solution of the PDE."""
    model = model_type.load(model_path)
    pde = PDE.load(pde_path)

    t = tensor(np.linspace(0, 1, 40)[:, None])
    x = tensor(np.linspace(0, np.pi, 100)[:, None])
    parameters = pde.parameters_test.values.detach().cpu()

    tx = cartesian_product_of_rows(t, x).float()

    txp, y = model.forward(tx, parameters)

    Nx = x.shape[0]
    Nt = t.shape[0]
    Np = parameters.shape[0]
    yy = np.array(
        [y[k::Np, :].reshape(Nt, Nx).detach().numpy() for k in range(Np)],
    )

    solutions = np.array(
        [to_numpy(analytical_solution(t.T, x, p)) for p in parameters],
    )

    mean_y = yy.mean(axis=0)
    std_y = yy.std(axis=0)

    mean_solution = solutions.mean(axis=0).T
    std_solution = solutions.std(axis=0).T

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    k = 0

    ax0_lines = axs[0].plot(x, solutions[:, :, k].T)

    (ax1_mean,) = axs[1].plot(x, mean_y[k], ".")
    axs[1].fill_between(
        x.squeeze(),
        mean_y[k] - std_y[k],
        mean_y[k] + std_y[k],
        color="blue",
        alpha=0.3,
        label="Uncertainty band (1 std)",
    )

    (ax2_line1,) = axs[2].plot(x, mean_solution[k])
    (ax2_line2,) = axs[2].plot(x, mean_y[k], "--")

    (ax3_line1,) = axs[3].plot(x, std_solution[k])
    (ax3_line2,) = axs[3].plot(x, std_y[k], "--")

    for a in axs:
        a.grid()

    def animate(
        i: int,
    ) -> tuple[Line2D, Line2D, Line2D, Line2D, Line2D, Line2D, Line2D]:
        """Animate the solution of the PDE."""
        fig.suptitle(f"t = {t[i].item():.2f}")

        for k, line in enumerate(ax0_lines):
            line.set_ydata(solutions[k, :, i].T)

        ax1_mean.set_ydata(mean_solution[i])

        axs[1].collections[0].remove()

        ax1_fill = axs[1].fill_between(
            x.squeeze(),
            mean_y[i] - std_y[i],
            mean_y[i] + std_y[i],
            color="blue",
            alpha=0.3,
            label="Uncertainty band (1 std)",
        )

        ax2_line1.set_ydata(mean_y[i])
        ax2_line2.set_ydata(mean_solution[i])
        ax3_line1.set_ydata(std_y[i])
        ax3_line2.set_ydata(std_solution[i])

        return (
            ax0_lines,
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
