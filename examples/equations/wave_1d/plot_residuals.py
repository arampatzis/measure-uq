"""
Plot the residuals of the 1D wave equation.

This module provides functionality to visualize the residuals of the 1D wave equation,
including both the residuals of the PDE and the initial and boundary conditions.
It creates animated plots showing the evolution of the residuals over time.

The main function `plot()` takes a trained model, PDE configuration, and model type
as inputs and returns a matplotlib figure with an animated visualization of the
residuals.
"""

# ruff: noqa: D103 ERA001

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.lines import Line2D

from examples.equations.wave_1d.pde import analytical_solution_2
from measure_uq.models import PINN, PINN_PCE
from measure_uq.pde import PDE
from measure_uq.utilities import KeyController

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
    model = model_type.load(model_path).to("cuda:0")
    pde = PDE.load(pde_path)

    parameters = pde.parameters_train.values
    parameters = parameters[:1]

    residual: dict[str, dict[str, np.ndarray]] = {}
    residual_exact: dict[str, dict[str, np.ndarray]] = {}

    tx = pde.conditions_train.conditions[0].points
    Nx = getattr(pde.conditions_train.conditions[0], "Nx", 100)
    Nt = getattr(pde.conditions_train.conditions[0], "Nt", 100)
    x = tx[0:Nx, 1].detach().cpu().numpy()
    t = tx[0 : Nx * Nt : Nx, 0].detach().cpu().numpy()

    # Residual for computed and analytical solution of the pde
    txp, yp = model.forward(tx, parameters)
    y = yp.reshape(Nt, Nx).T.detach().cpu().numpy()

    residual["pde"] = {}
    residual["pde"]["values"] = np.abs(
        pde.conditions_train.conditions[0]
        .eval(txp, yp)
        .reshape(Nt, Nx)
        .T.detach()
        .cpu()
        .numpy()
    )
    residual["pde"]["x"] = x

    txp, solutionp = analytical_solution_2(tx, parameters)
    solution = solutionp.reshape(Nt, Nx).T.detach().cpu().numpy()

    residual_exact["pde"] = {}
    residual_exact["pde"]["values"] = np.abs(
        pde.conditions_train.conditions[0]
        .eval(txp, solutionp)
        .reshape(Nt, Nx)
        .T.detach()
        .cpu()
        .numpy()
    )
    residual_exact["pde"]["x"] = x

    # Residual for computed ICs and BCs
    tx = pde.conditions_train.conditions[1].points
    txp, yp = model.forward(tx, parameters)
    residual["ic0"] = {}
    residual["ic0"]["values"] = np.abs(
        pde.conditions_train.conditions[1]
        .eval(txp, yp)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    residual["ic0"]["x"] = tx[:, 1].squeeze().detach().cpu().numpy()

    tx = pde.conditions_train.conditions[2].points
    txp, yp = model.forward(tx, parameters)
    residual["ic1"] = {}
    residual["ic1"]["values"] = np.abs(
        pde.conditions_train.conditions[2]
        .eval(txp, yp)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    residual["ic1"]["x"] = tx[:, 1].squeeze().detach().cpu().numpy()

    tx = pde.conditions_train.conditions[3].points
    txp, yp = model.forward(tx, parameters)
    residual["bc0"] = {}
    residual["bc0"]["values"] = np.abs(
        pde.conditions_train.conditions[3]
        .eval(txp, yp)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    residual["bc0"]["t"] = tx[:, 0].squeeze().detach().cpu().numpy()

    tx = pde.conditions_train.conditions[4].points
    txp, yp = model.forward(tx, parameters)
    residual["bc1"] = {}
    residual["bc1"]["values"] = np.abs(
        pde.conditions_train.conditions[4]
        .eval(txp, yp)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    residual["bc1"]["t"] = tx[:, 0].squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    axs[0].set_ylim(1.05 * y.min(), 1.05 * y.max())
    axs[1].set_ylim(
        1.05 * residual["pde"]["values"].min(), 1.05 * residual["pde"]["values"].max()
    )
    z = np.concatenate([residual["ic0"]["values"], residual["ic1"]["values"]])
    axs[2].set_ylim(1.05 * z.min(), 1.05 * z.max())
    z = np.concatenate([residual["bc0"]["values"], residual["bc1"]["values"]])
    axs[3].set_ylim(1.05 * z.min(), 1.05 * z.max())

    k = 0

    (ax0_lines1,) = axs[0].plot(x, solution[:, k].T)
    (ax0_lines2,) = axs[0].plot(x, y[:, k].T, "--")

    (ax1_line1,) = axs[1].plot(residual["pde"]["x"], residual["pde"]["values"][:, k])
    (ax1_line2,) = axs[1].plot(
        residual_exact["pde"]["x"], residual_exact["pde"]["values"][:, k]
    )

    (ax2_line1,) = axs[2].plot(residual["ic0"]["x"], residual["ic0"]["values"])
    (ax2_line2,) = axs[2].plot(residual["ic1"]["x"], residual["ic1"]["values"])

    (ax3_line1,) = axs[3].plot(residual["bc0"]["t"], residual["bc0"]["values"])
    (ax3_line2,) = axs[3].plot(residual["bc1"]["t"], residual["bc1"]["values"])

    controller = KeyController(
        use_toggle=True,
        key_bindings={"toggle": " ", "stop": "x"},
    )

    for a in axs:
        a.grid()

    def animate(
        i: int,
    ) -> tuple[Line2D, Line2D, Line2D, Line2D]:
        """
        Animate the solution of the PDE.

        Parameters
        ----------
        i : int
            The index of the frame.

        Returns
        -------
        tuple[Line2D, Line2D, Line2D, Line2D]
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

        ax0_lines1.set_ydata(solution[:, i].T)

        ax0_lines2.set_ydata(y[:, i].T)

        ax1_line1.set_ydata(residual["pde"]["values"][:, i])
        ax1_line2.set_ydata(residual_exact["pde"]["values"][:, i])

        return (
            ax0_lines1,
            ax0_lines2,
            ax1_line1,
            ax1_line2,
        )

    anime = animation.FuncAnimation(
        fig,
        animate,
        interval=200,
        blit=False,
        frames=t.shape[0],
    )

    return fig, axs, anime
