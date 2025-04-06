"""
Module for plotting functions.

The functions in this module are used for visualization of the results of
the training process.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from torch import Tensor, nn

from measure_uq.pde import Parameters
from measure_uq.trainers.trainer_data import TrainerData


def plot_losses(
    trainer_data: TrainerData,
    figsize: tuple = (20, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the losses during training and testing and the losses of each condition.

    Parameters
    ----------
    trainer_data: TrainerData
        The trainer_data that was used for training.
    figsize: tuple
        The size of the figure. Defaults to (20, 10).

    Returns
    -------
    fig: maatplotlib.figure.Figure
        The figure.
    ax: mtplotlib.axes.Axes
        The axes.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].set_title("Total losses")
    ax[0].plot(
        trainer_data.losses_train.i,
        trainer_data.losses_train.v,
        label="train",
    )
    ax[0].plot(
        trainer_data.losses_test.i,
        trainer_data.losses_test.v,
        "--",
        label="test",
    )

    ax[1].set_title("Train losses per condition")
    for c in trainer_data.pde.conditions_train:
        ax[1].plot(c.loss.i, c.loss.v, label=c.__class__.__name__)

    for a in ax:
        a.set_yscale("log")
        a.grid()
        a.legend()

    return fig, ax


@dataclass(kw_only=True)
class PlotDataOnGrid:
    """
    Dataclass for plotting data on a grid.

    Parameters
    ----------
    x1 : ArrayLike | list[ArrayLike]
        The data to be plotted on the x1 axis.
    x2 : ArrayLike | list[ArrayLike]
        The data to be plotted on the x2 axis.
    legend : str | list[str]
        The legend for the plot.
    title : str
        The title for the plot.
    """

    x1: ArrayLike | list[ArrayLike]
    x2: ArrayLike | list[ArrayLike]
    legend: str | list[str]
    title: str


def plot_1d_on_grid(
    data: list[PlotDataOnGrid],
    figsize: tuple = (10, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 1D data on a grid of subplots.

    Parameters
    ----------
    data : list[PlotDataOnGrid]
        A list of PlotDataOnGrid dataclasses. Each dataclass contains the data to be
        plotted on the x1 and x2 axes, the legend for the plot, and the title for the
        plot.
    figsize : tuple, optional
        The figure size for the plot. Default is (10, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : list[matplotlib.axes.Axes]
        The axes objects for the plot.
    """
    N = len(data)

    rows = int(np.ceil(np.sqrt(N)))  # Number of rows
    cols = int(np.ceil(N / rows))

    fig, ax = plt.subplots(rows, cols, figsize=figsize)

    ax = np.array([ax]) if N == 1 else ax.flatten()

    for i, d in enumerate(data):
        if not isinstance(d.x1, list):
            d.x1 = [d.x1]
        if not isinstance(d.x2, list):
            d.x2 = [d.x2]
        if not isinstance(d.legend, list):
            d.legend = [d.legend]

        assert len(d.x1) == len(d.x2) == len(d.legend)

        for x1, x2, s in zip(d.x1, d.x2, d.legend, strict=True):
            ax[i].plot(x1, x2, label=s)

        ax[i].set_title(d.title)

    for i in range(N, len(ax)):
        fig.delaxes(ax[i])

    for i in range(N):
        ax[i].grid()
        ax[i].legend()

    return fig, ax


def plot_ode_on_grid(
    *,
    model: nn.Module,
    t: Tensor,
    parameters: Parameters,
    analytical_solution: Callable,
    approximate_solution: Callable,
    figsize: tuple = (10, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the analytical and approximate solution of an ODE on a grid of subplots.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    t : Tensor
        The time points to evaluate the model at.
    parameters : Parameters
        The parameters to evaluate the model at.
    analytical_solution : Callable
        A function that takes a time array and a set of parameters and returns the
        exact solution of the ODE.
    approximate_solution : Callable
        A function that takes a time array and a set of parameters and returns the
        approximate solution of the ODE computed by the model.
    figsize : tuple, optional
        The size of the figure. Defaults to (10, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    """
    model.eval()
    with torch.no_grad():
        data = []
        for i, p in enumerate(parameters.values):
            u_exact = analytical_solution(t, p)
            u_aprox = approximate_solution(t, p)

            data.append(
                PlotDataOnGrid(
                    x1=[t, t],
                    x2=[u_exact, u_aprox],
                    legend=["exact", "approx"],
                    title=" ".join([f"{x:.1e}" for x in p.tolist()]),
                ),
            )

            print(
                f"{i:03}: {p.tolist()}  --  "
                f"l2 error: {torch.mean((u_exact - u_aprox)**2):.5e}",
            )

    return plot_1d_on_grid(data, figsize=figsize)
