"""
Module for plotting functions.

The functions in this module are used for visualization of the results of
the training process.

This module provides:
    - set_log_scale_with_latex: Apply log scale with LaTeX-style tick labels
    (e.g., $10^{-2}$).
    - plot_losses: Plot the losses during training and testing and the losses of
    each condition.
    - plot_1d_on_grid: Plot 1D data on a grid of subplots.
    - plot_ode_on_grid: Plot the analytical and approximate solution of an ODE
    on a grid of subplots.
    - BasePlotPanel: Base class for plot panels.
    - LossPanel: Panel for displaying loss plots.
    - ConditionLossPanel: Panel for displaying condition loss plots.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
from numpy.typing import ArrayLike
from torch import Tensor, nn

from measure_uq.pde import Parameters
from measure_uq.trainers.trainer_data import TrainerData
from measure_uq.utilities import to_numpy


def set_log_scale_with_latex(
    ax: plt.Axes,
    axis: str = "y",
    label_only_base: bool = False,
) -> None:
    """
    Apply log scale with LaTeX-style tick labels (e.g., $10^{-2}$).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to apply the log scale to.
    axis : str, optional
        The axis to apply the log scale to ('x' or 'y'). Default is 'y'.
    label_only_base : bool, optional
        Whether to label only the base of the log scale. Default is False.
    """
    formatter = LogFormatterMathtext()
    formatter.labelOnlyBase = label_only_base

    if axis == "y":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(formatter)
    elif axis == "x":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(formatter)


def plot_losses(
    trainer_data: TrainerData,
    figsize: tuple = (20, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the losses during training and testing and the losses of each condition.

    Parameters
    ----------
    trainer_data : TrainerData
        The trainer_data that was used for training.
    figsize : tuple, optional
        The size of the figure. Defaults to (20, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
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

    Attributes
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


@dataclass(kw_only=True)
class BasePlotPanel:
    """
    Base class for plot panels.

    This class provides a base for creating plot panels with a figure and grid
    specification.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure to plot on.
    grid_spec : matplotlib.gridspec.GridSpec
        The grid specification for the panel layout.
    position : int
        The position of the panel in the grid.
    """

    fig: plt.Figure
    grid_spec: plt.GridSpec
    position: int

    def __post_init__(self) -> None:
        """Initialize the BasePlotPanel."""
        self.ax = self.fig.add_subplot(self.grid_spec[self.position])
        self._initialized = False

    def update(self, trainer_data: TrainerData) -> None:
        """
        Update the plot panel with new trainer data.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data to update the plot with.
        """
        raise NotImplementedError("Override in subclass")

    def relayout(self) -> None:
        """
        Relayout the plot panel.

        This method adjusts the layout of the plot panel based on the current
        grid specification and position.
        """
        self.ax.relim()
        self.ax.autoscale_view()


@dataclass(kw_only=True)
class LossPanel(BasePlotPanel):
    """
    Panel for displaying loss plots.

    Inherits from BasePlotPanel to provide a specific implementation for
    plotting training and testing losses.
    """

    def __post_init__(self) -> None:
        """Initialize the LossPanel."""
        super().__post_init__()
        self.line = self.ax.plot([], [], linewidth=3, label="Total Loss")[0]
        self.ax.set_title("Total Loss")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        set_log_scale_with_latex(self.ax, axis="y")
        self.ax.grid(True)
        self.ax.legend()

    def update(self, trainer_data: TrainerData) -> None:
        """
        Update the loss panel with new trainer data.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data to update the loss plot with.
        """
        i = np.array(trainer_data.losses_train.i)
        v = np.array(trainer_data.losses_train.v)
        self.line.set_data(i, v)
        self.relayout()


@dataclass(kw_only=True)
class ConditionLossPanel(BasePlotPanel):
    """
    Panel for displaying condition loss plots.

    Inherits from BasePlotPanel to provide a specific implementation for
    plotting condition losses during training.
    """

    def __post_init__(self) -> None:
        """Initialize the ConditionLossPanel."""
        super().__post_init__()
        self.ax.set_title("Train Losses per Condition")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        cmap = cm.get_cmap("Set1")
        self.ax.set_prop_cycle(color=cmap(np.linspace(0, 1, 10)))
        set_log_scale_with_latex(self.ax, axis="y")
        self.ax.grid(True)
        self.lines: dict[int, plt.Line2D] = {}

    def update(self, trainer_data: TrainerData) -> None:
        """
        Update the condition loss panel with new trainer data.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data to update the condition loss plot with.
        """
        for c in trainer_data.pde.conditions_train:
            key = id(c)  # avoid "unhashable type" error
            if key not in self.lines:
                (line,) = self.ax.plot([], [], label=c.__class__.__name__, linewidth=3)
                self.lines[key] = line
            self.lines[key].set_data(c.loss.i, c.loss.v)

        self.ax.legend()
        self.relayout()


@dataclass(kw_only=True)
class ResidualPanel(BasePlotPanel):
    """
    Panel for displaying residual plots.

    Inherits from BasePlotPanel to provide a specific implementation for
    plotting residuals during training.
    """

    def __post_init__(self) -> None:
        """Initialize the ResidualPanel."""
        super().__post_init__()
        self.ax.set_title("Residuals")
        self.ax.set_xlabel("Input")
        self.ax.set_ylabel("Residual")
        set_log_scale_with_latex(self.ax, axis="y")
        self.ax.grid(True)
        self.lines: list[plt.Line2D] = []

    def update(self, trainer_data: TrainerData) -> None:
        """
        Update the residual panel with new trainer data.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data to update the residual plot with.
        """
        try:
            cond = trainer_data.pde.conditions_train[0]
            x = cond.points.detach().cpu().numpy().squeeze()
            try:
                residual = getattr(cond, "residual")  # noqa: B009
                res = residual.detach().cpu().numpy()
            except AttributeError as e:
                raise ValueError(
                    "The condition does not have a residual attribute.",
                ) from e

            if not self.lines:
                for i in range(res.shape[1]):
                    (line,) = self.ax.plot([], [], label=f"Residual[{i}]", linewidth=3)
                    self.lines.append(line)
                self.ax.legend()

            for i, line in enumerate(self.lines):
                line.set_data(x, np.abs(res[:, i]))
            self.relayout()
        except (AttributeError, IndexError, ValueError) as e:
            print(f"[ResidualPanel] Warning: {e}")


@dataclass(kw_only=True)
class SolutionComparisonPanel(BasePlotPanel):
    """Plot the solution of the model and the analytical solution."""

    analytical_solution: Callable

    def __post_init__(self) -> None:
        """Initialize the SolutionComparisonPanel."""
        super().__post_init__()
        self.ax.set_title("Model vs Analytical Solutions")
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("Solution")
        self.ax.grid(True)

    def update(self, trainer_data: TrainerData) -> None:
        """
        Update the SolutionComparisonPanel.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data.
        """
        try:
            pde = trainer_data.pde
            model = trainer_data.model
            parameters = pde.parameters_test.values
            t = pde.conditions_train[0].points

            try:
                T = getattr(pde.conditions_train[0], "T")  # noqa: B009
            except AttributeError as e:
                raise ValueError("The condition does not have a T attribute.") from e

            ic = (
                pde.conditions_train[1]
                .buffer["initial_values"]
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

            with torch.no_grad():
                _, y_pred = model.forward(t, parameters)

            x = t.squeeze().detach().cpu().numpy()
            yy = y_pred.detach().cpu().numpy()
            solutions = np.array(
                [
                    to_numpy(self.analytical_solution(x, ic, T, p))
                    for p in parameters.detach().cpu().numpy()
                ],
            )

            self.ax.clear()
            self.ax.set_title("Model vs Analytical Solutions")
            self.ax.set_xlabel("t")
            self.ax.set_ylabel("Solution")
            self.ax.grid(True)

            for i in range(solutions.shape[0]):
                self.ax.plot(x, solutions[i, 0, :], label="u", linewidth=3)
                self.ax.plot(x, solutions[i, 1, :], label="v", linewidth=3)

            self.ax.set_prop_cycle(None)  # type: ignore[call-overload]

            self.ax.plot(x, yy[:, 0], "--", label="u_pred", linewidth=3)
            self.ax.plot(x, yy[:, 1], "--", label="v_pred", linewidth=3)
            self.ax.legend(fontsize="small", loc="upper right")

            self.relayout()
        except (ValueError, RuntimeError, IndexError) as e:
            print(f"[SolutionComparisonPanel] Warning: {e}")
