"""
Provide a base class `Callback` and a class `CallbackList` for managing callbacks.

The module provides a base class `Callback` and a class `CallbackList` for managing
callbacks during the training process of a machine learning model. The `Callback` class
is used for creating custom actions at specific points during training, while the
`CallbackList` class manages a list of such callbacks, ensuring they are executed at the
appropriate times.

This module is a modification of the `callbacks` module, copied from `deepxde`.
Licensed under the GNU Lesser General Public License (LGPL) 2.1.
See the LICENSE.LGPL file in the root directory for details.
Original source: https://github.com/lululxvi/deepxde
"""

from dataclasses import dataclass
from tkinter import TclError

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from measure_uq.trainers.trainer_data import TrainerData
from measure_uq.utilities import KeyController


@dataclass(kw_only=True)
class Callback:
    """
    Abstract base class used to build new callbacks.

    Attributes
    ----------
    trainer_data : TrainerData
        The `TrainerData` instance containing the data of the trainer.

    Notes
    -----
    In order to activate the callbacks, pass it to the `Trainer` instance.

    The `Callback` class is an abstract class that every callback should
    inherit. It provides five methods that can be overridden to add new
    functionality to the training process.
    """

    def init(self, trainer_data: TrainerData) -> None:
        """
        Initialize the callback after setting a trainer.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.
        """

    def on_iteration_begin(self, trainer_data: TrainerData) -> None:
        """
        Call at the beginning of every iteration.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.
        """

    def on_iteration_end(self, trainer_data: TrainerData) -> None:
        """
        Call at the end of every iteration.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.
        """

    def on_train_begin(self, trainer_data: TrainerData) -> None:
        """
        Call at the beginning of trainer training.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.
        """

    def on_train_end(self, trainer_data: TrainerData) -> None:
        """
        Call at the end of trainer training.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.
        """

    def on_predict_begin(self, trainer_data: TrainerData) -> None:
        """
        Call at the beginning of prediction.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.
        """

    def on_predict_end(self, trainer_data: TrainerData) -> None:
        """
        Call at the end of prediction.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.
        """


@dataclass(kw_only=True)
class Callbacks:
    """
    List of callbacks to be executed at specific points during training.

    Attributes
    ----------
    callbacks : list[Callback]
        A list of callbacks to be executed. The list can be empty.

    Notes
    -----
    The list can be empty.
    """

    callbacks: list[Callback]
    trainer_data: TrainerData

    def init(self) -> None:
        """Initialize each callback in the list."""
        for callback in self.callbacks:
            callback.init(self.trainer_data)

    def on_iteration_begin(self) -> None:
        """Call at the beginning of every iteration."""
        for callback in self.callbacks:
            callback.on_iteration_begin(self.trainer_data)

    def on_iteration_end(self) -> None:
        """Call at the end of every iteration."""
        for callback in self.callbacks:
            callback.on_iteration_end(self.trainer_data)

    def on_train_begin(self) -> None:
        """Call at the beginning of trainer training."""
        for callback in self.callbacks:
            callback.on_train_begin(self.trainer_data)

    def on_train_end(self) -> None:
        """Call at the end of trainer training."""
        for callback in self.callbacks:
            callback.on_train_end(self.trainer_data)

    def on_predict_begin(self) -> None:
        """Call at the beginning of prediction."""
        for callback in self.callbacks:
            callback.on_predict_begin(self.trainer_data)

    def on_predict_end(self) -> None:
        """Call at the end of prediction."""
        for callback in self.callbacks:
            callback.on_predict_end(self.trainer_data)


def call_callbacks(
    callbacks: Callbacks,
    method: str = "on_iteration_end",
) -> None:
    """
    Call a specified method on a callbacks instance.

    This function calls the specified method on the callbacks instance and provides
    an interactive loop for controlling the display of any plots or output.

    Parameters
    ----------
    callbacks : Callbacks
        The callbacks instance containing the methods to call.
    method : str, default="on_iteration_end"
        The callback method to call. Must be a valid method name from the Callbacks
        class.

    Raises
    ------
    AttributeError
        If the specified method name is not a valid Callbacks method.
    TclError
        If the figure is not found when attempting to close it.

    Notes
    -----
    The function provides interactive control via keyboard:
    - Space bar to pause/resume
    - 'x' key to stop

    The callbacks are initialized and the specified method is called. The function
    handles cleanup of matplotlib figures when finished.
    """
    callbacks.trainer_data.iterations = callbacks.trainer_data.iteration + 1

    callbacks.init()

    try:
        method = getattr(callbacks, method)
        if callable(method):
            method()
    except AttributeError:
        print(f"Method '{method}' is not a valid Callbacks method.")

    controller = KeyController(
        use_toggle=True,
        key_bindings={"toggle": " ", "stop": "x"},
    )

    try:
        while True:
            controller.check_pause()
            if controller.stop_requested:
                print("Stop requested.")
                break
    finally:
        controller.close()
        try:
            if plt.get_fignums():
                plt.close("all")
        except TclError:
            print(
                "Warning: Attempted to close figures after application was destroyed.",
            )


@dataclass
class CallbackLog(Callback):
    """
    Callback for logging training progress.

    This callback prints training information at regular intervals during training,
    including iteration number, loss value, gradient norm, and learning rate.

    Attributes
    ----------
    print_every : int, default=100
        Number of iterations between logging outputs.
    show_grad_norm : bool, default=True
        Whether to show gradient norm in the output.
    show_lr : bool, default=True
        Whether to show learning rate in the output.
    log_file : str | None, default=None
        Path to file for saving logs. If None, logs are only printed to stdout.

    Notes
    -----
    The callback prints a line containing:
    - Current iteration number
    - Current loss value
    - Gradient norm (if show_grad_norm=True)
    - Learning rate (if show_lr=True and scheduler exists)
    """

    print_every: int = 100
    show_grad_norm: bool = True
    show_lr: bool = True
    log_file: str | None = None

    def on_iteration_end(self, trainer_data: TrainerData) -> None:
        """
        Call at the end of every iteration.

        This method is called at the end of each training iteration. It checks if the
        current iteration number is a multiple of `print_every` or if it's the final
        iteration, and if so, prints training information including the iteration
        number, loss value, gradient norm (if enabled), and learning rate (if enabled
        and scheduler exists).

        The output is printed to stdout and optionally written to a log file if one is
        specified.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.
        """
        if (
            trainer_data.iteration % self.print_every == 0
            or trainer_data.iteration == trainer_data.iterations - 1
        ):
            i = trainer_data.losses_train.i[-1]
            v = trainer_data.losses_train.v[-1]

            grad_str = ""
            if self.show_grad_norm:
                grad = sum(
                    (
                        param.grad.norm().item()
                        for _, param in trainer_data.model.named_parameters()
                        if param.grad is not None
                    ),
                    0.0,
                )
                grad_str = f"(grad: {grad:.2e})"

            lr_str = ""
            if self.show_lr and hasattr(trainer_data, "scheduler"):
                lr = trainer_data.optimizer.param_groups[0]["lr"]
                lr_str = f"(lr: {lr:.2e})"

            log_line = f"{i:10}: {v:.5e} {grad_str} {lr_str}"
            print(log_line)

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(log_line + "\n")


@dataclass(kw_only=True)
class ModularPlotCallback(Callback):
    """
    A callback that creates modular plots during training.

    This callback creates a figure with multiple panels that can be updated during
    training. Each panel is created from a panel class specified in the panels list.

    Attributes
    ----------
    panels : list
        List of panel classes. Each panel class should implement an update method
        that takes trainer_data as an argument.
    plot_every : int, default=100
        Number of iterations between plot updates.
    _fig : matplotlib.figure.Figure
        The figure to plot on.
    _grid_spec : matplotlib.gridspec.GridSpec
        The grid specification for the panel layout.
    _active_panels : list
        The list of active panels.

    Notes
    -----
    The callback creates a figure with panels arranged vertically. Each panel is
    updated every `plot_every` iterations by calling its update method with the
    current trainer_data.
    """

    panels: list

    plot_every: int = 100

    def __post_init__(self) -> None:
        """
        Initialize the callback after instantiation.

        This method is automatically called after the callback is instantiated. It sets
        up the figure and grid specification for the modular plots, creates the panel
        instances, and configures the interactive plotting mode.

        Notes
        -----
        The method creates a figure with dimensions proportional to the number of
        panels, arranges the panels vertically using GridSpec, and initializes
        interactive matplotlib mode for real-time updates.
        """
        self._fig = plt.figure(figsize=(12, 3.5 * len(self.panels)))
        self._grid_spec = GridSpec(len(self.panels), 1, figure=self._fig)

        self._active_panels = []
        for i, panel_entry in enumerate(self.panels):
            if isinstance(panel_entry, tuple):
                panel_class, extra_kwargs = panel_entry
            else:
                panel_class, extra_kwargs = panel_entry, {}
            panel = panel_class(
                fig=self._fig,
                grid_spec=self._grid_spec,
                position=i,
                **extra_kwargs,
            )
            self._active_panels.append(panel)

        plt.ion()
        plt.tight_layout()
        plt.show()

    def on_iteration_end(self, trainer_data: TrainerData) -> None:
        """
        Update the plots at the end of each iteration.

        This method checks if the current iteration is a multiple of `plot_every` or
        if it's the final iteration. If so, it updates all panels with the current
        training data and refreshes the figure display.

        Parameters
        ----------
        trainer_data : TrainerData
            The trainer data containing the data of the trainer.

        Notes
        -----
        The method handles the actual plotting updates by:
        - Checking if it's time to update based on iteration count
        - Calling update() on each panel with current trainer data
        - Redrawing the figure canvas
        - Flushing any pending display events
        """
        if (
            trainer_data.iteration % self.plot_every == 0
            or trainer_data.iteration == trainer_data.iterations - 1
        ):
            for panel in self._active_panels:
                panel.update(trainer_data)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
