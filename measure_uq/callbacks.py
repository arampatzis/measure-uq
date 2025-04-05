"""
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

from measure_uq.trainers.trainer_data import TrainerData


@dataclass(kw_only=True)
class Callback:
    """
    Abstract base class used to build new callbacks.

    Notes
    -----
    In order to activate the callbacks, pass it to the `Trainer` instance.

    The `Callback` class is an abstract class that every callback should
    inherit. It provides five methods that can be overridden to add new
    functionality to the training process.

    Parameters
    ----------
    trainer_data : TrainerData
        The `TrainerData` instance containing the data of the trainer.
    """

    trainer_data: TrainerData

    def init(self) -> None:
        """Init after setting a trainer."""

    def on_iteration_begin(self) -> None:
        """Called at the beginning of every iteration."""

    def on_iteration_end(self) -> None:
        """Called at the end of every iteration."""

    def on_train_begin(self) -> None:
        """Called at the beginning of trainer training."""

    def on_train_end(self) -> None:
        """Called at the end of trainer training."""

    def on_predict_begin(self) -> None:
        """Called at the beginning of prediction."""

    def on_predict_end(self) -> None:
        """Called at the end of prediction."""


@dataclass(kw_only=True)
class CallbackList:
    """
    A list of callbacks to be executed at specific points during training.

    Parameters
    ----------
    callbacks : list[Callback]
        A list of callbacks to be executed. The list can be empty.

    Notes
    -----
    The list can be empty.
    """

    callbacks: list[Callback]

    def init(self) -> None:
        """Initialize each callback in the list."""
        for callback in self.callbacks:
            callback.init()

    def on_iteration_begin(self) -> None:
        """Called at the beginning of every iteration."""
        for callback in self.callbacks:
            callback.on_iteration_begin()

    def on_iteration_end(self) -> None:
        """Called at the end of every iteration."""
        for callback in self.callbacks:
            callback.on_iteration_end()

    def on_train_begin(self) -> None:
        """Called at the beginning of trainer training."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self) -> None:
        """Called at the end of trainer training."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_predict_begin(self) -> None:
        """Called at the beginning of prediction."""
        for callback in self.callbacks:
            callback.on_predict_begin()

    def on_predict_end(self) -> None:
        """Called at the end of prediction."""
        for callback in self.callbacks:
            callback.on_predict_end()
