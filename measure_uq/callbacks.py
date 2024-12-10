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

from torch import nn


@dataclass(kw_only=True)
class Callback:
    """
    Base class for callbacks.

    Attributes
    ----------
    trainer : nn.Module or None
        Reference of the trainer being trained.
    """

    trainer: nn.Module | None = None

    def set_trainer(self, trainer):
        """
        Sets the trainer for the callback and calls init.

        Parameters
        ----------
        trainer: instance of ``trainer``
            The trainer to be set.
        """
        if trainer is not self.trainer:
            self.trainer = trainer
            self.init()

    def init(self):
        """Init after setting a trainer."""

    def on_iteration_begin(self):
        """Called at the beginning of every iteration."""

    def on_teration_end(self):
        """Called at the end of every iteration."""

    def on_train_begin(self):
        """Called at the beginning of trainer training."""

    def on_train_end(self):
        """Called at the end of trainer training."""

    def on_predict_begin(self):
        """Called at the beginning of prediction."""

    def on_predict_end(self):
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

    def set_trainer(self, trainer):
        """
        Set a trainer for each callback.

        Parameters
        ----------
        trainer : nn.Module
            The trainer to be set.
        """
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_iteration_begin(self):
        """Called at the beginning of every iteration."""
        for callback in self.callbacks:
            callback.on_iteration_begin()

    def on_iteration_end(self):
        """Called at the end of every iteration."""
        for callback in self.callbacks:
            callback.on_iteration_end()

    def on_train_begin(self):
        """Called at the beginning of trainer training."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        """Called at the end of trainer training."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_predict_begin(self):
        """Called at the beginning of prediction."""
        for callback in self.callbacks:
            callback.on_predict_begin()

    def on_predict_end(self):
        """Called at the end of prediction."""
        for callback in self.callbacks:
            callback.on_predict_end()

    def append(self, callback):
        """
        Append a callback to the list of callbacks.

        Args:
        ----
            callback: The callback to be appended.

        Raises:
        ------
            Exception: If the callback is not an instance of Callback.
        """
        if not isinstance(callback, Callback):
            raise TypeError(str(callback) + " is an invalid Callback object")
        self.callbacks.append(callback)
