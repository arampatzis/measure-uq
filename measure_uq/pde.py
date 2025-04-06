"""
Partial Differential Equation (PDE) module.

This module provides abstract base classes and implementations for defining and working
with partial differential equations (PDEs) in the measure-uq package.

The module includes:
- Abstract base class for PDE definitions
- Parameter management for PDE models
- Utilities for boundary conditions and domain specifications

Classes
-------
PDE
    Abstract base class for defining partial differential equations.

Parameters
----------
    A class to manage parameter values used in PDE models.

Notes
-----
This module forms the foundation for physics-informed neural networks (PINNs) by
providing the mathematical framework for the differential equations that the neural
networks aim to solve.

"""

# ruff: noqa: S301


import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Self

import torch
from torch import Tensor

from measure_uq.models import ModelWithCombinedInput
from measure_uq.utilities import (
    INT_INF,
    ArrayLike1DFloat,
    ArrayLike1DInt,
    SparseDynamicArray,
    extend_vector_tensor,
)


@dataclass(kw_only=True)
class Parameters:
    """
    Manage parameter values used in a model.

    Ensuring they are ready for gradient ready for gradient computation.

    Attributes
    ----------
    values : Optional[Tensor]
        A tensor containing parameter values. Should be initialized either
        directly or by the `sample_values` method to ensure it's ready for
        gradient computation.
    device : str
        The device to move the values of the parameters to.


    Methods
    -------
    sample_values():
        Should be implemented to assign sampled values to the `values` attribute.
    """

    values: Tensor = field(init=True, repr=True, default_factory=torch.Tensor)

    device: str = "cpu"

    def __post_init__(self) -> None:
        """
        Initialize the `values` attribute.

        Ensures that the `values` attribute is initialized and ready for gradient
        computation. If not initialized, it attempts to sample values.
        """
        if self.values.numel() == 0:
            self.sample_values()

        if self.values.numel() == 0:
            raise ValueError(
                "The attribute `values` is None after sampling. "
                "The `sample_values` method should assign it.",
            )

        if self.values.dim() != 2:
            raise ValueError(
                "The attribute `values` should be a 2D tensor. ",
            )

        self.values.requires_grad = True
        self.to_device()

    def sample_values(self) -> None:
        """
        Sample and assign values to `values`.

        This method is a placeholder and should be implemented in subclasses or
        instances. If it is not implemented, the `values` attribute should be set by the
        user, otherwise an error will be raised.
        """

    def to_device(self) -> None:
        """
        Move the `values` attribute to the device specified in the `device` attribute.

        This method is called in the `__post_init__` method to ensure the `values`
        attribute is on the correct device, and is ready for gradient computation.
        """
        self.values.requires_grad = True
        self.values = self.values.to(self.device)


@dataclass(kw_only=True)
class Condition(ABC):
    """
    Abstract base class for defining conditions to be satisfied by a model.

    Attributes
    ----------
    points : Optional[Tensor]
        A tensor containing points where the condition should be evaluated.
        When `None`, the `sample_points` method is called to sample points.

    loss : SparseDynamicArray
        A container for storing loss values related to the condition.
    """

    points: Tensor = field(init=True, repr=True, default_factory=torch.Tensor)

    loss: SparseDynamicArray = field(
        init=False,
        repr=True,
        default_factory=lambda: SparseDynamicArray(shape=1000, dtype=float),
    )

    def __post_init__(self) -> None:
        """
        Initialize the `points` attribute.

        Ensures that the `points` attribute is initialized and ready for
        gradient computation. If not initialized, it attempts to sample points.
        """
        if self.points.numel() == 0:
            self.sample_points()

        if self.points.numel() == 0:
            raise ValueError(
                "The attibute `points` is None after sampling."
                "The methods `sample_points` should assign it.",
            )

        if self.points.dim() != 2:
            raise ValueError(
                "The attribute `points` should be a 2D tensor.",
            )

        self.points.requires_grad = True

    def sample_points(self) -> None:
        """
        Sample and assign data to the attribute `points`.

        Placeholder method to sample and assign data to the attribute `points`.
        This method should be implemented in subclasses or instances.
        """

    def __call__(self, model: ModelWithCombinedInput, parameters: Tensor) -> Tensor:
        """
        Evaluate the condition for given model and parameters.

        Parameters
        ----------
        model : ModelWithCombinedInput
            The model to evaluate the condition for.
        parameters : Tensor
            The parameters to use for evaluation.

        Returns
        -------
        Tensor
            The result of evaluating the condition.
        """
        z, y = model(self.points, parameters)

        return self.eval(z, y)

    @abstractmethod
    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Abstract method to evaluate the condition for given output and input.

        Parameters
        ----------
        x : Tensor
            The input of the model.
        y : Tensor
            The output of the model.

        Returns
        -------
        Tensor
            The result of evaluating the condition.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class Conditions:
    """
    A container class for a collection of conditions.

    This class manages a list of `Condition` objects and provides methods for
    iterating through them, accessing individual conditions, and handling
    device placement and point sampling.

    Parameters
    ----------
    conditions : list[Condition], optional
        A list of condition objects to be managed. Default is an empty list.
    device : str, optional
        The device to place the condition points on. Default is "cpu".

    Attributes
    ----------
    n : int
        The number of conditions in the collection.
    conditions : list[Condition]
        The list of condition objects.
    device : str
        The device where condition points are placed.

    Notes
    -----
    After initialization, all condition points are moved to the specified device.
    The class provides iteration and indexing functionality to access the underlying
    conditions.
    """

    device: str = "cpu"

    conditions: list[Condition] = field(init=True, repr=True, default_factory=list)

    n: int = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Initialize the number of conditions and move all points to the device."""
        self.n = len(self.conditions)
        self.to_device()

    def __iter__(self) -> Iterator[Condition]:
        """
        Return an iterator over the conditions in the collection.

        Returns
        -------
        Iterator[Condition]
            An iterator that yields each condition in the collection.
        """
        return iter(self.conditions)

    def __getitem__(self, index: int) -> Condition:
        """
        Return the condition at the given index.

        Parameters
        ----------
        index : int
            The index of the condition to retrieve.

        Returns
        -------
        Condition
            The condition at the specified index.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        return self.conditions[index]

    def __len__(self) -> int:
        """Return the number of conditions."""
        return len(self.conditions)

    def to_device(self) -> None:
        """
        Move all points of the conditions to a device.

        The device is specified in the `device` attribute. This method is called during
        initialization and ensures that all condition points are on the correct device
        for computation.
        """
        for condition in self.conditions:
            condition.points = condition.points.to(self.device)

    def sample_points(self, iteration: int, resample_conditions_every: Tensor) -> None:
        """
        Sample new points for the conditions.

        Parameters
        ----------
        iteration : int
            The current iteration.
        resample_conditions_every : Tensor
            The number of iterations between resampling points for each condition.

        Notes
        -----
        This method is used to resample the points for the conditions at the
        specified iterations. The actual resampling is done by calling the
        `sample_points` method of each condition.
        """
        for i, condition in enumerate(self.conditions):
            if iteration > 0 and iteration % resample_conditions_every[i] == 0:
                condition.sample_points()
                condition.points.requires_grad = True
                condition.points = condition.points.to(self.device)

    def eval(
        self,
        model: ModelWithCombinedInput,
        parameters: Tensor,
        iteration: int,
    ) -> Tensor:
        """
        Evaluate the conditions.

        Parameters
        ----------
        model : ModelWithCombinedInput
            The model to evaluate the conditions for.
        parameters : Tensor
            The parameters to use for the evaluation.
        iteration : int
            The current iteration.

        Returns
        -------
        Tensor
            The value of the conditions at the current iteration.

        Notes
        -----
        This method is used to evaluate the conditions at each iteration. The
        actual evaluation is done by calling the `eval` method of each
        condition. The results are then stored in the `loss` attribute of each
        condition.
        """
        res = torch.zeros(self.n)
        for i, condition in enumerate(self.conditions):
            res[i] = torch.mean(condition(model, parameters) ** 2)
            condition.loss[iteration] = res[i].item()

        return res


@dataclass(kw_only=True)
class PDE:
    """
    Abstract base class for physics-informed neural networks (PINNs) for PDEs.

    Parameters
    ----------
    conditions_train : list[Condition]
        A list of conditions to be satisfied by the model during training.
    conditions_test : list[Condition]
        A list of conditions to be satisfied by the model during testing.
    parameters_train : Parameters
        Parameters for the model during training.
    parameters_test : Parameters
        Parameters for the model during testing.
    loss_weights : Optional[ArrayLike1DFloat], optional
        Loss weights for conditions. If not given, equal weights are assigned.
        by default None
    resample_conditions_every : Optional[ArrayLike1DInt], optional
        The number of iterations between resampling points for each condition.
        If not given, parameters are sampled once at the beginning.
        by default None

    Attributes
    ----------
    loss_weights_ : Tensor
        The loss weights as a tensor.
    resample_conditions_every_ : Tensor
        The number of iterations between resampling points as a tensor.
    """

    conditions_train: Conditions
    conditions_test: Conditions
    parameters_train: Parameters
    parameters_test: Parameters
    loss_weights_: Tensor = field(init=False)
    resample_conditions_every_: Tensor = field(init=False)
    resample_parameters_every: int = INT_INF

    loss_weights: InitVar[ArrayLike1DFloat | None] = None
    resample_conditions_every: InitVar[ArrayLike1DInt | None] = None

    def __post_init__(
        self,
        loss_weights: ArrayLike1DFloat | None = None,
        resample_conditions_every: ArrayLike1DInt | None = None,
    ) -> None:
        """
        Validate and initialize attributes after dataclass construction.

        Parameters
        ----------
        loss_weights : ArrayLike1DFloat | None, optional
            The weights for each condition's loss. If None, equal weights are used.
        resample_conditions_every : ArrayLike1DInt | None, optional
            The number of iterations between resampling points for each condition.
            If None, points are sampled once at initialization.

        Notes
        -----
        This method ensures that:
        - The number of training and test conditions are equal
        - The loss weights and resampling intervals are properly initialized
        - All tensors are properly shaped and on the correct device
        """
        n = self.conditions_train.n

        if loss_weights is not None:
            if not isinstance(loss_weights, Tensor):
                self.loss_weights_ = torch.tensor(loss_weights)
            else:
                self.loss_weights_ = loss_weights
        else:
            self.loss_weights_ = torch.ones(n)

        if resample_conditions_every is not None:
            if not isinstance(resample_conditions_every, Tensor):
                self.resample_conditions_every_ = torch.tensor(
                    resample_conditions_every,
                )
            else:
                self.resample_conditions_every_ = resample_conditions_every
        else:
            self.resample_conditions_every_ = torch.full((n,), INT_INF)

        if self.conditions_train.n != self.conditions_test.n:
            raise ValueError(
                "The number of training and test conditions must be equal.",
            )

        if self.loss_weights_.dim() != 1:
            raise ValueError("loss_weights_ tensor is not a vector (1D).")

        if self.resample_conditions_every_.dim() != 1:
            raise ValueError("resample_conditions_every_ tensor is not a vector (1D).")

        self.loss_weights_ = extend_vector_tensor(
            x=self.loss_weights_,
            n=self.conditions_train.n,
            default_value=1.0,
        )

        self.resample_conditions_every_ = extend_vector_tensor(
            x=self.resample_conditions_every_,
            n=self.conditions_train.n,
            default_value=INT_INF,
        )

    def loss_train(self, model: ModelWithCombinedInput, iteration: int) -> Tensor:
        """
        Compute the loss for the training conditions.

        Re-sample points on conditions or parameters if needed.

        Parameters
        ----------
        model : ModelWithCombinedInput
            The model to evaluate the conditions for.
        iteration : int
            The current training iteration.

        Returns
        -------
        Tensor
            The computed training loss.
        """
        self.conditions_train.sample_points(iteration, self.resample_conditions_every_)

        if iteration > 0 and iteration % self.resample_parameters_every == 0:
            self.parameters_train.sample_values()
            self.parameters_train.to_device()

        res = self.conditions_train.eval(model, self.parameters_train.values, iteration)

        return torch.dot(self.loss_weights_, res)

    def loss_test(self, model: ModelWithCombinedInput, iteration: int = 0) -> Tensor:
        """
        Compute the loss for the testing conditions.

        Parameters
        ----------
        model : ModelWithCombinedInput
            The model to evaluate the conditions for.
        iteration : int, optional
            The current testing iteration (default is 0).

        Returns
        -------
        Tensor
            The computed testing loss.
        """
        res = self.conditions_test.eval(model, self.parameters_test.values, iteration)

        return torch.dot(self.loss_weights_, res)

    def save(self, filename: str | Path = "pde.pickle") -> None:
        """
        Save the PDE to a file using pickling.

        Parameters
        ----------
        filename : str | Path
            The name of the file to save the PDE to, by default "pde.pickle".
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str | Path) -> Self:
        """
        Load a PDE instance from a file using pickling.

        Parameters
        ----------
        filename : str | Path
            The name of the file from which to load the PDE instance.

        Returns
        -------
        PDE
            The loaded PDE instance.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
