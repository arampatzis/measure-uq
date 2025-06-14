"""
Partial Differential Equation (PDE) module.

This module provides abstract base classes and implementations for defining and working
with partial differential equations (PDEs) in the measure-uq package.

The module provides:
    - Parameters: A class to manage parameter values used in PDE models.
    - Condition: A class to manage conditions in the PDE.
    - Conditions: A class to manage a collection of conditions.
    - PDE: A class to manage a PDE.
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
    Buffer,
    DeviceLikeType,
    SparseDynamicArray,
    extend_vector_tensor,
)


@dataclass(kw_only=True)
class Parameters:
    """
    Class for handling PDE parameters.

    This class provides methods to manage and manipulate parameters
    used in partial differential equations.

    Attributes
    ----------
    values : Optional[Tensor]
        A tensor containing parameter values. Should be initialized either
        directly or by the `sample_values` method to ensure it's ready for
        gradient computation.
    device : DeviceLikeType
        The device to move the values of the parameters to.

    Methods
    -------
    sample_values():
        Should be implemented to assign sampled values to the `values` attribute.
    to_device():
        Move the `values` attribute to the device specified in the `device` attribute.
    """

    values: Tensor = field(init=True, repr=True, default_factory=torch.Tensor)

    device: DeviceLikeType = "cpu"

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
        if self.values.is_leaf:
            self.values.requires_grad = True

        if not self.values.requires_grad:
            raise ValueError(
                "The attribute `values` is not a leaf tensor and "
                "`requires_grad` cannot be set to True.",
            )

        self.values = self.values.to(self.device)

    def is_on_device(self, device: DeviceLikeType) -> bool:
        """
        Check if the `values` attribute is on the specified device.

        Parameters
        ----------
        device : DeviceLikeType
            The device to check against.

        Returns
        -------
        bool
            True if the `values` attribute is on the specified device, False otherwise.
        """
        return self.values.device == torch.device(device)


@dataclass(kw_only=True)
class Condition(ABC):
    """
    Abstract base class for conditions in the PDE.

    This class represents a condition in the PDE and provides methods to sample points,
    move data to a specified device, and evaluate the condition using a given model
    and parameters.

    Attributes
    ----------
    points : Tensor
        A tensor containing the points for the condition.
    loss : SparseDynamicArray
        A dynamic array to store the loss values.
    buffer : Buffer
        A buffer to store additional data for the condition.

    Methods
    -------
    __post_init__()
        Initialize the `points` attribute.
    to(device: str)
        Move the `points` and `buffer` attributes to the specified device.
    sample_points()
        Sample and assign data to the attribute `points`.
    __call__(model: ModelWithCombinedInput, parameters: Tensor) -> Tensor
        Evaluate the condition using the given model and parameters.
    """

    points: Tensor = field(init=True, repr=True, default_factory=torch.Tensor)

    loss: SparseDynamicArray = field(
        init=False,
        repr=True,
        default_factory=lambda: SparseDynamicArray(shape=1000, dtype=float),
    )

    buffer: Buffer = field(init=False, repr=True, default_factory=Buffer)

    def __post_init__(self) -> None:
        """
        Initialize the Condition class.

        Ensures that the `points` attribute is initialized and ready for
        gradient computation. If not initialized, it attempts to sample points.

        Important: A child class that implements a `__post_init__` method should
        call the `super().__post_init__()` method at the end of the method to ensure
        that the `points` attribute is initialized and ready for gradient computation.
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

    def to(self, device: DeviceLikeType) -> None:
        """
        Move the condition to the specified device.

        Parameters
        ----------
        device : DeviceLikeType
            The device to move the condition to.
        """
        self.points = self.points.to(device)
        self.buffer.to(device)

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

    Attributes
    ----------
    conditions : list[Condition]
        A list of condition objects to be managed.
    device : DeviceLikeType
        The device to place the condition points on.
    n : int
        The number of conditions in the collection, automatically set during
        initialization.

    Notes
    -----
    After initialization, all condition points are moved to the specified device.
    The class provides iteration and indexing functionality to access the underlying
    conditions.
    """

    device: DeviceLikeType = "cpu"

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
        """
        Return the number of conditions.

        Returns
        -------
        int
            The number of conditions.
        """
        return len(self.conditions)

    def to_device(self) -> None:
        """
        Move all points and buffers of the conditions to a device.

        The device is specified in the `device` attribute. This method is called during
        initialization and ensures that all condition points and buffers are on the
        correct device for computation.
        """
        for condition in self.conditions:
            condition.to(self.device)

    def is_on_device(self, device: DeviceLikeType) -> bool:
        """
        Check if all points and buffers of the conditions are on the specified device.

        Parameters
        ----------
        device : torch.device or str
            The device to check against.

        Returns
        -------
        bool
            True if all points and buffers of the conditions are on the specified
            device, False otherwise.
        """
        return all(
            condition.points.device == torch.device(device)
            for condition in self.conditions
        )

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
        if iteration == 0:
            return

        for i, condition in enumerate(self.conditions):
            if iteration % resample_conditions_every[i] == 0:
                condition.sample_points()
                condition.points.requires_grad = True
                condition.to(self.device)

    def l2_loss(self, model: ModelWithCombinedInput, parameters: Tensor) -> Tensor:
        """
        Compute the L2 loss for the given model and parameters.

        This method evaluates the conditions using the provided model and parameters,
        and computes the L2 loss for each condition.

        Parameters
        ----------
        model : ModelWithCombinedInput
            The model to evaluate the conditions for.
        parameters : Tensor
            The parameters to use for the evaluation.

        Returns
        -------
        Tensor
            A tensor containing the L2 loss for each condition.

        Notes
        -----
        The L2 loss is computed as the mean of the squared L2 norm of the residuals
        for each condition.
        """
        res = torch.zeros(self.n)
        for i, condition in enumerate(self.conditions):
            res[i] = torch.mean(
                torch.linalg.vector_norm(
                    condition(model, parameters),
                    ord=2,
                    dim=1,
                )
                ** 2,
            )

        return res

    def eval(
        self,
        model: ModelWithCombinedInput,
        parameters: Tensor,
        iteration: int,
    ) -> Tensor:
        """
        Evaluate the conditions and store the loss.

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
        res = self.l2_loss(model, parameters)
        for i, condition in enumerate(self.conditions):
            condition.loss[iteration] = res[i].item()

        return res

    def eval_for_closure(
        self,
        model: ModelWithCombinedInput,
        parameters: Tensor,
    ) -> Tensor:
        """
        Evaluate the conditions for the closure function.

        This method is used by optimizers that require a closure function. It evaluates
        the conditions using the provided model and parameters, and returns the result
        as a tensor. It does not store the loss.

        Parameters
        ----------
        model : ModelWithCombinedInput
            The model to evaluate the conditions for.
        parameters : Tensor
            The parameters to use for the evaluation.

        Returns
        -------
        Tensor
            The value of the conditions evaluated for the closure function.
        """
        return self.l2_loss(model, parameters)


@dataclass(kw_only=True)
class PDE:
    """
    A class representing a Partial Differential Equation (PDE).

    This class provides functionality for handling both training and testing aspects
    of a PDE, including conditions, parameters, loss weights, and resampling strategies.
    It manages the training and testing conditions separately, along with their
    corresponding parameters, and provides methods for computing losses and resampling
    data during the training process.

    The class is designed to work with neural network models that solve PDEs, providing
    a structured way to define the problem, compute losses, and handle the training
    process. It supports dynamic resampling of both conditions and parameters, which
    can help in achieving better training results by exploring different parts of the
    solution space.

    Attributes
    ----------
    conditions_train : Conditions
        The conditions used for training the PDE.
    conditions_test : Conditions
        The conditions used for testing the PDE.
    parameters_train : Parameters
        The parameters used for training the PDE.
    parameters_test : Parameters
        The parameters used for testing the PDE.
    loss_weights_ : Tensor
        The weights for each condition's loss, initialized during post-construction.
    resample_conditions_every_ : Tensor
        The number of iterations between resampling points for each condition.
    resample_parameters_every : int
        The number of iterations between resampling parameters, default is infinity.
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
        loss_weights: ArrayLike1DFloat | None,
        resample_conditions_every: ArrayLike1DInt | None,
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

    def resample_conditions(self, iteration: int) -> None:
        """
        Resample the conditions for the PDE.

        This method updates the conditions by resampling them
        according to the current state of the PDE.

        Parameters
        ----------
        iteration : int
            The current training iteration.

        Returns
        -------
        None
            This method does not return anything.
        """
        self.conditions_train.sample_points(iteration, self.resample_conditions_every_)

        if iteration > 0 and iteration % self.resample_parameters_every == 0:
            self.parameters_train.sample_values()
            self.parameters_train.to_device()

    def loss_train(self, model: ModelWithCombinedInput, iteration: int) -> Tensor:
        """
        Compute the training loss for the given model and iteration.

        This method evaluates the training conditions using the provided model and
        parameters, and computes the weighted loss based on the specified loss weights.

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
        res = self.conditions_train.eval(
            model,
            self.parameters_train.values,
            iteration,
        )

        return torch.dot(self.loss_weights_, res)

    def loss_train_for_closure(self, model: ModelWithCombinedInput) -> Tensor:
        """
        Compute the training loss for the given model.

        This method evaluates the training conditions using the provided model and
        parameters, and computes the weighted loss based on the specified loss weights.
        It is intended to be used as a closure for optimizers that require a closure
        function.

        Parameters
        ----------
        model : ModelWithCombinedInput
            The model to evaluate the conditions for.

        Returns
        -------
        Tensor
            The computed training loss.
        """
        res = self.conditions_train.eval_for_closure(
            model,
            self.parameters_train.values,
        )

        return torch.dot(self.loss_weights_, res)

    def loss_test(self, model: ModelWithCombinedInput, iteration: int = 0) -> Tensor:
        """
        Compute the test loss for the given model and iteration.

        This method evaluates the test conditions using the provided model and
        parameters, and computes the weighted loss based on the specified loss weights.

        Parameters
        ----------
        model : ModelWithCombinedInput
            The model to evaluate the conditions for.
        iteration : int, optional
            The current test iteration, by default 0.

        Returns
        -------
        Tensor
            The computed test loss.
        """
        res = self.conditions_test.eval(
            model,
            self.parameters_test.values,
            iteration,
        )

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
