"""
Defines classes and functions for handling parameters and conditions
in the context of physics-informed machine learning. The module includes:

- A `Parameters` class for managing parameter values used in the model. This class
  provides functionality to sample and initialize these values, ensuring they
  are ready for gradient computation.

- A `Condition` abstract base class for defining conditions that the model must
  satisfy. This class includes a `loss` attribute for tracking the loss values
  related to the condition.

Additionally, a utility function `extend_vector_tensor` is provided to extend a
given tensor to a specified size, filling with a default value if necessary.

Note: The module relies on PyTorch for tensor operations and gradient
computation.
"""

# ruff: noqa: S301

import pickle
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from measure_uq.models import ModelWithCombinedInput
from measure_uq.typing import ArrayLike1DFloat, ArrayLike1DInt
from measure_uq.utilities import INT_INF, LossContainer, extend_vector_tensor


@dataclass(kw_only=True)
class Parameters:
    """
    A class to manage parameter values used in a model, ensuring they are
    ready for gradient computation.

    Attributes
    ----------
    values : Optional[Tensor]
        A tensor containing parameter values. Should be initialized either
        directly or by the `sample_values` method to ensure it's ready for
        gradient computation.

    Methods
    -------
    sample_values():
        Should be implemented to assign sampled values to the `values` attribute.
    """

    values: Tensor = field(init=True, repr=True, default_factory=torch.Tensor)

    def __post_init__(self):
        """
        Ensures that the `values` attribute is initialized and ready for
        gradient computation. If not initialized, it attempts to sample values.
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

    def sample_values(self):
        """
        Placeholder method to sample and assign values to `values`.
        This method should be implemented in subclasses or instances.
        """


@dataclass(kw_only=True)
class Condition(ABC):
    """
    Abstract base class for defining conditions to be satisfied by a model.

    Attributes
    ----------
    points : Optional[Tensor]
        A tensor containing points where the condition should be evaluated.
        When `None`, the `sample_points` method is called to sample points.

    loss : LossContainer
        A container for storing loss values related to the condition.
    """

    points: Tensor = field(init=True, repr=True, default_factory=torch.Tensor)

    loss: LossContainer = field(init=False, repr=True, default_factory=LossContainer)

    def __post_init__(self):
        """
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

    def sample_points(self):
        """
        Placeholder method to sample and assign points to `points`.
        This method should be implemented in subclasses or instances.
        """

    def __call__(self, model: ModelWithCombinedInput, parameters: Tensor) -> Tensor:
        """
        Evaluates the condition for given model and parameters.

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
        self.points.requires_grad = True
        parameters.requires_grad = True

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
class PDE(ABC):
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

    conditions_train: list[Condition]
    conditions_test: list[Condition]
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
    ):
        """
        Validates and initializes attributes after dataclass construction.
        Ensures equal number of train/test conditions and initializes vectors.
        """
        n = len(self.conditions_train)

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

        if len(self.conditions_train) != len(self.conditions_test):
            raise ValueError(
                "The number of training and test conditions must be equal.",
            )

        if self.loss_weights_.dim() != 1:
            raise ValueError("loss_weights_ tensor is not a vector (1D).")

        if self.resample_conditions_every_.dim() != 1:
            raise ValueError("resample_conditions_every_ tensor is not a vector (1D).")

        self.loss_weights_ = extend_vector_tensor(
            x=self.loss_weights_,
            n=len(self.conditions_train),
            default_value=1.0,
        )

        self.resample_conditions_every_ = extend_vector_tensor(
            x=self.resample_conditions_every_,
            n=len(self.conditions_train),
            default_value=INT_INF,
        )

    def loss_train(self, model: ModelWithCombinedInput, iteration: int):
        """
        Computes the loss for the training conditions. Re-sample points on conditions
        or parameters if needed.

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
        res = torch.zeros(len(self.conditions_train))

        for i, condition in enumerate(self.conditions_train):
            if iteration > 0 and iteration % self.resample_conditions_every_[i] == 0:
                condition.sample_points()

        if iteration > 0 and iteration % self.resample_parameters_every == 0:
            self.parameters_train.sample_values()

        for i, condition in enumerate(self.conditions_train):
            res[i] = torch.mean(condition(model, self.parameters_train.values) ** 2)
            condition.loss(iteration, res[i].item())

        return torch.dot(self.loss_weights_, res)

    def loss_test(self, model: ModelWithCombinedInput, iteration: int = 0):
        """
        Computes the loss for the testing conditions.

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
        res = torch.zeros(len(self.conditions_test))

        for i, condition in enumerate(self.conditions_test):
            res[i] = torch.mean(condition(model, self.parameters_test.values) ** 2)
            condition.loss(iteration, res[i].item())

        return torch.dot(self.loss_weights_, res)

    def save(self, filename: str | Path = "pde.pickle"):
        """
        Saves the PDE to a file using pickling.

        Parameters
        ----------
        filename : str | Path
            The name of the file to save the PDE to, by default "pde.pickle".
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str | Path):
        """
        Loads a PDE instance from a file using pickling.

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
