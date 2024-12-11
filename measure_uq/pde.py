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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from measure_uq.typing import ArrayLike1D
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

    def __call__(self, model: nn.Module, parameters: Tensor) -> Tensor:
        """
        Evaluates the condition for given model and parameters.

        Parameters
        ----------
        model : nn.Module
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

        z = model.combine_input(self.points, parameters)

        y = model(z)

        return self.eval(y, z)

    @abstractmethod
    def eval(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Abstract method to evaluate the condition for given output and input.

        Parameters
        ----------
        y : Tensor
            The output of the model.
        x : Tensor
            The input of the model.

        Returns
        -------
        Tensor
            The result of evaluating the condition.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class PDE(ABC):
    """
    Abstract base class for representing partial differential equations (PDEs)
    within a physics-informed machine learning framework.

    Attributes
    ----------
    conditions_train : list[Condition]
        A list of conditions to be satisfied during the training phase.
    conditions_test : list[Condition]
        A list of conditions to be satisfied during the testing phase.
    parameters_train : Parameters
        Parameters used during training.
    parameters_test : Parameters
        Parameters used during testing.
    loss_weights : Optional[Tensor]
        Weights applied to each condition's loss contribution.
    resample_conditions_every : Optional[Tensor]
        Frequency at which conditions are resampled.
    resample_parameters_every : Optional[int]
        Frequency at which parameters are resampled.
    """

    conditions_train: list[Condition]
    conditions_test: list[Condition]
    parameters_train: Parameters
    parameters_test: Parameters
    loss_weights: Tensor = field(init=False)
    resample_conditions_every: Tensor = field(init=False)
    resample_parameters_every: int = INT_INF

    def __post_init__(
        self,
        loss_weights: ArrayLike1D | None = None,
        resample_conditions_every: ArrayLike1D | None = None,
    ):
        """
        Validates and initializes attributes after dataclass construction.
        Ensures equal number of train/test conditions and initializes vectors.
        """
        n = len(self.conditions_train)

        if loss_weights is not None:
            self.loss_weights = torch.tensor(loss_weights)
        else:
            self.loss_weights = torch.ones(n)

        if resample_conditions_every is not None:
            self.resample_conditions_every = torch.tensor(resample_conditions_every)
        else:
            self.resample_conditions_every = torch.full((n,), INT_INF)

        if len(self.conditions_train) != len(self.conditions_test):
            raise ValueError(
                "The number of training and test conditions must be equal.",
            )

        if self.loss_weights.dim() != 1:
            raise ValueError("loss_weights tensor is not a vector (1D).")

        if self.resample_conditions_every.dim() != 1:
            raise ValueError("resample_conditions_every tensor is not a vector (1D).")

        self.loss_weights = extend_vector_tensor(
            x=self.loss_weights,
            n=len(self.conditions_train),
            default_value=1.0,
        )

        self.resample_conditions_every = extend_vector_tensor(
            x=self.resample_conditions_every,
            n=len(self.conditions_train),
            default_value=INT_INF,
        )

    def loss_train(self, model: nn.Module, iteration: int):
        """
        Computes the loss for the training conditions. Re-sample points on conditions
        or parameters if needed.

        Parameters
        ----------
        model : nn.Module
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
            if iteration > 0 and iteration % self.resample_conditions_every[i] == 0:
                condition.sample_points()

        if iteration > 0 and iteration % self.resample_parameters_every == 0:
            self.parameters_train.sample_values()

        for i, condition in enumerate(self.conditions_train):
            res[i] = torch.mean(condition(model, self.parameters_train.values) ** 2)
            condition.loss(iteration, res[i].item())

        return torch.dot(self.loss_weights, res)

    def loss_test(self, model: nn.Module, iteration: int = 0):
        """
        Computes the loss for the testing conditions.

        Parameters
        ----------
        model : nn.Module
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

        return torch.dot(self.loss_weights, res)
