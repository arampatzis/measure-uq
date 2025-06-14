"""
Provide classes for creating and managing neural networks.

The module includes classes for managing the networks, such as the `PINN` and `PINN_PCE`
classes, as well as functions for creating the networks, such as the `feedforward`
function.

This module provides:
    - feedforward: Create a feedforward neural network.
    - ResidualBlock: Create a residual block.
    - feedforward_resnet: Create a residual feedforward neural network.
    - ModelWithCombinedInput: A base class for models that combine input and parameters.
    - PINN: A Physics Informed Neural Network.
    - PINN_PCE: A Physics Informed Neural Network with a PCE.
"""

# ruff: noqa: N801

from abc import abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch import Tensor, nn

from measure_uq.networks import NetworkBuilder
from measure_uq.utilities import ArrayLike1DInt, PolyExpansion, torch_numpoly_call


class ModelWithCombinedInput(nn.Module):
    """
    A model that combines input and parameters.

    This class inherits from `nn.Module` and provides functionality to combine
    input tensors with parameter tensors. It serves as a base class for more
    complex models that require combined input and parameters.

    Parameters
    ----------
    network_builder : NetworkBuilder
        The network builder.
    """

    network_builder: NetworkBuilder
    network: nn.Sequential

    def __init__(self, network_builder: NetworkBuilder) -> None:
        """
        Initialize the ModelWithCombinedInput.

        This constructor sets up the base structure for models that combine
        input and parameters.

        Parameters
        ----------
        network_builder : NetworkBuilder
            The network builder.
        """
        super().__init__()

        self.network_builder = network_builder
        self.network = self.network_builder()

    def combine_input(
        self,
        x: Tensor,
        p: Tensor,
    ) -> Tensor:
        """
        Combine input and parameters into a single tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input to combine with parameters.
        p : torch.Tensor
            Parameters to combine with input.

        Returns
        -------
        torch.Tensor
            Combined input and parameters.
        """
        x_repeated = x.repeat_interleave(p.shape[0], dim=0)

        p_tiled = p.repeat(x.shape[0], 1)

        return torch.cat([x_repeated, p_tiled], dim=1)

    @abstractmethod
    def save(self, file_path: str | Path) -> None:
        """
        Save the model's state and parameters to a file.

        Parameters
        ----------
        file_path : str | Path
            The path to the file where the model will be saved.
        """
        raise NotImplementedError("Must implement in subclass.")

    @abstractmethod
    def load(self, file_path: str | Path) -> Self:
        """
        Load the model's state and parameters from a file.

        Parameters
        ----------
        file_path : str | Path
            The path to the file from which to load the model.
        """
        raise NotImplementedError("Must implement in subclass.")


class PINN(ModelWithCombinedInput):
    """
    Define a PINN.

    A Physics Informed Neural Network (PINN) is a neural network that is
    constrained to satisfy a set of PDEs. The neural network is designed to
    satisfy the PDEs by minimizing a loss function that measures the difference
    between the neural network's output and the solution to the PDEs.

    Parameters
    ----------
    network_builder : NetworkBuilder
        The network builder.

    Attributes
    ----------
    network : nn.Sequential
        The neural network.
    """

    def __init__(self, network_builder: NetworkBuilder) -> None:
        """
        Initialize a Physics Informed Neural Network (PINN) with n hidden layers.

        Parameters
        ----------
        network_builder : NetworkBuilder
            The network builder.
        """
        super().__init__(network_builder=network_builder)

    def forward(
        self,
        x: Tensor,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the output of the neural network for the given input.

        Parameters
        ----------
        x : torch.Tensor
            Input to combine with parameters.
        p : torch.Tensor
            Parameters to combine with input.

        Returns
        -------
        torch.Tensor
            The output of the neural network.
        """
        z = self.combine_input(x, p)

        return z, self.network(z)

    def combine_input(
        self,
        x: Tensor,
        p: Tensor,
    ) -> Tensor:
        """
        Combine input and parameters into a single tensor.

        Assert that the dimensions are correct.

        Parameters
        ----------
        x : torch.Tensor
            Input to combine with parameters.
        p : torch.Tensor
            Parameters to combine with input.

        Returns
        -------
        torch.Tensor
            Combined input and parameters.
        """
        if x.shape[1] + p.shape[1] != self.network[0].in_features:
            raise ValueError(
                f"Input dimensions do not match. Expected "
                f"{self.network[0].in_features}, but got {x.shape[1] + p.shape[1]}.",
            )

        z = super().combine_input(x, p)

        assert z.shape[1] == self.network[0].in_features

        return z

    def save(self, file_path: str | Path) -> None:
        """
        Save the model's state and parameters to a file.

        Parameters
        ----------
        file_path : str | Path
            The path to the file where the model will be saved.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "network_builder": self.network_builder,
                "network": self.network,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        """
        Load a model from a file and return an instance.

        Parameters
        ----------
        file_path : str | Path
            The path to the file from which to load the model.

        Returns
        -------
        PINN
            The loaded model instance.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        model = cls(network_builder=checkpoint["network_builder"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


class PINN_PCE(ModelWithCombinedInput):
    """
    Define a PINN with a PCE.

    This class implements a neural network model that combines the principles of
    Physics Informed Neural Networks (PINNs) with Polynomial Chaos Expansion (PCE).
    The model is designed to handle inputs and parameters, and it computes the
    output based on the given polynomial expansion.

    Parameters
    ----------
    network_builder : NetworkBuilder
        The network builder.
    expansion : PolyExpansion
        The polynomial expansion used in the model.

    Attributes
    ----------
    coefficients : torch.Tensor
        The coefficients of the polynomial expansion.
        :no-index:
    exponents : torch.Tensor
        The exponents of the polynomial expansion.
        :no-index:
    np : int
        The number of polynomial terms in the expansion.
        :no-index:
    n : ArrayLike1DInt
        The architecture of the neural network.
        :no-index:
    Nx : int
        The number of input features.
        :no-index:
    expansion : PolyExpansion
        The polynomial expansion used in the model.
        :no-index:
    """

    coefficients: torch.Tensor
    exponents: torch.Tensor
    np: int
    n: ArrayLike1DInt
    Nx: int
    expansion: PolyExpansion

    def __init__(
        self,
        network_builder: NetworkBuilder,
        expansion: PolyExpansion,
    ) -> None:
        """
        Initialize the PINN_PCE model.

        Parameters
        ----------
        network_builder : NetworkBuilder
            The network builder.
        expansion : PolyExpansion
            The polynomial expansion used in the model.
        """
        super().__init__(network_builder=network_builder)

        self.np = len(expansion)

        n = self.network_builder.layer_sizes

        assert n[-1] == self.np, (
            "Last layer must match the polynomial expansion dimension."
        )

        self.Nx = int(n[0])

        self.expansion = expansion

        self.register_buffer(
            "exponents",
            torch.from_numpy(expansion.exponents).long(),
        )
        self.register_buffer(
            "coefficients",
            torch.tensor(np.array(expansion.coefficients), dtype=torch.float64),
        )

    def forward(
        self,
        x: Tensor,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the output of the PCE-PINN for the given input and parameters.

        Parameters
        ----------
        x : Tensor
            Input to combine with parameters.
        p : Tensor
            Parameters to combine with input.

        Returns
        -------
        Tensor
            The output of the neural network.
        """
        # This operation can be executed on GPU.
        phi = torch_numpoly_call(
            self.exponents,
            self.coefficients,
            p.double(),
        )

        # Same as phi.repeat(n, 1) without copy data. Not gradient broadcasting safe.
        n = x.shape[0]
        phi = phi[None, ...].expand(n, -1, -1).reshape(n * phi.shape[0], phi.shape[1])

        z = self.combine_input(x, p)

        c = self.network(z[:, 0 : self.Nx])

        res = torch.sum(c * phi, axis=1, keepdim=True)  # type: ignore [call-overload]

        return z, res

    def save(self, file_path: str | Path) -> None:
        """
        Save the model's state and parameters to a file.

        Parameters
        ----------
        file_path : str | Path
            The path to the file where the model will be saved.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "network_builder": self.network_builder,
                "expansion": self.expansion,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        """
        Load a model from a file and return an instance.

        Parameters
        ----------
        file_path : str | Path
            The path to the file from which to load the model.

        Returns
        -------
        PINN_PCE
            The loaded model instance.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        model = cls(
            network_builder=checkpoint["network_builder"],
            expansion=checkpoint["expansion"],
        )
        model.load_state_dict(checkpoint["state_dict"])

        return model
