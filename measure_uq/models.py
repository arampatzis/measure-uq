"""
Provide classes for creating and managing neural networks.

The module includes classes for managing the networks, such as the `PINN` and `PINN_PCE`
classes, as well as functions for creating the networks, such as the `feedforward`
function.
"""

# ruff: noqa: N801

from typing import Self

import numpy as np
import torch
from torch import Tensor, nn

from measure_uq.utilities import ArrayLike1DInt, PolyExpansion, torch_numpoly_call


def feedforward(n: ArrayLike1DInt) -> nn.Sequential:
    """
    Create a feedforward NN with the given number of neurons in each layer.

    The output of the last layer is the output of the network.

    Parameters
    ----------
    n : list of int
        List of number of neurons in each layer, including the output layer.

    Returns
    -------
    nn.Sequential
        The created feedforward neural network.
    """
    min_layers = 3

    assert len(n) >= min_layers

    layers = [
        nn.Linear(n[0], n[1]),  # type: ignore [arg-type]
        nn.Tanh(),
    ]
    for i in range(2, len(n) - 1):
        layers += [
            nn.Linear(n[i - 1], n[i]),  # type: ignore [arg-type]
            nn.Tanh(),
        ]
    layers += [nn.Linear(n[-2], n[-1])]  # type: ignore [arg-type]

    ff = nn.Sequential(*layers)

    for layer in ff:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    return ff


class ModelWithCombinedInput(nn.Module):
    """
    A model that combines input and parameters.

    This class inherits from `nn.Module` and provides functionality to combine
    input tensors with parameter tensors. It serves as a base class for more
    complex models that require combined input and parameters.

    Attributes
    ----------
    None
    """

    def __init__(self) -> None:
        super().__init__()

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


class PINN(ModelWithCombinedInput):
    """
    Define a PINN.

    A Physics Informed Neural Network (PINN) is a neural network that is
    constrained to satisfy a set of PDEs. The neural network is designed to
    satisfy the PDEs by minimizing a loss function that measures the difference
    between the neural network's output and the solution to the PDEs.

    Parameters
    ----------
    n : list of int
        List of number of neurons in each layer, including the output layer.

    Attributes
    ----------
    network : torch.nn.Sequential
        The neural network.
    n : list of int
        List of number of neurons in each layer, including the output layer.
    """

    def __init__(self, n: ArrayLike1DInt) -> None:
        """
        Initialize a Physics Informed Neural Network (PINN) with n hidden layers.

        Parameters
        ----------
        n : ArrayLike1DInt
            List of number of neurons in each layer, including the output layer.

        Returns
        -------
        None
        """
        super().__init__()

        self.network = feedforward(n)

        self.n = n

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
        assert x.shape[1] + p.shape[1] == self.network[0].in_features

        z = super().combine_input(x, p)

        assert z.shape[1] == self.network[0].in_features

        return z

    def save(self, file_path: str) -> None:
        """Save the model's state and parameters to a file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n": self.n,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str) -> Self:
        """Load a model from a file and return an instance."""
        checkpoint = torch.load(file_path, weights_only=False)
        n = checkpoint["n"]
        model = cls(n=n)
        model.load_state_dict(checkpoint["state_dict"])
        return model


class PINN_PCE(ModelWithCombinedInput):
    """Define a PINN with a PCE.

    This class implements a neural network model that combines the principles of
    Physics Informed Neural Networks (PINNs) with Polynomial Chaos Expansion (PCE).
    The model is designed to handle inputs and parameters, and it computes the
    output based on the given polynomial expansion.

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
    net : torch.nn.Module
        The feedforward neural network.
        :no-index:
    """

    coefficients: torch.Tensor
    exponents: torch.Tensor
    np: int
    n: ArrayLike1DInt
    Nx: int
    expansion: PolyExpansion
    net: torch.nn.Module

    def __init__(
        self,
        n: ArrayLike1DInt,
        expansion: PolyExpansion,
    ) -> None:
        super().__init__()
        """Initialize the PINN_PCE model.

        Parameters
        ----------
        n : ArrayLike1DInt
            The architecture of the neural network.
        expansion : PolyExpansion
            The polynomial expansion used in the model.
        """
        self.np = len(expansion)

        assert (
            n[-1] == self.np
        ), "Last layer must match the polynomial expansion dimension."

        self.n = n

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

        self.net = feedforward(n)

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

        c = self.net(z[:, 0 : self.Nx])

        res = torch.sum(c * phi, axis=1, keepdim=True)  # type: ignore [call-overload]

        return z, res

    def save(self, file_path: str) -> None:
        """Save the model's state and parameters to a file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n": self.n,
                "expansion": self.expansion,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str) -> Self:
        """Load a model from a file and return an instance."""
        checkpoint = torch.load(file_path, weights_only=False)
        model = cls(
            n=checkpoint["n"],
            expansion=checkpoint["expansion"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
