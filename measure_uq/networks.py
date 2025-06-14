"""Implementations of neural networks architecture builders."""

from typing import Any

import torch
from torch import Tensor, nn

from measure_uq.activations import Snake


class ResidualBlock(nn.Module):
    """
    Residual block implementation.

    A modular residual block for neural networks with LayerNorm and customizable
    activation.

    Parameters
    ----------
    width : int
        Number of features for the input and output of the block.
    activation : nn.Module
        Activation module (e.g., nn.Tanh(), custom Snake()).
    """

    def __init__(self, width: int, activation: nn.Module) -> None:
        """
        Initialize a residual block.

        Parameters
        ----------
        width : int
            Number of features for the input and output of the block.
        activation : nn.Module
            Activation module (e.g., nn.Tanh(), custom Snake()).
        """
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.activation = activation
        self.linear2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the weights of the residual block."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        # Optional: scale second layer to stabilize residual training
        with torch.no_grad():
            self.linear2.weight *= 0.1

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the residual block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        residual = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.norm2(out)
        return residual + out


class ActivationBuilder:
    """
    A builder for activation functions.

    Parameters
    ----------
    name : str
        Name of the activation function.
    **kwargs : dict
        Additional arguments to pass to the activation (e.g., alpha_init for Snake).
    """

    def __init__(self, name: str, **kwargs: Any):
        """
        Initialize an activation builder.

        Parameters
        ----------
        name : str
            Name of the activation function. Available options are:
            - "tanh"
            - "relu"
            - "snake"
        **kwargs : dict
            Additional arguments to pass to the activation (e.g., alpha_init for Snake).
        """
        self.name = name.lower()
        self.kwargs = kwargs

    def build(self, in_features: int) -> nn.Module:
        """
        Build an activation function.

        Parameters
        ----------
        in_features : int
            Number of input features.

        Returns
        -------
        nn.Module
            The activation function.
        """
        if self.name == "tanh":
            return nn.Tanh()
        if self.name == "relu":
            return nn.ReLU()
        if self.name == "snake":
            return Snake(in_features=in_features)
        raise ValueError(f"Unknown activation function: {self.name}")


class NetworkBuilder:
    """
    A builder for neural networks.

    Parameters
    ----------
    layer_sizes : list[int]
        List of number of features for each layer.
    activation : str
        Name of the activation function.
    **activation_kwargs : dict
        Additional arguments to pass to the activation (e.g., alpha_init for Snake).
    """

    activation_builder: ActivationBuilder
    layer_sizes: list[int]

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "tanh",
        **activation_kwargs: Any,
    ):
        """
        Initialize a network builder.

        Parameters
        ----------
        layer_sizes : list[int]
            List of number of features for each layer.
        activation : str
            Name of the activation function. Available options are:
            - "tanh"
            - "relu"
            - "snake"
        **activation_kwargs : dict
            Additional arguments to pass to the activation (e.g., alpha_init for Snake).
        """
        self.layer_sizes = layer_sizes
        self.activation_builder = ActivationBuilder(
            name=activation,
            **activation_kwargs,
        )

    def __call__(self) -> nn.Sequential:
        """
        Build a network.

        Returns
        -------
        nn.Sequential
            The network.
        """
        raise NotImplementedError("Must implement in subclass.")


class FeedforwardBuilder(NetworkBuilder):
    """A builder for feedforward networks."""

    def __call__(self) -> nn.Sequential:
        """
        Build a feedforward network.

        Returns
        -------
        nn.Sequential
            The network.
        """
        if len(self.layer_sizes) < 3:
            raise ValueError("Must have at least 3 layers.")

        layers: list[nn.Module] = []
        for i in range(len(self.layer_sizes) - 2):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layers.append(self.activation_builder.build(self.layer_sizes[i + 1]))

        layers.append(nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))

        return self._initialize(nn.Sequential(*layers))

    def _initialize(self, model: nn.Sequential) -> nn.Sequential:
        """
        Initialize the weights of the network.

        Parameters
        ----------
        model : nn.Sequential
            The network.

        Returns
        -------
        nn.Sequential
            The initialized network.
        """
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        return model


class ResidualBuilder(NetworkBuilder):
    """
    Builds a residual feedforward neural network with customizable activation.

    Parameters
    ----------
    layer_sizes : list[int]
        List of number of features for each layer.
    activation : str
        Name of the activation function ("tanh", "relu", "snake", etc.).
    **activation_kwargs :
        Additional arguments to pass to the activation (e.g., alpha_init for Snake).
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "tanh",
        **activation_kwargs: Any,
    ):
        """
        Initialize a residual builder.

        Parameters
        ----------
        layer_sizes : list[int]
            List of number of features for each layer.
        activation : str
            Name of the activation function. Available options are:
            - "tanh"
            - "relu"
            - "snake"
        **activation_kwargs : dict
            Additional arguments to pass to the activation (e.g., alpha_init for Snake).
        """
        super().__init__(layer_sizes, activation, **activation_kwargs)

    def __call__(self) -> nn.Sequential:
        """
        Construct a residual feedforward neural network.

        Returns
        -------
        nn.Sequential
            Residual network model.
        """
        if len(self.layer_sizes) < 3:
            raise ValueError("Must have at least 3 layers: input, hidden, output.")

        input_dim = self.layer_sizes[0]
        output_dim = self.layer_sizes[-1]
        hidden_layers = self.layer_sizes[1:-1]

        if any(width != hidden_layers[0] for width in hidden_layers):
            raise ValueError(
                "All hidden layers in a residual network must have the same width."
            )

        width = hidden_layers[0]
        depth = len(hidden_layers)

        layers: list[nn.Module] = []

        # Input layer
        linear_in = nn.Linear(input_dim, width)

        layers += [
            linear_in,
            self.activation_builder.build(width),
        ]

        # Residual blocks
        for _ in range(depth - 1):
            layers.append(ResidualBlock(width, self.activation_builder.build(width)))

        # Output layer
        linear_out = nn.Linear(width, output_dim)
        layers.append(linear_out)

        # Initialize and return the model
        model = nn.Sequential(*layers)
        return self._initialize(model)

    def _initialize(self, model: nn.Sequential) -> nn.Sequential:
        """
        Initialize the weights of the network.

        Parameters
        ----------
        model : nn.Sequential
            The network.

        Returns
        -------
        nn.Sequential
            The initialized network.
        """
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        return model
