"""
Provides classes for creating and managing neural networks.

The module includes classes for managing the networks, such as the `PINN` and `PINN_PCE`
classes, as well as functions for creating the networks, such as the `feedforward`
function.

"""

# ruff: noqa: N801


import torch
from torch import Tensor, nn

from measure_uq.typing import ArrayLike1DInt, PolyExpansion


def feedforward(n: ArrayLike1DInt) -> nn.Sequential:
    """
    Creates a feedforward neural network with the given number of neurons in each layer.
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
    Abstract base class for neural networks with a combined input of x and p.

    The class provides a method `combine_input` to combine the input `x` and the
    parameters `p` into a single tensor. The method must be implemented by
    subclasses.

    Methods
    -------
    combine_input(x, p)
        Combine input and parameters into a single tensor.

    Notes
    -----
    The class is an abstract base class, meaning that it cannot be instantiated
    directly. Instead, it must be subclassed and the method `combine_input` must
    be implemented.
    """

    def __init__(
        self,
    ):
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
    hidden : torch.nn.Sequential
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
        Combine input and parameters into a single tensor. Assert that the dimensions
        are correct.

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

    def save(self, filepath):
        """Save the model's state and parameters to a file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n": self.n,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath):
        """Load a model from a file and return an instance."""
        checkpoint = torch.load(filepath)
        n = checkpoint["n"]
        model = cls(n=n)
        model.load_state_dict(checkpoint["state_dict"])
        return model


class PINN_PCE(ModelWithCombinedInput):
    """
    A Physics Informed Neural Network with a Polynomial Chaos Expansion (PINN_PCE)
    is a neural network that is constrained to satisfy a set of PDEs. The neural
    network is designed to satisfy the PDEs by minimizing a loss function that
    measures the difference between the neural network's output and the solution
    to the PDEs. The output of the neural network is a polynomial chaos expansion.

    Parameters
    ----------
    n : list of int
        List of number of neurons in each layer, including the output layer.
    expansion : PolyExpansion
        The polynomial chaos expansion to use.

    Attributes
    ----------
    n : list of int
        List of number of neurons in each layer, including the output layer.
    expansion : PolyExpansion
        The polynomial chaos expansion to use.
    net : torch.nn.Sequential
        The neural network.
    Nx : int
        The number of input neurons.
    np : int
        The number of output neurons.
    """

    def __init__(self, n: ArrayLike1DInt, expansion: PolyExpansion) -> None:
        """
        Initialize a Physics Informed Neural Network with a Polynomial Chaos Expansion
        (PINN_PCE).

        Parameters
        ----------
        n : ArrayLike1DInt
            List of number of neurons in each layer, including the output layer.
        expansion : PolyExpansion
            The polynomial chaos expansion to use.

        Returns
        -------
        None
        """
        super().__init__()

        self.np = len(expansion)

        assert n[-1] == self.np, "The last layer must have dimension 1."

        self.n = n

        self.Nx = n[0]

        self.expansion = expansion

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
        phi = torch.tensor(
            self.expansion(
                *p.T.detach().numpy(),
            ).T,
        )

        n = x.shape[0]

        # instead of phi.repeat(n, 1)
        phi = (
            phi.view(
                1,
                phi.shape[0],
                phi.shape[1],
            )
            .expand(
                n,
                phi.shape[0],
                phi.shape[1],
            )
            .reshape(
                n * phi.shape[0],
                phi.shape[1],
            )
        )

        z = self.combine_input(x, p)

        c = self.net(z[:, 0 : self.Nx])

        res = torch.sum(c * phi, axis=1, keepdim=True)  # type: ignore [call-overload]

        return z, res

    def save(self, filepath):
        """Save the model's state and parameters to a file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n": self.n,
                "expansion": self.expansion,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath):
        """Load a model from a file and return an instance."""
        checkpoint = torch.load(filepath)
        n = checkpoint["n"]
        expansion = checkpoint["expansion"]
        model = cls(n=n, expansion=expansion)
        model.load_state_dict(checkpoint["state_dict"])
        return model
