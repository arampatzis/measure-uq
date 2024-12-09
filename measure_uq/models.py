"""
Provides a simple implementation of a Physics Informed Neural Network
(PINN).

This module provides a class, `PINN`, that can be used to create a PINN and train
it to satisfy a set of PDEs.
"""
import torch
from torch import Tensor, nn


class PINN(nn.Module):
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

    def __init__(self, n) -> None:
        """
        Initialize a Physics Informed Neural Network (PINN) with n hidden layers.

        Parameters
        ----------
        n : list of int
            List of number of neurons in each layer, including the output layer.

        Returns
        -------
        None
        """
        super().__init__()

        min_layers = 2

        assert len(n) > min_layers

        layers = [
            nn.Linear(n[0], n[1]),
            nn.Tanh(),
        ]
        for i in range(2, len(n) - 1):
            layers += [
                nn.Linear(n[i - 1], n[i]),
                nn.Tanh(),
            ]
        layers += [nn.Linear(n[-2], n[-1])]

        self.hidden = nn.Sequential(*layers)

        for layer in self.hidden:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.n = n

    def forward(self, z: Tensor) -> Tensor:
        """
        Compute the output of the neural network for the given input.

        Parameters
        ----------
        z : torch.Tensor
            Input to evaluate the neural network at.

        Returns
        -------
        torch.Tensor
            The output of the neural network.
        """
        return self.hidden(z)

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
        assert x.shape[1] + p.shape[1] == self.hidden[0].in_features

        x_repeated = x.repeat_interleave(p.shape[0], dim=0)

        p_tiled = p.repeat(x.shape[0], 1)

        z = torch.cat([x_repeated, p_tiled], dim=1)

        assert z.shape[1] == self.hidden[0].in_features

        return z
