#!/usr/bin/env python3
"""
Demonstrates the computation and visualization of gradients
using a simple neural network. The network consists of two fully connected
layers with sine and cosine activations. The script computes gradients
using both a custom jacobian function and manual computation, comparing
the results through plots.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn

from measure_uq.gradients import jacobian

torch.set_default_dtype(torch.float64)


class SimpleNN(nn.Module):
    """
    A simple neural network with two layers and non-linear activations.

    Attributes
    ----------
        fc (list): List of fully connected layers.
        activation (list): List of activation functions applied after each layer.
    """

    def __init__(self, nx, ny):
        super().__init__()
        self.fc = [
            nn.Linear(nx, 5, bias=False),
            nn.Linear(5, ny, bias=False),
        ]

        self.activation = [
            torch.sin,
            torch.cos,
        ]

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        ----
            x (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Output tensor after passing through layers and activations.
        """
        x = self.activation[0](self.fc[0](x))
        return self.activation[1](self.fc[1](x))


def compare_gradients():
    """Main function to execute the computation and visualization of gradients."""
    nx = 3
    ny = 2

    model = SimpleNN(nx, ny)
    x = (
        torch.linspace(
            0,
            20,
            1000,
        )[:, None]
        .repeat(1, nx)
        .requires_grad_(True)
    )

    y = model(x)

    z = torch.linspace(
        100,
        200,
        1000,
    )[:, None]

    xx = torch.cat((x, z), dim=1)

    computed_gradients = torch.stack(
        [jacobian(y, xx, i=i) for i in range(ny)],
        dim=1,
    )

    z = [model.fc[0](x)]
    z[0].retain_grad()
    model.activation[0](z[0]).sum().backward(retain_graph=True)
    d_z = [z[0].grad]

    for i in range(1, len(model.fc)):
        z += [model.fc[i](model.activation[i - 1](z[i - 1]))]
        z[i].retain_grad()
        model.activation[i](z[i]).sum().backward(retain_graph=True)
        d_z += [z[i].grad]

    exact_gradients = torch.einsum(
        "bi,ij,bj,jk->bik",
        d_z[1],
        model.fc[1].weight,
        d_z[0],
        model.fc[0].weight,
    )

    # Plot the results
    fig, axes = plt.subplots(
        ny,
        1,
        figsize=(14, 10),
    )
    for i in range(ny):
        for j in range(nx):
            (line_computed,) = axes[i].plot(
                x[:, j].detach().numpy(),
                computed_gradients[:, i, j].detach().numpy(),
                label=f"Computed Grad x[{j}]",
            )
            axes[i].plot(
                x[:, j].detach().numpy(),
                exact_gradients[:, i, j].detach().numpy(),
                ".-",
                color=line_computed.get_color(),
                label=f"Exact Grad x[{j}]",
            )
        axes[i].set_title(
            f"Gradients for Output y[:, {i}]",
        )
        axes[i].legend()
        axes[i].set_xlabel(
            "Input",
        )
        axes[i].set_ylabel(
            "Gradient Value",
        )

    plt.tight_layout()
    plt.show()


def main():
    """Entry point of the script."""
    compare_gradients()


if __name__ == "__main__":
    main()
