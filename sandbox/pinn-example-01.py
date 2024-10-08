#!/usr/bin/env python3
"""Script to test the PINN for solving the ODE dy/dt = -a * y."""

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class PINN(nn.Module):
    """
    Physics Informed Neural Network (PINN) for solving the ODE: dy/dt = -a * y

    This class implements a neural network with two inputs, time and parameter,
    and one output, the solution of the ODE.

    The neural network is composed of three fully connected layers with
    hyperbolic tangent activation functions.
    """

    def __init__(self) -> None:
        """
        Initialize the neural network.

        This function takes no arguments and returns nothing.
        """
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 20),  # Input now has two dimensions: time and parameter
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, t: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        Compute the output of the neural network for the given time and parameter.

        Parameters
        ----------
        t : torch.Tensor
            Time point(s) to evaluate the solution at.
        param : torch.Tensor
            Parameter(s) to evaluate the solution at.

        Returns
        -------
        torch.Tensor
            The solution of the ODE at the given time and parameter.
        """
        if param.dim() == 1:
            param = param.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        inputs = torch.cat((t, param), dim=1)
        return self.hidden(inputs)


def ode_function(
    y: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the right-hand side of the ODE: dy/dt = -a * y

    Parameters
    ----------
    y : torch.Tensor
        Current value of the solution
    a : torch.Tensor
        Parameter of the ODE

    Returns
    -------
    torch.Tensor
        Right-hand side of the ODE
    """
    return -a * y


def pinn_loss(
    model: PINN,
    t: torch.Tensor,
    param: torch.Tensor,
    y0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the loss for the Physics Informed Neural Network.

    Parameters
    ----------
    model : PINN
        The Physics Informed Neural Network
    t : torch.Tensor
        Time points where to evaluate the solution
    param : torch.Tensor
        Parameter of the ODE
    y0 : torch.Tensor
        Initial condition y(0)

    Returns
    -------
    loss : torch.Tensor
        The loss of the PINN
    """
    y_pred = model(t, param)

    y_pred_t = torch.autograd.grad(
        outputs=y_pred,
        inputs=t,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]

    # ODE residual
    f = y_pred_t - ode_function(y_pred, param)
    # Model evaluation at t=0
    ym0 = model(torch.tensor([[0.0]], dtype=torch.float32), param[0].unsqueeze(0))

    # Loss function
    return torch.mean(f**2) + (ym0 - y0) ** 2


# Training the PINN
def main():
    # Initial condition y(0) = 1
    """
    Train a Physics Informed Neural Network (PINN) to solve the ODE: dy/dt = -a * y
    with y(0) = 1 and a in [0.5, 1.0, 1.5].

    The PINN is trained using the Adam optimizer with a learning rate of 0.01 and
    a batch size of 32. The training loop runs for 5000 epochs.

    The model prediction is plotted on top of the exact ODE solution for a = 1.0.
    """
    y0 = torch.tensor([[1.0]], dtype=torch.float32)

    # Time domain for training
    t = torch.linspace(0, 2, 100).reshape(-1, 1)
    t = t.requires_grad_()

    # Parameters a for the ODE
    a_values = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
    t_repeated = t.repeat(a_values.shape[0], 1)
    a_repeated = a_values.repeat_interleave(t.shape[0]).reshape(-1, 1)

    # Create a dataset and data loader for batching
    dataset = TensorDataset(t_repeated, a_repeated)
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = PINN()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 5000
    for epoch in range(epochs):
        for batch in data_loader:
            t_batch, a_batch = batch
            optimizer.zero_grad()
            loss = pinn_loss(model, t_batch, a_batch, y0)
            loss.backward()
            optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Plotting the model prediction on top of the exact ODE solution
    t_eval = torch.linspace(0, 2, 100).reshape(-1, 1)
    a_eval = torch.tensor(1.0, dtype=torch.float32).expand(t_eval.shape[0], 1)
    y_pred = model(t_eval, a_eval).detach().numpy()

    # Exact solution for dy/dt = -a * y with y(0) = 1 and a = 1.0
    y_exact = torch.exp(-a_eval[0, 0] * t_eval).numpy()

    plt.plot(t_eval.numpy(), y_pred, label="PINN Prediction", linestyle="--")
    plt.plot(t_eval.numpy(), y_exact, label="Exact Solution", linestyle="-")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.title("PINN Prediction vs Exact Solution")
    plt.show()


if __name__ == "__main__":
    main()
