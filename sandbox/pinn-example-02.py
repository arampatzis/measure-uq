#!/usr/bin/env python3
"""Script for the solution of the heat equation using PINNs."""

import torch
import wandb
from box import Box
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class PINN(nn.Module):
    """Physics Informed Neural Network (PINN) architecture."""

    def __init__(self):
        """
        Initialize the neural network.

        This function takes no arguments and returns nothing.

        The neural network is composed of three fully connected layers with
        hyperbolic tangent activation functions.
        """
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 20),  # Input now has two dimensions: space and time
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the output of the neural network for the given space and time points.

        Parameters
        ----------
        x : torch.Tensor
            Space point(s) to evaluate the solution at.
        t : torch.Tensor
            Time point(s) to evaluate the solution at.

        Returns
        -------
        torch.Tensor
            The solution of the heat equation at the given space and time points.
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        inputs = torch.cat((x, t), dim=1)
        return self.hidden(inputs)


class HeatEquation:
    """Heat equation class"""

    def __init__(
        self,
        t0: float,
        t1: float,
        x0: float,
        x1: float,
        alpha: float,
    ) -> None:
        """
        Initialize the heat equation object.

        Parameters
        ----------
        t0 : float
            Initial time.
        t1 : float
            Final time.
        x0 : float
            Initial space coordinate.
        x1 : float
            Final space coordinate.
        alpha : float
            Thermal diffusivity of the material.

        Returns
        -------
        None
        """
        self.alpha = alpha

        self.t0, self.t1 = (t0, t1)
        self.x0, self.x1 = (x0, x1)

    def u0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the initial condition of the heat equation.

        Parameters
        ----------
        x : torch.Tensor
            Space coordinates where to evaluate the initial condition.

        Returns
        -------
        torch.Tensor
            The initial condition of the heat equation at the given space coordinates.
        """
        return torch.sin(torch.pi * x)

    def u_exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Exact solution of the heat equation.

        Parameters
        ----------
        x : torch.Tensor
            Space coordinates where to evaluate the exact solution
            (shape: (n_samples,)).
        t : torch.Tensor
            Time points where to evaluate the exact solution (shape: (n_samples,)).

        Returns
        -------
        torch.Tensor
            The exact solution of the heat equation at the given space and time points
            (shape: (n_samples,)).
        """
        return torch.sin(torch.pi * x) * torch.exp(-self.alpha * torch.pi**2 * t)

    def residual(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the residual of the heat equation.

        Parameters
        ----------
        u : torch.Tensor
            The solution of the heat equation to compute the residual for.
        x : torch.Tensor
            Space coordinates where to compute the residual.
        t : torch.Tensor
            Time points where to compute the residual.

        Returns
        -------
        torch.Tensor
            The residual of the heat equation at the given space and time points.
        """
        u_t = torch.autograd.grad(
            u,
            t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        u_x = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
        )[0]
        return u_t - self.alpha * u_xx


class Trainer:
    """Class for training a PINN."""

    def __init__(self, config: dict, pde: HeatEquation):
        """
        Initialize the Trainer object.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing hyperparameters.
        pde : HeatEquation
            Heat equation object.
        """
        self.cfg = Box(config)
        self.pde = pde

        self.epochs = self.cfg.epochs
        self.batch_size = self.cfg.batch_size

        self.model = PINN()
        wandb.watch(self.model)

        x = torch.linspace(pde.t0, pde.t1, 100, requires_grad=True).reshape(-1, 1)
        t = torch.linspace(pde.x0, pde.x1, 100, requires_grad=True).reshape(-1, 1)

        self.u0 = pde.u0(x)

        self.x0 = x
        self.t0 = torch.zeros_like(x) + t[0]

        self.create_dataloaders(x, t)

        self.epoch = 0
        self.create_optimizer_scheduler()

        self.train_losses: list[float] = []
        self.validation_losses: list[float] = []

    def create_dataloaders(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Create train and validation data loaders from x and t coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Space coordinates (shape: (n_samples,)).
        t : torch.Tensor
            Time coordinates (shape: (n_samples,)).

        Returns
        -------
        tuple[DataLoader, DataLoader]
            Tuple containing the training and validation data loaders.
        """
        x_repeated = x.repeat(t.shape[0], 1)
        t_repeated = t.repeat(x.shape[0], 1)
        dataset = TensorDataset(x_repeated, t_repeated)
        train_size = int(0.8 * len(dataset))
        validation_size = len(dataset) - train_size
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def create_optimizer_scheduler(self):
        """Create the optimizer and scheduler."""
        match self.cfg.optimizer.type.lower():
            case "adam":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.cfg.optimizer.lr,
                )
            case _:
                print("Unknown optimizer type, using Adam.")
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.cfg.optimizer.lr,
                )

        cfg_scheduler = self.cfg.optimizer.scheduler
        match cfg_scheduler.type:
            case "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=cfg_scheduler.step_size,
                    gamma=cfg_scheduler.gamma,
                )
            case "multi":
                self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                    self.optimizer,
                    lr_lambda=lambda epoch: cfg_scheduler.gamma**epoch,
                )
            case _:
                print(
                    "Unknown scheduler type, using constant learning rate "
                    f"{self.cfg.optimizer.lr}.",
                )
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=10,
                    gamma=1,
                )

    def loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss of the model.

        The loss is a combination of the mean squared residual of the heat equation
        and the mean squared error of the initial condition.

        Parameters
        ----------
        x : torch.Tensor
            Space coordinates where to evaluate the loss.
        t : torch.Tensor
            Time points where to evaluate the loss.

        Returns
        -------
        torch.Tensor
            The loss of the model.
        """
        u_pred = self.model(x, t)
        u0_pred = self.model(self.x0, self.t0)
        f = self.pde.residual(u_pred, x, t)
        return torch.mean(f**2) + torch.mean((u0_pred - self.u0) ** 2)

    def train_step(self):
        """
        Perform one training step on the model.

        This method first sets the model in training mode, then iterates over the
        training data loader. For each batch, it computes the loss, backpropagates
        the gradients, and updates the parameters using the optimizer. The total
        loss is accumulated over the batches and saved in the `train_losses` list.
        Finally, the scheduler is stepped to update the learning rate, and the
        epoch number is incremented.
        """
        self.model.train()
        epoch_train_loss = 0.0
        for batch in self.train_loader:
            x_batch, t_batch = batch
            loss = self.loss(x_batch, t_batch)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(self.train_loader)
        self.train_losses.append(epoch_train_loss)
        self.scheduler.step()
        self.epoch += 1

    def validation_step(self):
        """
        Perform one validation step on the model.

        This method first sets the model in evaluation mode, then iterates over the
        validation data loader. For each batch, it computes the loss, and
        accumulates the total loss over the batches. The result is saved in the
        `validation_losses` list.
        """
        self.model.eval()
        epoch_validation_loss = 0.0
        for batch in self.validation_loader:
            x_batch, t_batch = batch
            loss = self.loss(x_batch, t_batch)
            epoch_validation_loss += loss.item()
        epoch_validation_loss /= len(self.validation_loader)
        self.validation_losses.append(epoch_validation_loss)


class EarlyStopper:
    """Early stopper for PyTorch training process."""

    def __init__(self, patience: int = 1, min_delta: float = 0) -> None:
        """
        Initialize the EarlyStopper object.

        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait before stopping training when the validation
            loss does not decrease. Default is 1.
        min_delta : float, optional
            Minimum difference between the current and previous validation loss to
            consider the loss as decreased. Default is 0.

        Attributes
        ----------
        patience : int
            Number of epochs to wait before stopping training.
        min_delta : float
            Minimum difference between the current and previous validation loss.
        counter : int
            Number of epochs since the last decrease of the validation loss.
        min_validation_loss : float
            Minimum validation loss encountered during training.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def __call__(self, validation_loss: float) -> bool:
        """
        Check if the validation loss has decreased and update the counter.

        If the validation loss has decreased, the counter is reset to 0.
        If the validation loss has not decreased for a number of epochs equal to
        or greater than the patience, the method returns True, indicating that
        the training should be stopped.

        Parameters
        ----------
        validation_loss : float
            Validation loss of the current epoch.

        Returns
        -------
        bool
            Whether the training should be stopped due to no improvement in the
            validation loss.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    """
    Train a Physics Informed Neural Network (PINN) to solve the heat equation
    on a 1D domain.

    The PINN is trained using the Adam optimizer with a learning rate of 0.01
    and a batch size of 16. The training loop runs for 1 epoch.

    The model prediction is plotted on top of the exact solution for t=0.5.

    This script is meant to be run with the Weights & Biases library installed,
    and will log the training and validation loss, as well as the learning rate,
    to the Weights & Biases dashboard.

    The script also logs a plot of the predicted solution vs. the exact solution
    at t=0.5.
    """
    config = {
        "project_name": "pinn-for-heat-equation",
        "batch_size": 16,
        "epochs": 100,
        "optimizer": {
            "type": "adam",
            "lr": 0.01,
            "scheduler": {
                "type": "step",
                "step_size": 10,
                "gamma": 0.5,
            },
        },
    }

    wandb.init(
        project=config["project_name"],
        config=config,
    )

    pde = HeatEquation(0, 1, 0, 1, 0.01)

    trainer = Trainer(config, pde)
    early_stopper = EarlyStopper(patience=10)

    for epoch in range(trainer.epochs):
        trainer.epoch = epoch
        trainer.train_step()
        trainer.validation_step()

        if epoch % 1 == 0:
            print(
                f"Epoch {epoch} -- "
                f"Train Loss: {trainer.train_losses[-1]} "
                f"Validation Loss: {trainer.validation_losses[-1]}",
            )

        wandb.log(
            {
                "train_loss": trainer.train_losses[-1],
                "validation_loss": trainer.validation_losses[-1],
                "lrs": trainer.scheduler.get_last_lr()[0],
            },
        )

        if early_stopper(trainer.validation_losses[-1]):
            print(
                f"Validation error has not improved for {early_stopper.patience}"
                "epochs, stopping training.",
            )
            break

    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    t = torch.tensor([[0.5]], dtype=torch.float32).expand(x.shape[0], 1)
    u_pred = trainer.model(x, t)
    u_exact = pde.u_exact(x, t)

    ys = torch.cat((u_pred, u_exact), dim=1).T.tolist()
    xs = x.squeeze().tolist()

    wandb.log(
        {
            "computed_vs_exact": wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=["predicted", "exact"],
                title="Predicted vs. Exact at t=0.5",
                xname="x",
            ),
        },
    )

    wandb.finish()


if __name__ == "__main__":
    main()
