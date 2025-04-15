r"""Description of the heat equation on the line.

.. math::
    u_t - a / k^2 u_xx = 0
    u(0, x) = \sin(k x)
    u(t, 0) = 0
    u(t, \pi) = exp(-a t) \sin(\pi k)
"""

from dataclasses import dataclass

import chaospy
import numpy as np
import torch
from torch import Tensor, tensor

from measure_uq.callbacks import Callback
from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters
from measure_uq.utilities import cartesian_product_of_rows


def analytical_solution(
    t: np.ndarray | torch.Tensor,
    x: np.ndarray | torch.Tensor,
    p: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Analytical solution of the heat equation on the line."""
    if isinstance(t, torch.Tensor) and isinstance(x, torch.Tensor):
        return torch.exp(-p[0] * t) * torch.sin(p[1] * x)
    return np.exp(-p[0] * t) * np.sin(p[1] * x)


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual condition for the heat equation.

    This class represents the residual condition of the heat equation on the line.
    It is used to sample points and evaluate the residual of the PDE.

    Parameters
    ----------
    Nt : int
        Number of time points to sample.
    Nx : int
        Number of spatial points to sample.

    Methods
    -------
    sample_points()
        Sample points for the PDE.
    eval(x: Tensor, y: Tensor) -> Tensor
        Evaluate the residual of the PDE.

    """

    Nt: int
    Nx: int

    def __post_init__(self) -> None:
        """Initialize the Residual condition."""
        assert self.Nt > 0
        assert self.Nx > 0
        super().__post_init__()

    def sample_points(self) -> None:
        """Sample points for the PDE."""
        print("Re-sample PDE variables for Residual")

        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, 1, (self.Nt, 1))),
            tensor(np.random.uniform(0, np.pi, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the residual of the PDE."""
        assert x.shape[0] == y.shape[0]

        dy_dt = jacobian(y, x, j=0)
        dy_dx = jacobian(y, x, j=1)
        dy_d2 = jacobian(dy_dx, x, j=1)

        c = x[:, 2][:, None] / x[:, 3][:, None] ** 2
        f = c * dy_d2

        assert dy_dt.shape == f.shape

        return dy_dt - f


@dataclass(kw_only=True)
class InitialCondition(Condition):
    """
    Initial condition of the PDE.

    This class represents the initial condition of the PDE and provides methods
    to sample points and evaluate the initial condition.

    Parameters
    ----------
    Nx : int
        Number of spatial points to sample.

    Methods
    -------
    sample_points()
        Sample points for the initial condition of the PDE.
    eval(x: Tensor, y: Tensor) -> Tensor
        Evaluate the initial condition of the PDE.
    """

    Nx: int

    def sample_points(self) -> None:
        """Sample points for the initial condition of the PDE."""
        print("Re-sample PDE variables for InitialCondition")
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.0]]),
            tensor(np.random.uniform(0, np.pi, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the initial condition of the PDE."""
        assert x.shape[0] == y.shape[0]

        kx = x[:, 3][:, None] * x[:, 1][:, None]

        return y - torch.sin(kx)


@dataclass(kw_only=True)
class BoundaryConditionLeft(Condition):
    """
    Boundary condition at the left boundary of the PDE.

    This class represents the boundary condition at the left boundary of the PDE
    and provides methods to sample points and evaluate the boundary condition.

    Parameters
    ----------
    Nt : int
        Number of temporal points to sample.

    Methods
    -------
    sample_points()
        Sample points for the boundary condition at the left boundary of the PDE.
    eval(x: Tensor, y: Tensor) -> Tensor
        Evaluate the boundary condition at the left boundary of the PDE.
    """

    Nt: int

    def sample_points(self) -> None:
        """Sample points for the boundary condition at the left boundary of the PDE."""
        print("Re-sample PDE variables for BoundaryConsitionLeft")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, 1, (self.Nt, 1))),
            torch.tensor([[0.0]]),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: ARG002
        """Evaluate the boundary condition at the left boundary of the PDE."""
        return y


@dataclass(kw_only=True)
class BoundaryConditionRight(Condition):
    """
    Boundary condition at the right boundary of the PDE.

    This class represents the boundary condition at the right boundary of the PDE
    and provides methods to sample points and evaluate the boundary condition.

    Parameters
    ----------
    Nt : int
        Number of temporal points to sample.

    Methods
    -------
    sample_points()
        Sample points for the boundary condition at the right boundary of the PDE.
    eval(x: Tensor, y: Tensor) -> Tensor
        Evaluate the boundary condition at the right boundary of the PDE.
    """

    Nt: int

    def sample_points(self) -> None:
        """Sample points for the boundary condition at the right boundary of the PDE."""
        print("Re-sample PDE variables for BoundaryConsitionRight")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, 1, (self.Nt, 1))),
            torch.tensor([[np.pi]]),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the boundary condition at the right boundary of the PDE."""
        at = x[:, 2][:, None] * x[:, 0][:, None]
        kpi = np.pi * x[:, 3][:, None]

        return y - torch.exp(-at) * torch.sin(kpi)


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    """
    Random parameters for the PDE.

    This class represents the random parameters for the PDE and provides methods
    to sample values.

    Parameters
    ----------
    joint : chaospy.J
        The joint distribution of the random parameters.
    N : int
        Number of samples to generate.

    Methods
    -------
    sample_values()
        Sample values for the random parameters.
    """

    joint: chaospy.J
    N: int

    def sample_values(self) -> None:
        """Sample values for the random parameters."""
        print("Re-sample PDE parameters")
        self.values = tensor(self.joint.sample(self.N).T).float()


@dataclass
class CallbackLog(Callback):
    """A callback that prints the loss value at each iteration."""

    print_every: int = 10

    def on_iteration_end(self) -> None:
        """Print the loss value at each iteration."""
        if (
            self.trainer_data.iteration % self.print_every == 0
            or self.trainer_data.iteration == self.trainer_data.iterations - 1
        ):
            print(
                f"{self.trainer_data.losses_train.i[-1]:10}:  "
                f"{self.trainer_data.losses_train.v[-1]:.5e}",
            )
