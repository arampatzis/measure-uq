r"""
Description of the heat equation on the line.

.. math::
    u_t - a / k^2 u_xx = 0
    u(0, x) = \sin(k x)
    u(t, 0) = 0
    u(t, \pi) = exp(-a t) \sin(\pi k)
"""

from dataclasses import dataclass, field

import chaospy
import numpy as np
import torch
from torch import Tensor, tensor

from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters
from measure_uq.utilities import cartesian_product_of_rows


def analytical_solution(
    t: torch.Tensor | np.ndarray,
    x: torch.Tensor | np.ndarray,
    p: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """
    Compute the analytical solution of the heat equation.

    Parameters
    ----------
    t : torch.Tensor | np.ndarray
        Time coordinates.
    x : torch.Tensor | np.ndarray
        Spatial coordinates.
    p : torch.Tensor | np.ndarray
        Parameters of the PDE, where p[0] is the thermal diffusivity.

    Returns
    -------
    torch.Tensor | np.ndarray
        The analytical solution of the heat equation at the given coordinates.
    """
    if isinstance(t, torch.Tensor) and isinstance(x, torch.Tensor):
        tt, xx = torch.meshgrid(t.squeeze(), x.squeeze(), indexing="ij")
        return torch.exp(-p[0] * tt) * torch.sin(p[1] * xx)

    ttt, xxx = np.meshgrid(t.squeeze(), x.squeeze(), indexing="ij")
    return np.exp(-p[0] * ttt) * np.sin(p[1] * xxx)


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual condition for the heat equation.

    This class represents the residual condition of the heat equation on the line.
    It is used to sample points and evaluate the residual of the PDE.

    Attributes
    ----------
    Nt : int
        Number of time points to sample.
    Nx : int
        Number of spatial points to sample.
    T : float
        Maximum time.
    X : float
        Maximum spatial coordinate.
    residual : Tensor
        Residual of the heat equation.
    """

    Nt: int
    Nx: int
    T: float
    X: float

    residual: Tensor = field(init=False, default_factory=lambda: torch.tensor([]))

    def __post_init__(self) -> None:
        """Initialize the Residual condition."""
        assert self.Nt > 0
        assert self.Nx > 0
        super().__post_init__()

    def sample_points(self) -> None:
        """Sample points for the PDE."""
        print("Re-sample PDE variables for Residual")

        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            tensor(np.random.uniform(0, self.X, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual of the PDE.

        Parameters
        ----------
        x : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor.

        Returns
        -------
        Tensor
            The residual of the PDE at the given points.
        """
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

    Attributes
    ----------
    Nx : int
        Number of spatial points to sample.
    """

    Nx: int
    X: float

    def sample_points(self) -> None:
        """Sample points for the initial condition of the PDE."""
        print("Re-sample PDE variables for InitialCondition")
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.0]]),
            tensor(np.random.uniform(0, self.X, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the initial condition of the PDE.

        Parameters
        ----------
        x : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor.

        Returns
        -------
        Tensor
            The difference between the predicted and true initial condition.
        """
        assert x.shape[0] == y.shape[0]

        kx = x[:, 3][:, None] * x[:, 1][:, None]

        return y - torch.sin(kx)


@dataclass(kw_only=True)
class BoundaryConditionLeft(Condition):
    """
    Boundary condition at the left boundary of the PDE.

    This class represents the boundary condition at the left boundary of the PDE
    and provides methods to sample points and evaluate the boundary condition.

    Attributes
    ----------
    Nt : int
        Number of temporal points to sample.
    """

    Nt: int
    T: float

    def sample_points(self) -> None:
        """Sample points for the boundary condition at the left boundary of the PDE."""
        print("Re-sample PDE variables for BoundaryConsitionLeft")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            torch.tensor([[0.0]]),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: ARG002
        """
        Evaluate the boundary condition at the left boundary of the PDE.

        Parameters
        ----------
        x : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor.

        Returns
        -------
        Tensor
            The difference between the predicted and true boundary condition.
        """
        return y


@dataclass(kw_only=True)
class BoundaryConditionRight(Condition):
    """
    Boundary condition at the right boundary of the PDE.

    This class represents the boundary condition at the right boundary of the PDE
    and provides methods to sample points and evaluate the boundary condition.

    Attributes
    ----------
    Nt : int
        Number of temporal points to sample.
    """

    Nt: int
    T: float
    X: float

    def sample_points(self) -> None:
        """Sample points for the boundary condition at the right boundary of the PDE."""
        print("Re-sample PDE variables for BoundaryConsitionRight")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            torch.tensor([[self.X]]),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the boundary condition at the right boundary of the PDE.

        Parameters
        ----------
        x : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor.

        Returns
        -------
        Tensor
            The difference between the predicted and true boundary condition.
        """
        at = x[:, 2][:, None] * x[:, 0][:, None]
        kpi = self.X * x[:, 3][:, None]

        return y - torch.exp(-at) * torch.sin(kpi)


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    """
    Random parameters for the PDE.

    This class represents the random parameters for the PDE and provides methods
    to sample values.

    Attributes
    ----------
    joint : chaospy.J
        The joint distribution of the random parameters.
    N : int
        Number of samples to generate.
    """

    joint: chaospy.J
    N: int

    def sample_values(self) -> None:
        """Sample values for the random parameters."""
        print("Re-sample PDE parameters")
        self.values = tensor(self.joint.sample(self.N).T).float()
