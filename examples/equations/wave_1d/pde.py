r"""
Description of the wave equation on the line.

.. math::
    u_{tt} - a u_{xx} = 0, \quad (t,x) \in [0,1] \times [0,\pi]

    u(0, x) = \sin(k x)

    u_t(0, x) = 0

    u(t, 0) = 0

    u(t, \pi) = \sin(k \pi) \cos(\sqrt{a} k t)
"""

from dataclasses import dataclass, field

import chaospy
import numpy as np
import torch
from torch import Tensor, tensor

from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters
from measure_uq.utilities import cartesian_product_of_rows


def reference_solution(
    t: torch.Tensor | np.ndarray,
    x: torch.Tensor | np.ndarray,
    p: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """
    Compute the analytical solution of the wave equation.

    Parameters
    ----------
    t : torch.Tensor | np.ndarray
        Time coordinates.
    x : torch.Tensor | np.ndarray
        Spatial coordinates.
    p : torch.Tensor | np.ndarray
        Parameters of the PDE, where p[0] is a and p[1] is k.

    Returns
    -------
    torch.Tensor | np.ndarray
        The analytical solution of the wave equation at the given coordinates. The
        shape of the output is (Nt, Nx).
    """
    p = p.squeeze()

    tt, xx = np.meshgrid(t.squeeze(), x.squeeze(), indexing="ij")
    y = np.sin(p[1] * xx) * np.cos(np.sqrt(p[0]) * p[1] * tt)

    return y.T


def analytical_solution_2(
    tx: Tensor,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Compute the analytical solution of the wave equation.

    Parameters
    ----------
    tx : Tensor
        Time and spatial coordinates.
    p : Tensor
        Parameters of the PDE.

    Returns
    -------
    tuple[Tensor, Tensor]
        The analytical solution of the wave equation and the true solution.
    """
    tx_repeated = tx.repeat_interleave(p.shape[0], dim=0)
    p_tiled = p.repeat(tx.shape[0], 1)
    txp = torch.cat([tx_repeated, p_tiled], dim=1)

    y = torch.zeros(size=(txp.shape[0], 1))
    for (
        i,
        z,
    ) in enumerate(txp):
        y[i] = torch.sin(z[3] * z[1]) * torch.cos(torch.sqrt(z[2]) * z[3] * z[0])

    return txp, y


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual of the wave equation.

    Attributes
    ----------
    Nt : int
        Number of time points.
    Nx : int
        Number of spatial points.
    T : float
        Maximum time.
    X : float
        Maximum spatial coordinate.
    residual : Tensor
        Residual of the wave equation.
    """

    Nt: int
    Nx: int
    T: float
    X: float
    residual: Tensor = field(init=False, default_factory=lambda: torch.tensor([]))

    def __post_init__(self) -> None:
        """Initialize the residual of the wave equation."""
        assert self.Nt > 0
        assert self.Nx > 0
        super().__post_init__()

    def sample_points(self) -> None:
        """Sample points for the residual of the wave equation."""
        print("Re-sample PDE variables for Residual")

        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            tensor(np.random.uniform(0, self.X, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual of the wave equation.

        Parameters
        ----------
        x : Tensor
            Input coordinates (t, x) and parameters (a, k).
        y : Tensor
            Predicted output values.

        Returns
        -------
        Tensor
            Residual of the wave equation.
        """
        assert x.shape[0] == y.shape[0]

        dy_dt = jacobian(y, x, j=0)
        dy_tt = jacobian(dy_dt, x, j=0)
        dy_dx = jacobian(y, x, j=1)
        dy_xx = jacobian(dy_dx, x, j=1)

        a = x[:, 2][:, None]

        self.residual = dy_tt - a * dy_xx

        return self.residual


@dataclass(kw_only=True)
class InitialCondition(Condition):
    """
    Initial condition for the wave equation.

    Attributes
    ----------
    Nx : int
        Number of spatial points.
    X : float
        Maximum spatial coordinate.
    """

    Nx: int
    X: float

    def sample_points(self) -> None:
        """Sample points for the initial condition."""
        print("Re-sample PDE variables for InitialCondition")
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.0]]),
            tensor(np.random.uniform(0, self.X, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the initial condition.

        Parameters
        ----------
        x : Tensor
            Input coordinates (t, x) and parameters (a, k).
        y : Tensor
            Predicted output values.

        Returns
        -------
        Tensor
            Difference between predicted and true initial condition.
        """
        assert x.shape[0] == y.shape[0]
        kx = x[:, 3][:, None] * x[:, 1][:, None]
        return y - torch.sin(kx)


@dataclass(kw_only=True)
class InitialVelocity(Condition):
    """
    Initial velocity condition for the wave equation.

    Attributes
    ----------
    Nx : int
        Number of spatial points.
    X : float
        Maximum spatial coordinate.
    """

    Nx: int
    X: float

    def sample_points(self) -> None:
        """Sample points for the initial velocity condition."""
        print("Re-sample PDE variables for InitialVelocity")
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.0]]),
            tensor(np.random.uniform(0, self.X, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the initial velocity condition.

        Parameters
        ----------
        x : Tensor
            Input coordinates (t, x) and parameters (a, k).
        y : Tensor
            Predicted output values.

        Returns
        -------
        Tensor
            Difference between predicted and true initial velocity condition.
        """
        assert x.shape[0] == y.shape[0]
        return jacobian(y, x, j=0)


@dataclass(kw_only=True)
class BoundaryConditionLeft(Condition):
    """
    Boundary condition at the left boundary (x = 0) of the wave equation.

    Attributes
    ----------
    Nt : int
        Number of time points.
    T : float
        Maximum time.
    """

    Nt: int
    T: float

    def sample_points(self) -> None:
        """Sample points for the boundary condition at x = 0."""
        print("Re-sample PDE variables for BoundaryConditionLeft")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            torch.tensor([[0.0]]),
        ).float()

    def eval(self, _: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the boundary condition at the left boundary.

        Parameters
        ----------
        _ : Tensor
            Unused input coordinates (t, x) and parameters (a, k).
        y : Tensor
            Predicted output values.

        Returns
        -------
        Tensor
            Difference between predicted and true boundary condition.
        """
        return y


@dataclass(kw_only=True)
class BoundaryConditionRight(Condition):
    """
    Boundary condition at the right boundary (x = pi) of the wave equation.

    Attributes
    ----------
    Nt : int
        Number of time points.
    T : float
        Maximum time.
    X : float
        Maximum spatial coordinate.
    """

    Nt: int
    T: float
    X: float

    def sample_points(self) -> None:
        """Sample points for the boundary condition at x = pi."""
        print("Re-sample PDE variables for BoundaryConditionRight")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),  # t values
            torch.tensor([[self.X]]),  # x values
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the boundary condition at x = pi.

        Parameters
        ----------
        x : Tensor
            Input coordinates (t, x) and parameters (a, k).
        y : Tensor
            Predicted output values.

        Returns
        -------
        Tensor
            Difference between predicted and true boundary condition.
        """
        t = x[:, 0][:, None]
        a = x[:, 2][:, None]
        k = x[:, 3][:, None]

        sin_kpi = torch.sin(k * self.X)
        cos_term = torch.cos(torch.sqrt(a) * k * t)

        boundary_value = sin_kpi * cos_term

        return y - boundary_value


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    """
    Random parameters for the PDE.

    Attributes
    ----------
    joint : chaospy.J
        Joint distribution of the parameters.
    N : int
        Number of samples.
    """

    joint: chaospy.J
    N: int

    def __post_init__(self) -> None:
        """Initialize the parameters of the PDE."""
        super().__post_init__()

    def sample_values(self) -> None:
        """Re-sample the parameters of the PDE."""
        print("Re-sample PDE parameters")
        self.values = tensor(self.joint.sample(self.N).T).float()
