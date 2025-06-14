r"""
Solve a nonlinear 1D reaction-diffusion equation with random coefficients.

.. math::
    \frac{\partial u}{\partial t}
    - D \frac{\partial^2 u}{\partial x^2}
    + g(x) u^3 = f(x),
    \quad t \in [0, T],\ x \in [-1, 1]

with:

**Initial condition:**

.. math::
    u(0, x) = 0.5 \cos^2(\pi x)

**Boundary conditions:**

.. math::
    u(t, -1) = u(t, 1) = 0.5

**Reaction coefficient:**

.. math::
    g(x) = 0.2 + e^{r_1 x} \cos^2(r_2 x),\quad
    r_1 \sim \mathcal{U}(0.5, 1),\quad
    r_2 \sim \mathcal{U}(3, 4)

**Forcing term:**

.. math::
    f(x) = \exp\left( -\frac{(x - 0.25)^2}{2 k_1^2} \right) \sin^2(k_2 x),\quad
    k_1 \sim \mathcal{U}(0.2, 0.8),\quad
    k_2 \sim \mathcal{U}(1, 4)
"""

from dataclasses import dataclass, field

import chaospy
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import Tensor, tensor

from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters
from measure_uq.utilities import cartesian_product_of_rows


def reference_solution(
    t: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    D: float,
) -> np.ndarray:
    """
    Analytical solution of the reaction-diffusion equation.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    x : np.ndarray
        Spatial points.
    p : np.ndarray
        Parameters of the reaction-diffusion equation.
    D : float
        Diffusion coefficient.

    Returns
    -------
    np.ndarray
        Numerical solution of the reaction-diffusion equation of shape (Nx, Nt).
    """
    dx = x[1] - x[0]
    Nx = len(x)
    t_eval = t

    r1, r2, k1, k2 = p

    g = 0.2 + np.exp(r1 * x) * np.cos(r2 * x) ** 2
    f = np.exp(-((x - 0.25) ** 2) / (2 * k1**2)) * np.sin(k2 * x) ** 2

    u0 = 0.5 * np.cos(np.pi * x) ** 2

    def apply_bc(u: np.ndarray) -> np.ndarray:
        """
        Apply boundary conditions.

        Parameters
        ----------
        u : np.ndarray
            Solution tensor.

        Returns
        -------
        Tensor
            Solution tensor with boundary conditions applied.
        """
        u[0] = 0.5
        u[-1] = 0.5
        return u

    def laplacian_matrix(N: int, dx: float) -> np.ndarray:
        """
        Laplacian matrix.

        Parameters
        ----------
        N : int
            Number of spatial points.
        dx : float
            Spatial step size.

        Returns
        -------
        np.ndarray
            Laplacian matrix.
        """
        L = np.zeros((N, N))
        for i in range(1, N - 1):
            L[i, i - 1] = 1
            L[i, i] = -2
            L[i, i + 1] = 1
        return L / dx**2

    L_mat = laplacian_matrix(Nx, dx)

    def rhs(_: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the reaction-diffusion equation.

        Parameters
        ----------
        _ : np.ndarray
            Input coordinates tensor.
        u : np.ndarray
            Solution tensor.

        Returns
        -------
        np.ndarray
            Right-hand side of the reaction-diffusion equation.
        """
        u = apply_bc(u.copy())
        du = D * (L_mat @ u) - g * u**3 + f
        du[0] = 0.0
        du[-1] = 0.0
        return du

    sol = solve_ivp(rhs, [t[0], t[-1]], u0, t_eval=t_eval, method="RK45")

    return sol.y


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
    D: float = 0.01

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
            tensor(np.random.uniform(-1, 1, (self.Nx, 1))),
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

        xx = x[:, 1][:, None]
        r1 = x[:, 2][:, None]
        r2 = x[:, 3][:, None]
        k1 = x[:, 4][:, None]
        k2 = x[:, 5][:, None]

        g = 0.2 + torch.exp(r1 * xx) * torch.cos(r2 * xx) ** 2
        f = torch.exp(-((xx - 0.25) ** 2) / (2 * k1**2)) * torch.sin(k2 * xx) ** 2

        rhs = self.D * dy_d2 - g * y**3 + f

        assert dy_dt.shape == rhs.shape

        return dy_dt - rhs


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

    def sample_points(self) -> None:
        """Sample points for the initial condition of the PDE."""
        print("Re-sample PDE variables for InitialCondition")
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.0]]),
            tensor(np.random.uniform(-1, 1, (self.Nx, 1))),
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

        xx = x[:, 1][:, None]

        return y - 0.5 * torch.cos(np.pi * xx) ** 2


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
            torch.tensor([[-1.0]]),
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
        return y - 0.5


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

    def sample_points(self) -> None:
        """Sample points for the boundary condition at the right boundary of the PDE."""
        print("Re-sample PDE variables for BoundaryConsitionRight")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            torch.tensor([[1.0]]),
        ).float()

    def eval(self, _: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the boundary condition at the right boundary of the PDE.

        Parameters
        ----------
        _ : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor.

        Returns
        -------
        Tensor
            The difference between the predicted and true boundary condition.
        """
        return y - 0.5


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
