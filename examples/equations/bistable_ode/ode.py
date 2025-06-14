r"""
Bistable ODE Simulation.

This script simulates the bistable ODE:

.. math::

    \frac{dy}{dt} = r (y - 1) (2 - y) (y - 3), \quad y(0) = y_0

where :math:`r \sim \mathcal{U}(0.8, 1.2)` and :math:`y_0 \sim \mathcal{U}(0, 4)`.
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
    p: np.ndarray,
) -> np.ndarray:
    """
    Compute the analytical solution of the wave equation.

    Parameters
    ----------
    t : np.ndarray
        Time coordinates.
    p : np.ndarray
        Parameters of the PDE, where p[0] is a and p[1] is k.

    Returns
    -------
    np.ndarray
        The analytical solution of the wave equation at the given coordinates. The
        shape of the output is (Nt, Nx).
    """

    def bistable_ode(_: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
        """
        Right hand side of the bistable ODE.

        Parameters
        ----------
        _ : np.ndarray
            Input coordinates tensor.
        y : np.ndarray
            Solution tensor.
        r : float
            Reaction coefficient.

        Returns
        -------
        np.ndarray
            Right-hand side of the bistable ODE.
        """
        return r * (y - 1) * (2 - y) * (y - 3)

    p = p.squeeze()

    t_span = (0, t[-1])
    y0 = p[0]
    r = p[1]

    sol = solve_ivp(bistable_ode, t_span, [y0], t_eval=t, args=(r,), method="Radau")
    return sol.y[0]


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual of the ODE.

    Attributes
    ----------
    Nt : int
        Number of time points.
    T : float
        Maximum time.
    residual : Tensor
        Residual of the wave equation.
    """

    Nt: int
    T: float
    residual: Tensor = field(init=False, default_factory=lambda: torch.tensor([]))

    def __post_init__(self) -> None:
        """Initialize the residual of the wave equation."""
        assert self.Nt > 0
        super().__post_init__()

    def sample_points(self) -> None:
        """Sample points for the residual of the wave equation."""
        print("Re-sample PDE variables for Residual")

        self.points = tensor(np.random.uniform(0, self.T, (self.Nt, 1))).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual of the ODE.

        Parameters
        ----------
        x : Tensor
            Input coordinates (t, x) and parameters (a, k).
        y : Tensor
            Predicted output values.

        Returns
        -------
        Tensor
            Residual of the ODE.
        """
        assert x.shape[0] == y.shape[0]

        dy_dt = jacobian(y, x, j=0)

        r = x[:, 2][:, None]

        self.residual = dy_dt - r * (y - 1) * (2 - y) * (y - 3)

        return self.residual


@dataclass(kw_only=True)
class InitialCondition(Condition):
    """Initial condition for the wave equation."""

    def sample_points(self) -> None:
        """Sample points for the initial condition."""
        print("Re-sample PDE variables for InitialCondition")
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.0]]),
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
        y0 = x[:, 1][:, None]
        return y - y0


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
