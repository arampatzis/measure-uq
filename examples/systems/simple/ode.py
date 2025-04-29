r"""
Solution of a decdecoupled system of ODEs.

.. math::
    \frac{du}{dt} = c_1 u
    \frac{dv}{dt} = c_2 v
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import Tensor

from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters
from measure_uq.utilities import numpy_array_like


def simple(
    _t: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    p: np.ndarray | torch.Tensor,
) -> list[np.ndarray | torch.Tensor]:
    """
    Return the right hand side of the simple system of ODEs.

    Parameters
    ----------
    _t : np.ndarray | torch.Tensor
        Time, not used in this system.
    y : np.ndarray | torch.Tensor
        State.
    p : np.ndarray | torch.Tensor
        Parameters.

    Returns
    -------
    list[np.ndarray | torch.Tensor]
        Right hand side of the system of ODEs.
    """
    dy1dt = p[0] * y[0]
    dy2dt = p[1] * y[1]
    return [dy1dt, dy2dt]


def analytical_solution(
    t: np.ndarray,
    y0: numpy_array_like,
    t_final: float,
    p: numpy_array_like,
) -> np.ndarray:
    """
    Analytical solution of the system of ODEs.

    Parameters
    ----------
    t : np.ndarray
        Time.
    y0 : numpy_array_like
        Initial condition.
    t_final : float
        Final time.
    p : numpy_array_like
        Parameters.

    Returns
    -------
    np.ndarray
        Solution of the system of ODEs.
    """
    sol = solve_ivp(simple, (0, t_final), y0, args=(p,), t_eval=t, method="Radau")
    return sol.y


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual condition for the system of ODEs.

    This class represents the residual condition of the heat equation on the line.
    It is used to sample points and evaluate the residual of the PDE.

    Attributes
    ----------
    N : int
        Number of points to sample.
    T : float
        Final time.
    residual : Tensor
        Residual of the PDE.

    Methods
    -------
    sample_points()
        Sample points for the PDE.
    eval(x: Tensor, y: Tensor) -> Tensor
        Evaluate the residual of the PDE.
    """

    N: int
    T: float
    residual: Tensor = field(init=False, default_factory=lambda: torch.tensor([]))

    def __post_init__(self) -> None:
        """Initialize the Residual condition."""
        assert self.N > 0
        super().__post_init__()

    def sample_points(self) -> None:
        """Sample points for the PDE."""
        print("Re-sample PDE variables for Residual")

        self.points = torch.linspace(0, self.T, self.N).view(-1, 1).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual of the PDE.

        Parameters
        ----------
        x : Tensor
            The points at which to evaluate the residual.
        y : Tensor
            The values of the state at the points.

        Returns
        -------
        Tensor
            The residual of the PDE.
        """
        assert x.shape[0] == y.shape[0]

        dy1_dt = jacobian(y, x, i=0, j=0)
        dy2_dt = jacobian(y, x, i=1, j=0)

        c1 = x[:, 1].unsqueeze(1)
        c2 = x[:, 2].unsqueeze(1)

        y1 = y[:, 0].unsqueeze(1)
        y2 = y[:, 1].unsqueeze(1)

        r1 = dy1_dt - c1 * y1
        r2 = dy2_dt - c2 * y2

        r = torch.cat([r1, r2], dim=1)

        assert r.shape == y.shape

        self.residual = r

        return r


@dataclass(kw_only=True)
class InitialCondition(Condition):
    """
    Initial condition of the PDE.

    This class represents the initial condition of the PDE and provides methods
    to sample points and evaluate the initial condition.

    Methods
    -------
    sample_points()
        Sample points for the initial condition of the PDE.
    eval(x: Tensor, y: Tensor) -> Tensor
        Evaluate the initial condition of the PDE.
    """

    initial_values: Tensor = field(
        default_factory=lambda: torch.tensor([[1, 4]]).float(),
    )

    def __post_init__(self) -> None:
        """Initialize the InitialCondition."""
        self.buffer.register("initial_values", self.initial_values)

        super().__post_init__()

    def sample_points(self) -> None:
        """Sample points for the initial condition of the PDE."""
        print("Re-sample PDE variables for InitialCondition")
        self.points = torch.tensor([[0.0]]).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the initial condition of the PDE.

        Parameters
        ----------
        x : Tensor
            The points at which to evaluate the initial condition.
        y : Tensor
            The values of the initial condition.

        Returns
        -------
        Tensor
            The values of the initial condition.
        """
        assert x.shape[0] == y.shape[0]

        return y - self.buffer["initial_values"]


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    """
    Random parameters for the PDE.

    This class represents the random parameters for the PDE and provides methods
    to sample values.

    Methods
    -------
    sample_values()
        Sample values for the random parameters.
    """

    def __post_init__(self) -> None:
        """Initialize the RandomParameters."""
        self.values = torch.tensor(
            [
                [-1.0, 1.0],
            ],
        ).float()
        super().__post_init__()
