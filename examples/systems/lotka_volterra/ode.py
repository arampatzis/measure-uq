r"""
ODE for the Lotka-Volterra system.

.. math::
    \frac{dx}{dt} =  a x - b x y
    \frac{dy}{dt} = -c y + d x y
"""

# ruff: noqa: ERA001

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import Tensor

from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters
from measure_uq.utilities import numpy_array_like


def lotka_volterra(
    _t: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    p: np.ndarray | torch.Tensor,
) -> list[np.ndarray | torch.Tensor]:
    """
    Lotka-Volterra system.

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
    np.ndarray | torch.Tensor
        Right hand side of the Lotka-Volterra system.
    """
    dy1dt = p[0] * y[0] - p[1] * y[0] * y[1]
    dy2dt = -p[2] * y[1] + p[3] * y[0] * y[1]
    return [dy1dt, dy2dt]


def analytical_solution(
    t: np.ndarray,
    y0: numpy_array_like,
    t_final: float,
    p: np.ndarray,
) -> np.ndarray:
    """
    Analytical solution of the Lotka-Volterra system.

    Parameters
    ----------
    t : np.ndarray
        Time.
    y0 : np.ndarray
        Initial condition.
    t_final : float
        Final time.
    p : np.ndarray
        Parameters.

    Returns
    -------
    np.ndarray
        Solution of the Lotka-Volterra system.
    """
    sol = solve_ivp(
        lotka_volterra,
        (0, t_final),
        y0,
        args=(p,),
        t_eval=t,
        method="Radau",
    )
    return sol.y


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual condition for the Lotka-Volterra system.

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
            Time.
        y : Tensor
            State.

        Returns
        -------
        Tensor
            Residual of the PDE.
        """
        assert x.shape[0] == y.shape[0]

        dy1_dt = jacobian(y, x, i=0, j=0)
        dy2_dt = jacobian(y, x, i=1, j=0)

        x[:, 0].unsqueeze(1)
        a = x[:, 1].unsqueeze(1)
        b = x[:, 2].unsqueeze(1)
        c = x[:, 3].unsqueeze(1)
        d = x[:, 4].unsqueeze(1)

        y1 = y[:, 0].unsqueeze(1)
        y2 = y[:, 1].unsqueeze(1)
        y1y2 = y1 * y2

        r1 = dy1_dt - a * y1 + b * y1y2
        r2 = dy2_dt + c * y2 - d * y1y2

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
    """

    initial_values: Tensor = field(
        default_factory=lambda: torch.tensor([[1.75, 1.75]]).float(),
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
            Time.
        y : Tensor
            State.

        Returns
        -------
        Tensor
            The initial condition of the PDE.
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
                [1.0, 0.1, 1.5, 0.075],  # y0 = [15, 10] T=
                # [.5, 0.8, 1.5, 0.5], # y0 = [2, 1], T=20
            ],
        ).float()
        super().__post_init__()
