"""
Passive membrane ODE example.

We consider the classic RC membrane model

    tau dV/dt = -V + R I,   V(0) = V_rest

which yields the residual used by a PINN

    tau dV/dt + V - R I = 0.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, tensor

import chaospy
from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters
from measure_uq.utilities import cartesian_product_of_rows


def reference_solution(
    t: np.ndarray,
    p: np.ndarray,
    tau: float = 10.0,
    R: float = 5.0,
) -> np.ndarray:
    """
    Analytical solution of the passive membrane equation.

    Parameters
    ----------
    t : np.ndarray
        Time coordinates.
    p : np.ndarray
        Parameters, where p[0] is V_rest and p[1] is I.
    tau : float, default=10.0
        Membrane time constant.
    R : float, default=5.0
        Membrane resistance.

    Returns
    -------
    np.ndarray
        V(t) for the given parameters.
    """
    # Ensure t is a column vector for broadcasting
    if t.ndim == 1:
        t = t[:, np.newaxis]
    
    v_rest = p[:, 0]
    I = p[:, 1]
    
    # Broadcasting takes care of matching shapes
    return R * I + (v_rest - R * I) * np.exp(-t / tau)


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual of the membrane ODE.

    Attributes
    ----------
    Nt : int
        Number of time points.
    T : float
        Final time.
    tau : float
        Membrane time constant.
    R : float
        Membrane resistance.
    """

    Nt: int
    T: float
    tau: float = 10.0
    R: float = 5.0

    def sample_points(self) -> None:
        """Sample time points in (0, T)."""
        self.points = torch.tensor(np.random.uniform(0, self.T, (self.Nt, 1)), dtype=torch.float32)

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate tau dV/dt + V - R I.
        
        Assumes input x has columns: t, V_rest, I.
        """
        assert x.shape[0] == y.shape[0]

        dV_dt = jacobian(y, x, j=0)
        # Extract I from the third column of the input tensor x
        I = x[:, 2:3]

        return self.tau * dV_dt + y - self.R * I


@dataclass(kw_only=True)
class InitialCondition(Condition):
    """Initial condition V(0) = V_rest."""

    def sample_points(self) -> None:
        """The point for the initial condition is always at t=0."""
        self.points = torch.tensor([[0.0]], dtype=torch.float32)

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate V(0) - V_rest.
        
        Assumes input x has columns: t, V_rest, I.
        """
        # Extract V_rest from the second column of the input tensor x
        v_rest = x[:, 1:2]
        return y - v_rest


@dataclass(kw_only=True)
class RandomParameters(Parameters):
    """
    Random parameters for the ODE.

    Attributes
    ----------
    N : int
        Number of samples.
    joint : chaospy.J | None
        A chaospy joint distribution for sampling parameters.
    """

    N: int
    joint: chaospy.J | None = None

    def sample_values(self) -> None:
        """Sample parameters from `joint` if provided, else uniform defaults."""
        if self.joint is not None:
            self.values = torch.tensor(self.joint.sample(self.N).T, dtype=torch.float32)
        else:
            # Sample V_rest and I from uniform distributions
            v0 = np.random.uniform(-70.0, -60.0, (self.N, 1))
            I = np.random.uniform(1.0, 3.0, (self.N, 1))
            # Concatenate horizontally to create the parameter matrix
            self.values = torch.tensor(np.concatenate([v0, I], axis=1), dtype=torch.float32)