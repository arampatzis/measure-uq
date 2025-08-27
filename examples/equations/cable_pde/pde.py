"""
Description of the Cable Equation, a model for signal propagation in a passive
neuron dendrite or axon.

This model describes how voltage evolves in time and space along a 1D cable.

The governing PDE and the residual used for training a PINN are

    τ ∂V/∂t + V − λ² ∂²V/∂x² = 0.

We define the problem on a finite domain: t in [0, T_max], x in [0, L_max].

Conditions for this specific problem:
1.  Initial Condition: The dendrite is at rest.
    V(0, x) = V_rest
2.  Boundary Condition (Left, x=0): A constant voltage is applied.
    V(t, 0) = V_inject
3.  Boundary Condition (Right, x=L_max): The end is sealed (no current flows).
    ∂V/∂x (t, L_max) = 0  (Neumann boundary condition)

This file is implemented against the common `measure_uq` framework so it can be
used with the provided trainers and plotting utilities.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, tensor

from measure_uq.gradients import jacobian
from measure_uq.pde import Condition, Parameters
from measure_uq.utilities import cartesian_product_of_rows

# --- Problem-Specific Classes ---

@dataclass(kw_only=True)
class ModelParameters(Parameters):
    """
    Defines and holds the fixed parameters for the Cable Equation model.
    """
    tau: float = 20.0       # ms (Membrane time constant)
    lambda_p: float = 1.0   # mm (Membrane length constant)
    V_rest: float = -70.0   # mV (Resting potential)
    V_inject: float = -50.0 # mV (Injected voltage at x=0)

    # parameter indices in the combined input tensor (t, x, tau, lambda_p, V_rest, V_inject)
    _TAU_IDX = 2
    _LAMBDA_P_IDX = 3
    _V_REST_IDX = 4
    _V_INJECT_IDX = 5

    def __post_init__(self) -> None:
        """Combine parameters into a single tensor for the PINN."""
        # Shape: (1, 4) -> [tau, lambda_p, V_rest, V_inject]
        self.values = tensor(
            [[self.tau, self.lambda_p, self.V_rest, self.V_inject]]
        ).float()
        # Ensure parent invariants (device placement, requires_grad, shape checks)
        super().__post_init__()

    def sample_values(self) -> None:
        """For fixed parameters, we don't need to resample."""
        print("Using fixed model parameters.")


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual condition for the Cable Equation PDE.

    Attributes
    ----------
    Nt : int
        Number of time points to sample.
    Nx : int
        Number of spatial points to sample.
    T_max : float
        Maximum time for the simulation.
    L_max : float
        Length of the dendrite.
    """
    Nt: int
    Nx: int
    T_max: float
    L_max: float

    def sample_points(self) -> None:
        """Sample (t, x) points for the PDE residual."""
        print("Re-sampling (t, x) points for PDE Residual.")
        t_points = tensor(np.random.uniform(0, self.T_max, (self.Nt, 1)))
        x_points = tensor(np.random.uniform(0, self.L_max, (self.Nx, 1)))
        self.points = cartesian_product_of_rows(t_points, x_points).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual: τ*∂V/∂t + V - λ²*∂²V/∂x².
        """
        tau = x[:, ModelParameters._TAU_IDX : ModelParameters._TAU_IDX + 1]
        lambda_p = x[:, ModelParameters._LAMBDA_P_IDX : ModelParameters._LAMBDA_P_IDX + 1]

        # First-order derivative w.r.t. time (input column 0)
        dv_dt = jacobian(y, x, j=0)

        # First-order derivative w.r.t. space (input column 1)
        dv_dx = jacobian(y, x, j=1)
        # Second-order derivative w.r.t. space
        d2v_dx2 = jacobian(dv_dx, x, j=1)

        residual = tau * dv_dt + y - (lambda_p**2) * d2v_dx2
        return residual


@dataclass(kw_only=True)
class InitialCondition(Condition):
    """Initial condition: V(0, x) = V_rest."""
    Nx: int
    L_max: float

    def sample_points(self) -> None:
        """Sample points at t=0 across the spatial domain."""
        print("Re-sampling points for Initial Condition.")
        t0 = torch.tensor([[0.0]])
        x_points = tensor(np.random.uniform(0, self.L_max, (self.Nx, 1)))
        self.points = cartesian_product_of_rows(t0, x_points).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the condition: y - V_rest."""
        v_rest = x[:, ModelParameters._V_REST_IDX : ModelParameters._V_REST_IDX + 1]
        return y - v_rest


@dataclass(kw_only=True)
class BoundaryConditionLeft(Condition):
    """Left boundary condition (Dirichlet): V(t, 0) = V_inject."""
    Nt: int
    T_max: float

    def sample_points(self) -> None:
        """Sample points at x=0 across the time domain."""
        print("Re-sampling points for Left Boundary Condition.")
        t_points = tensor(np.random.uniform(0, self.T_max, (self.Nt, 1)))
        x0 = torch.tensor([[0.0]])
        self.points = cartesian_product_of_rows(t_points, x0).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the condition: y - V_inject."""
        v_inject = x[:, ModelParameters._V_INJECT_IDX : ModelParameters._V_INJECT_IDX + 1]
        return y - v_inject


@dataclass(kw_only=True)
class BoundaryConditionRight(Condition):
    """Right boundary condition (Neumann): ∂V/∂x(t, L_max) = 0."""
    Nt: int
    T_max: float
    L_max: float

    def sample_points(self) -> None:
        """Sample points at x=L_max across the time domain."""
        print("Re-sampling points for Right Boundary Condition.")
        t_points = tensor(np.random.uniform(0, self.T_max, (self.Nt, 1)))
        xL = torch.tensor([[self.L_max]])
        self.points = cartesian_product_of_rows(t_points, xL).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the condition: ∂V/∂x - 0."""
        # This condition is on the derivative, so we compute it here.
        dv_dx = jacobian(y, x, j=1)
        return dv_dx  # The loss is |dv_dx - 0|^2

# --- Example Usage ---
if __name__ == '__main__':
    # Define simulation domain
    T_MAX = 100.0 # ms
    L_MAX = 3.0   # mm

    # 1. Define the problem components
    params = ModelParameters()
    residual_cond = Residual(Nt=100, Nx=100, T_max=T_MAX, L_max=L_MAX)
    initial_cond = InitialCondition(Nx=100, L_max=L_MAX)
    bc_left = BoundaryConditionLeft(Nt=100, T_max=T_MAX)
    bc_right = BoundaryConditionRight(Nt=100, T_max=T_MAX, L_max=L_MAX)

    # 2. Sample points for all conditions
    params.sample_values()
    residual_cond.sample_points()
    initial_cond.sample_points()
    bc_left.sample_points()
    bc_right.sample_points()

    print("\n--- Parameters ---")
    print(f"Param values (tau, lambda, V_rest, V_inject): {params.values}")

    print("\n--- Residual Condition ---")
    print(f"Shape of residual points (t, x): {residual_cond.points.shape}")
    print("Example residual points:")
    print(residual_cond.points[:5])

    print("\n--- Initial Condition ---")
    print(f"Shape of initial condition points (t, x): {initial_cond.points.shape}")
    print("Example initial condition points (t=0):")
    print(initial_cond.points[:5])

    print("\n--- Left Boundary Condition ---")
    print(f"Shape of BC Left points (t, x): {bc_left.points.shape}")
    print("Example BC Left points (x=0):")
    print(bc_left.points[:5])

    print("\n--- Right Boundary Condition ---")
    print(f"Shape of BC Right points (t, x): {bc_right.points.shape}")
    print("Example BC Right points (x=L_max):")
    print(bc_right.points[:5])