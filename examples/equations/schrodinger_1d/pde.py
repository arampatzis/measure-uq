r"""
Description of the 1D time-dependent Schrödinger equation.

.. math::
    i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \psi}{\partial x^2} + V(x) \psi
    \psi(0, x) = \psi_0(x)
    \psi(t, x_{min}) = 0
    \psi(t, x_{max}) = 0

Where:
- \psi(t, x) is the complex wave function
- \hbar is the reduced Planck constant
- m is the particle mass
- V(x) is the potential energy function
- \psi_0(x) is the initial wave function

For implementation, we split the complex wave function into real and imaginary parts:
\psi(t, x) = u(t, x) + i v(t, x)

This gives us two coupled PDEs:
\frac{\partial u}{\partial t} = \frac{\hbar}{2m} \frac{\partial^2 v}{\partial x^2} - \frac{V(x)}{\hbar} v
\frac{\partial v}{\partial t} = -\frac{\hbar}{2m} \frac{\partial^2 u}{\partial x^2} + \frac{V(x)}{\hbar} u
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
    t: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the analytical solution of the 1D Schrödinger equation for a Gaussian wave packet.

    Parameters
    ----------
    t : np.ndarray
        Time coordinates.
    x : np.ndarray
        Spatial coordinates.
    p : np.ndarray
        Parameters of the PDE, where:
        p[0] = hbar (reduced Planck constant)
        p[1] = m (particle mass)
        p[2] = k0 (initial wave number)
        p[3] = sigma (width of the initial Gaussian wave packet)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The analytical solution of the Schrödinger equation at the given coordinates
        as (real_part, imaginary_part), each with shape: (Nx, Nt).
    """
    # Extract parameters
    hbar = p[0]
    m = p[1]
    k0 = p[2]
    sigma = p[3]
    
    # Create meshgrid
    tt, xx = np.meshgrid(t.squeeze(), x.squeeze(), indexing="ij")
    
    # Calculate time-dependent width
    sigma_t = sigma * np.sqrt(1 + (hbar * tt / (m * sigma**2))**2)
    
    # Calculate normalization factor
    norm_factor = (sigma / sigma_t) * np.exp(1j * k0 * xx - 1j * (hbar * k0**2 / (2 * m)) * tt)
    
    # Calculate exponent
    exponent = -((xx)**2) / (2 * sigma_t**2)
    
    # Calculate complex wave function
    psi = norm_factor * np.exp(exponent)
    
    # Return real and imaginary parts
    return psi.real.T, psi.imag.T  # shape: (Nx, Nt)


@dataclass(kw_only=True)
class Residual(Condition):
    """
    Residual condition for the 1D Schrödinger equation.

    This class represents the residual condition of the Schrödinger equation.
    It is used to sample points and evaluate the residual of the PDE.

    Attributes
    ----------
    Nt : int
        Number of time points to sample.
    Nx : int
        Number of spatial points to sample.
    T : float
        Maximum time.
    X_min : float
        Minimum spatial coordinate.
    X_max : float
        Maximum spatial coordinate.
    residual : Tensor
        Residual of the Schrödinger equation.
    """

    Nt: int
    Nx: int
    T: float
    X_min: float
    X_max: float

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
            tensor(np.random.uniform(self.X_min, self.X_max, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual of the PDE.

        Parameters
        ----------
        x : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor (contains both real and imaginary parts).

        Returns
        -------
        Tensor
            The residual of the PDE at the given points.
        """
        assert x.shape[0] == y.shape[0]
        
        # Split output into real and imaginary parts
        u = y[:, 0:1]  # Real part
        v = y[:, 1:2]  # Imaginary part
        
        # Calculate derivatives
        du_dt = jacobian(u, x, j=0)
        dv_dt = jacobian(v, x, j=0)
        
        du_dx = jacobian(u, x, j=1)
        dv_dx = jacobian(v, x, j=1)
        
        du_dxx = jacobian(du_dx, x, j=1)
        dv_dxx = jacobian(dv_dx, x, j=1)
        
        # Extract parameters
        hbar = x[:, 2][:, None]  # Reduced Planck constant
        m = x[:, 3][:, None]     # Particle mass
        
        # For simplicity, we use a zero potential V(x) = 0
        # The equations become:
        # du/dt = (hbar/2m) * d²v/dx²
        # dv/dt = -(hbar/2m) * d²u/dx²
        
        # Calculate the residuals for both real and imaginary parts
        residual_u = du_dt - (hbar/(2*m)) * dv_dxx
        residual_v = dv_dt + (hbar/(2*m)) * du_dxx
        
        # Combine residuals
        return torch.cat([residual_u, residual_v], dim=1)


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
    X_min : float
        Minimum spatial coordinate.
    X_max : float
        Maximum spatial coordinate.
    """

    Nx: int
    X_min: float
    X_max: float

    def sample_points(self) -> None:
        """Sample points for the initial condition of the PDE."""
        print("Re-sample PDE variables for InitialCondition")
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.0]]),
            tensor(np.random.uniform(self.X_min, self.X_max, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the initial condition of the PDE.

        Parameters
        ----------
        x : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor (contains both real and imaginary parts).

        Returns
        -------
        Tensor
            The difference between the predicted and true initial condition.
        """
        assert x.shape[0] == y.shape[0]
        
        # Extract parameters
        hbar = x[:, 2][:, None]  # Reduced Planck constant
        m = x[:, 3][:, None]     # Particle mass
        k0 = x[:, 4][:, None]    # Initial wave number
        sigma = x[:, 5][:, None] # Width of the Gaussian wave packet
        
        # Spatial coordinate
        xx = x[:, 1][:, None]
        
        # Initial Gaussian wave packet: exp(i*k0*x) * exp(-(x^2)/(2*sigma^2))
        # Real part: cos(k0*x) * exp(-(x^2)/(2*sigma^2))
        # Imaginary part: sin(k0*x) * exp(-(x^2)/(2*sigma^2))
        
        gaussian = torch.exp(-(xx**2)/(2*sigma**2))
        real_part = torch.cos(k0*xx) * gaussian
        imag_part = torch.sin(k0*xx) * gaussian
        
        # Split the predicted output
        u_pred = y[:, 0:1]  # Real part
        v_pred = y[:, 1:2]  # Imaginary part
        
        # Calculate the difference
        diff_real = u_pred - real_part
        diff_imag = v_pred - imag_part
        
        return torch.cat([diff_real, diff_imag], dim=1)


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
    T : float
        Maximum time.
    X_min : float
        Minimum spatial coordinate.
    """

    Nt: int
    T: float
    X_min: float

    def sample_points(self) -> None:
        """Sample points for the boundary condition at the left boundary of the PDE."""
        print("Re-sample PDE variables for BoundaryConditionLeft")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            torch.tensor([[self.X_min]]),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the boundary condition at the left boundary of the PDE.

        Parameters
        ----------
        x : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor (contains both real and imaginary parts).

        Returns
        -------
        Tensor
            The difference between the predicted and true boundary condition.
        """
        # For Schrödinger equation with infinite potential well, wave function is zero at boundaries
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
    T : float
        Maximum time.
    X_max : float
        Maximum spatial coordinate.
    """

    Nt: int
    T: float
    X_max: float

    def sample_points(self) -> None:
        """Sample points for the boundary condition at the right boundary of the PDE."""
        print("Re-sample PDE variables for BoundaryConditionRight")
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            torch.tensor([[self.X_max]]),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the boundary condition at the right boundary of the PDE.

        Parameters
        ----------
        x : Tensor
            Input coordinates tensor.
        y : Tensor
            Output values tensor (contains both real and imaginary parts).

        Returns
        -------
        Tensor
            The difference between the predicted and true boundary condition.
        """
        # For Schrödinger equation with infinite potential well, wave function is zero at boundaries
        return y


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