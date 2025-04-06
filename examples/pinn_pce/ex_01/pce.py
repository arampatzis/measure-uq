#!/usr/bin/env python3
"""
Solves an ODE using a PINN-PCE model.

.. math::
    y' = -beta * y
    y(0) = alpha

The script performs the following steps:
1. Defines the probability distributions for the parameters using chaospy.
2. Generates polynomial chaos expansion for the defined distributions.
3. Samples points using Gaussian quadrature.
4. Solves the ODE for each sample using the model solver.
5. Fits a regression model to approximate the solution using polynomial chaos expansion.
6. Computes the mean and standard deviation of the approximate solution.
7. Plots the mean and standard deviation of the solution.

The results are displayed using matplotlib.
"""

# ruff: noqa: D103

import chaospy
import numpoly
import numpy as np


def model_solver(
    parameters: tuple[float, float],
    t: np.ndarray,
) -> np.ndarray:
    """
    Solve the ODE using the given parameters and time points.

    Parameters
    ----------
    parameters : tuple[float, float]
        A tuple containing the parameters (alpha, beta) for the ODE.
    t : np.ndarray
        Array of time points at which the solution is evaluated.

    Returns
    -------
    np.ndarray
        Array containing the solution of the ODE at the given time points.
    """
    alpha, beta = parameters
    return np.exp(-beta * t) * alpha


def collocation_pce(
    t: np.ndarray,
    joint: chaospy.Distribution,
    expansion: numpoly.baseclass.ndpoly,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform collocation-based PCE to approximate the solution of the ODE.

    Parameters
    ----------
    t : np.ndarray
        Array of time points at which the solution is evaluated.
    joint : chaospy.Distribution
        Joint probability distribution of the parameters.
    expansion : numpoly.baseclass.ndpoly
        Polynomial chaos expansion basis.
    samples : np.ndarray
        Sample points generated using Gaussian quadrature.

    Returns
    -------
    mean : np.ndarray
        Mean of the approximate solution.
    std : np.ndarray
        Standard deviation of the approximate solution.
    """
    evals = [model_solver(sample, t) for sample in samples.T]

    u_approx = chaospy.fit_regression(expansion, samples, evals)

    mean = chaospy.E(u_approx, joint)
    std = chaospy.Std(u_approx, joint)

    return mean, std


def main() -> None:
    """
    Set up and solve the ODE using Polynomial Chaos Expansion (PCE).

    This function defines the probability distributions for the parameters, generates
    polynomial chaos expansion, samples points using Gaussian quadrature, solves the ODE
    for each sample using the model solver, fits a regression model to approximate the
    solution using polynomial chaos expansion, and computes the mean and standard
    deviation of the approximate solution.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    alpha = chaospy.Normal(1.5, 0.2)
    beta = chaospy.Uniform(0.1, 0.2)
    joint = chaospy.J(alpha, beta)

    t = np.linspace(0, 10, 1000)

    expansion = chaospy.generate_expansion(3, joint, normed=True)

    samples, _ = chaospy.generate_quadrature(8, joint, rule="gaussian")
    m, s = collocation_pce(
        t,
        joint,
        expansion,
        samples,
    )

    np.savez("data/pce.npz", m=m, s=s)


if __name__ == "__main__":
    main()
