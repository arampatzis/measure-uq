#!/usr/bin/env python3
"""Script for the presentation of the chaospy package on a simple ODE."""

import sys
from typing import Any

import chaospy
import matplotlib.pyplot as plt
import numpoly
import numpy as np
from scipy.integrate import odeint

plt.rc("figure", figsize=[12, 8])


def model_solver(
    parameters: tuple[float, float],
    t: np.ndarray,
) -> np.ndarray:
    """
    Return the numerical solution of the ODE.

    The ODE to be solved is:

        dy/dt = -beta * y

    with the initial condition y(0) = alpha.

    Parameters
    ----------
    parameters : tuple[float, float]
        (alpha, beta) parameters of the ODE
    t : np.ndarray
        Time points where to evaluate the solution

    Returns
    -------
    np.ndarray
        Solution of the ODE at the given time points
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
    Point collocation formulation of PCE. See also:
    https://github.com/jonathf/chaospy/blob/master/docs/user_guide/main_usage/point_collocation.ipynb

    Parameters
    ----------
    t : np.ndarray
        Time points
    joint : chaospy.Distribution
        Joint distribution of the parameters
    expansion : chaospy.poly.Poly
        PCE expansion
    samples : np.ndarray
        Collocation points
    """
    import inspect

    print(inspect.getmro(expansion.__class__))
    sys.exit()
    evals = [model_solver(sample, t) for sample in samples.T]

    u_approx = chaospy.fit_regression(expansion, samples, evals)

    mean = chaospy.E(u_approx, joint)
    std = chaospy.Std(u_approx, joint)

    return mean, std


def galerkin_pce(
    t: np.ndarray,
    joint: chaospy.Distribution,
    expansion: numpoly.baseclass.ndpoly,
    norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Galerkin formulation of PCE. See also:
    https://github.com/jonathf/chaospy/blob/master/docs/user_guide/main_usage/intrusive_galerkin.ipynb

    Parameters
    ----------
    t : np.ndarray
        Time array
    joint : chaospy.Distribution
        Joint distribution of the random parameters
    expansion : chaospy.poly.Poly
        PCE expansion
    norms : np.ndarray
        Norms of the polynomials

    Returns
    -------
    mean : np.ndarray
        Mean of the solution
    std : np.ndarray
        Standard deviation of the solution
    """
    alpha, beta = chaospy.variable(2)

    phi_phi = chaospy.outer(
        expansion,
        expansion,
    )

    e_beta_phi_phi = chaospy.E(beta * phi_phi, joint)

    def right_hand_side(c: np.ndarray, _: Any) -> np.ndarray:
        """
        Right-hand side of the ODE to solve

        Parameters
        ----------
        c : np.ndarray
            Coefficients
        _ : Any
            Ignored time argument

        Returns
        -------
        np.ndarray
            Right-hand side of the ODE
        """
        return -np.sum(c * e_beta_phi_phi, -1) / norms

    e_alpha_phi = chaospy.E(alpha * expansion, joint)
    initial_condition = e_alpha_phi / norms

    coefficients = odeint(
        func=right_hand_side,
        y0=initial_condition,
        t=t,
    )

    u_approx = chaospy.sum(expansion * coefficients, -1)
    mean = chaospy.E(u_approx, joint)
    std = chaospy.Std(u_approx, joint)

    return mean, std


def main():
    """
    Run the main function of the script.

    This example shows how to solve an ODE with a PCE using the Galerkin and
    point collocation methods. The ODE is a simple exponential decay with two
    parameters: the initial value and the decay rate. The parameters are
    distributed according to the given distributions.

    The ODE is solved using the Galerkin method and the point collocation method
    with two different sets of collocation points: Sobol points and Gauss points.

    The results are plotted in three subplots: the Galerkin method, the point
    collocation method with Sobol points, and the point collocation method with
    Gauss points. Each subplot shows the mean and standard deviation of the
    solution as a function of time.
    """
    alpha = chaospy.Normal(1.5, 0.2)
    beta = chaospy.Uniform(0.1, 0.2)
    joint = chaospy.J(alpha, beta)

    t = np.linspace(0, 10, 1000)

    expansion, norms = chaospy.generate_expansion(3, joint, retall=True)

    mean = []
    std = []
    title = []

    m, s = galerkin_pce(t, joint, expansion, norms)
    mean.append(m)
    std.append(s)
    title.append("Galerkin")

    samples = joint.sample(10, rule="sobol")
    m, s = collocation_pce(
        t,
        joint,
        expansion,
        samples,
    )
    mean.append(m)
    std.append(s)
    title.append("Point collocation (Sobol)")

    samples, _ = chaospy.generate_quadrature(8, joint, rule="gaussian")
    m, s = collocation_pce(
        t,
        joint,
        expansion,
        samples,
    )
    mean.append(m)
    std.append(s)
    title.append("Point collocation (Gauss)")

    _, axs = plt.subplots(1, len(mean))

    for k, ax in enumerate(axs):
        ax.set_xlabel("t")
        ax.set_ylabel("model approximation")
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 2])
        ax.fill_between(t, mean[k] - std[k], mean[k] + std[k], alpha=0.3)
        ax.plot(t, mean[k])
        ax.grid(True)
        ax.title.set_text(title[k])

    plt.show()


if __name__ == "__main__":
    main()
