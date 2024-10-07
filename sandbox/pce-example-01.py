#!/usr/bin/env python3
"""Script for the presentation of the chaospy package on a simple ODE."""

import chaospy
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

plt.rc("figure", figsize=[12, 8])


def model_solver(parameters, t):
    """Return the numerical solution of the ode."""
    alpha, beta = parameters
    return alpha * np.e ** (-t * beta)


def collocation_pce(t, joint, expansion, samples):
    """Point collocation formulation of PCE"""
    evals = np.array([model_solver(sample, t) for sample in samples.T])

    u_approx = chaospy.fit_regression(expansion, samples, evals)

    mean = chaospy.E(u_approx, joint)
    std = chaospy.Std(u_approx, joint)

    return mean, std


def galerkin_pce(
    t,
    joint,
    expansion,
    norms,
):
    """Galerkin formulation of PCE"""
    alpha, beta = chaospy.variable(2)

    phi_phi = chaospy.outer(
        expansion,
        expansion,
    )

    e_beta_phi_phi = chaospy.E(beta * phi_phi, joint)

    def right_hand_side(c, _):
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
    """Run the main function of the script."""
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
