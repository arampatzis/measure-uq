#!/usr/bin/env python3
"""Solution of ODE using the PINN-PCE method."""

# ruff: noqa: D103

import chaospy
import numpoly
import numpy as np


def model_solver(
    parameters: tuple[float, float],
    t: np.ndarray,
) -> np.ndarray:
    alpha, beta = parameters
    return np.exp(-beta * t) * alpha


def collocation_pce(
    t: np.ndarray,
    joint: chaospy.Distribution,
    expansion: numpoly.baseclass.ndpoly,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    evals = [model_solver(sample, t) for sample in samples.T]

    u_approx = chaospy.fit_regression(expansion, samples, evals)

    mean = chaospy.E(u_approx, joint)
    std = chaospy.Std(u_approx, joint)

    return mean, std


def main():
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
