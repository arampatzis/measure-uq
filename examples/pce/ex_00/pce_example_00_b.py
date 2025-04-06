#!/usr/bin/env python3
"""Script for the presentation of the chaospy package on polynomial evaluation."""

# ruff: noqa: D103

import chaospy
import numpy as np


def show_orthogonality_of_expansion() -> None:
    """
    Demonstrate the orthogonality properties of polynomial chaos expansion.

    This function shows how the polynomials in the expansion are orthogonal with
    respect to the uniform distribution and how their expectations and variances
    behave.
    """
    a, b = -1.0, 1.0

    joint = chaospy.Uniform(a, b)

    expansion = chaospy.generate_expansion(
        3,
        joint,
        normed=True,
    )

    q0 = chaospy.variable(1)

    p = chaospy.polynomial(q0)
    print(p, " --> ", (a + b) / 2, " --> ", chaospy.E(p, joint))
    p = chaospy.polynomial(q0)
    print(p, " --> ", (b - a) ** 2 / 12, " --> ", chaospy.Var(p, joint))

    print("---------------")
    for e in expansion:
        print(chaospy.E(e, joint), " -- ", e)

    print("---------------")
    for i, e1 in enumerate(expansion):
        for j, e2 in enumerate(expansion):
            print(
                i,
                j,
                "---",
                chaospy.E(e1 * e2, joint),
                " -- ",
                e1.round(1),
                " * ",
                e2.round(1),
            )


def show_expansion() -> None:
    """
    Demonstrate polynomial chaos expansion fitting and properties.

    This function shows how to fit a polynomial chaos expansion to trigonometric
    functions and compare the statistical properties of the expansion with the
    original data.
    """
    a, b = -1.0, 1.0

    joint = chaospy.Uniform(a, b)

    expansion = chaospy.generate_expansion(
        10,
        joint,
        normed=True,
    )

    samples = joint.sample(100_000, rule="random")
    evals = np.array([np.cos(samples), np.sin(samples)]).T

    print(evals.shape)

    y = chaospy.fit_regression(expansion, samples, evals)

    print(chaospy.E(y, joint))
    print(np.mean(evals, axis=0))
    print("----------")
    print(chaospy.Var(y, joint))
    print(np.var(evals, axis=0))
    print("----------")
    print(y.round(4))


def main() -> None:
    """
    Run the demonstration of polynomial chaos expansion properties.

    This function executes both the orthogonality and expansion demonstrations
    to showcase the capabilities of polynomial chaos expansion.
    """
    show_orthogonality_of_expansion()

    show_expansion()


if __name__ == "__main__":
    main()
