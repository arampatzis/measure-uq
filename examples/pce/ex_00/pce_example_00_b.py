#!/usr/bin/env python3
"""Script for the presentation of the chaospy package on polynomial evaluation."""

# ruff: noqa: D103

import chaospy
import numpy as np


def show_orthogonality_of_expansion() -> None:
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
    show_orthogonality_of_expansion()

    show_expansion()


if __name__ == "__main__":
    main()
