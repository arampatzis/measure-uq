#!/usr/bin/env python3
"""Script for the presentation of the chaospy package on polynomial evaluation."""

# ruff: noqa: D103

import chaospy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

plt.rc("figure", figsize=[16, 10])


def main():
    grid = np.mgrid[-2.0:2.0:100j, -2.0:2.0:100j]

    fig, ax = plt.subplots(1, 2)
    ax.flatten()

    joint = chaospy.MvNormal(
        [0.0, 0.0],
        [
            [0.3, 0.1],
            [0.1, 0.2],
        ],
    )

    ax[0].contourf(grid[0], grid[1], joint.pdf(grid), 30, cmap=cm.jet)
    ax[0].scatter(*joint.sample(100, rule="random"))
    ax[0].set_aspect("equal", adjustable="box")

    expansion = chaospy.generate_expansion(4, joint, normed=False)

    for i, e in enumerate(expansion):
        print(i, "  ", e.round(1))

    f = expansion[6](*grid)

    c1 = ax[1].contourf(grid[0], grid[1], f, 30, cmap=cm.jet)
    ax[1].set_aspect("equal", adjustable="box")
    fig.colorbar(c1, ax=ax[1])
    plt.show()


if __name__ == "__main__":
    main()
