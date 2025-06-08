#!/usr/bin/env python3
r"""
Solution of the heat equation on the line using a PCE.

.. math::
    u_t - a / k^2 u_xx = 0
    u(0, x) = \sin(k x)
    u(t, 0) = 0
    u(t, \pi) = exp(-a t) \sin(\pi k)
"""

import chaospy
import matplotlib.pyplot as plt
import numpy as np
from chaospy import J
from matplotlib import animation
from matplotlib.lines import Line2D

from examples.equations.heat_1d.pde import analytical_solution


def main() -> None:
    """Train the PCE."""
    joint = J(
        chaospy.Uniform(1, 3),
        chaospy.Uniform(1, 3),
    )

    order = 5
    expansion = chaospy.generate_expansion(order, joint, normed=True)

    samples = joint.sample(1000, rule="sobol")

    x = np.linspace(0, np.pi, 100)[:, None]
    t = np.linspace(0, 1, 40)[None, :]

    solutions = np.array([analytical_solution(t, x, p) for p in samples.T])

    pce_coefficients = chaospy.fit_regression(expansion, samples, solutions)

    mean_solution = chaospy.E(pce_coefficients, joint).T
    std_solution = chaospy.Std(pce_coefficients, joint).T

    x = np.squeeze(x)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    k = 0

    ax0_lines = axs[0].plot(x, solutions[:, :, k].T)

    (ax1_mean,) = axs[1].plot(x, mean_solution[k], ".")

    axs[1].fill_between(
        x.T,
        mean_solution[k] - std_solution[k],
        mean_solution[k] + std_solution[k],
        color="blue",
        alpha=0.3,
        label="Uncertainty band (1 std)",
    )

    (ax2_line1,) = axs[2].plot(x, np.mean(solutions[:, :, k], axis=0))
    (ax2_line2,) = axs[2].plot(x, mean_solution[k], "--")
    (ax3_line1,) = axs[3].plot(x, np.std(solutions[:, :, k], axis=0))
    (ax3_line2,) = axs[3].plot(x, std_solution[k], "--")

    for ax in axs:
        ax.grid()

    def animate(
        i: int,
    ) -> tuple[Line2D, Line2D, Line2D, Line2D, Line2D, Line2D, Line2D]:
        """
        Animate the solution of the PDE.

        Parameters
        ----------
        i : int
            The time step index to animate.

        Returns
        -------
        tuple[Line2D, Line2D, Line2D, Line2D, Line2D, Line2D, Line2D]
            A tuple containing the updated line objects for the animation.
        """
        for k, line in enumerate(ax0_lines):
            line.set_ydata(solutions[k, :, i].T)

        ax1_mean.set_ydata(mean_solution[i])

        axs[1].collections[0].remove()

        ax1_fill = axs[1].fill_between(
            x.T,
            mean_solution[i] - std_solution[i],
            mean_solution[i] + std_solution[i],
            color="blue",
            alpha=0.3,
            label="Uncertainty band (1 std)",
        )

        ax2_line1.set_ydata(np.mean(solutions[:, :, i], axis=0))
        ax2_line2.set_ydata(mean_solution[i])
        ax3_line1.set_ydata(np.std(solutions[:, :, i], axis=0))
        ax3_line2.set_ydata(std_solution[i])

        return (
            ax0_lines,
            ax1_mean,
            ax1_fill,
            ax2_line1,
            ax2_line2,
            ax3_line1,
            ax3_line2,
        )

    animation.FuncAnimation(
        fig,
        animate,
        interval=200,
        blit=False,
        frames=t.shape[1],
    )

    plt.show()


if __name__ == "__main__":
    main()
