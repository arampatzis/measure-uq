#!/usr/bin/env python3
"""
Visualizes the exact and approximate solutions of the ordinary differential
equation (ODE) given by:

.. math::
    y' = p1 * y
    y(0) = p2

The script performs the following steps:
1. Loads the pre-trained Physics Informed Neural Network (PINN) model and the
corresponding PDE.
2. Samples test points and parameters for evaluation.
3. Computes the exact solution of the ODE using the analytical solution function.
4. Plots the mean and standard deviation of the exact solution.
5. Fills the area between the mean ± 1 standard deviation and mean ± 2 standard
deviations for better visualization.

The results are displayed using matplotlib.
"""

# ruff: noqa: D103

import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.pinn.ex_04.pde import analytical_solution
from measure_uq.models import PINN
from measure_uq.pde import PDE

plt.rc("figure", figsize=[16, 9])


def main() -> None:
    model = PINN.load("model-b.pt")
    pde = PDE.load("pde-b.pickle")

    pde.conditions_test[0].sample_points()
    t, _ = torch.sort(pde.conditions_test[0].points, dim=0)
    pde.parameters_test.sample_values()
    p = pde.parameters_test.values

    y = np.zeros((t.shape[0], p.shape[0]))
    for i in range(p.shape[0]):
        y[:, i] = analytical_solution(t, p[i, :]).numpy().squeeze()

    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()

    ymean_ex = np.mean(y, axis=1)
    ystd_ex = np.std(y, axis=1)
    ax[0].plot(t, ymean_ex)
    ax[0].fill_between(
        t.squeeze(),
        ymean_ex - 2 * ystd_ex,
        ymean_ex + 2 * ystd_ex,
        alpha=0.4,
    )
    ax[0].fill_between(t.squeeze(), ymean_ex - ystd_ex, ymean_ex + ystd_ex, alpha=0.4)

    ax[0].plot(t, y, alpha=0.1)
    ax[0].title.set_text("Exact solutions")

    tp, ym = model(t, p)
    ym = ym.reshape(t.shape[0], p.shape[0]).detach().numpy()

    ymean_c = np.mean(ym, axis=1)
    ystd_c = np.std(ym, axis=1)
    ax[1].plot(t, ymean_c)
    ax[1].fill_between(
        t.squeeze(),
        ymean_c - 2 * ystd_c,
        ymean_c + 2 * ystd_c,
        alpha=0.4,
    )
    ax[1].fill_between(t.squeeze(), ymean_c - ystd_c, ymean_c + ystd_c, alpha=0.4)

    ax[1].plot(t, ym, alpha=0.05)

    ax[1].title.set_text("Approximate solutions")

    ax[2].plot(t, ymean_ex)
    ax[2].plot(t, ymean_c)

    ax[2].title.set_text("Exact and Approximate mean")

    ax[3].plot(t, ystd_ex)
    ax[3].plot(t, ystd_c)

    ax[3].title.set_text("Exact and Approximate standard deviation")

    for a in ax:
        a.grid()

    plt.show()


if __name__ == "__main__":
    main()
