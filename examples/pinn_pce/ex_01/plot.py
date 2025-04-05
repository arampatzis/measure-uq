#!/usr/bin/env python3
"""
Visualizes the solution of an ordinary differential equation (ODE) using a
Physics Informed Neural Network with Polynomial Chaos Expansion (PINN-PCE).

The script performs the following steps:
1. Loads the trained PINN-PCE model and the saved PDE.
2. Samples points and parameters for testing.
3. Computes the analytical solution of the ODE for the sampled points and parameters.
4. Plots the mean and standard deviation of the analytical solution.
5. Uses the trained PINN-PCE model to predict the solution for the sampled points and
parameters.
6. Plots the mean and standard deviation of the predicted solution.
7. Compares the analytical and predicted solutions visually.

The results are displayed using matplotlib.

"""

# ruff: noqa: D103 ERA001

import matplotlib.pyplot as plt
import numpy as np
from torch import tensor

from examples.pinn_pce.ex_01.pde import analytical_solution
from measure_uq.models import PINN_PCE
from measure_uq.pde import PDE

plt.rc("figure", figsize=[16, 9])


def main() -> None:
    """
    Main function to visualize the solution of an ordinary differential equation (ODE)
    using a Physics Informed Neural Network with Polynomial Chaos Expansion (PINN-PCE).
    """
    model = PINN_PCE.load("data/model.pt")
    pde = PDE.load("data/pde.pickle")

    # pde.conditions_test[0].sample_ponts()
    t = np.sort(pde.conditions_test[0].points.detach().numpy(), axis=0)
    # pde.parameters_test.sample_values()
    p = pde.parameters_test.values

    y = np.zeros((p.shape[0], t.shape[0]))
    for i in range(p.shape[0]):
        y[i, :] = analytical_solution(tensor(t), p[i, :]).detach().numpy().squeeze()

    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()

    ymean_ex = np.mean(y, axis=0)
    ystd_ex = np.std(y, axis=0)
    ax[0].plot(t, ymean_ex)
    ax[0].fill_between(
        t.squeeze(),
        ymean_ex - 2 * ystd_ex,
        ymean_ex + 2 * ystd_ex,
        alpha=0.4,
    )
    ax[0].fill_between(t.squeeze(), ymean_ex - ystd_ex, ymean_ex + ystd_ex, alpha=0.4)

    ax[0].plot(t, y.T, alpha=0.25, color="black")

    tp, ym = model(tensor(t), p)
    ym = ym.reshape(t.shape[0], p.shape[0]).detach().numpy()

    ymean_c = np.mean(ym, axis=1)
    ystd_c = np.std(ym, axis=1)

    ax[1].plot(t, ymean_c)
    ax[1].plot(t, ymean_ex, "--")
    ax[1].fill_between(
        t.squeeze(),
        ymean_c - 2 * ystd_c,
        ymean_c + 2 * ystd_c,
        alpha=0.4,
    )
    ax[1].fill_between(t.squeeze(), ymean_c - ystd_c, ymean_c + ystd_c, alpha=0.4)

    yyy = model.net(tensor(t)).detach().numpy().squeeze()
    yyy[:, 0]
    np.sqrt(np.sum(yyy**2, axis=1))

    # ax[2].plot(t, mean, label="1st coefficient")
    ax[2].plot(t, ymean_c, "-", label="Monte Carlo")
    ax[2].plot(t, ymean_ex, "--", label="exact")

    # ax[3].plot(t, std, label="sum of coefficient")
    ax[3].plot(t, ystd_c, "-", label="Monte Carlo")
    ax[3].plot(t, ystd_ex, "--", label="exact")

    for a in ax:
        a.grid()

    plt.show()


if __name__ == "__main__":
    main()
