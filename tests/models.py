"""
Unit tests for the models module.

This module contains unit tests for the models module of measure-uq.
"""

import chaospy
import numpy as np
import torch

from measure_uq.models import PINN_PCE
from measure_uq.networks import FeedforwardBuilder


def test_pinn_pce() -> None:
    """
    Test the evaluation of the PINN_PCE model.

    The output for the point `x_i` and the parameter `p_j` is at row
    `i * Np + j` and must equal `sum_k c_k(x_i) phi_k(p_j)`, where `c` is the
    output of the network and `phi` is the expansion evaluated by chaospy.
    """
    joint = chaospy.J(
        chaospy.Uniform(0, 4),
        chaospy.Uniform(0.8, 1.2),
    )
    expansion = chaospy.generate_expansion(3, joint, normed=True)

    model = PINN_PCE(
        network_builder=FeedforwardBuilder([1, 8, len(expansion)]),
        expansion=expansion,
    )

    Nx, Np = 7, 5
    x = torch.linspace(0, 1, Nx)[:, None]
    p = torch.tensor(joint.sample(Np, seed=0).T, dtype=torch.float32)

    z, y = model(x, p)

    with torch.no_grad():
        c = model.network(x).double().numpy()
    phi = expansion(*p.double().numpy().T)
    expected = (c @ phi).reshape(-1, 1)

    assert z.shape == (Nx * Np, 3)
    assert torch.equal(z[:, :1], x.repeat_interleave(Np, dim=0))
    assert torch.equal(z[:, 1:], p.repeat(Nx, 1))
    assert y.shape == (Nx * Np, 1)
    assert np.allclose(y.detach().numpy(), expected, rtol=1e-6, atol=1e-10)
