#!/usr/bin/env python3
import chaospy
import numpy as np
import torch

from measure_uq.utilities import torch_numpoly_call

joint = chaospy.J(
    chaospy.Uniform(1, 3),
    chaospy.Uniform(-2, 1),
)

expansion = chaospy.generate_expansion(
    5,
    joint,
    normed=True,
)

x = np.random.uniform(1, 3, (2, 10))

e1 = expansion(*x)


e2 = torch_numpoly_call(
    torch.tensor(expansion.exponents, dtype=torch.float64),
    torch.tensor(np.array(expansion.coefficients), dtype=torch.float64),
    torch.tensor(x.T, dtype=torch.float64),
).T

print("exponents: ", expansion.exponents.shape)
print("coefficients: ", np.array(expansion.coefficients).shape)
print("input: ", x.T.shape)
print("output: ", e2.shape)

print(np.allclose(e1, e2.detach().numpy(), atol=1e-12))
