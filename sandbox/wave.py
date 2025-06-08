#!/usr/bin/env python3
r"""
Description of the wave equation on the line.

.. math::
    u_{tt} - a u_{xx} = 0, \quad (t,x) \in [0,1] \times [0,\pi]

    u(0, x) = \sin(k x)

    u_t(0, x) = 0

    u(t, 0) = 0

    u(t, \pi) = \sin(k \pi) \cos(\sqrt{a} k t)
"""

import numpy as np
from sympy import cos, simplify, sin, symbols, pi, sqrt
from sympy.plotting import plot3d

t, x, a, b, k, u1, u2, y, L = symbols("t,x,a,b,k,u1,u2,y,L")

u = sin(k * x) * cos(sqrt(a) * k * t)

print("u_tt - a u_xx = ", simplify(u.diff(t, t) - a * u.diff(x, x)))

print("\nu(t,0)       = ", u.subs(x, 0))
print("u(t,L)      = ", u.subs(x, L))
print("u(0,x)       = ", u.subs(t, 0))
print("u_t(0,x)     = ", u.diff(t).subs(t, 0))

print("\nu(t,x)       = ", u, "\n")
