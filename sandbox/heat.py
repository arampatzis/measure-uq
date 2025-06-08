#!/usr/bin/env python3

import numpy as np
from sympy import exp, simplify, sin, symbols
from sympy.plotting import plot3d

t, x, a, b, k, u1, u2, L, pi, y = symbols("t,x,a,b,k,u1,u2,L,pi,y")

u = sin(k * x) * exp(-b * t)
print("u_t - b/k^2 u_xx = ", simplify(u.diff(t) - (b / k**2) * u.diff(x).diff(x)))

print("\nu(t,0)       = ", u.subs(x, 0))
print("u(t,L)       = ", u.subs(x, L))
print("u(0,x)       = ", u.subs(t, 0))

print("\nu(t,x)       = ", u, "\n")
