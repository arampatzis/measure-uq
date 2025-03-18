#!/usr/bin/env python3

import numpy as np

from sympy import symbols, simplify
from sympy import sin, exp
from sympy.plotting import plot, plot3d


t, x, a, b, k, u1, u2, L, pi, y = symbols("t,x,a,b,k,u1,u2,L,pi,y")

# u = u1 + x*(u2 - u1)/L + sin(k*x) * exp(-b*t)
# print("\nu_t - b u_xx = ", u.diff(t) - b*u.diff(x).diff(x)/(k**2))

u = sin(k*x) * exp(-b*t)
print("u_t - b/k^2 u_xx = ", simplify(u.diff(t) - (b/k**2)*u.diff(x).diff(x)))

print("\nu(t,0)       = ", u.subs(x, 0))
print("u(t,L)       = ", u.subs(x, L))
print("u(0,x)       = ", u.subs(t, 0))

print("\nu(t,x)       = ", u, "\n")

y = u.subs(L, np.pi).subs(k, 5).subs(u1, 1).subs(u2, 2).subs(b, 2).subs(pi, np.pi)
# y = y.subs(t, 1)

# plot(y, (x, 0, np.pi))

plot3d(y, (t, 0, 1), (x, 0, np.pi))
