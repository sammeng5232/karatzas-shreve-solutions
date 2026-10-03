"""Exercise 5.5.36: dX = (1+X)(1+X^2) dt + (1+X^2) dW explodes in finite expected time.

The solution claims  E S = g(arctan x)  for X_0 = x, where
  g(theta) = (pi/2) coth(pi) - theta - pi exp(-2 theta) / (2 sinh(pi)).
We check this (a) against the Green's-function formula of Remark 5.5.33,
  M(x) = int G(x,y) m(dy),  with p(x) = (1 - exp(-2 arctan x))/2,
  m(dy) = 2 exp(2 arctan y)/(1+y^2) dy,  computed by quadrature on the real line,
and (b) against an Euler simulation of the equation for X itself, with the
state-dependent step h/(1+X^2) and explosion declared when |X| > 300.
"""
import numpy as np
from scipy.integrate import quad

def g(th):
    return np.pi / 2 / np.tanh(np.pi) - th - np.pi * np.exp(-2 * th) / (2 * np.sinh(np.pi))

p = lambda x: (1 - np.exp(-2 * np.arctan(x))) / 2
pm, pp = (1 - np.exp(np.pi)) / 2, (1 - np.exp(-np.pi)) / 2
dm = lambda y: 2 * np.exp(2 * np.arctan(y)) / (1 + y * y)

def M(x):
    lo = quad(lambda y: (p(y) - pm) * (pp - p(x)) * dm(y), -np.inf, x)[0]
    hi = quad(lambda y: (p(x) - pm) * (pp - p(y)) * dm(y), x, np.inf)[0]
    return (lo + hi) / (pp - pm)

rng = np.random.default_rng(20261002)
h, n, big = 0.004, 4000, 300.0
for x0 in (-1.0, 0.0, 1.5):
    X = np.full(n, x0)
    T = np.zeros(n)
    alive = np.arange(n)
    while alive.size:
        x = X[alive]
        dt = h / (1 + x * x)
        x = x + (1 + x) * (1 + x * x) * dt + (1 + x * x) * np.sqrt(dt) * rng.standard_normal(x.size)
        X[alive] = x
        T[alive] += dt
        alive = alive[np.abs(x) <= big]
    print(f"x0={x0:5.2f}: closed form {g(np.arctan(x0)):.4f}   Green quadrature {M(x0):.4f}"
          f"   simulation {T.mean():.4f} (s.e. {T.std() / np.sqrt(n):.4f})"
          f"   P[X_S=+inf]: formula {(p(x0) - pm) / (pp - pm):.4f}  sim {np.mean(X > 0):.4f}")
