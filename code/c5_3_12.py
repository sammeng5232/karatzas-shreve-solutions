"""Exercise 5.3.12: check of formula (3.13) for dX = -sgn(X) dt + dW, X_0 = 0.

By Theorem 3.6.17 and (2.8.2), the right-hand side of (3.13) has the density
  q_t(x) = exp(-t/2-|x|)/sqrt(2 pi t^3) * int_0^inf (b+|x|) exp(b-(b+|x|)^2/(2t)) db.
We check (a) that q_t integrates to 1, and (b) that Q[|X_t| <= c] computed from q_t
agrees with an Euler simulation of the equation.
"""
import numpy as np
from scipy.integrate import quad

def q(x, t):
    x = abs(x)
    f = lambda b: (b + x) * np.exp(b - (b + x) ** 2 / (2 * t))
    return np.exp(-t / 2 - x) / np.sqrt(2 * np.pi * t ** 3) * quad(f, 0, np.inf)[0]

rng = np.random.default_rng(20261002)
for t in (0.5, 1.0, 2.0):
    mass = 2 * quad(lambda x: q(x, t), 0, np.inf)[0]
    n, steps = 200_000, 2000
    h = t / steps
    X = np.zeros(n)
    for _ in range(steps):
        X += -np.where(X > 0, 1.0, -1.0) * h + np.sqrt(h) * rng.standard_normal(n)
    print(f"t={t}: total mass of q_t = {mass:.6f}")
    for c in (0.3, 1.0):
        th = 2 * quad(lambda x: q(x, t), 0, c)[0]
        mc = np.mean(np.abs(X) <= c)
        se = np.sqrt(mc * (1 - mc) / n)
        print(f"   Q[|X_t|<={c}]: formula {th:.4f}   simulation {mc:.4f} (s.e. {se:.4f})")
