"""Exercise 6.3.5 of Karatzas & Shreve: numerical check of formula (3.19).

Checked, for several t:
  (a) q_t of (3.19) equals e^{-t/2} e^{-|a|} int_0^inf e^b f_t(a,b) db, with f_t the density (3.17);
  (b) q_t integrates to 1;
  (c) P[|X_t| <= c] from an Euler scheme for dX = -sgn(X) dt + dW, X_0 = 0, against int_{-c}^{c} q_t.
"""
import numpy as np
from scipy import integrate
from scipy.stats import norm


def q(a, t):
    a = abs(a)
    tail = np.sqrt(2 * np.pi * t) * norm.sf((a - t) / np.sqrt(t))
    return (np.exp(-(a + t) ** 2 / (2 * t)) + np.exp(-2 * a) * tail) / np.sqrt(2 * np.pi * t)


def q_direct(a, t):
    a = abs(a)
    f = lambda b: (b + a) / np.sqrt(2 * np.pi * t ** 3) * np.exp(b - (b + a) ** 2 / (2 * t))
    return np.exp(-t / 2 - a) * integrate.quad(f, 0, np.inf)[0]


for t in (0.3, 1.0, 2.5):
    err = max(abs(q(a, t) - q_direct(a, t)) for a in (0.0, 0.2, -0.7, 1.5, -3.0))
    mass = 2 * integrate.quad(q, 0, np.inf, args=(t,))[0]
    print(f"t={t}: max |q - direct| = {err:.2e},  integral of q = {mass:.10f}")

rng = np.random.default_rng(12345)
t, n, paths, c = 1.0, 2000, 200_000, 0.5
dt = t / n
x = np.zeros(paths)
for _ in range(n):
    x += -np.sign(x) * dt + np.sqrt(dt) * rng.standard_normal(paths)
mc = np.mean(np.abs(x) <= c)
exact = 2 * integrate.quad(q, 0, c, args=(t,))[0]
print(f"P[|X_1| <= {c}]: Euler Monte Carlo = {mc:.4f} (s.e. {np.sqrt(mc*(1-mc)/paths):.4f}), formula = {exact:.4f}")
