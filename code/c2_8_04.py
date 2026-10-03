"""Exercise 2.8.4 of Karatzas & Shreve: numerical check of (8.6).

Checks, for several (b, alpha), that the passage-time density (8.5)
    f(t) = b / sqrt(2 pi t^3) * exp(-b^2 / (2t))
integrates to 1 and has Laplace transform exp(-b sqrt(2 alpha)) (quadrature),
and that a Monte-Carlo estimate of E exp(-alpha T_b) agrees with the same
value, where T_b is the first time a simulated random walk approximation of
Brownian motion (step dt) reaches level b (so the simulation does not use
the density being tested).
"""
import numpy as np
from scipy.integrate import quad

rng = np.random.default_rng(20261002)


def f(t, b):
    return b / np.sqrt(2 * np.pi * t**3) * np.exp(-b * b / (2 * t))


def integral(g):
    return quad(g, 0, 1, limit=200)[0] + quad(g, 1, np.inf, limit=200)[0]


print("total mass of (8.5):")
for b in (0.5, 1.0, 2.0):
    print(f"  b={b}: {integral(lambda t: f(t, b)):.6f}")

print("b, alpha, quadrature of Laplace transform, exp(-b sqrt(2 alpha))")
for b in (0.5, 1.0, 2.0):
    for alpha in (0.1, 1.0, 3.0):
        lt = integral(lambda t: np.exp(-alpha * t) * f(t, b))
        exact = np.exp(-b * np.sqrt(2 * alpha))
        print(f"  {b}, {alpha}: {lt:.6f}  {exact:.6f}")
        assert abs(lt - exact) < 1e-6

# Monte Carlo with simulated paths: b = 1, horizon 12 (e^{-alpha*12} negligible
# for the alphas used), dt = 1e-3; discrete monitoring biases T_b upwards a bit.
b, dt, horizon, npaths = 1.0, 1e-3, 12.0, 20000
nsteps = int(horizon / dt)
w = np.zeros(npaths)
hit = np.full(npaths, np.inf)
for k in range(1, nsteps + 1):
    w += np.sqrt(dt) * rng.standard_normal(npaths)
    new = (w >= b) & np.isinf(hit)
    hit[new] = k * dt
print("b, alpha, Monte Carlo (simulated paths), exp(-b sqrt(2 alpha))")
for alpha in (1.0, 3.0):
    mc = np.mean(np.exp(-alpha * hit))
    exact = np.exp(-b * np.sqrt(2 * alpha))
    print(f"  {b}, {alpha}: {mc:.4f}  {exact:.4f}")
    assert abs(mc - exact) < 0.02
print("all checks passed")
