"""Exercise 6.5.3 of Karatzas & Shreve: checks of formula (5.15).

(a) v(t,x) from (5.15) is compared with the quadrature of z^2 against the
    transition density (5.14) (theta = 1), whose total mass is also printed;
(b) the residual of v_t = v_xx/2 - sgn(x) v_x is evaluated by finite differences;
(c) v(t,x) is compared with an Euler Monte-Carlo estimate of E[Z_t^2] for
    dZ = -sgn(Z) dt + dW;
(d) the limits t -> 0 and t -> infinity.
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def v(t, x):
    r = abs(x)
    s = np.sqrt(t)
    return (0.5 + np.sqrt(t / (2 * np.pi)) * (r - t - 1) * np.exp(-(r - t) ** 2 / (2 * t))
            + ((r - t) ** 2 + t - 0.5) * norm.cdf((r - t) / s)
            + np.exp(2 * r + norm.logsf((r + t) / s)) * (r + t - 0.5))


def dens(t, x, z):
    """(5.14) with theta = 1, for x >= 0."""
    s = np.sqrt(t)
    c = 1 / np.sqrt(2 * np.pi * t)
    if z > 0:
        return c * np.exp(-(x - z - t) ** 2 / (2 * t)) + np.exp(-2 * z) * norm.sf((x + z - t) / s)
    return c * np.exp(2 * x - (x - z + t) ** 2 / (2 * t)) + np.exp(2 * z) * norm.sf((x - z - t) / s)


print("(a) quadrature of (5.14) against formula (5.15)")
for t, x in [(0.3, 0.0), (0.3, 1.2), (1.0, 0.5), (2.5, 3.0), (6.0, 1.0)]:
    mass = sum(quad(lambda z: dens(t, x, z), a, b, limit=200)[0] for a, b in [(-40, 0), (0, 40)])
    m2 = sum(quad(lambda z: z * z * dens(t, x, z), a, b, limit=200)[0] for a, b in [(-40, 0), (0, 40)])
    print(f"  t={t:4.1f} x={x:4.1f}  mass={mass:.8f}  quad={m2:.8f}  (5.15)={v(t, x):.8f}  v(t,-x)={v(t, -x):.8f}")

print("(b) PDE residual v_t - v_xx/2 + sgn(x) v_x (finite differences)")
h = 1e-3
for t, x in [(0.3, 0.0), (0.3, 1.2), (1.0, -0.5), (2.5, 3.0), (1.0, 0.0)]:
    vt = (v(t + h, x) - v(t - h, x)) / (2 * h)
    vx = (v(t, x + h) - v(t, x - h)) / (2 * h)
    vxx = (v(t, x + h) - 2 * v(t, x) + v(t, x - h)) / h ** 2
    print(f"  t={t:4.1f} x={x:5.1f}  residual={vt - 0.5 * vxx + np.sign(x) * vx:+.2e}  v_x={vx:+.5f}")

print("(c) Euler Monte-Carlo for dZ = -sgn(Z) dt + dW")
rng = np.random.default_rng(653)
n, dt = 40_000, 5e-3
for t, x in [(0.5, 0.0), (1.0, 1.5), (2.0, -0.7)]:
    z = np.full(n, x)
    for _ in range(int(round(t / dt))):
        z += -np.sign(z) * dt + np.sqrt(dt) * rng.standard_normal(n)
    est, se = np.mean(z ** 2), np.std(z ** 2) / np.sqrt(n)
    print(f"  t={t:3.1f} x={x:4.1f}  MC={est:.4f} +- {se:.4f}  (5.15)={v(t, x):.4f}")

print("(d) limits")
for x in [0.0, 0.8, -2.0]:
    print(f"  x={x:4.1f}  v(1e-8,x)={v(1e-8, x):.6f}  x^2={x * x:.6f}  v(60,x)={v(60.0, x):.6f}")
