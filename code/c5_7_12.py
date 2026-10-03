"""Exercise 5.7.12 (Karatzas & Shreve): Brownian motion with constant drift mu.

Checks, by central finite differences at random points, that the transition density
    G(t,x;tau,xi) = (2 pi (tau-t))^{-1/2} exp(-(xi - x - mu (tau-t))^2 / (2 (tau-t)))
satisfies
    backward:  -G_t   = (1/2) G_xx   + mu G_x      in (t, x),
    forward:    G_tau = (1/2) G_xixi - mu G_xi     in (tau, xi),
and, by Monte Carlo, that G is the density of X_tau = x + mu (tau-t) + (W_tau - W_t).
"""
import numpy as np

rng = np.random.default_rng(5712)


def G(t, x, tau, xi, mu):
    s = tau - t
    return np.exp(-(xi - x - mu * s) ** 2 / (2 * s)) / np.sqrt(2 * np.pi * s)


h = 1e-4
worst_b = worst_f = 0.0
for _ in range(2000):
    mu = rng.uniform(-2, 2)
    t = rng.uniform(0, 1)
    tau = t + rng.uniform(0.3, 2)
    x, xi = rng.uniform(-2, 2, size=2)
    g = G(t, x, tau, xi, mu)
    Gt = (G(t + h, x, tau, xi, mu) - G(t - h, x, tau, xi, mu)) / (2 * h)
    Gx = (G(t, x + h, tau, xi, mu) - G(t, x - h, tau, xi, mu)) / (2 * h)
    Gxx = (G(t, x + h, tau, xi, mu) - 2 * g + G(t, x - h, tau, xi, mu)) / h**2
    Gtau = (G(t, x, tau + h, xi, mu) - G(t, x, tau - h, xi, mu)) / (2 * h)
    Gxi = (G(t, x, tau, xi + h, mu) - G(t, x, tau, xi - h, mu)) / (2 * h)
    Gxixi = (G(t, x, tau, xi + h, mu) - 2 * g + G(t, x, tau, xi - h, mu)) / h**2
    worst_b = max(worst_b, abs(-Gt - 0.5 * Gxx - mu * Gx))
    worst_f = max(worst_f, abs(Gtau - 0.5 * Gxixi + mu * Gxi))
print("max residual, backward equation:", worst_b)
print("max residual, forward equation: ", worst_f)

# Monte Carlo: P[a < X_tau <= b] against the integral of G over (a, b]
mu, t, x, tau, a, b = 0.7, 0.5, -0.3, 1.8, 0.0, 1.0
X = x + mu * (tau - t) + np.sqrt(tau - t) * rng.standard_normal(2_000_000)
grid = np.linspace(a, b, 20001)
vals = G(t, x, tau, grid, mu)
integral = np.sum((vals[1:] + vals[:-1]) / 2) * (grid[1] - grid[0])
print("P[a < X_tau <= b]: Monte Carlo", np.mean((X > a) & (X <= b)), " integral of G", integral)
