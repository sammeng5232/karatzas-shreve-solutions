"""Numerical check for Exercise 6.4.6 of Karatzas & Shreve.

Checks, for the density of (4.21)
    g_a(t;b,tau) = b / (pi tau^{1/2} (t-tau)^{3/2}) exp{-b^2/(2(t-tau)) - (a+b)^2/(2 tau)},
(1) total mass:  int int g_a db dtau = P[W_t >= a];
(2) Laplace transform (4.20): int_0^inf e^{-alpha t} int int e^{-beta tau - gamma b} g_a db dtau dt
    against the closed form;
(3) Monte Carlo: P[W_t >= a, L_t(0) <= b0, Gamma_+(t) <= tau0] against the integral of g_a,
    with L_t(0) computed from Tanaka's formula |W_t| = int sgn(W) dW + 2 L_t(0) (book's normalisation).
"""
import numpy as np
from scipy.stats import norm

XL, WL = np.polynomial.legendre.leggauss(200)


def mass(t, a, bmax=None, taumax=None, beta=0.0, gam=0.0):
    """int_0^bmax int_0^taumax e^{-beta tau - gam b} g_a(t;b,tau) dtau db.

    With u = b / sqrt(t - tau):  g_a dtau = 2 e^{-u^2/2} (pi sqrt(tau))^{-1} e^{-(a+b)^2/(2 tau)} du,
    tau = t - b^2/u^2, u from b/sqrt(t) to b/sqrt(t - taumax).  Gauss-Legendre in b and in u.
    """
    bmax = 12 * np.sqrt(t) if bmax is None else bmax
    b = 0.5 * bmax * (XL + 1)[:, None]
    wb = 0.5 * bmax * WL[:, None]
    lo = b / np.sqrt(t)
    hi = lo + 10
    if taumax is not None and taumax < t:
        hi = np.minimum(hi, b / np.sqrt(t - taumax))
    u = lo + 0.5 * (hi - lo) * (XL + 1)[None, :]
    wu = 0.5 * (hi - lo) * WL[None, :]
    tau = np.maximum(t - b * b / (u * u), 1e-300)
    val = 2 * np.exp(-u * u / 2 - (a + b) ** 2 / (2 * tau) - beta * tau - gam * b) / (np.pi * np.sqrt(tau))
    return float(np.sum(val * wb * wu))


t, a = 1.0, 0.3
print("(1) mass of g_a      :", round(mass(t, a), 6), "  P[W_t>=a] :", round(norm.sf(a / np.sqrt(t)), 6))

alpha, beta, gam = 0.7, 1.3, 0.9
mu, nu = np.sqrt(2 * alpha), np.sqrt(2 * (alpha + beta))
closed = 2 * np.exp(-a * nu) / (nu * (gam + mu + nu))
T = 60.0                                   # e^{-alpha T} is negligible
ts = 0.5 * T * (XL + 1)
lt = sum(0.5 * T * w * np.exp(-alpha * s) * mass(s, a, beta=beta, gam=gam) for s, w in zip(ts, WL))
print("(2) Laplace transform:", round(lt, 6), "  closed form (4.20):", round(closed, 6))

rng = np.random.default_rng(20261002)
n, paths, b0, tau0 = 2000, 40000, 0.4, 0.6
dt = t / n
W = np.zeros(paths)
I = np.zeros(paths)
G = np.zeros(paths)
for _ in range(n):
    dW = rng.standard_normal(paths) * np.sqrt(dt)
    I += np.sign(W) * dW
    G += (W > 0) * dt
    W += dW
L = (np.abs(W) - I) / 2
mc = np.mean((W >= a) & (L <= b0) & (G <= tau0))
ex = mass(t, a, bmax=b0, taumax=tau0)
print("(3) Monte Carlo      :", round(mc, 4), "+/-", round(3 * np.sqrt(mc * (1 - mc) / paths), 4),
      "  integral of g_a:", round(ex, 4))
