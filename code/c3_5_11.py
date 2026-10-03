"""Exercise 3.5.11 (Robbins & Siegmund): numerical check.

R_c = inf{t : exp(nu W_t - nu^2 t/2) = c} is the passage time of W_t - (nu/2) t to the level
b = log(c)/nu.  Under P the process W_t - (nu/2)t is a Brownian motion with drift -nu/2, under
P^(nu) it is a Brownian motion with drift +nu/2.  With the (possibly defective) density
h(t; b, mu) = b (2 pi t^3)^(-1/2) exp(-(b - mu t)^2 / (2t)) of (5.12) we check

   (a)  int_0^oo h(t; b, -nu/2) dt   = 1/c            (= P[R_c < oo]),
   (b)  int_0^oo h(t; b, +nu/2) dt   = 1              (R_c < oo a.s. P^(nu)),
   (c)  int_0^oo t h(t; b, +nu/2) dt = 2 log c / nu^2 (= E^(nu) R_c),

by quadrature, and (a), (c) again by a seeded Monte Carlo simulation of the two drifted
Brownian motions on a fine grid (the simulation has a small discretisation bias).
"""
import numpy as np
from scipy.integrate import quad


def h(t, b, mu):
    return b / np.sqrt(2 * np.pi * t**3) * np.exp(-(b - mu * t) ** 2 / (2 * t))


print("quadrature")
for nu, c in [(1.0, 2.0), (0.7, 3.5), (2.0, 1.3)]:
    b = np.log(c) / nu
    a = quad(h, 0, np.inf, args=(b, -nu / 2))[0]
    m0 = quad(h, 0, np.inf, args=(b, nu / 2))[0]
    m1 = quad(lambda t: t * h(t, b, nu / 2), 0, np.inf)[0]
    print(f" nu={nu}, c={c}:  P[R<oo]={a:.6f} vs 1/c={1/c:.6f};  "
          f"mass under P^(nu)={m0:.6f};  E^(nu)R={m1:.6f} vs 2log(c)/nu^2={2*np.log(c)/nu**2:.6f}")

print("Monte Carlo (nu=1, c=2)")
rng = np.random.default_rng(20261002)
nu, c = 1.0, 2.0
b = np.log(c) / nu
dt, horizon, npaths = 2e-3, 40.0, 20000
nsteps = int(horizon / dt)
for mu, label in [(-nu / 2, "P"), (nu / 2, "P^(nu)")]:
    x = np.zeros(npaths)
    hit = np.full(npaths, np.inf)
    alive = np.ones(npaths, dtype=bool)
    for k in range(1, nsteps + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        x[idx] += mu * dt + np.sqrt(dt) * rng.standard_normal(idx.size)
        crossed = idx[x[idx] >= b]
        hit[crossed] = k * dt
        alive[crossed] = False
    finite = np.isfinite(hit)
    if mu < 0:
        print(f" under {label}: fraction of paths reaching b = {finite.mean():.4f}   (1/c = {1/c:.4f})")
    else:
        print(f" under {label}: fraction reaching b = {finite.mean():.4f}, mean of R = "
              f"{hit[finite].mean():.4f}   (2 log c / nu^2 = {2*np.log(c)/nu**2:.4f})")
