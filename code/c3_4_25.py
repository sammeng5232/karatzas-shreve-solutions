"""Numerical check for Exercise 3.4.25 of Karatzas & Shreve (Elworthy, Li & Yor).

For a continuous nonnegative local martingale M with M_0 = m and M_infinity = 0 the
solution shows

    x P[sup_t M_t >= x] = m                              for every x >= m,
    y P[sqrt(<M>_infinity) >= y] = y (2 Phi(m/y) - 1)    for every y > 0,

and the right-hand side of the second line tends to sqrt(2/pi) m as y -> infinity.

Test case (not a stopped Brownian motion): M_t = m exp(W_t - t/2), for which
<M>_infinity = int_0^infinity M_t^2 dt.  log M is simulated exactly on a grid; the
maximum of log M over each grid interval is sampled exactly from the law of the
maximum of a Brownian bridge, so sup M carries no discretisation bias; the integral
is computed by the trapezoidal rule and truncated at a horizon where M is negligible.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(3425)
m, n_paths, dt, horizon = 1.5, 30000, 0.025, 100.0
n_steps = int(round(horizon / dt))

L = np.full(n_paths, np.log(m))          # log M on the grid
sup_log = L.copy()
qv = np.zeros(n_paths)                   # int_0^t M_s^2 ds
for _ in range(n_steps):
    L_new = L + np.sqrt(dt) * rng.standard_normal(n_paths) - 0.5 * dt
    u = rng.random(n_paths)
    bridge_max = 0.5 * (L + L_new + np.sqrt((L_new - L) ** 2 - 2.0 * dt * np.log(u)))
    np.maximum(sup_log, bridge_max, out=sup_log)
    qv += 0.5 * dt * (np.exp(2.0 * L) + np.exp(2.0 * L_new))
    L = L_new
sup_M = np.exp(sup_log)
root_qv = np.sqrt(qv)

print(f"m = {m},  paths = {n_paths},  largest M at the horizon = {np.exp(L).max():.2e}")
print("x P[sup M >= x]   (theory: m for every x >= m)")
for x in (2.0, 4.0, 8.0, 16.0):
    p = np.mean(sup_M >= x)
    se = x * np.sqrt(p * (1 - p) / n_paths)
    print(f"  x = {x:5.1f}:  simulated {x * p:.4f} +- {se:.4f}   theory {m:.4f}")
print("y P[sqrt(<M>_inf) >= y]   (theory: y (2 Phi(m/y) - 1) -> sqrt(2/pi) m)")
for y in (1.0, 2.0, 4.0, 8.0, 16.0):
    p = np.mean(root_qv >= y)
    se = y * np.sqrt(p * (1 - p) / n_paths)
    print(f"  y = {y:5.1f}:  simulated {y * p:.4f} +- {se:.4f}   theory {y * (2 * norm.cdf(m / y) - 1):.4f}")
print(f"limits:  l = m = {m:.4f},  sigma = sqrt(2/pi) m = {np.sqrt(2 / np.pi) * m:.4f},"
      f"  sqrt(pi/2) sigma = {m:.4f}")
