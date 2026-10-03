"""Exercise 3.5.9 check.  For Brownian motion with drift mu>0, P[-W_* >= b] should be
exp(-2 mu b).  (a) Integrate the defective passage density (5.12) for level -b over t>0.
(b) Monte Carlo on a fine grid (slightly biased downward by discretisation)."""
import numpy as np
from scipy.integrate import quad
rng = np.random.default_rng(0)
mu = 0.7
for b in (0.3, 1.0, 2.0):
    h = lambda t: b/np.sqrt(2*np.pi*t**3)*np.exp(-(b+mu*t)**2/(2*t))
    print(f"b={b}: integral of (5.12) = {quad(h,0,np.inf)[0]:.6f}, exp(-2 mu b) = {np.exp(-2*mu*b):.6f}")
dt, T, n = 2e-3, 30.0, 4000
m = np.zeros(n); w = np.zeros(n)
for _ in range(int(T/dt)):
    w += mu*dt + np.sqrt(dt)*rng.standard_normal(n)
    np.minimum(m, w, out=m)
for b in (0.3, 1.0, 2.0):
    print(f"b={b}: MC P[-W_*>=b] = {np.mean(-m>=b):.4f}, exact {np.exp(-2*mu*b):.4f}")
print(f"MC mean of -W_* = {np.mean(-m):.4f}, exact 1/(2mu) = {1/(2*mu):.4f}")
