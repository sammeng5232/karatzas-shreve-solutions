"""Problem 2.8.17 (Karatzas & Shreve): numerical checks of the joint density of (M_t, theta_t).

Checked:
 1. integrating the trivariate density (8.31) over a gives g(b,s) = b exp(-b^2/2s)/(pi sqrt(s^3 (t-s)));
 2. integrating g over s in (0,t) gives the density sqrt(2/(pi t)) exp(-b^2/2t) of M_t, cf. (8.3);
 3. integrating g over b gives the arc-sine density, whose integral over (0,s) is (2/pi) arcsin sqrt(s/t);
 4. Monte Carlo (Gaussian random walk): distribution function of theta_t and E[M_t^2 | theta_t] = 2 theta_t.
"""
import numpy as np
from scipy.integrate import quad

t = 1.7
f = lambda a, b, s: b*(b-a)/(np.pi*np.sqrt(s**3*(t-s)**3))*np.exp(-b*b/(2*s)-(b-a)**2/(2*(t-s)))
g = lambda b, s: b/(np.pi*np.sqrt(s**3*(t-s)))*np.exp(-b*b/(2*s))
h = lambda s: 1/(np.pi*np.sqrt(s*(t-s)))

for b, s in [(0.3, 0.2), (1.1, 0.9), (2.0, 1.5)]:
    v = quad(lambda a: f(a, b, s), -np.inf, b)[0]
    print(f"1. b={b}, s={s}:  int f da = {v:.8f}   g = {g(b, s):.8f}")
for b in [0.2, 1.0, 2.5]:
    v = quad(lambda s: g(b, s), 0, t, limit=200)[0]
    print(f"2. b={b}:  int g ds = {v:.8f}   density of M_t = {np.sqrt(2/(np.pi*t))*np.exp(-b*b/(2*t)):.8f}")
for s in [0.1, 0.85, 1.6]:
    v = quad(lambda b: g(b, s), 0, np.inf)[0]
    c = quad(h, 0, s)[0]
    print(f"3. s={s}:  int g db = {v:.8f}   h = {h(s):.8f};  int_0^s h = {c:.8f}   "
          f"(2/pi)arcsin sqrt(s/t) = {2/np.pi*np.arcsin(np.sqrt(s/t)):.8f}")

rng = np.random.default_rng(20817)
n, paths = 2000, 20000
W = np.cumsum(rng.standard_normal((paths, n), dtype=np.float32)*np.sqrt(t/n), axis=1)
W = np.concatenate([np.zeros((paths, 1), dtype=np.float32), W], axis=1)
idx = W.argmax(axis=1)
theta, M = idx*t/n, W[np.arange(paths), idx]
for s in [0.2, 0.85, 1.4]:
    print(f"4. P[theta<={s}]: MC {np.mean(theta <= s):.4f}   exact {2/np.pi*np.arcsin(np.sqrt(s/t)):.4f}")
for lo, hi in [(0.2, 0.4), (0.8, 1.0), (1.3, 1.5)]:
    m = (theta > lo) & (theta <= hi)
    print(f"4. E[M^2 | {lo}<theta<={hi}]: MC {np.mean(M[m]**2):.4f}   2 E[theta | same] = {2*np.mean(theta[m]):.4f}")
