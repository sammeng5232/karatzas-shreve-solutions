"""Exercise 2.8.11 (Karatzas & Shreve): density (8.24) of the exit time T_0 ^ T_a
of Brownian motion started at x in [0, a].

Checks:
 (1) the image series (8.24) agrees with the independent eigenfunction expansion
       f(t) = sum_{k odd} (2 k pi / a^2) sin(k pi x / a) exp(-k^2 pi^2 t / (2 a^2));
 (2) the series integrates to 1 over (0, infinity) for 0 < x < a;
 (3) the series vanishes identically for x = 0 and x = a;
 (4) a Monte Carlo estimate of P^x[t1 < T <= t2] (Euler walk, small step) agrees
     with the integral of (8.24) over (t1, t2].
"""
import numpy as np
from scipy.integrate import quad

def f_images(t, x, a, N=60):
    n = np.arange(-N, N + 1)
    c1 = 2 * n * a + x
    c2 = 2 * n * a + a - x
    s = np.sum(c1 * np.exp(-c1**2 / (2 * t)) + c2 * np.exp(-c2**2 / (2 * t)))
    return s / np.sqrt(2 * np.pi * t**3)

def f_eigen(t, x, a, K=4001):
    k = np.arange(1, K, 2)
    return np.sum(2 * k * np.pi / a**2 * np.sin(k * np.pi * x / a)
                  * np.exp(-k**2 * np.pi**2 * t / (2 * a**2)))

a = 1.7
print("(1) images vs eigenfunction expansion")
for x in (0.2, 0.85, 1.3):
    for t in (0.05, 0.3, 1.0, 4.0):
        u, v = f_images(t, x, a), f_eigen(t, x, a)
        print(f"  x={x:4.2f} t={t:4.2f}  images={u:.12f}  eigen={v:.12f}  diff={u-v:.1e}")

print("(2) total mass")
for x in (0.2, 0.85, 1.3):
    m = quad(f_images, 1e-9, 1, args=(x, a), limit=400)[0] + quad(f_images, 1, 80, args=(x, a), limit=200)[0]
    print(f"  x={x:4.2f}  integral={m:.10f}")

print("(3) boundary points")
for x in (0.0, a):
    print(f"  x={x:4.2f}  max |series| over t grid = "
          f"{max(abs(f_images(t, x, a)) for t in np.linspace(0.01, 5, 200)):.2e}")

print("(4) Monte Carlo")
rng = np.random.default_rng(20261002)
x, t1, t2, dt, M = 0.5, 0.2, 0.8, 2e-4, 40000
w = np.full(M, x); T = np.full(M, np.inf); alive = np.ones(M, bool)
for i in range(1, int(t2 / dt) + 1):
    w[alive] += np.sqrt(dt) * rng.standard_normal(alive.sum())
    hit = alive & ((w <= 0) | (w >= a))
    T[hit] = i * dt; alive &= ~hit
mc = np.mean((T > t1) & (T <= t2))
ex = quad(f_images, t1, t2, args=(x, a))[0]
print(f"  P[{t1} < T <= {t2}]  MC={mc:.4f} (s.e. {np.sqrt(mc*(1-mc)/M):.4f})  formula={ex:.4f}")
