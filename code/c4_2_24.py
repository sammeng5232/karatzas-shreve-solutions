"""Exercise 4.2.24 (Karatzas & Shreve): exit distribution of Brownian motion from the ball B_r.

Checks formula (2.26),
    P^x[W_tau in dy] = r^(d-2) (r^2 - |x|^2) |x - y|^(-d) mu_r(dy),   |y| = r,
in two independent ways.
 (1) Normalisation: the integral of the kernel against the uniform law mu_r equals 1
     (exact quadrature for d = 2, Gauss-Legendre in the polar angle for d = 3, 4).
 (2) Simulation by "walk on spheres", which uses only the fact (2.5) that Brownian motion
     started at the centre of a ball leaves it at a uniformly distributed point, together
     with the strong Markov property: from the current point z jump to a uniform point of
     the sphere of radius r - |z| about z, until r - |z| < 1e-7.  The simulated values of
     E^x f(W_tau) and of P^x[first coordinate of W_tau > r/2] are compared with the
     integrals of the kernel (computed from an independent uniform sample of the sphere).
"""
import numpy as np
from scipy import integrate, special

rng = np.random.default_rng(4224)


def kernel(x, y, r):
    d = x.shape[-1]
    return r ** (d - 2) * (r * r - np.sum(x * x, -1)) / np.linalg.norm(x - y, axis=-1) ** d


def uniform_sphere(n, d):
    g = rng.standard_normal((n, d))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


print("(1) total mass of the kernel against mu_r (should be 1)")
# d = 2: y = r(cos t, sin t), mu_r(dy) = dt / (2 pi)
r, x = 1.7, np.array([0.9, -0.8])
val, _ = integrate.quad(
    lambda t: kernel(x, r * np.array([np.cos(t), np.sin(t)]), r) / (2 * np.pi), 0, 2 * np.pi,
    limit=200)
print(f"   d=2, r={r}, x={x}:  {val:.12f}")
# d >= 3: by rotation invariance take x = rho e_1; the law of the polar angle under mu_r
# has density  Gamma(d/2) / (sqrt(pi) Gamma((d-1)/2)) * sin(t)^(d-2)  on (0, pi).
for d, r, rho in [(3, 1.0, 0.6), (3, 2.5, 2.4), (4, 1.3, 0.5), (5, 0.7, 0.65)]:
    c = special.gamma(d / 2) / (np.sqrt(np.pi) * special.gamma((d - 1) / 2))
    val, _ = integrate.quad(
        lambda t: c * np.sin(t) ** (d - 2) * r ** (d - 2) * (r * r - rho * rho)
        / (r * r + rho * rho - 2 * r * rho * np.cos(t)) ** (d / 2), 0, np.pi, limit=200)
    print(f"   d={d}, r={r}, |x|={rho}:  {val:.12f}")

print("(2) walk on spheres against the kernel")


def walk_on_spheres(x, r, n, tol=1e-7):
    d = x.size
    z = np.tile(x, (n, 1))
    active = np.ones(n, dtype=bool)
    while active.any():
        dist = r - np.linalg.norm(z[active], axis=1)
        idx = np.flatnonzero(active)
        done = dist < tol
        active[idx[done]] = False
        idx, dist = idx[~done], dist[~done]
        z[idx] += dist[:, None] * uniform_sphere(idx.size, d)
    return r * z / np.linalg.norm(z, axis=1, keepdims=True)


def f(y):
    return np.exp(y[:, 0]) + y[:, 1] * y[:, -1] + np.abs(y[:, -1])


for x, r in [(np.array([0.5, 0.3]), 1.0), (np.array([0.3, 0.2, -0.4]), 1.0),
             (np.array([1.0, -0.6, 0.5, 0.8]), 2.0)]:
    d, n = x.size, 400_000
    w = walk_on_spheres(x, r, n)
    y = r * uniform_sphere(2_000_000, d)
    k = kernel(x, y, r)
    for name, g in [("E f(W_tau)       ", f), ("P[W_tau^1 > r/2] ", lambda y: (y[:, 0] > r / 2) * 1.0)]:
        sim, se = g(w).mean(), g(w).std() / np.sqrt(n)
        ker = (g(y) * k).mean()
        print(f"   d={d}  {name} simulated {sim:.4f} (+-{se:.4f})   kernel {ker:.4f}")
