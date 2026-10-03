"""Exercise 6.4.10 (Karatzas & Shreve): Cameron-Martin formula

    E^0 exp(-beta * int_0^b W_t^2 dt) = cosh(b*sqrt(2*beta))^(-1/2).

Two independent checks for several (beta, b):
 (1) exact Gaussian computation for the Riemann sum S_n = (b/n) sum_k W_{kb/n}^2:
     E exp(-beta S_n) = det(I + 2*beta*(b/n)*C)^(-1/2),  C_ij = min(t_i, t_j),
     which converges to the Laplace transform as n -> infinity;
 (2) Monte Carlo simulation of Brownian paths (trapezoidal rule for the integral).
"""
import numpy as np

rng = np.random.default_rng(20261002)


def exact(beta, b):
    return 1.0 / np.sqrt(np.cosh(b * np.sqrt(2.0 * beta)))


def riemann_det(beta, b, n=1500):
    t = b * np.arange(1, n + 1) / n
    C = np.minimum.outer(t, t)
    sign, logdet = np.linalg.slogdet(np.eye(n) + 2.0 * beta * (b / n) * C)
    return np.exp(-0.5 * logdet)


def monte_carlo(beta, b, n=400, paths=40000):
    h = b / n
    W = np.cumsum(rng.standard_normal((paths, n)) * np.sqrt(h), axis=1)
    W2 = W ** 2
    integral = h * (W2[:, :-1].sum(axis=1) + 0.5 * W2[:, -1])  # trapezoid, W_0 = 0
    vals = np.exp(-beta * integral)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(paths)


print(" beta    b   formula    Riemann-det   Monte Carlo (s.e.)")
for beta, b in [(0.5, 1.0), (1.0, 1.0), (2.0, 1.5), (0.3, 3.0)]:
    f = exact(beta, b)
    d = riemann_det(beta, b)
    m, se = monte_carlo(beta, b)
    print(f"{beta:5.2f} {b:4.1f}  {f:.6f}   {d:.6f}     {m:.6f} ({se:.6f})")
    assert abs(d - f) < 2e-3 and abs(m - f) < 4 * se + 2e-3
print("all checks passed")
