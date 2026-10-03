"""Problem 5.6.11 (Karatzas & Shreve): check of formula (6.28).

The process X of (6.26) is Gaussian with mean m(t) = a(1-t/T) + b t/T (6.27)
and covariance rho(s,t) = min(s,t) - st/T (6.25).  We compare, at random
points, the density on the right-hand side of (6.28) with the multivariate
normal density having that mean vector and covariance matrix, and check that
the right-hand side of (6.28) integrates to 1 for n = 2.
"""
import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

rng = np.random.default_rng(5611)


def p(t, x, y):
    return np.exp(-(x - y) ** 2 / (2 * t)) / np.sqrt(2 * np.pi * t)


def rhs628(ts, xs, a, b, T):
    tt = np.concatenate(([0.0], ts))
    xx = np.concatenate(([a], xs))
    val = np.prod(p(np.diff(tt), xx[:-1], xx[1:]))
    return val * p(T - ts[-1], xs[-1], b) / p(T, a, b)


maxrel = 0.0
for trial in range(200):
    n = rng.integers(1, 6)
    T = rng.uniform(0.5, 3.0)
    a, b = rng.normal(size=2)
    ts = np.sort(rng.uniform(0.05 * T, 0.95 * T, size=n))
    mean = a * (1 - ts / T) + b * ts / T
    cov = np.minimum.outer(ts, ts) - np.outer(ts, ts) / T
    xs = rng.multivariate_normal(mean, cov)
    ref = multivariate_normal(mean, cov).pdf(xs)
    val = rhs628(ts, xs, a, b, T)
    maxrel = max(maxrel, abs(val - ref) / ref)
print("max relative error, (6.28) vs Gaussian density (200 trials):", maxrel)

a, b, T, ts = 0.3, -1.1, 2.0, np.array([0.5, 1.4])
tot, err = dblquad(lambda y, x: rhs628(ts, np.array([x, y]), a, b, T),
                   -12, 12, -12, 12)
print("integral of the right-hand side of (6.28), n = 2:", tot)
