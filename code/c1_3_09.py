"""Problem 1.3.9 (Karatzas & Shreve): check of the exact formula
    E(Z - mu)^+ = exp(-mu) mu^(m+1) / m!,   m = floor(mu),   Z ~ Poisson(mu),
against direct summation of the series, and of the limit
    E(Z - mu)^+ / sqrt(mu) -> 1/sqrt(2 pi)   as mu -> infinity.
Deterministic (no random numbers)."""
from math import floor, exp, lgamma, log, sqrt, pi
import numpy as np
from scipy.stats import poisson

print("     mu      direct sum   closed form   closed/sqrt(mu)   1/sqrt(2pi)")
for mu in [0.3, 1.0, 2.7, 10.0, 50.5, 400.0, 10000.9]:
    k = np.arange(0, int(mu + 60 * sqrt(mu) + 60))
    direct = (np.maximum(k - mu, 0.0) * poisson.pmf(k, mu)).sum()
    m = floor(mu)
    closed = exp(-mu + (m + 1) * log(mu) - lgamma(m + 1))
    assert abs(direct - closed) < 1e-9 * max(1.0, closed)
    print(f"{mu:9.1f} {direct:13.8f} {closed:13.8f} {closed / sqrt(mu):15.8f} {1 / sqrt(2 * pi):13.8f}")
