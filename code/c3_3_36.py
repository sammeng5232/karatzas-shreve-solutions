"""Exercise 3.3.36 (Karatzas & Shreve): numerical check of the moment formula

    E(M_t^p) = Gamma((d - p(d-2))/2) / ( (2t)^{p(d-2)/2} Gamma(d/2) ),   M_t = R_t^{-(d-2)},

for the d-dimensional Bessel process R started at 0 (R_t = |W_t|, W_t ~ N(0, t I_d)).
For p = 1 the right-hand side is strictly decreasing in t, which is why M is not a
martingale.  The exponents p are chosen with 2p(d-2) < d, so that the Monte Carlo
estimator has finite variance.  Prints Monte Carlo estimate, standard error and formula.
"""
import numpy as np
from scipy.special import gamma

rng = np.random.default_rng(3336)
n = 2_000_000


def formula(d, p, t):
    q = p * (d - 2)
    return gamma((d - q) / 2) / ((2 * t) ** (q / 2) * gamma(d / 2))


worst = 0.0
for d, p in [(3, 1.0), (3, 1.4), (4, 0.9), (5, 0.7), (6, 0.5)]:
    for t in (1.0, 2.5, 10.0):
        w = np.sqrt(t) * rng.standard_normal((n, d))
        m = np.sum(w * w, axis=1) ** (-p * (d - 2) / 2)
        est, se = m.mean(), m.std(ddof=1) / np.sqrt(n)
        ex = formula(d, p, t)
        z = (est - ex) / se
        worst = max(worst, abs(z))
        print(f"d={d} p={p:.1f} t={t:5.1f}  MC={est:.6f} (se {se:.6f})  formula={ex:.6f}  z={z:+.2f}")
print(f"largest |z| = {worst:.2f}")
print("d=3: E(1/R_1) formula =", formula(3, 1.0, 1.0), " sqrt(2/pi) =", np.sqrt(2 / np.pi))
