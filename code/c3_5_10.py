"""Check of Exercise 3.5.10 (Karatzas & Shreve).

The density (5.12) of T_b under P^(mu) is
    h(t; b, mu) = |b| / sqrt(2 pi t^3) * exp(-(b - mu t)^2 / (2 t)),  t > 0.
We integrate exp(-alpha t) h(t; b, mu) numerically over (0, infinity) and compare
with the closed form exp(mu b - |b| sqrt(mu^2 + 2 alpha)).
"""
import numpy as np
from scipy.integrate import quad


def h(t, b, mu):
    return abs(b) / np.sqrt(2 * np.pi * t**3) * np.exp(-(b - mu * t) ** 2 / (2 * t))


worst = 0.0
for b in (1.0, -0.7, 2.5):
    for mu in (0.8, -0.5, 1.5, -2.0):
        for alpha in (0.1, 1.0, 3.0):
            num = sum(quad(lambda t: np.exp(-alpha * t) * h(t, b, mu), lo, hi, limit=200)[0]
                      for lo, hi in ((0, 1), (1, 10), (10, np.inf)))
            exact = np.exp(mu * b - abs(b) * np.sqrt(mu**2 + 2 * alpha))
            worst = max(worst, abs(num - exact))
            print(f"b={b:5.2f} mu={mu:5.2f} alpha={alpha:4.1f}  numeric={num:.10f}  formula={exact:.10f}")
print("largest absolute discrepancy:", worst)
