"""Problem 3.5.8: numerical check of h(.;b1+b2,mu) = h(.;b1,mu) * h(.;b2,mu) for b1*b2 > 0.

For several (b1, b2, mu, t) the convolution integral int_0^t h(s;b1,mu) h(t-s;b2,mu) ds is computed
by adaptive quadrature and compared with h(t;b1+b2,mu).  A case with b1*b2 < 0 is printed to show
that the identity fails there.
"""
import numpy as np
from scipy.integrate import quad


def h(t, b, mu):
    return abs(b) / np.sqrt(2 * np.pi * t**3) * np.exp(-(b - mu * t) ** 2 / (2 * t))


def conv(t, b1, b2, mu):
    f = lambda s: h(s, b1, mu) * h(t - s, b2, mu)
    return quad(f, 0, t / 2, limit=200)[0] + quad(f, t / 2, t, limit=200)[0]


cases = [(1.0, 0.5, 0.0, 1.0), (1.0, 2.0, 0.7, 2.5), (0.3, 0.4, -1.2, 0.8),
         (-1.0, -0.5, 0.9, 1.7), (-2.0, -1.0, -0.4, 3.0), (1.0, -0.4, 0.3, 1.0)]
for b1, b2, mu, t in cases:
    lhs, rhs = h(t, b1 + b2, mu), conv(t, b1, b2, mu)
    print(f"b1={b1:5.2f} b2={b2:5.2f} mu={mu:5.2f} t={t:4.2f}  h(t;b1+b2)={lhs:.10f}  conv={rhs:.10f}"
          f"  diff={lhs - rhs:+.2e}" + ("   (b1*b2<0: no identity expected)" if b1 * b2 < 0 else ""))
