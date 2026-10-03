"""Check of Problem 2.8.2 (Karatzas & Shreve).

(8.3): under P^0, M_t, |W_t| and M_t - W_t all have density sqrt(2/(pi t)) exp(-b^2/2t) on (0,inf).
We simulate Brownian paths on a fine grid and compare, for each of the three variables, the empirical
mean with sqrt(2t/pi) and the empirical tail P[X >= b] with 2 P[W_t >= b].  (The discrete-time maximum
is biased downward by about 0.58*sqrt(dt).)
(8.3)': we compare P[max|W_u| >= b], 4 P[W_t >= b] and sqrt(t/2pi)(4/b)exp(-b^2/2t).
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20261002)
t, n, paths = 2.0, 2000, 20000
dt = t / n
W = np.zeros(paths); M = np.zeros(paths); A = np.zeros(paths)
for _ in range(n):
    W += np.sqrt(dt) * rng.standard_normal(paths)
    np.maximum(M, W, out=M)
    np.maximum(A, np.abs(W), out=A)

print("t =", t, " grid bias of a discrete maximum ~", round(0.5826 * np.sqrt(dt), 4))
print("exact mean sqrt(2t/pi) = %.4f" % np.sqrt(2 * t / np.pi))
for name, X in (("M_t", M), ("|W_t|", np.abs(W)), ("M_t-W_t", M - W)):
    print("%-8s mean %.4f" % (name, X.mean()), end="  tails:")
    for b in (0.5, 1.5, 3.0):
        print("  b=%.1f: %.4f (exact %.4f)" % (b, (X >= b).mean(), 2 * norm.sf(b / np.sqrt(t))), end="")
    print()
print("(8.3)':")
for b in (0.5, 1.5, 3.0, 4.0):
    lhs = (A >= b).mean()
    mid = 4 * norm.sf(b / np.sqrt(t))
    rhs = np.sqrt(t / (2 * np.pi)) * 4 / b * np.exp(-b * b / (2 * t))
    se = np.sqrt(max(lhs * (1 - lhs), 1e-12) / paths)
    print("  b=%.1f: P[max|W|>=b] = %.4f (s.e. %.4f) <= %.4f <= %.4f : %s"
          % (b, lhs, se, mid, rhs, lhs - 3 * se <= mid <= rhs))
