"""Numerical check of formula (3.37) of Problem 4.3.11 (Karatzas & Shreve).

Mixing distribution F: masses 1/2 at y = 1 and y = 2, so that
    v(t, x) = 0.5*exp(x - t/2) + 0.5*exp(2x - 2t),
and A(t, b) solves v(t, A(t, b)) = b (explicit: a quadratic in exp(x)).

(a) Right-hand side of (3.37).
(b) Quadrature of P[W_s >= A(s)] + (1/b) * int_{-inf}^{A(s)} v(s, a) p(s; 0, a) da,
    i.e. (3.36) integrated against the law of W_s.
(c) Monte Carlo estimate of P[W_t >= A(t, b) for some t >= s] on a time grid, with the
    standard continuity correction (boundary lowered by 0.5826*sqrt(dt)) and horizon T.
(d) Linear case F = point mass at mu: (3.37) against the classical formula
    P[sup_u (B_u - beta*u) >= c] = exp(-2*beta*c) integrated against the law of W_s.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

b, s = 3.0, 1.0
ys = np.array([1.0, 2.0])
ws = np.array([0.5, 0.5])


def v(t, x):
    return sum(w * np.exp(y * x - 0.5 * y * y * t) for w, y in zip(ws, ys))


def A(t):
    # 0.5 e^{-2t} u^2 + 0.5 e^{-t/2} u - b = 0,  u = e^x
    p, q = 0.5 * np.exp(-2 * t), 0.5 * np.exp(-t / 2)
    u = 2 * b / (q + np.sqrt(q * q + 4 * p * b))  # stable root
    return np.log(u)


As = A(s)
assert abs(v(s, As) - b) < 1e-10

rhs = 1 - norm.cdf(As / np.sqrt(s)) + sum(
    w * norm.cdf(As / np.sqrt(s) - y * np.sqrt(s)) for w, y in zip(ws, ys)) / b
integral, _ = quad(lambda a: v(s, a) * norm.pdf(a, scale=np.sqrt(s)), -np.inf, As)
quadr = 1 - norm.cdf(As / np.sqrt(s)) + integral / b
print(f"(a) formula (3.37)              : {rhs:.5f}")
print(f"(b) quadrature of (3.36)        : {quadr:.5f}")

rng = np.random.default_rng(20261002)
n, dt, T = 40000, 0.01, 60.0
W = np.sqrt(s) * rng.standard_normal(n)
hit = W >= As
t = s
corr = 0.5826 * np.sqrt(dt)
for _ in range(int(round((T - s) / dt))):
    t += dt
    W += np.sqrt(dt) * rng.standard_normal(n)
    hit |= W >= A(t) - corr
mc = hit.mean()
se = np.sqrt(mc * (1 - mc) / n)
print(f"(c) Monte Carlo                 : {mc:.5f}  (s.e. {se:.5f})")

beta, gamma = 0.7, 0.4
lin = 1 - norm.cdf(gamma / np.sqrt(s) + beta * np.sqrt(s)) + np.exp(-2 * beta * gamma) * norm.cdf(
    gamma / np.sqrt(s) - beta * np.sqrt(s))
c0 = gamma + beta * s
i2, _ = quad(lambda a: np.exp(-2 * beta * (c0 - a)) * norm.pdf(a, scale=np.sqrt(s)), -np.inf, c0)
print(f"(d) linear: (3.39) {lin:.6f}  vs classical {1 - norm.cdf(c0 / np.sqrt(s)) + i2:.6f}")
