"""Exercise 5.8.15 (Karatzas & Shreve): numerical check of formula (8.37) and of its
corrected form for payoffs with kinks.

With tau = T - t and xi = exp(sigma*sqrt(tau)*Z + (r - sigma^2/2)*tau), Z ~ N(0,1), we compare
    LHS(x) = exp(-r*tau) * E[h(x*xi)]                       (Gauss-Hermite quadrature / closed form)
with
    RHS(x) = int_0^infty h''(q) v_q(tau, x) dq + sum_a theta_a v_a(tau, x),
where v_q is the Black-Scholes function (8.34) and theta_a is the jump of h' at the kink a.
Cases:  (1) h(x) = x^2 (C^2; closed form x^2 exp((r+sigma^2) tau));
        (2) h(x) = ((x-1)^+)^2, which is C^1 with h'' = 2*1_{x>1};
        (3) h(x) = (x-1)^+ + x^2/2, which has a kink at 1 (theta = 1): (8.37) as printed fails.
Also checks (8.34) itself against a Monte Carlo estimate of exp(-r tau) E (x xi - q)^+.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

r, sigma, tau, x = 0.05, 0.3, 0.75, 1.2

def v(q, x):
    s = sigma * np.sqrt(tau)
    rp = (np.log(x / q) + tau * (r + sigma**2 / 2)) / s
    rm = (np.log(x / q) + tau * (r - sigma**2 / 2)) / s
    return x * norm.cdf(rp) - q * np.exp(-r * tau) * norm.cdf(rm)

nodes, weights = np.polynomial.hermite_e.hermegauss(200)
weights = weights / np.sqrt(2 * np.pi)
xi = np.exp(sigma * np.sqrt(tau) * nodes + (r - sigma**2 / 2) * tau)

def lhs(h):
    return np.exp(-r * tau) * np.sum(weights * h(x * xi))

def rhs(h2, lo=0.0, kinks=()):
    val = quad(lambda q: h2(q) * v(q, x), lo, 60.0, limit=400, epsabs=1e-12)[0] if lo > 0 else \
          quad(lambda q: h2(q) * v(q, x), 1e-12, 60.0, limit=400, epsabs=1e-12)[0]
    return val + sum(th * v(a, x) for a, th in kinks)

rng = np.random.default_rng(5815)
Z = rng.standard_normal(4_000_000)
S = x * np.exp(sigma * np.sqrt(tau) * Z + (r - sigma**2 / 2) * tau)
q0 = 1.0
print("(8.34) v_q(t,x)          :", v(q0, x), "  Monte Carlo:", np.exp(-r * tau) * np.maximum(S - q0, 0).mean())

print("case 1  closed form      :", x**2 * np.exp((r + sigma**2) * tau))
print("case 1  LHS, RHS         :", lhs(lambda y: y**2), rhs(lambda q: 2.0))
print("case 2  LHS, RHS         :", lhs(lambda y: np.maximum(y - 1, 0)**2), rhs(lambda q: 2.0, lo=1.0))
l3 = lhs(lambda y: np.maximum(y - 1, 0) + y**2 / 2)
print("case 3  LHS              :", l3)
print("case 3  RHS as printed   :", rhs(lambda q: 1.0), "  (differs by v_1 =", v(1.0, x), ")")
print("case 3  RHS with kink    :", rhs(lambda q: 1.0, kinks=[(1.0, 1.0)]))
