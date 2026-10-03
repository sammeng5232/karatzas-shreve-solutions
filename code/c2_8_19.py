"""Exercise 2.8.19 (Karatzas & Shreve): numerical checks of the quadrivariate density
h(a,b,s,u) = -2ab^2 (2 pi u(s-u)(t-s))^{-3/2} exp{-s b^2/(2u(s-u)) - a^2/(2(t-s))}.

(1) The key lemma: for b>0, a<0, 0<sigma<T,
      int_{-inf}^0 k_sigma(b-y) q_{T-sigma}(y,a) dy = int_0^sigma k_v(b) k_{T-v}(-a) dv,
    with k_v(x) = x (2 pi v^3)^{-1/2} exp(-x^2/2v), q_tau(y,a) = p_tau(y-a) - p_tau(y+a).
(2) Integrating h over a<0<b gives 1/(2 pi s sqrt(s(t-s))), whose integral over 0<u<s<t is 1/2 = P[W_t<0].
(3) Monte Carlo (random walk approximation, seeded): P[W_t<0, gamma_t<=s0, theta_t<=u0]
    against the integral of the (gamma,theta)-marginal du ds / (2 pi s sqrt(s(t-s))).
"""
import numpy as np
from scipy import integrate

def p(v, x): return np.exp(-x * x / (2 * v)) / np.sqrt(2 * np.pi * v)
def k(v, x): return x / v * p(v, x)
def q(tau, y, a): return p(tau, y - a) - p(tau, y + a)

print("(1) lemma: lhs, rhs")
for (b, a, T, sig) in [(1.0, -0.5, 2.0, 0.7), (0.3, -1.2, 1.0, 0.9), (2.0, -0.1, 3.0, 1.5)]:
    lhs = integrate.quad(lambda y: k(sig, b - y) * q(T - sig, y, a), -np.inf, 0)[0]
    rhs = integrate.quad(lambda v: k(v, b) * k(T - v, -a), 0, sig)[0]
    print(f"   b={b}, a={a}, T={T}, sigma={sig}:  {lhs:.10f}  {rhs:.10f}")

t = 1.0
def h(a, b, s, u):
    return (-2 * a * b * b / (2 * np.pi * u * (s - u) * (t - s)) ** 1.5
            * np.exp(-s * b * b / (2 * u * (s - u)) - a * a / (2 * (t - s))))
# integrate a and b numerically, then (s,u); substitute to tame endpoint singularities
def ab_mass(s, u):
    ia = integrate.quad(lambda a: -a * np.exp(-a * a / (2 * (t - s))), -np.inf, 0)[0]
    ib = integrate.quad(lambda b: b * b * np.exp(-s * b * b / (2 * u * (s - u))), 0, np.inf)[0]
    return 2 * ia * ib / (2 * np.pi * u * (s - u) * (t - s)) ** 1.5
print("(2) mass of h in (a,b) at fixed (s,u), against 1/(2 pi s sqrt(s(t-s))):")
for (s_, u_) in [(0.5, 0.2), (0.9, 0.05), (0.3, 0.29)]:
    print(f"   s={s_}, u={u_}:  {ab_mass(s_, u_):.10f}  {1 / (2 * np.pi * s_ * np.sqrt(s_ * (t - s_))):.10f}")
print(f"    integral of the latter over 0<u<s<t: "
      f"{integrate.quad(lambda phi: 1 / np.pi, 0, np.pi / 2)[0]:.6f} (s = t sin^2 phi; exact 0.5)")

rng = np.random.default_rng(20261002)
n, paths, s0, u0 = 2000, 20000, 0.6, 0.25
W = np.cumsum(rng.standard_normal((paths, n)) * np.sqrt(t / n), axis=1)
W = np.concatenate([np.zeros((paths, 1)), W], axis=1)
theta = np.argmax(W, axis=1) * t / n
sign_change = (W[:, :-1] * W[:, 1:] <= 0)
gamma = (n - 1 - np.argmax(sign_change[:, ::-1], axis=1)) * t / n
mc = np.mean((W[:, -1] < 0) & (gamma <= s0) & (theta <= u0))
exact = integrate.dblquad(lambda u, s: 1 / (2 * np.pi * s * np.sqrt(s * (t - s))),
                          0, s0, 0, lambda s: min(s, u0))[0]
print(f"(3) P[W_t<0, gamma<= {s0}, theta<= {u0}]: Monte Carlo {mc:.4f}, formula {exact:.4f}")
