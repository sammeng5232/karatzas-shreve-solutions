"""Exercise 5.6.20 (Brownian oscillator): numerical check of the closed forms.

Checked, in the underdamped, critical and overdamped cases:
  (a) e^{tA} = [[u'+alpha u, u], [-beta u, u']] against scipy.linalg.expm;
  (b) V = int_0^infty e^{sA} b b^T e^{sA^T} ds = (sigma^2/2alpha) diag(1/beta, 1) (quadrature);
  (c) the covariance e^{tau A} V against the stated formulas;
  (d) a Monte-Carlo run of the exact one-step recursion Z_{k+1} = e^{hA} Z_k + N(0, Q_h),
      started from N(0, V): empirical covariances at lag tau versus formula.
"""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad_vec

rng = np.random.default_rng(20)


def S_funcs(alpha, beta):
    D = alpha**2 - 4 * beta
    if D > 1e-12:
        g = 0.5 * np.sqrt(D)
        return (lambda t: np.sinh(g * t) / g), (lambda t: np.cosh(g * t))
    if D < -1e-12:
        w = 0.5 * np.sqrt(-D)
        return (lambda t: np.sin(w * t) / w), (lambda t: np.cos(w * t))
    return (lambda t: t), (lambda t: 1.0 + 0 * t)


for alpha, beta, sigma in [(1.0, 2.0, 0.7), (2.0, 1.0, 1.3), (3.0, 1.0, 0.5)]:
    A = np.array([[0.0, 1.0], [-beta, -alpha]])
    b = np.array([[0.0], [sigma]])
    S, Sp = S_funcs(alpha, beta)
    u = lambda t: np.exp(-alpha * t / 2) * S(t)
    up = lambda t: np.exp(-alpha * t / 2) * (Sp(t) - alpha / 2 * S(t))
    Psi = lambda t: np.array([[up(t) + alpha * u(t), u(t)], [-beta * u(t), up(t)]])

    err_a = max(np.abs(expm(t * A) - Psi(t)).max() for t in [0.0, 0.3, 1.0, 2.5, 6.0])
    V, _ = quad_vec(lambda s: expm(s * A) @ b @ b.T @ expm(s * A).T, 0, np.inf)
    V0 = sigma**2 / (2 * alpha) * np.diag([1 / beta, 1.0])
    err_b = np.abs(V - V0).max()

    tau = 0.8
    rho_formula = sigma**2 / (2 * alpha) * np.array(
        [[(up(tau) + alpha * u(tau)) / beta, u(tau)], [-u(tau), up(tau)]])
    err_c = np.abs(expm(tau * A) @ V0 - rho_formula).max()

    # exact transition over a step h = tau: Z' = e^{hA} Z + N(0, V0 - e^{hA} V0 e^{hA^T})
    n = 400_000
    E = expm(tau * A)
    Q = V0 - E @ V0 @ E.T
    Z0 = rng.multivariate_normal([0, 0], V0, size=n)
    Z1 = Z0 @ E.T + rng.multivariate_normal([0, 0], Q, size=n)
    rho_mc = Z1.T @ Z0 / n          # E[Z_{t+tau} Z_t^T]
    V_mc = Z1.T @ Z1 / n            # should again be V0 (stationarity)
    print(f"alpha={alpha}, beta={beta}, sigma={sigma}")
    print(f"  (a) max|expm - Psi|      = {err_a:.2e}")
    print(f"  (b) max|V_quad - V0|     = {err_b:.2e}")
    print(f"  (c) max|e^(tau A)V - formula| = {err_c:.2e}")
    print(f"  (d) MC  max|rho_mc - formula| = {np.abs(rho_mc - rho_formula).max():.2e},"
          f"  max|V_mc - V0| = {np.abs(V_mc - V0).max():.2e}")
