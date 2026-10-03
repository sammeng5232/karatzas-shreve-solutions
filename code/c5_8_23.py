"""Exercise 5.8.23 (Karatzas & Shreve): power utility U(c) = c^delta, constant coefficients.

Checks, for d = 2 and both a case k != 0 and the case k = 0:
  (a) the algebraic identity  k + g(|th|^2/2 - beta + r) + g^2 |th|^2/2 = r + g |th|^2,  g = 1/(1-delta);
  (b) an Euler scheme for the closed-loop wealth equation
          dX = X[(r + g|th|^2 - 1/f(s)) ds + g th^T dW],  X_t = x,
      reproduces the closed form  X*_s = x f(s)/f(t) exp{g th^T(W_s-W_t) + g(|th|^2/2 - beta + r)(s-t)},
      and X*_T = 0;
  (c) Monte Carlo:  E int_t^T H_s C*_s ds = x   (budget equality), with H_s = e^{-r(s-t)} Z^t_s,
      and  E int_t^T e^{-beta s} (C*_s)^delta ds = e^{-beta t} f(t)^{1-delta} x^delta = V(t,x).
"""
import numpy as np

rng = np.random.default_rng(5823)


def run(beta, r, delta, b, sigma, t, T, x, label):
    theta = np.linalg.solve(sigma, b - r)
    th2 = float(theta @ theta)
    g = 1.0 / (1.0 - delta)
    k = g * (beta - r * delta - 0.5 * th2 * delta * g)

    def f(s):
        return (T - s) if abs(k) < 1e-12 else (1.0 - np.exp(-k * (T - s))) / k

    print(f"--- {label}: k = {k:.6f}")
    lhs = k + g * (0.5 * th2 - beta + r) + 0.5 * g * g * th2
    print(f"(a) identity: lhs = {lhs:.12f}, rhs = {r + g * th2:.12f}")

    n, paths = 2000, 20000
    h = (T - t) / n
    s = t + h * np.arange(n + 1)
    dW = rng.standard_normal((paths, n, 2)) * np.sqrt(h)
    B = np.concatenate([np.zeros((paths, 1)), np.cumsum(dW @ theta, axis=1)], axis=1)  # theta^T(W_s - W_t)
    Lam = np.exp(g * B + g * (0.5 * th2 - beta + r) * (s - t))
    Xstar = x * f(s) / f(t) * Lam

    # (b) Euler scheme for the closed-loop SDE on a few paths, stopped one step before T
    m = 200
    X = np.full(m, x)
    err = 0.0
    for i in range(n - 1):
        X = X * (1.0 + (r + g * th2 - 1.0 / f(s[i])) * h + g * (dW[:m, i, :] @ theta))
        err = max(err, float(np.max(np.abs(X - Xstar[:m, i + 1]) / Xstar[:m, i + 1])))
    print(f"(b) max relative error Euler vs closed form on [t, T-h]: {err:.4f};  max X*_T = {Xstar[:, -1].max():.3e}")

    # (c) budget equality and value
    C = x / f(t) * Lam
    H = np.exp(-r * (s - t)) * np.exp(-B - 0.5 * th2 * (s - t))
    w = np.full(n + 1, h); w[0] = w[-1] = h / 2          # trapezoidal weights
    budget = (H * C) @ w
    util = (np.exp(-beta * s) * C ** delta) @ w
    V = np.exp(-beta * t) * f(t) ** (1 - delta) * x ** delta
    print(f"(c) E int H C* ds = {budget.mean():.4f} +- {2 * budget.std() / np.sqrt(paths):.4f}   (x = {x})")
    print(f"    E int e^(-beta s) U(C*) ds = {util.mean():.4f} +- {2 * util.std() / np.sqrt(paths):.4f}   (V(t,x) = {V:.4f})")


sigma = np.array([[0.25, 0.05], [0.10, 0.30]])
b = np.array([0.08, 0.10])
run(beta=0.07, r=0.03, delta=0.4, b=b, sigma=sigma, t=0.25, T=1.5, x=2.0, label="k != 0")

# k = 0: choose beta = r*delta + |theta|^2 delta / (2(1-delta))
r0, d0 = 0.03, 0.4
th = np.linalg.solve(sigma, b - r0)
beta0 = r0 * d0 + 0.5 * float(th @ th) * d0 / (1 - d0)
run(beta=beta0, r=r0, delta=d0, b=b, sigma=sigma, t=0.25, T=1.5, x=2.0, label="k = 0")
