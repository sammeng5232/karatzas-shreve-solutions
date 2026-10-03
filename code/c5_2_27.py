"""Check of Exercise 5.2.27 (Karatzas & Shreve).

Claim: the solution of dX = (sqrt(1+X^2) + X/2) dt + sqrt(1+X^2) dW, X_0 = x, is
    X_t = sinh(asinh(x) + t + W_t).
We (a) integrate the equation by the Euler-Maruyama scheme along simulated Brownian
paths and compare pathwise with the closed form at time T, for two step sizes (the
error must shrink with the step), and (b) compare the Monte-Carlo mean of the Euler
scheme with E X_T = ((x+r) e^{3T/2} + (x-r) e^{-T/2})/2, r = sqrt(1+x^2).
"""
import numpy as np

rng = np.random.default_rng(5227)
x0, T, npaths, nfine = 0.7, 1.0, 20000, 4096
dW = rng.standard_normal((npaths, nfine)) * np.sqrt(T / nfine)
WT = dW.sum(axis=1)
exact = np.sinh(np.arcsinh(x0) + T + WT)

for n in (256, 4096):
    inc = dW.reshape(npaths, n, nfine // n).sum(axis=2)
    h = T / n
    X = np.full(npaths, x0)
    for k in range(n):
        s = np.sqrt(1.0 + X * X)
        X = X + (s + 0.5 * X) * h + s * inc[:, k]
    err = np.abs(X - exact)
    print(f"Euler n={n:5d}: mean |X_T - closed form| = {err.mean():.5f}, "
          f"median = {np.median(err):.5f}")

r = np.sqrt(1 + x0 * x0)
mean_formula = 0.5 * ((x0 + r) * np.exp(1.5 * T) + (x0 - r) * np.exp(-0.5 * T))
se = X.std() / np.sqrt(npaths)
print(f"E X_T: Euler MC = {X.mean():.4f} (s.e. {se:.4f}), "
      f"closed-form sample mean = {exact.mean():.4f}, formula = {mean_formula:.4f}")
