"""Exercise 5.2.28 (ii) of Karatzas & Shreve.

Checks that X_t = (1+t)^2 [xi + a (W_t - t)] solves the Ito equation
    dX = [2 X/(1+t) - a (1+t)^2] dt + a (1+t)^2 dW,   X_0 = xi,
by comparing it, along the same Brownian paths, with the Euler-Maruyama
scheme on finer and finer grids (the pathwise error must shrink), and by
comparing the mean and variance at time T with the closed-form values
    E X_T = (1+T)^2 (xi - a T),   Var X_T = a^2 (1+T)^4 T.
"""
import numpy as np

rng = np.random.default_rng(52028)
a, xi, T, npaths = 0.7, 1.3, 2.0, 4000
nfine = 2 ** 12
dWf = rng.normal(0.0, np.sqrt(T / nfine), size=(npaths, nfine))
WT = dWf.sum(axis=1)
exact = (1 + T) ** 2 * (xi + a * (WT - T))

for n in (2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12):
    dW = dWf.reshape(npaths, n, nfine // n).sum(axis=2)
    dt = T / n
    X = np.full(npaths, xi)
    for k in range(n):
        t = k * dt
        X = X + (2 * X / (1 + t) - a * (1 + t) ** 2) * dt + a * (1 + t) ** 2 * dW[:, k]
    print(f"n = {n:5d}:  mean |Euler - formula| at T = {np.mean(np.abs(X - exact)):.5f}")

print("mean of formula   :", exact.mean(), "  closed form:", (1 + T) ** 2 * (xi - a * T))
print("variance, formula :", exact.var(), "  closed form:", a ** 2 * (1 + T) ** 4 * T)
print("Euler mean, var   :", X.mean(), X.var())
