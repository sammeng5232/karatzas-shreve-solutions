"""Check of Problem 5.6.6 (Karatzas & Shreve).

For a random stable A (all eigenvalues with negative real part) and random sigma:
  * V of (6.19) is computed by quadrature of e^{sA} sigma sigma^T e^{sA^T} over [0, S];
  * the residual of (6.20), A V + V A^T + sigma sigma^T, is printed;
  * V(t) of (6.14)' with V(0) = V is computed by quadrature and compared with V.
"""
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

rng = np.random.default_rng(0)
d, r = 3, 2
B = rng.standard_normal((d, d))
A = B - (np.max(np.linalg.eigvals(B).real) + 0.7) * np.eye(d)
sigma = rng.standard_normal((d, r))
Q = sigma @ sigma.T
print("eigenvalues of A:", np.linalg.eigvals(A))


def simpson(f, a, b, n):
    s = np.linspace(a, b, 2 * n + 1)
    w = np.ones(2 * n + 1); w[1:-1:2] = 4; w[2:-1:2] = 2
    return sum(wi * f(si) for wi, si in zip(w, s)) * (b - a) / (6 * n)


V = simpson(lambda s: expm(s * A) @ Q @ expm(s * A.T), 0.0, 60.0, 6000)
print("max |A V + V A^T + sigma sigma^T| :", np.abs(A @ V + V @ A.T + Q).max())
print("max |V - Lyapunov solver|         :",
      np.abs(V - solve_continuous_lyapunov(A, -Q)).max())
for t in (0.3, 1.0, 2.5):
    K = V + simpson(lambda s: expm(-s * A) @ Q @ expm(-s * A.T), 0.0, t, 1000)
    Vt = expm(t * A) @ K @ expm(t * A.T)
    print(f"t = {t}: max |V(t) - V| = {np.abs(Vt - V).max():.2e}")
