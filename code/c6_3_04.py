"""Check of Problem 6.3.4 (Karatzas & Shreve): joint law of (W_t, 2L_t(0)) under P^0.

(a) Quadrature: the densities (3.17) on R x (0,inf) and (3.18) on (0,inf)^2 integrate to 1,
    and the marginal of (3.17) in a is the N(0,t) density.
(b) Monte Carlo: for a simple random walk S with n steps, Tanaka's formula in discrete time gives
    |S_n| = sum sgn(S_{k-1}) xi_k + #{0 <= k < n : S_k = 0}, so Z_n/sqrt(n) -> 2L_1(0) jointly with
    S_n/sqrt(n) -> W_1, where Z_n is the number of visits to 0.  We compare the empirical value of
    P[S_n/sqrt n > a, Z_n/sqrt n > b] with the tail formula P[W_1 > a + b] implied by (3.17).
"""
import numpy as np
from scipy import integrate
from scipy.stats import norm

t = 1.7
f17 = lambda b, a: (b + abs(a)) / np.sqrt(2 * np.pi * t**3) * np.exp(-(b + abs(a))**2 / (2 * t))
f18 = lambda b, a: 2 * (a + b) / np.sqrt(2 * np.pi * t**3) * np.exp(-(a + b)**2 / (2 * t))
m17 = 2 * integrate.dblquad(f17, 0, np.inf, 0, np.inf)[0]
m18 = integrate.dblquad(f18, 0, np.inf, 0, np.inf)[0]
print(f"total mass (3.17): {m17:.8f}   total mass (3.18): {m18:.8f}   (both should be 1)")
for a in (-1.3, 0.4, 2.0):
    marg = integrate.quad(lambda b: f17(b, a), 0, np.inf)[0]
    print(f"a={a:5.2f}: marginal of (3.17) {marg:.8f}   N(0,t) density {norm.pdf(a, scale=np.sqrt(t)):.8f}")

rng = np.random.default_rng(634)
n, paths, chunk = 2500, 40000, 4000
S_end, Z = [], []
for _ in range(paths // chunk):
    xi = rng.integers(0, 2, size=(chunk, n), dtype=np.int8) * 2 - 1
    S = np.cumsum(xi, axis=1, dtype=np.int32)
    S_end.append(S[:, -1])
    Z.append(1 + (S[:, :-1] == 0).sum(axis=1))       # visits to 0 at times 0,...,n-1
X = np.concatenate(S_end) / np.sqrt(n)
Y = np.concatenate(Z) / np.sqrt(n)
print("P[W_1 > a, 2L_1(0) > b]:  random walk (n=2500, 40000 paths)  vs  1 - Phi(a+b)")
for a, b in ((0.0, 0.5), (0.3, 0.3), (0.5, 1.0), (1.0, 0.2)):
    emp = np.mean((X > a) & (Y > b))
    print(f"  a={a:.1f}, b={b:.1f}:  {emp:.4f}   {norm.sf(a + b):.4f}")
