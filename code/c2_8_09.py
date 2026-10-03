"""Check of Exercise 2.8.9: density of phi(W_t) under P^x, where phi is the
2a-periodic even 'tent' map onto [0,a].  Compares the series
q(t;x,y) = sum_n p_+(t;x,y+2na) with (i) its normalisation over (0,a) and
(ii) Monte Carlo probabilities P^x[phi(W_t) in bin]."""
import numpy as np
from scipy.integrate import quad

rng = np.random.default_rng(20261002)
a, t, x = 1.3, 0.9, 0.4

def p(t, x, y):
    return np.exp(-(x - y) ** 2 / (2 * t)) / np.sqrt(2 * np.pi * t)

def q(t, x, y, N=60):
    n = np.arange(-N, N + 1)
    z = y + 2 * n * a
    return np.sum(p(t, x, z) + p(t, x, -z))

def phi(z):
    r = np.mod(z, 2 * a)
    return np.where(r <= a, r, 2 * a - r)

print("integral of q over (0,a):", quad(lambda y: q(t, x, y), 0, a)[0])
W = x + np.sqrt(t) * rng.standard_normal(4_000_000)
Y = phi(W)
edges = np.linspace(0, a, 6)
for lo, hi in zip(edges[:-1], edges[1:]):
    mc = np.mean((Y > lo) & (Y < hi))
    ex = quad(lambda y: q(t, x, y), lo, hi)[0]
    print(f"bin ({lo:.2f},{hi:.2f}): Monte Carlo {mc:.5f}   formula {ex:.5f}")
print("q(t;x,y) vs q(t;phi(x'),y) for x' = -x+4a:", q(t, x, 0.7), q(t, -x + 4 * a, 0.7))
