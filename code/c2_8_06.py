"""Check of Problem 2.8.6: P^x[W_t in dy, T_0 > t] = [p(t;x,y) - p(t;x,-y)] dy.

(1) total mass: int_0^inf p_-(t;x,y) dy = 1 - 2 P^0[W_t >= x]   (cf. (8.4));
(2) Chapman-Kolmogorov: int_0^inf p_-(s;x,z) p_-(t;z,y) dz = p_-(s+t;x,y);
(3) Monte Carlo: P^x[a < W_t < b, T_0 > t] from random-walk paths on a fine grid
    (the grid misses some crossings, so the estimate is biased slightly upwards).
"""
import numpy as np
from scipy import integrate, stats

def p(t, x, y):
    return np.exp(-(x - y) ** 2 / (2 * t)) / np.sqrt(2 * np.pi * t)

def pm(t, x, y):
    return p(t, x, y) - p(t, x, -y)

t, x = 0.7, 0.6
mass = integrate.quad(lambda y: pm(t, x, y), 0, np.inf)[0]
print("mass  quad = %.8f   1-2P0[W_t>=x] = %.8f" % (mass, 1 - 2 * stats.norm.sf(x / np.sqrt(t))))

s, y = 0.4, 1.1
ck = integrate.quad(lambda z: pm(s, x, z) * pm(t, z, y), 0, np.inf)[0]
print("C-K   quad = %.8f   p_-(s+t;x,y)  = %.8f" % (ck, pm(s + t, x, y)))

rng = np.random.default_rng(20261002)
a, b, n, N = 0.5, 1.5, 2000, 40000
hit = 0
for _ in range(4):
    w = np.full(N // 4, x); alive = np.ones(N // 4, bool)
    for _ in range(n):
        w += rng.standard_normal(N // 4) * np.sqrt(t / n)
        alive &= w > 0
    hit += np.sum(alive & (w > a) & (w < b))
exact = integrate.quad(lambda y: pm(t, x, y), a, b)[0]
print("MC          = %.4f   formula       = %.4f   (MC s.e. %.4f)" % (hit / N, exact, np.sqrt(exact * (1 - exact) / N)))
