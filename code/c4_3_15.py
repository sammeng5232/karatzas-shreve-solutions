"""Exercise 4.3.15 (finite rod): numerical check of the kernels used in the solution.

With b the length of the rod, the solution writes (3.46) as
    u(t,x) = int_0^b f(y) q(t;x,y) dy + int_0^t g(t-s) H(s,x) ds + int_0^t k(t-s) H(s,b-x) ds,
    q(t;x,y) = sum_n [p(t;x,y+2nb) - p(t;x,-y-2nb)]          (Proposition 2.8.10),
    H(s,x)   = sum_n (x+2nb) (2 pi s^3)^(-1/2) exp(-(x+2nb)^2/2s)   (formula (2.8.25)).
We compare, for constant data, with the classical Fourier sine series of the same
boundary value problem for u_t = u_xx/2 on (0,b):
  (a) f=1,g=k=0:  sum_{n odd} 4/(n pi) sin(n pi x/b) exp(-n^2 pi^2 t/(2 b^2));
  (b) f=0,g=1,k=0: (b-x)/b - sum_n 2/(n pi) sin(n pi x/b) exp(-n^2 pi^2 t/(2 b^2));
  (c) f=g=k=1:    u = 1;
and (d) with a seeded Monte Carlo simulation of (3.46) itself (Gaussian random walk with
Brownian-bridge correction for the exits at 0 and b) for f(y)=y, g(t)=cos t, k(t)=1/(1+t).
"""
import numpy as np
from scipy import integrate

b = 1.3
N = 40


def p(t, x, y):
    return np.exp(-(x - y) ** 2 / (2 * t)) / np.sqrt(2 * np.pi * t)


def q(t, x, y):
    return sum(p(t, x, y + 2 * n * b) - p(t, x, -y - 2 * n * b) for n in range(-N, N + 1))


def H(s, x):
    return sum((x + 2 * n * b) * np.exp(-(x + 2 * n * b) ** 2 / (2 * s)) for n in range(-N, N + 1)) \
        / np.sqrt(2 * np.pi * s ** 3)


def u(t, x, f, g, k):
    a1 = integrate.quad(lambda y: f(y) * q(t, x, y), 0, b, limit=200)[0]
    a2 = integrate.quad(lambda s: g(t - s) * H(s, x), 0, t, limit=200)[0]
    a3 = integrate.quad(lambda s: k(t - s) * H(s, b - x), 0, t, limit=200)[0]
    return a1 + a2 + a3


def fourier_a(t, x):
    n = np.arange(1, 400, 2)
    return np.sum(4 / (n * np.pi) * np.sin(n * np.pi * x / b) * np.exp(-n ** 2 * np.pi ** 2 * t / (2 * b * b)))


def fourier_b(t, x):
    n = np.arange(1, 4000)
    return (b - x) / b - np.sum(2 / (n * np.pi) * np.sin(n * np.pi * x / b)
                                * np.exp(-n ** 2 * np.pi ** 2 * t / (2 * b * b)))


one = lambda z: 1.0
zero = lambda z: 0.0
worst = 0.0
for t, x in [(0.05, 0.2), (0.3, 0.65), (1.0, 1.1), (2.5, 0.4)]:
    va, fa = u(t, x, one, zero, zero), fourier_a(t, x)
    vb, fb = u(t, x, zero, one, zero), fourier_b(t, x)
    vc = u(t, x, one, one, one)
    worst = max(worst, abs(va - fa), abs(vb - fb), abs(vc - 1))
    print(f"t={t:4.2f} x={x:4.2f}  (a) {va:.8f} vs {fa:.8f}   (b) {vb:.8f} vs {fb:.8f}   (c) {vc:.8f} vs 1")
print("largest deviation in (a)-(c):", worst)

# (d) Monte Carlo for non-constant data
rng = np.random.default_rng(20261002)
f = lambda y: y
g = np.cos
k = lambda t: 1 / (1 + t)
t, x = 0.8, 0.5
M, n = 50000, 800
dt = t / n
pos = np.full(M, x)
val = np.zeros(M)
alive = np.ones(M, dtype=bool)
for i in range(1, n + 1):
    new = pos + np.sqrt(dt) * rng.standard_normal(M)
    # probability that the Brownian bridge between pos and new crossed 0 resp. b
    with np.errstate(over="ignore"):
        p0 = np.where((new <= 0), 1.0, np.exp(-2 * pos * np.maximum(new, 0) / dt))
        pb = np.where((new >= b), 1.0, np.exp(-2 * (b - pos) * np.maximum(b - new, 0) / dt))
    r = rng.random(M)
    hit0 = alive & (r < p0)
    hitb = alive & ~hit0 & (r < p0 + pb)
    tau = (i - 0.5) * dt
    val[hit0] = g(t - tau)
    val[hitb] = k(t - tau)
    alive &= ~(hit0 | hitb)
    pos = new
val[alive] = f(pos[alive])
mc, se = val.mean(), val.std() / np.sqrt(M)
ex = u(t, x, f, g, k)
print(f"(d) formula {ex:.5f}   Monte Carlo {mc:.5f} +- {se:.5f}")
