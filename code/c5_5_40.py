"""Exercise 5.5.40 (Karatzas & Shreve): numerical check of the formulas of parts (i), (ii)
and of the limit (5.81), for the Ornstein-Uhlenbeck process dX = -X dt + dW on I = R.

Here (c = 0 in (5.42)) p'(x) = exp(x^2), m(dx) = 2 exp(-x^2) dx, m(I) = 2 sqrt(pi),
p(-inf) = -inf, p(+inf) = +inf, and X_t is Gaussian, so (5.80) holds.

Compared (x = 0, u = 1, z = 0.5):
  E^x T_u, E^u T_x                       formulas of part (i)        vs Monte Carlo (Euler scheme)
  E^x S_{u;x} = (p(u)-p(x)) m(I)         part (i)                    vs Monte Carlo
  E^x int_0^S 1{X_t<z} dt                (p(u)-p(x)) m((l,z)), (ii)  vs Monte Carlo
  lim_t P^x(X_t<z) = m((l,z))/m(I)       (5.81)                      vs exact N(0,1/2) limit law
The Euler scheme detects crossings only on the grid, which biases hitting times upward
by O(sqrt(dt)); agreement within a few percent is what is expected.
"""
import numpy as np
from scipy import integrate, stats

x0, u, z = 0.0, 1.0, 0.5

dp = lambda y: np.exp(y * y)                       # p'
p = lambda y: integrate.quad(dp, 0.0, y)[0]        # scale function
dm = lambda y: 2.0 * np.exp(-y * y)                # density of the speed measure
m = lambda lo, hi: integrate.quad(dm, lo, hi)[0]
mI = m(-np.inf, np.inf)

# part (i): E^x T_b  = -int_x^b (p(y)-p(x)) m(dy) + (p(b)-p(x)) m((l,b))
ExTu = -integrate.quad(lambda y: (p(y) - p(x0)) * dm(y), x0, u)[0] + (p(u) - p(x0)) * m(-np.inf, u)
#           E^u T_a  = -int_a^u (p(u)-p(y)) m(dy) + (p(u)-p(a)) m((a,r)),  with a = x0
EuTx = -integrate.quad(lambda y: (p(u) - p(y)) * dm(y), x0, u)[0] + (p(u) - p(x0)) * m(x0, np.inf)
ES = (p(u) - p(x0)) * mI
Eocc = (p(u) - p(x0)) * m(-np.inf, z)

rng = np.random.default_rng(540)
n, dt = 12000, 5e-4
sq = np.sqrt(dt)
X = np.full(n, x0)
phase = np.zeros(n, dtype=int)          # 0: before T_u, 1: between T_u and S, 2: done
Tu = np.zeros(n); S = np.zeros(n); occ = np.zeros(n)
alive = np.arange(n)
t = 0.0
while alive.size:
    Xa = X[alive]
    occ[alive] += dt * (Xa < z)
    Xa = Xa - Xa * dt + sq * rng.standard_normal(alive.size)
    X[alive] = Xa
    t += dt
    ph = phase[alive]
    hitu = (ph == 0) & (Xa >= u)
    Tu[alive[hitu]] = t
    phase[alive[hitu]] = 1
    hitx = (ph == 1) & (Xa <= x0)
    S[alive[hitx]] = t
    phase[alive[hitx]] = 2
    alive = alive[~hitx]

def show(name, formula, sample):
    se = sample.std(ddof=1) / np.sqrt(sample.size)
    print(f"{name:34s} formula {formula:8.4f}   Monte Carlo {sample.mean():8.4f} +- {se:.4f}")

show("E^x T_u", ExTu, Tu)
show("E^u T_x", EuTx, S - Tu)
show("E^x S = (p(u)-p(x)) m(I)", ES, S)
show("E^x int_0^S 1{X<z} dt", Eocc, occ)
print(f"check E^x T_u + E^u T_x - (p(u)-p(x)) m(I) = {ExTu + EuTx - ES:.2e}")
print(f"m((l,z))/m(I) = {m(-np.inf, z) / mI:.6f}   N(0,1/2) cdf at z = {stats.norm.cdf(z, scale=np.sqrt(0.5)):.6f}")
