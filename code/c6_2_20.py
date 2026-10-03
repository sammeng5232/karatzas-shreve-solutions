"""Numerical check for Exercise 6.2.20 of Karatzas & Shreve.

nu is a Poisson random measure on (0,inf)^2 with intensity db * dl / sqrt(2 pi l^3).
L_b^{eps} = total length of the jumps of size l < eps up to "time" b.  We simulate the
jumps of size in [d0, eps) exactly (Poisson number, sizes by inverse c.d.f.) and add the
deterministic mean of the neglected jumps of size < d0 (their variance is O(d0^{3/2})).

Checked: (i)  E L_b^eps = b sqrt(2 eps/pi),  Var L_b^eps = (2b/3) sqrt(eps^3/(2 pi))  (2.44);
         (ii) the Laplace transform (2.47) of V_b(eps) = L_b^eps - b sqrt(2 eps/pi);
         (iii) the pathwise identity (2.45) (truncated at d0) by quadrature on one path;
         (iv) the law of large numbers (2.46): L_b/b for a large b.
"""
import numpy as np
from scipy import integrate

rng = np.random.default_rng(6220)
eps, b, d0 = 0.5, 2.0, 1e-5
c = np.sqrt(2 / np.pi)


def sample_jumps(bb, size_paths):
    """jump sizes in [d0, eps) for each of size_paths independent paths on (0, bb]."""
    rate = bb * c * (d0 ** -0.5 - eps ** -0.5)
    n = rng.poisson(rate, size_paths)
    u = rng.random(n.sum())
    ell = (d0 ** -0.5 - u * (d0 ** -0.5 - eps ** -0.5)) ** -2.0
    return n, ell


npaths = 20000
n, ell = sample_jumps(b, npaths)
idx = np.repeat(np.arange(npaths), n)
small = b * c * np.sqrt(d0)                      # mean of jumps of size < d0
L = np.bincount(idx, weights=ell, minlength=npaths) + small
m1 = np.sqrt(2 * eps / np.pi)
m2 = (2 / 3) * np.sqrt(eps ** 3 / (2 * np.pi))
se = L.std() / np.sqrt(npaths)
print("(i)  mean  L_b: MC %.5f +- %.5f   formula b*sqrt(2eps/pi)        = %.5f" % (L.mean(), 2 * se, b * m1))
print("     var   L_b: MC %.5f            formula (2b/3)sqrt(eps^3/2pi) = %.5f" % (L.var(), b * m2))
q1 = integrate.quad(lambda l: l / np.sqrt(2 * np.pi * l ** 3), 0, eps)[0]
q2 = integrate.quad(lambda l: l * l / np.sqrt(2 * np.pi * l ** 3), 0, eps)[0]
print("     quadrature of int l mu(dl), int l^2 mu(dl): %.6f %.6f   closed forms %.6f %.6f" % (q1, q2, m1, m2))

V = L - b * m1
for alpha in (0.5, 2.0, 5.0):
    psi = integrate.quad(lambda l: (1 - np.exp(-alpha * l)) / np.sqrt(2 * np.pi * l ** 3), 0, eps)[0]
    rhs = np.exp(alpha * b * m1 - b * psi)
    x = np.exp(-alpha * V)
    print("(ii) alpha=%.1f  E exp(-alpha V_b): MC %.5f +- %.5f   (2.47) = %.5f"
          % (alpha, x.mean(), 2 * x.std() / np.sqrt(npaths), rhs))

# (iii) one path: sum of jumps in [d0,eps) = -eps*N^eps + int_0^eps N^l dl, with N^l counting
# jumps of size in [max(l,d0), inf); jumps >= eps cancel, so we may take N^eps = 0.
n1, e1 = sample_jumps(b, 1)
e1 = np.sort(e1)
grid = np.linspace(0, eps, 2_000_001)
Nl = e1.size - np.searchsorted(e1, grid, side="left")     # number of jumps >= grid point
print("(iii) sum of jumps %.6f   int_0^eps N^l dl (trapezoid) %.6f" % (e1.sum(), integrate.trapezoid(Nl, grid)))

# (iv) LLN
B = 2000.0
nB, eB = sample_jumps(B, 1)
print("(iv) L_b/b at b=%g: %.5f   sqrt(2eps/pi) = %.5f" % (B, (eB.sum() + B * c * np.sqrt(d0)) / B, m1))
