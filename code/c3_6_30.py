"""Check of Problem 3.6.30, (6.56) at t=1, for f = indicator of [-1,1] (||f||_1 = 2).

The limit X_1 = 2||f||_1 L_1(0) has the law of ||f||_1 |Z|, Z standard normal, so
E X_1 = 2 sqrt(2/pi), E X_1^2 = 4, P[X_1 <= 2] = 2 Phi(1) - 1.
(a) exact mean of X^(n)_1 = n^{-1/2} int_0^n P[|W_s| <= 1] ds by quadrature, n large;
(b) Monte Carlo for mean, second moment and one value of the distribution function.
"""
import numpy as np
from scipy import integrate, special

lim_mean = 2*np.sqrt(2/np.pi)
print("limit: mean %.4f  second moment %.4f  P[X<=2] %.4f" % (lim_mean, 4.0, special.erf(1/np.sqrt(2))))
for n in (10**2, 10**4, 10**6):
    val, _ = integrate.quad(lambda u: special.erf(1/np.sqrt(2*n*u)), 0, 1, limit=200)
    print("n=%8d  exact E X^(n)_1 = %.4f" % (n, np.sqrt(n)*val))

rng = np.random.default_rng(0)
n, dt, paths = 400, 0.02, 4000
steps = int(n/dt)
w = np.zeros(paths); occ = np.zeros(paths)
for _ in range(steps):
    occ += (np.abs(w) <= 1)*dt
    w += np.sqrt(dt)*rng.standard_normal(paths)
x = occ/np.sqrt(n)
val, _ = integrate.quad(lambda u: special.erf(1/np.sqrt(2*n*u)), 0, 1, limit=200)
print("MC n=%d: mean %.4f (exact %.4f)  second moment %.4f  P[X<=2] %.4f"
      % (n, x.mean(), np.sqrt(n)*val, (x**2).mean(), (x <= 2).mean()))
