"""Check of Exercise 2.8.20 (Seshadri): under P^0, U = M_t (M_t - W_t) and W_t are
independent and (2/t) U is standard exponential.

We integrate the joint density (8.2) of (W_t, M_t),
    f(a,b) = 2(2b-a)/sqrt(2 pi t^3) * exp(-(2b-a)^2/(2t)),  b >= max(a,0),
numerically over the set {a <= c, b(b-a) >= t x / 2} and compare with
    Phi(c/sqrt(t)) * exp(-x).
For fixed a the inner b-integral runs over b >= b0(a) = (a + sqrt(a^2 + 2 t x))/2.
"""
import numpy as np
from scipy import integrate
from scipy.stats import norm

def f(b, a, t):
    return 2*(2*b - a)/np.sqrt(2*np.pi*t**3)*np.exp(-(2*b - a)**2/(2*t))

def prob(c, x, t):
    def inner(a):
        b0 = 0.5*(a + np.sqrt(a*a + 2*t*x))
        b0 = max(b0, a, 0.0)
        return integrate.quad(f, b0, b0 + 12*np.sqrt(t) + abs(a), args=(a, t))[0]
    return integrate.quad(inner, -12*np.sqrt(t), c, limit=200)[0]

worst = 0.0
for t in (0.5, 1.0, 3.0):
    for c in (-1.0, 0.0, 0.7, 2.0, 12*np.sqrt(t)):
        for x in (0.0, 0.3, 1.0, 2.5):
            num = prob(c, x, t)
            exact = norm.cdf(c/np.sqrt(t))*np.exp(-x)
            worst = max(worst, abs(num - exact))
            print(f"t={t:3.1f} c={c:6.3f} x={x:3.1f}  numeric={num:.8f}  Phi(c/sqrt t)e^-x={exact:.8f}")
print("max abs difference:", worst)
