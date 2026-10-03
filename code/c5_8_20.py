"""Check of Exercise 5.8.20 (power utility U(c)=c^delta, constant coefficients).

The representations (8.55), (8.56),
  G(t,y) = E int_t^T e^{-beta(s-t)} U(I(Y_s)) ds,   S(t,y) = y E int_t^T e^{-r(s-t)} Z_s I(Y_s) ds,
with Y_s = y e^{(beta-r)(s-t)} Z_s, Z_s = exp(-theta.(W_s-W_t) - |theta|^2 (s-t)/2), involve only the
one-time marginals theta.(W_s-W_t) = |theta| sqrt(s-t) xi, xi ~ N(0,1).  We evaluate them by
Gauss-Hermite quadrature in xi and Gauss-Legendre quadrature in s (exact in law), and compare with
  G = f(t) (y/delta)^{delta/(delta-1)},  S = delta G,
  f(t) = (1-e^{-k(T-t)})/k  (k != 0),   f(t) = T-t  (k = 0).
V(t,x) = e^{-beta t} G(t, Y(t,x)) (8.66), with Y(t,.) obtained by numerically inverting
X(t,y) = S(t,y)/y built from the quadrature S, is compared with e^{-beta t} f^{1-delta} x^delta.
Three parameter sets: k > 0, k = 0 (beta chosen so that k vanishes; f = T-t), k < 0.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

xi, wx = hermegauss(80); wx = wx/np.sqrt(2*np.pi)        # E h(xi) ~ sum wx h(xi)
sl, ws = leggauss(60)


def check(delta, beta, r, th, T, t, label, kzero=False):
    k = (beta - r*delta - 0.5*th**2*delta/(1-delta))/(1-delta)
    if kzero:
        assert abs(k) < 1e-14                              # k = 0 up to rounding
        f = T - t
    else:
        f = (1-np.exp(-k*(T-t)))/k
    u = 0.5*(T-t)*(sl+1); wu = 0.5*(T-t)*ws               # nodes/weights in s-t on [0,T-t]
    U, I = (lambda c: c**delta), (lambda y: (y/delta)**(1/(delta-1)))
    Z = np.exp(-th*np.sqrt(u)[:, None]*xi[None, :] - 0.5*th**2*u[:, None])
    G = lambda y: wu @ (np.exp(-beta*u) * (U(I(y*np.exp((beta-r)*u)[:, None]*Z)) @ wx))
    S = lambda y: y*(wu @ (np.exp(-r*u) * ((Z*I(y*np.exp((beta-r)*u)[:, None]*Z)) @ wx)))
    err = 0.0
    print(f"[{label}] k = {k:.6g}, f(t) = {f:.6g}")
    for y in (0.2, 0.7, 1.5):
        Gex = f*(y/delta)**(delta/(delta-1))
        g, s = G(y), S(y)
        err = max(err, abs(g/Gex-1), abs(s/(delta*Gex)-1))
        print(f"  y={y}: G quad={g:.10f} exact={Gex:.10f} | S quad={s:.10f} exact={delta*Gex:.10f}")
    for x in (0.5, 2.0):
        yy = brentq(lambda y: S(y)/y - x, 1e-6, 1e6, xtol=1e-14, rtol=1e-14)
        v, vex = np.exp(-beta*t)*G(yy), np.exp(-beta*t)*f**(1-delta)*x**delta
        err = max(err, abs(v/vex-1))
        print(f"  x={x}: V quad={v:.10f} exact={vex:.10f}")
    print(f"  max relative error {err:.2e}")
    assert err < 1e-9
    return k


th = np.sqrt(0.3**2 + 0.2**2)                               # |theta| for theta = (0.3, -0.2)
assert check(0.4, 0.07, 0.03, th, 2.0, 0.5, "k>0") > 0
d, r = 0.4, 0.03
check(d, r*d + 0.5*th**2*d/(1-d), r, th, 2.0, 0.5, "k=0", kzero=True)
assert check(0.5, 0.05, 0.03, 0.8, 2.0, 0.5, "k<0") < 0
print("all checks passed")
