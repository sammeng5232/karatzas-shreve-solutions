"""Exercise 6.5.4 of Karatzas & Shreve: numerical check.

With g(t,r) the right-hand side of (5.15) (so v(t,x) = g(t,|x|)) we check
 (a) the closed forms used in the solution,
       g_r  = 2(r-t)Phi(a) + 2 e^{2r}(r+t)(1-Phi(c)),
       g_rr = 2Phi(a) - 4 sqrt(t) phi(a) + 2 e^{2r}(2r+2t+1)(1-Phi(c)),
       g_t  = -2 sqrt(t) phi(a) + (1-2(r-t))Phi(a) + e^{2r}(1-Phi(c)),
     a=(r-t)/sqrt(t), c=(r+t)/sqrt(t), against finite differences of g;
 (b) g_t = g_rr/2 - g_r   (equation (5.16)),  g_r(t,0) = 0,  g_r > 0 for r > 0
     (hence the Hamilton-Jacobi-Bellman equation (5.17));
 (c) v(t,x) = E^x Z_t^2 and E^x Z_t^2 <= E^x X_t^2 for two other drifts with
     values in [-1,1], by an Euler scheme (Monte Carlo).
"""
import numpy as np
from scipy.stats import norm

def g(t, r):
    a, c = (r - t) / np.sqrt(t), (r + t) / np.sqrt(t)
    return (0.5 + np.sqrt(t / (2 * np.pi)) * (r - t - 1) * np.exp(-(r - t) ** 2 / (2 * t))
            + ((r - t) ** 2 + t - 0.5) * norm.cdf(a)
            + np.exp(2 * r) * (r + t - 0.5) * norm.sf(c))

def g_r(t, r):
    a, c = (r - t) / np.sqrt(t), (r + t) / np.sqrt(t)
    return 2 * (r - t) * norm.cdf(a) + 2 * np.exp(2 * r) * (r + t) * norm.sf(c)

def g_rr(t, r):
    a, c = (r - t) / np.sqrt(t), (r + t) / np.sqrt(t)
    return (2 * norm.cdf(a) - 4 * np.sqrt(t) * norm.pdf(a)
            + 2 * np.exp(2 * r) * (2 * r + 2 * t + 1) * norm.sf(c))

def g_t(t, r):
    a, c = (r - t) / np.sqrt(t), (r + t) / np.sqrt(t)
    return (-2 * np.sqrt(t) * norm.pdf(a) + (1 - 2 * (r - t)) * norm.cdf(a)
            + np.exp(2 * r) * norm.sf(c))

T, Rr = np.meshgrid(np.linspace(0.05, 4, 80), np.linspace(0.0, 5, 101), indexing="ij")
h = 1e-4
e_r = np.abs(g_r(T, Rr) - (g(T, Rr + h) - g(T, Rr - h)) / (2 * h)).max()
e_rr = np.abs(g_rr(T, Rr) - (g_r(T, Rr + h) - g_r(T, Rr - h)) / (2 * h)).max()
e_t = np.abs(g_t(T, Rr) - (g(T + h, Rr) - g(T - h, Rr)) / (2 * h)).max()
print("(a) max |g_r  - FD| :", e_r)
print("    max |g_rr - FD| :", e_rr)
print("    max |g_t  - FD| :", e_t)
assert max(e_r, e_rr, e_t) < 1e-6

pde = np.abs(g_t(T, Rr) - 0.5 * g_rr(T, Rr) + g_r(T, Rr)).max()
gr0 = np.abs(g_r(T[:, 0], 0.0)).max()
grmin = g_r(T[:, 1:], Rr[:, 1:]).min()
print("(b) max |g_t - g_rr/2 + g_r| :", pde)
print("    max |g_r(t,0)| :", gr0)
print("    min g_r over r>0 :", grmin, "(must be > 0)")
assert pde < 1e-12 and gr0 < 1e-12 and grmin > 0
g_small, g_large = g(1e-6, 1.3), g(60.0, 1.3)
print("    g(1e-6, 1.3) =", g_small, " (x^2 = 1.69);  g(60, 1.3) =", g_large, " (limit 1/2)")
assert abs(g_small - 1.69) < 1e-4 and abs(g_large - 0.5) < 1e-4

rng = np.random.default_rng(654)
N, STEPS = 40000, 100
def second_moment(drift, x, t, n=N, steps=STEPS):
    dt = t / steps
    X = np.full(n, x)
    for k in range(steps):
        X = X + drift(k * dt, X) * dt + np.sqrt(dt) * rng.standard_normal(n)
    return (X ** 2).mean(), (X ** 2).std() / np.sqrt(n)

OPT = "optimal u(x) = -sgn-type"
drifts = {OPT: lambda s, y: np.where(y <= 0, 1.0, -1.0),
          "b = -tanh(x)": lambda s, y: -np.tanh(y),
          "b = -x cut to [-1,1], t<0.5": lambda s, y: np.clip(-y, -1, 1) * (s < 0.5)}
print("(c) Monte Carlo (Euler, n=%d, %d steps), second moment at time t started from x:" % (N, STEPS))
for (x, t) in [(0.0, 1.0), (1.3, 0.7), (-2.0, 2.0)]:
    v = g(t, abs(x))
    print("    x=%5.2f t=%4.2f  v(t,x)=%.4f" % (x, t, v))
    for name, b in drifts.items():
        m, se = second_moment(b, x, t)
        if name == OPT:
            # the Euler scheme for the discontinuous drift carries a bias of order dt
            tol = 4 * se + 3 * t / STEPS
            ok = abs(m - v) < tol
            verdict = "|MC - v| = %.4f < %.4f" % (abs(m - v), tol)
        else:
            ok = m - 4 * se > v
            verdict = "MC - 4 se = %.4f > v" % (m - 4 * se)
        print("        %-30s %.4f +- %.4f   %s : %s" % (name, m, se, verdict, "PASS" if ok else "FAIL"))
        assert ok, (name, x, t)
print("ALL CHECKS PASSED")
