"""Numerical check for Problem 5.6.1 of Karatzas & Shreve.

For a 2-dimensional linear SDE dX = [A(t)X + a(t)]dt + sigma(t)dW with
time-dependent coefficients and a non-Gaussian square-integrable X_0, we compare
  (i)  the formulas (6.10), (6.11) for m(t) and rho(s,t), computed from the
       fundamental solution Phi (ODE solver + quadrature),
  (ii) a Monte Carlo estimate of m(t), rho(s,t) from an Euler scheme for (6.1),
  (iii) the solution of the ODEs (6.12), (6.13) for m and V.
"""
import numpy as np
from scipy.integrate import solve_ivp, trapezoid

rng = np.random.default_rng(20261002)


def A(t):
    return np.array([[-1.0, np.sin(3 * t)], [0.5 + t, -0.3 * t]])


def a(t):
    return np.array([1.0 + t, np.cos(t)])


def sig(t):
    return np.array([[0.8, 0.2 * t], [-0.3, 0.5 + np.sin(t) ** 2]])


s_, t_ = 0.7, 1.5

# X_0: non-Gaussian, correlated components
N = 50_000
U = rng.exponential(1.0, size=N)
X0 = np.stack([U - 0.5, 0.5 * U + rng.uniform(-1, 1, size=N)], axis=1)
m0 = np.array([0.5, 0.5])                       # exact mean of X_0
V0 = np.array([[1.0, 0.5], [0.5, 0.25 + 1.0 / 3.0]])  # exact covariance of X_0

# (i) formulas: fine grid, Phi by a high-accuracy ODE solve
grid = np.linspace(0.0, t_, 3001)
sol = solve_ivp(lambda t, y: (A(t) @ y.reshape(2, 2)).ravel(), (0, t_),
                np.eye(2).ravel(), t_eval=grid, rtol=1e-10, atol=1e-12)
Phi = sol.y.T.reshape(-1, 2, 2)
Pinv = np.linalg.inv(Phi)
b = np.array([Pinv[k] @ a(u) for k, u in enumerate(grid)])
Psi = np.array([Pinv[k] @ sig(u) for k, u in enumerate(grid)])
PP = Psi @ np.transpose(Psi, (0, 2, 1))


def trapz_to(F, k):
    return trapezoid(F[: k + 1], grid[: k + 1], axis=0)


ks, kt = np.argmin(abs(grid - s_)), len(grid) - 1
m_formula = Phi[kt] @ (m0 + trapz_to(b, kt))
rho_formula = Phi[ks] @ (V0 + trapz_to(PP, ks)) @ Phi[kt].T
V_formula = Phi[kt] @ (V0 + trapz_to(PP, kt)) @ Phi[kt].T

# (ii) Monte Carlo, Euler scheme
n = 600                      # dt = 0.0025; s_ = 0.7 is the grid point k = 280
dt = t_ / n
k_s = round(s_ / dt)
assert abs(k_s * dt - s_) < 1e-12
X = X0.copy()
Xs = None
for k in range(n):
    u = k * dt
    if k == k_s:
        Xs = X.copy()
    dW = rng.normal(0.0, np.sqrt(dt), size=(N, 2))
    X = X + (X @ A(u).T + a(u)) * dt + dW @ sig(u).T
m_mc = X.mean(axis=0)
rho_mc = (Xs - Xs.mean(axis=0)).T @ (X - m_mc) / N
V_mc = np.cov(X.T)

# (iii) the ODEs (6.12), (6.13)
def rhs(t, y):
    m, V = y[:2], y[2:].reshape(2, 2)
    return np.concatenate([A(t) @ m + a(t),
                           (A(t) @ V + V @ A(t).T + sig(t) @ sig(t).T).ravel()])


ode = solve_ivp(rhs, (0, t_), np.concatenate([m0, V0.ravel()]), rtol=1e-10, atol=1e-12)
m_ode, V_ode = ode.y[:2, -1], ode.y[2:, -1].reshape(2, 2)

np.set_printoptions(precision=4, suppress=True)
print("m(t):   formula (6.10)", m_formula, " ODE (6.12)", m_ode, " MC", m_mc)
print("V(t):   formula (6.14)\n", V_formula, "\n ODE (6.13)\n", V_ode, "\n MC\n", V_mc)
print("rho(s,t), s=0.7, t=1.5: formula (6.11)\n", rho_formula, "\n MC\n", rho_mc)
print("max |formula - ODE| :", max(abs(m_formula - m_ode).max(), abs(V_formula - V_ode).max()))
print("max |formula - MC|  :", max(abs(m_formula - m_mc).max(), abs(V_formula - V_mc).max(),
                                   abs(rho_formula - rho_mc).max()))
