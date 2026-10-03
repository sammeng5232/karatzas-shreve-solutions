"""Exercise 4.3.13: P^0[W_t >= beta*t + gamma for some t >= 0] = exp(-2*beta*gamma).

Two independent checks (seeded, numpy/scipy only):
 (a) the s -> 0 limit of the right-hand side of (3.39), and the integral over (0, infinity) of
     the first-passage density (3.5.12) with b = gamma, mu = -beta, against exp(-2*beta*gamma);
 (b) a Monte Carlo estimate on a fine grid over [0, T], with the Brownian-bridge correction for
     crossings between grid points, against formula (3.41)-type finite-horizon value obtained
     from (3.39)-style reflection:  P[cross before T] = 1 - Phi((gamma+beta*T)/sqrt(T))
     + exp(-2*beta*gamma) * Phi((beta*T-gamma)/sqrt(T)), and against exp(-2*beta*gamma).
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

rng = np.random.default_rng(4313)
for beta, gamma in [(0.5, 1.0), (1.0, 0.7), (2.0, 0.3)]:
    exact = np.exp(-2 * beta * gamma)
    s = 1e-8
    lim339 = 1 - norm.cdf(gamma / np.sqrt(s) + beta * np.sqrt(s)) \
        + exact * norm.cdf(gamma / np.sqrt(s) - beta * np.sqrt(s))
    dens = lambda t: gamma / np.sqrt(2 * np.pi * t**3) * np.exp(-(gamma + beta * t)**2 / (2 * t))
    integ = quad(dens, 0, 1)[0] + quad(dens, 1, np.inf)[0]
    # Monte Carlo on [0, T]
    T, n, m = 20.0, 2000, 20000
    dt = T / n
    hit = np.zeros(m, dtype=bool)
    x = np.zeros(m)                      # X_t = W_t - beta t
    logsurv = np.zeros(m)                # log-probability of no crossing between grid points
    for _ in range(n):
        y = x + np.sqrt(dt) * rng.standard_normal(m) - beta * dt
        hit |= y >= gamma
        ok = (x < gamma) & (y < gamma)
        # bridge of the drifted motion between x and y exceeds gamma w.p. exp(-2(g-x)(g-y)/dt)
        pr = np.where(ok, np.exp(-2 * (gamma - x) * (gamma - y) / dt), 0.0)
        logsurv += np.log1p(-np.minimum(pr, 1 - 1e-16))
        x = y
    mc = np.mean(np.where(hit, 1.0, 1 - np.exp(logsurv)))
    finT = 1 - norm.cdf((gamma + beta * T) / np.sqrt(T)) + exact * norm.cdf((beta * T - gamma) / np.sqrt(T))
    print(f"beta={beta}, gamma={gamma}: exp(-2bg)={exact:.6f}  lim(3.39)={lim339:.6f}  "
          f"int(5.12)={integ:.6f}  MC[0,{T:g}]={mc:.4f} (se~{np.sqrt(exact*(1-exact)/m):.4f})  finite-T={finT:.6f}")
