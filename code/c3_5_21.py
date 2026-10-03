"""Exercise 3.5.21 (Karatzas & Shreve): numerical check of the counterexample.

d = 1, W a standard Brownian motion under P,
    tau = inf{t in [0,1) : W_t + log(1-t) + 1 <= 0}   (tau < 1 for every path),
    X_t = 1/(1-t) on {t <= tau}, 0 otherwise,
so that int_0^infty X_t^2 dt = tau/(1-tau) < infinity for every path, and
    Z_1(X) = Z_tau,   Z_t = exp( I_t - t/(2(1-t)) ),   I_t = int_0^t dW_s/(1-s).
The solution claims  E Z_1(X) = 2(1 - Phi(1)) = 0.3173... < 1,  and more precisely
    E[ Z_1(X) ; tau < s ] = 2(1 - Phi(s^{-1/2})),  0 < s < 1
(the probability that a Brownian motion reaches -1 before time s).

Method (direct simulation under P, no change of measure).
  * Grid: step dt_k = min(1/N0, h (1 - t_k)), i.e. uniform in t on [0, 1 - 1/(N0 h)] and
    uniform in log(1-t) afterwards, up to 1 - t = EPS, so that the step stays small
    relative to 1 - t where the barrier b(t) = -1 - log(1-t) becomes steep.
  * (W, I) is simulated exactly at grid times: over [a, c] the increments are jointly
    Gaussian with Var dW = c - a, Var dI = 1/(1-c) - 1/(1-a), Cov = log((1-a)/(1-c)).
  * Crossings between grid times are accounted for by the Brownian-bridge correction
    for a linear barrier: given W at both ends of a step, both above the barrier, the
    barrier is hit in between with probability exp(-2 d_k d_{k+1} / dt_k), d = W - b.
    Instead of killing paths at random we carry the conditional survival weight q and
    add q * p * Z_tau at each step (conditional Monte Carlo), with Z_tau approximated
    by exp(I_t + (b(t) - W_t)/(1-t) - t/(2(1-t))), using W_tau = b(tau).
  * Paths are advanced step by step as a vector, and paths that have crossed are
    dropped, so memory is O(M) and the run takes well under a minute.
Outcome.  For s <= 0.95 the estimates of E[Z_1; tau < s] agree with the claimed values
to within about two standard errors (grid biases are below 0.001).  The remaining
mass 2(Phi(0.95^{-1/2}) - Phi(1)) = 0.0124 of E Z_1(X) is carried by paths with
tau >= 0.95, which have P-probability about 0.02 (printed); on them Z_1 is extremely
variable (e.g. the share 0.0022 from tau in (0.99, 1)
is typically missed by every one of 10^5 simulated paths), a heavy-tailed
contribution that plain Monte Carlo under P underestimates and whose standard error
is unreliable (different seeds give totals 0.307-0.315 with erratic error bars).
The total is therefore printed for information only (expect 0.305-0.315).  The
point is that E Z_1(X) is near 0.317, far from the value 1 that the martingale
property would require.
"""
import time
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(3521)
M = 100_000                 # paths
N0, h, EPS = 4000, 0.002, 1e-7
levels = [0.25, 0.5, 0.75, 0.9, 0.95]


def b(u):
    return -1.0 - np.log1p(-u)


t0 = time.time()
W = np.zeros(M); I = np.zeros(M); q = np.ones(M)      # state of the alive paths
a_tot = np.zeros(M)                                    # running contribution to Z_1
a_lev = {s: np.zeros(M) for s in levels}               # ... and to Z_1 1{tau < s}
S1 = {key: 0.0 for key in ['tot'] + levels}            # sums over finished paths
S2 = {key: 0.0 for key in ['tot'] + levels}            # (for the standard errors)
surv = {}                                              # P[tau > s] (grid version)


def flush(mask):
    for key, arr in [('tot', a_tot)] + [(s, a_lev[s]) for s in levels]:
        x = arr[mask]
        S1[key] += x.sum(); S2[key] += (x * x).sum()


t, k = 0.0, 0
while 1.0 - t > EPS and W.size:
    dt = min(1.0 / N0, h * (1.0 - t))
    c = (k + 1) / N0 if (k + 1) / N0 <= t + dt + 1e-12 else t + dt
    dt = c - t
    vI = 1.0 / (1.0 - c) - 1.0 / (1.0 - t)
    cov = np.log((1.0 - t) / (1.0 - c))
    m = W.size
    dW = np.sqrt(dt) * rng.standard_normal(m)
    dI = cov / dt * dW + np.sqrt(max(vI - cov * cov / dt, 0.0)) * rng.standard_normal(m)
    d0 = W - b(t)
    # Z at a crossing time tau in (t, c]: W_tau = b(tau), I_tau - I_t ~ (W_tau - W_t)/(1-t)
    Zc = np.exp(I - d0 / (1.0 - t) - 0.5 * t / (1.0 - t))
    W = W + dW; I = I + dI
    d1 = W - b(c)
    disc = d1 <= 0.0                                   # below the barrier at time c
    p = np.where(disc, 1.0, np.exp(-2.0 * d0 * np.maximum(d1, 0.0) / dt))
    contrib = q * p * Zc                               # mass with tau in (t, c]
    a_tot += contrib
    for s in levels:
        if c <= s + 1e-12:
            a_lev[s] += contrib
    q = q * (1.0 - p)
    for s in levels:
        if c <= s + 1e-12:
            surv[s] = (q.sum() - q[disc].sum()) / M
    keep = (~disc) & (q > 1e-300)
    flush(~keep)
    W, I, q, a_tot = W[keep], I[keep], q[keep], a_tot[keep]
    a_lev = {s: a_lev[s][keep] for s in levels}
    t = c; k += 1

alive = np.sum(q * np.exp(I - 0.5 * t / (1.0 - t))) / M if W.size else 0.0
flush(np.ones(W.size, dtype=bool))


def report(name, key, exact):
    mean = S1[key] / M
    se = np.sqrt((S2[key] / M - mean ** 2) / (M - 1))
    print(f"{name:22s} MC = {mean:.4f} +- {se:.4f}   claimed = {exact:.4f}")


print(f"{k} grid steps, {M} paths, {time.time()-t0:.1f} s")
print(f"E[Z; tau > 1-EPS] (neglected): {alive:.2e}")
for s in levels:
    report(f"E[Z_1; tau < {s}]", s, 2 * (1 - norm.cdf(s ** -0.5)))
print(f"P[tau >= 0.95] = {surv[0.95]:.2e}")
report("E Z_1(X) (see above)", "tot", 2 * (1 - norm.cdf(1.0)))
print("martingale value would be 1.0000")
