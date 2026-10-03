"""Problem 2.8.18: arc-sine law for gamma_t = sup{0 <= s <= t; W_s = 0} under P^0.

Checks P^0[gamma_t <= s] = (2/pi) arcsin sqrt(s/t) in two independent ways:
 (a) simulated Brownian paths on a grid of mesh dt, and the time theta_t of the
     maximum on the same paths (Problems 8.8 / 8.17);
 (b) the identity P^0[gamma_t < s] = P[sqrt(t-s)|Z2| < sqrt(s)|Z1|] (Markov property
     at time s and (8.4)), by Monte Carlo over two independent standard normals.

On the grid estimate of gamma_t.  The last sign change of the grid skeleton
misses every zero of the path that lies after it, so it underestimates gamma_t,
with a bias of order sqrt(dt) (column "skeleton", shown only to exhibit this bias).
The column "MC gamma" removes it: conditionally on the skeleton, the path on
[j dt, (j+1) dt] is a Brownian bridge from a = W_{j dt} to b = W_{(j+1) dt}, and
for a b > 0 it hits 0 with probability exp(-2 a b / dt), independently over j.
We sample these events and take the last step containing a zero.  For s on the
grid the event {gamma_t <= s} = {no zero in (s, t]} is then simulated exactly, so
"MC gamma" should agree with the arc-sine value within about 3 standard errors
(column "s.e."); the mean uses the midpoint of the last step (error <= dt / 2).
"""
import numpy as np

rng = np.random.default_rng(20818)
t, n, paths, chunk = 2.0, 4000, 20000, 2000
dt = t / n
gamma, skeleton, theta = [], [], []
for _ in range(paths // chunk):
    W = np.cumsum(rng.standard_normal((chunk, n)) * np.sqrt(dt), axis=1)
    W = np.concatenate([np.zeros((chunk, 1)), W], axis=1)
    prod = W[:, :-1] * W[:, 1:]
    change = prod <= 0                                  # skeleton changes sign on step j
    hit = change | (rng.random((chunk, n)) < np.exp(-2 * np.maximum(prod, 0) / dt))
    for flags, out in ((change, skeleton), (hit, gamma)):
        last = n - 1 - np.argmax(flags[:, ::-1], axis=1)   # step 0 is always flagged (W_0 = 0)
        out.append((last + 0.5) * dt)
    theta.append(np.argmax(W, axis=1) * dt)
gamma, skeleton, theta = map(np.concatenate, (gamma, skeleton, theta))
Z1, Z2 = rng.standard_normal((2, 10**6))

worst = 0.0
print("   s     arcsine   MC gamma    s.e.   skeleton  MC theta  two-normal")
for s in (0.1, 0.4, 1.0, 1.5, 1.9):
    exact = 2 / np.pi * np.arcsin(np.sqrt(s / t))
    se = np.sqrt(exact * (1 - exact) / paths)
    est = np.mean(gamma <= s)
    worst = max(worst, abs(est - exact) / se)
    print(f"{s:5.2f}  {exact:8.4f}  {est:8.4f}  {se:7.4f}  {np.mean(skeleton <= s):8.4f}"
          f"  {np.mean(theta <= s):8.4f}"
          f"  {np.mean(np.sqrt(t - s) * abs(Z2) < np.sqrt(s) * abs(Z1)):8.4f}")
se_mean = gamma.std() / np.sqrt(paths)
print(f"E gamma_t: exact {t / 2:.4f}   MC {gamma.mean():.4f} (s.e. {se_mean:.4f})"
      f"   skeleton {skeleton.mean():.4f}")
worst = max(worst, abs(gamma.mean() - t / 2) / se_mean)
print(f"largest deviation of the bridge-corrected estimates: {worst:.2f} standard errors")
assert worst < 3.5, "bridge-corrected estimate of gamma_t disagrees with the arc-sine law"
