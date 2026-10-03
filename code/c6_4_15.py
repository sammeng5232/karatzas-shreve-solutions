"""Numerical check of Exercise 6.4.15 (ii) of Karatzas & Shreve.

Claim:  E^b[exp{-beta L_{T_a}(b) - alpha L_{T_a}(0)}] = 1 / f_2(a, alpha; a+|b|, beta)
        with f_2(a, alpha; a+|b|, beta) = (1 + alpha a)(1 + beta |b|) + beta a,
for b < 0 < a, where L is Brownian local time in the book's normalisation
(occupation density = 2L).

Check: simple random walk on the lattice h*Z, h = 1/n, time step h^2.  The number of
visits V(x) to a site x before reaching a approximates  2 L_{T_a}(x) / h,  i.e.
L ~ h V / 2.  The discrete Feynman-Kac functional
    v(x) = E^x exp{-(h/2) (beta V(b) + alpha V(0))}
solves  v(x) = exp(-c(x)) (v(x+h) + v(x-h))/2,  v(a) = 1, and v(x) = v(b) for x < b
(from below b the walk returns to b with probability one, at no cost).  The linear
system is solved exactly and v(b) is compared with 1/f_2 as h -> 0.
"""
import numpy as np
from scipy.linalg import solve_banded


def f2(t1, g1, t2, g2):
    # (4.22), (4.23) with n = 2, expanded by the recursion itself
    f0 = 1.0
    f1 = lambda t, g: f0 + t * g * f0
    return f1(t2 - t1, g2) + t1 * (g1 * f1(t2 - t1, g2) + g2 * f0)


def walk_value(a, b, alpha, beta, n):
    h = 1.0 / n
    ia, ib = int(round(a * n)), int(round(b * n))   # lattice indices of a, b
    m = ia - ib                                     # unknowns v(ib), ..., v(ia-1)
    cost = np.zeros(m)
    cost[0] += beta * h / 2.0                       # site b
    cost[-ib] += alpha * h / 2.0                    # site 0
    disc = np.exp(-cost)
    ab = np.zeros((3, m))
    ab[1, :] = 1.0
    ab[0, 1:] = -0.5 * disc[:-1]                    # coefficient of v(x+h)
    ab[2, :-1] = -0.5 * disc[1:]                    # coefficient of v(x-h)
    ab[1, 0] = 1.0 - 0.5 * disc[0]                  # v(b-h) = v(b)
    rhs = np.zeros(m)
    rhs[-1] = 0.5 * disc[-1]                        # v(a) = 1
    return solve_banded((1, 1), ab, rhs)[0]


if __name__ == "__main__":
    np.random.seed(0)
    worst = 0.0
    for (a, b, alpha, beta) in [(1.0, -1.0, 1.0, 1.0), (0.5, -2.0, 3.0, 0.7),
                                (2.0, -0.25, 0.4, 5.0), (1.5, -1.5, 2.0, 2.0)]:
        exact = 1.0 / f2(a, alpha, a + abs(b), beta)
        closed = 1.0 / ((1 + alpha * a) * (1 + beta * abs(b)) + beta * a)
        vals = [walk_value(a, b, alpha, beta, n) for n in (100, 1000, 10000)]
        print(f"a={a}, b={b}, alpha={alpha}, beta={beta}")
        print(f"   1/f_2 = {exact:.8f}   closed form = {closed:.8f}")
        print("   random walk, n = 100, 1000, 10000: " + ", ".join(f"{v:.8f}" for v in vals))
        worst = max(worst, abs(vals[-1] - exact), abs(exact - closed))
    print(f"largest discrepancy at n = 10000: {worst:.2e}")
    assert worst < 1e-3
