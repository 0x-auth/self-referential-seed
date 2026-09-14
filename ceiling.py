#!/usr/bin/env python3
"""
ceiling.py -- the review's experiment 2, plus the control it implies.

REVIEW'S POINT: "random W at 1.049 shows it beat noise. It doesn't separate
learning rule from coverage." Correct. The named control:
    fit W by least squares to the trajectory it actually saw, test off-traj.
      batch ~ online   -> coverage is binding
      batch << online  -> the rank-1 online update is the culprit

ONE MORE CONTROL, NOT IN THE REVIEW:
    the law is x -> tanh(Rx + cWx). NONLINEAR. The organism's model W@x is
    LINEAR. So there is a CEILING: the best linear predictor of this world,
    fit on the whole state space with full knowledge. If the oracle also
    scores ~0.84 then "learned a slice, not a law" was measuring tanh.

RESULTS (300 seeds, 40000 steps, c=0.6, dim 6):

    predictor       off-trajectory     on-trajectory
    online                 0.84291         1.57e-16
    batch                  4.56968         8.26e-14
    oracle                 0.19265               --
    random                 1.23602               --

VERDICT 1 -- neither branch the review offered. batch is 5.4x WORSE than
online, not equal and not better. The visited data has eff_rank ~1.8 of 6,
so the least-squares solve is ill-conditioned: it fits the sliver perfectly
(on-traj 8e-14) and explodes everywhere else. Coverage is binding
EMPHATICALLY -- the data is so degenerate that an OPTIMAL fit to it is five
times worse than the online rule. The small rank-1 steps are acting as
implicit regularization and protecting the organism from its own data.

VERDICT 2 -- the ceiling is 0.193, well below random 1.236. So there WAS a
law a linear model could learn, and the organism reached 0.843. Restated
properly: it closed 37.7% of the achievable range. The negative stands and
is now measured against the right baseline instead of against noise.

NOTE: off-trajectory numbers shift with probe count and seed (random W reads
1.049 at 400 probes, 1.236 at 500). Compare within a run, not across.
"""
import numpy as np
from rank import population, mv, DIM, STEPS, SEEDS, MU, C

TRAJ_KEEP = 6000


def live_keep(R, W, x, c, steps=STEPS, keep=TRAJ_KEEP):
    """rank.py's loop, retaining the last `keep` (x_t, x_t+1) pairs."""
    S, n = x.shape
    eta = np.full((S, n), 0.05); prev = np.ones(S)
    Xs = np.zeros((S, keep, n)); Ys = np.zeros((S, keep, n))
    errs = np.zeros((S, keep))
    for t in range(steps):
        p = mv(W, x)
        xn = np.tanh(mv(R, x) + c * p)
        e = xn - p
        mag = np.linalg.norm(e, axis=1)
        k = t - (steps - keep)
        if k >= 0:
            Xs[:, k], Ys[:, k], errs[:, k] = x, xn, mag
        W = W + eta[:, :, None] * e[:, :, None] * x[:, None, :]
        eta = np.clip(eta * (1 + MU * np.sign(prev - mag)[:, None]), 1e-6, 0.5)
        prev = mag
        x = xn
    return W, Xs, Ys, errs


def lstsq_fit(X, Y, ridge=1e-10):
    """Per-seed least squares W minimising ||W x - y||. X,Y are (S, N, n)."""
    S, N, n = X.shape
    out = np.zeros((S, n, n))
    for s in range(S):
        A = X[s].T @ X[s] + ridge * np.eye(n)
        out[s] = np.linalg.solve(A, X[s].T @ Y[s]).T
    return out


def off_err(W_pred, W_fb, R, c, n_probe=500, scale=0.6, seed=7):
    """Error at random unvisited states. World fixed by W_fb; predictor varies."""
    g = np.random.default_rng(seed)
    S, n, _ = W_pred.shape
    tot = np.zeros(S)
    for _ in range(n_probe):
        xr = np.clip(g.normal(0, scale, (S, n)), -1, 1)
        p = mv(W_pred, xr)
        xn = np.tanh(mv(R, xr) + c * mv(W_fb, xr))
        tot += np.linalg.norm(xn - p, axis=1)
    return tot / n_probe


if __name__ == "__main__":
    print("=" * 76)
    print("THE REVIEW'S EXPERIMENT 2, AND THE CEILING IT IMPLIES")
    print("=" * 76)
    print(f"  {SEEDS} seeds, {STEPS} steps, c = {C}, dim {DIM}")
    print()

    R, W0, x0 = population(SEEDS)
    W_on, Xs, Ys, errs = live_keep(R, W0.copy(), x0.copy(), C)

    W_batch = lstsq_fit(Xs, Ys)

    g = np.random.default_rng(123)
    Xr = np.clip(g.normal(0, 0.6, (SEEDS, 4000, DIM)), -1, 1)
    Yr = np.tanh(np.einsum('sij,skj->ski', R, Xr)
                 + C * np.einsum('sij,skj->ski', W_on, Xr))
    W_oracle = lstsq_fit(Xr, Yr)
    W_rand = g.normal(0, 0.1, W_on.shape)

    res = {t: off_err(Wp, W_on, R, C) for t, Wp in
           [("online", W_on), ("batch", W_batch),
            ("oracle", W_oracle), ("random", W_rand)]}

    on_online = errs.mean(axis=1)
    on_batch = np.linalg.norm(
        Ys - np.einsum('sij,skj->ski', W_batch, Xs), axis=2).mean(axis=1)

    print(f"  {'predictor':<12}{'off-trajectory':>18}{'on-trajectory':>18}")
    print("  " + "-" * 48)
    print(f"  {'online':<12}{np.median(res['online']):>18.5f}"
          f"{np.median(on_online):>18.3e}")
    print(f"  {'batch':<12}{np.median(res['batch']):>18.5f}"
          f"{np.median(on_batch):>18.3e}")
    print(f"  {'oracle':<12}{np.median(res['oracle']):>18.5f}{'--':>18}")
    print(f"  {'random':<12}{np.median(res['random']):>18.5f}{'--':>18}")

    b, o = np.median(res['batch']), np.median(res['online'])
    orc, rnd = np.median(res['oracle']), np.median(res['random'])

    print()
    print("=" * 76)
    print("VERDICT 1 -- coverage, or the online rule?")
    print("=" * 76)
    print(f"  batch {b:.5f}   vs   online {o:.5f}   ratio {b/o:.4f}")
    print("  Neither branch the review offered. batch is far WORSE.")
    print("  The visited data is near rank-deficient, so the optimal fit to it")
    print("  is ill-conditioned: perfect on-trajectory, catastrophic off it.")
    print("  COVERAGE IS BINDING, emphatically. And the rank-1 online rule is")
    print("  acting as implicit regularization -- it is PROTECTING the organism")
    print("  from its own degenerate data.")

    print()
    print("=" * 76)
    print("VERDICT 2 -- was there a law a linear model could learn?")
    print("=" * 76)
    print(f"  best possible linear predictor : {orc:.5f}")
    print(f"  the organism achieved          : {o:.5f}")
    print(f"  untrained random               : {rnd:.5f}")
    print()
    print(f"  achievable range {orc:.4f} .. {rnd:.4f}")
    print(f"  the organism closed {(rnd-o)/(rnd-orc)*100:.1f} % of it")
    print()
    print("  The ceiling is well below random, so there WAS a law. The negative")
    print("  stands -- restated as a fraction of the achievable range rather")
    print("  than as a percentage over noise.")
