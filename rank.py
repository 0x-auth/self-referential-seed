#!/usr/bin/env python3
"""
rank.py -- does the arrow of time carry the law?

PREDICTION, stated before running:
  The update rule is W += eta * outer(e, x). Every update is rank-1 along x.
  So the only part of W that can ever be constrained is the span of states
  the organism actually VISITS.

  A settled organism sits at x*. Span = 1 direction. Five of six dimensions
  of its self-model are never touched by evidence. It has learned a POINT.

  An unsettled organism keeps visiting new x. Span grows. It is forced to
  constrain more of W.

  Therefore: the runs that FAIL to reach zero error should generalize
  BETTER. Error is not a defect being removed. It is the only channel a
  law travels through.

RESULT: BOTH HALVES OF THAT PREDICTION ARE FALSE.
  - settled runs do NOT have eff_rank ~ 1 (median 1.79, max 4.71), and zero
    error does not even imply a static state -- many track a moving orbit
    exactly.
  - settled runs do NOT generalize worse (0.8395 vs 0.8482, p = 0.84).
  - residual error predicts generalization NOT AT ALL (rho +0.045, p 0.53).

What survived is a different variable: COVERAGE. eff_rank vs off-trajectory
error gives rho = -0.152, p = 0.032 (filtered). How much of the world got
visited predicts generalization; how much error remained does not. I had been
conflating those two; the data separates them.

And the headline negative: median off-trajectory error 0.841 against 1.049
for an UNTRAINED random W. After 40k steps of successful self-prediction the
median organism is 20% better than chance on states it never visited. It
learned a slice, not a law. No intelligence appears here as a side effect.

MEASURED per run:
  eff_rank   participation ratio of the covariance of visited states
  on_err     mean |e| over the final stretch of the trajectory
  off_err    |Wx - tanh(Rx + cWx)| at random x the organism NEVER visited
"""
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

DIM, STEPS, SEEDS, MU, C = 6, 40000, 300, 0.02, 0.6
TAIL = 8000


def population(n, dim=DIM, rho=1.05):
    R = np.zeros((n, dim, dim)); W = np.zeros((n, dim, dim)); x = np.zeros((n, dim))
    for s in range(n):
        g = np.random.default_rng(s)
        r = g.normal(0, 1, (dim, dim)) / np.sqrt(dim)
        R[s] = r * (rho / max(abs(np.linalg.eigvals(r)).max(), 1e-9))
        W[s] = g.normal(0, 0.1, (dim, dim))
        x[s] = g.normal(0, 0.5, dim)
    return R, W, x


def mv(M, v):
    return np.einsum('sij,sj->si', M, v)


def live(R, W, x, c, steps=STEPS, tail=TAIL):
    S, n = x.shape
    eta = np.full((S, n), 0.05); prev = np.ones(S)
    errs = np.zeros((S, tail))
    M2 = np.zeros((S, n, n)); mean = np.zeros((S, n)); cnt = 0
    for t in range(steps):
        p = mv(W, x)
        xn = np.tanh(mv(R, x) + c * p)
        e = xn - p
        mag = np.linalg.norm(e, axis=1)
        if t >= steps - tail:
            errs[:, t - (steps - tail)] = mag
            M2 += x[:, :, None] * x[:, None, :]
            mean += x
            cnt += 1
        W = W + eta[:, :, None] * e[:, :, None] * x[:, None, :]
        eta = np.clip(eta * (1 + MU * np.sign(prev - mag)[:, None]), 1e-6, 0.5)
        prev = mag
        x = xn
    mean /= cnt
    cov = M2 / cnt - mean[:, :, None] * mean[:, None, :]
    return W, x, errs, cov


def eff_rank(cov):
    """Participation ratio of the eigenvalue spectrum. 1 = a point, DIM = full.

    CAUTION: for a run that truly froze, trace(cov) ~ 1e-30 (and can go
    slightly negative from floating point). The ratio then divides noise by
    noise and comes out HIGH, which is backwards. Always filter on trace(cov)
    first -- 27.4% of settled runs fail that filter.
    """
    out = np.zeros(len(cov))
    for i, c in enumerate(cov):
        lam = np.abs(np.linalg.eigvalsh((c + c.T) / 2))
        s1, s2 = lam.sum(), (lam ** 2).sum()
        out[i] = (s1 ** 2 / s2) if s2 > 1e-300 else 1.0
    return out


def off_traj_error(W, R, c, n_probe=400, scale=0.6, seed=7):
    """Error at random states the organism never visited. One step, no learning."""
    g = np.random.default_rng(seed)
    S, n, _ = W.shape
    tot = np.zeros(S)
    for _ in range(n_probe):
        xr = np.clip(g.normal(0, scale, (S, n)), -1, 1)
        p = mv(W, xr)
        xn = np.tanh(mv(R, xr) + c * p)
        tot += np.linalg.norm(xn - p, axis=1)
    return tot / n_probe


if __name__ == "__main__":
    print("=" * 78)
    print("DOES ERROR CARRY THE LAW?")
    print("=" * 78)
    print(f"  {SEEDS} seeds, {STEPS} steps, c = {C}, DIM = {DIM}")
    print()

    R, W0, x0 = population(SEEDS)
    W, x, errs, cov = live(R, W0.copy(), x0.copy(), C)

    er = eff_rank(cov)
    on = errs.mean(axis=1)
    off = off_traj_error(W, R, C)
    settled = on < 1e-9

    print(f"  settled (on-trajectory err < 1e-9): {settled.sum()} / {SEEDS}")
    print(f"  unsettled:                          {(~settled).sum()} / {SEEDS}")
    print()
    print(f"  {'group':<14}{'eff_rank':>12}{'on_err':>14}{'off_err':>14}")
    print("  " + "-" * 52)
    for tag, m in [("settled", settled), ("unsettled", ~settled)]:
        if m.sum():
            print(f"  {tag:<14}{np.median(er[m]):>12.4f}{np.median(on[m]):>14.3e}"
                  f"{np.median(off[m]):>14.5f}")

    print()
    print("  PREDICTION 1: settled runs have eff_rank ~ 1 -- FALSE")
    print(f"    settled   eff_rank: min {er[settled].min():.4f}  "
          f"max {er[settled].max():.4f}  median {np.median(er[settled]):.4f}")

    print()
    print("  PREDICTION 2: settled runs generalize worse -- FALSE")
    if settled.sum() and (~settled).sum():
        a, b = np.median(off[settled]), np.median(off[~settled])
        u, p = mannwhitneyu(off[settled], off[~settled], alternative='greater')
        print(f"    settled {a:.5f}   unsettled {b:.5f}   ratio {a/b:.3f}x   p = {p:.3e}")

    print()
    print("=" * 78)
    print("MANDATORY FILTER -- drop runs whose covariance is numerical noise")
    print("=" * 78)
    tr = np.array([np.trace(c) for c in cov])
    ok = tr > 1e-12
    print(f"  runs with real state variance: {ok.sum()} / {SEEDS}")
    print(f"  settled runs failing the filter: "
          f"{np.mean(tr[settled] < 1e-20)*100:.1f} %  (their eff_rank is meaningless)")
    r, p = spearmanr(er[ok], off[ok])
    r2, p2 = spearmanr(np.log10(np.maximum(on[ok], 1e-18)), off[ok])
    print(f"  eff_rank    vs off_err  (filtered)  rho = {r:+.4f}  p = {p:.2e}")
    print(f"  log(on_err) vs off_err  (filtered)  rho = {r2:+.4f}  p = {p2:.2e}")
    print()
    print("  -> coverage predicts generalization. residual error does not.")

    print()
    print("=" * 78)
    print("DID ANY OF THEM LEARN THE LAW?")
    print("=" * 78)
    g = np.random.default_rng(1)
    base = off_traj_error(g.normal(0, 0.1, W.shape), R, C)
    print(f"  best off-trajectory error of {SEEDS} runs : {off.min():.5f}")
    print(f"  median                                 : {np.median(off):.5f}")
    print(f"  UNTRAINED random W                     : {np.median(base):.5f}")
    print()
    print(f"  the median organism is {(1 - np.median(off)/np.median(base))*100:.0f}% better"
          f" than random")
    print("  on states it never visited. It learned a slice, not a law.")
    print("  No intelligence appears here as a side effect of learning.")

    print()
    print("=" * 78)
    print("CONTROL -- is it the rank, or just the error?")
    print("=" * 78)
    u_m = (~settled) & ok
    if u_m.sum() > 20:
        r4, p4 = spearmanr(er[u_m], off[u_m])
        r5, p5 = spearmanr(np.log10(np.maximum(on[u_m], 1e-18)), off[u_m])
        print(f"    eff_rank    vs off_err  (unsettled only)  rho = {r4:+.4f}  p = {p4:.2e}")
        print(f"    log(on_err) vs off_err  (unsettled only)  rho = {r5:+.4f}  p = {p5:.2e}")
