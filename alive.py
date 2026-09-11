#!/usr/bin/env python3
"""
alive.py -- ORDER and COVERAGE have mirror-image blind spots.

duo_order.py scores ORDER: average per-step displacement, normalised by the
system's own scale. It catches the PARKED case that collapse misses -- frozen
at a large state, which passes a ||x|| test and is just as dead. Good measure,
and it finds ~5x more death than collapse did (34-36% vs 2.5-8%).

But ORDER measures MOVEMENT, and movement is not exploration. A period-2 orbit
flipping between +x and -x along a single axis scores MAXIMAL ORDER and visits
exactly two points.

rank.py scores COVERAGE: participation ratio of the covariance of visited
states. It catches cycling. But when a run truly freezes, the covariance is
floating-point noise spread evenly across dimensions, so the participation
ratio comes out HIGH -- exactly backwards.

Each is blind where the other sees. Measured at c = 0.6, dim 6, 200 seeds:

    ORDER-frozen   n= 69   median ORDER 0.000   median eff_rank 2.768  <- noise
    ORDER-alive    n=131   median ORDER 1.435   median eff_rank 1.858

    among ORDER-alive:
      eff_rank < 1.2    37.4 %   a 2-cycle on one axis
      1.2 - 2.5         52.7 %   a short cycle
      > 2.5              9.9 %   actually exploring

    joint (moving AND exploring):  13 / 200 = 6.5 %

Which recovers the trichotomy: ORDER separates settled from the rest,
eff_rank separates cycling from drifting. Three fates, in the organism.
"""
import numpy as np
from duo_order import make, mv, DIM, SEEDS, STEPS, MU, ALIVE

RANK_EXPLORING = 2.5
VAR_FLOOR = 1e-12


def run_tracked(R, WA, WB, x, c, mode, steps=STEPS, tail=8000):
    """duo_order's loop, plus a streaming second moment for coverage."""
    S, n = x.shape
    WA, WB, x = WA.copy(), WB.copy(), x.copy()
    if mode == 'clone':
        WB = WA.copy()
    duo = mode != 'solo'
    eA = np.full((S, n), 0.05); eB = np.full((S, n), 0.05)
    pvA = np.ones(S); pvB = np.ones(S)
    moved = np.zeros(S); scale = np.zeros(S); cnt = 0; half = steps // 2
    M2 = np.zeros((S, n, n)); mean = np.zeros((S, n)); tc = 0

    for t in range(steps):
        pA = mv(WA, x)
        pB = mv(WB, x) if duo else None
        xn = np.tanh(mv(R, x) + c * ((pA + pB) / 2 if duo else pA))
        dA = xn - pA; mA = np.linalg.norm(dA, axis=1)
        WA = WA + eA[:, :, None] * dA[:, :, None] * x[:, None, :]
        eA = np.clip(eA * (1 + MU * np.sign(pvA - mA)[:, None]), 1e-6, 0.5); pvA = mA
        if duo:
            dB = xn - pB; mB = np.linalg.norm(dB, axis=1)
            WB = WB + eB[:, :, None] * dB[:, :, None] * x[:, None, :]
            eB = np.clip(eB * (1 + MU * np.sign(pvB - mB)[:, None]), 1e-6, 0.5); pvB = mB
        if t >= half:
            moved += np.linalg.norm(xn - x, axis=1)
            scale = np.maximum(scale, np.linalg.norm(xn, axis=1)); cnt += 1
        if t >= steps - tail:
            M2 += x[:, :, None] * x[:, None, :]; mean += x; tc += 1
        x = xn

    order = np.where(scale > VAR_FLOOR, moved / (cnt * np.maximum(scale, VAR_FLOOR)), 0.0)
    mean /= tc
    cov = M2 / tc - mean[:, :, None] * mean[:, None, :]
    er = np.zeros(S); tr = np.zeros(S)
    for i, cv in enumerate(cov):
        lam = np.abs(np.linalg.eigvalsh((cv + cv.T) / 2))
        s1, s2 = lam.sum(), (lam ** 2).sum()
        er[i] = (s1 ** 2 / s2) if s2 > 1e-300 else 1.0
        tr[i] = np.trace(cv)
    return order, er, tr


def classify(order, er, tr):
    """The three fates, jointly. ORDER decides settled; rank decides the rest."""
    frozen = order < ALIVE
    valid = tr > VAR_FLOOR
    exploring = (~frozen) & valid & (er > RANK_EXPLORING)
    cycling = (~frozen) & ~exploring
    return frozen, cycling, exploring


if __name__ == "__main__":
    print("=" * 76)
    print("  TWO MEASURES, MIRROR-IMAGE BLIND SPOTS")
    print(f"  {SEEDS} seeds, {STEPS} steps, dim {DIM}")
    print("=" * 76)

    for c in (0.4, 0.6, 0.8, 1.0):
        R, WA, WB, x = make(0, c)
        print(f"\n  c = {c}")
        print(f"  {'condition':<18}{'settled':>10}{'cycling':>10}{'exploring':>12}"
              f"{'med ORDER':>13}{'med rank':>11}")
        print("  " + "-" * 74)
        for mode, label in (("solo", "solo"), ("clone", "duo, clones"),
                            ("diff", "duo, different")):
            o, er, tr = run_tracked(R, WA, WB, x, c, mode)
            f, cy, ex = classify(o, er, tr)
            a = ~f
            print(f"  {label:<18}{f.mean()*100:>9.1f}%{cy.mean()*100:>9.1f}%"
                  f"{ex.mean()*100:>11.1f}%{np.median(o[a]) if a.sum() else 0:>13.3f}"
                  f"{np.median(er[a & (tr>VAR_FLOOR)]) if a.sum() else 0:>11.3f}")

    print("""
  READING IT

  settled    ORDER below threshold. Stopped, wherever it parked.
  cycling    moving, but confined to a low-dimensional orbit. A 2-cycle
             scores maximal ORDER and visits two points -- ORDER alone
             cannot tell it from genuine exploration.
  exploring  moving AND spanning more than 2.5 of DIM dimensions.

  Only 'exploring' can constrain more than a sliver of the self-model,
  because every update is rank-1 along the state visited. See rank.py:
  coverage predicts generalization (rho -0.15, p 0.03); residual error
  does not (rho +0.05, p 0.53).

  NOTE the trap in the other direction: for a truly frozen run the
  covariance is floating-point noise spread evenly, so eff_rank comes out
  HIGH. Always gate on trace(cov) before reading a rank. Here the frozen
  group shows median eff_rank 2.768, higher than the alive group's 1.858,
  which is an artefact and not a finding.
""")
