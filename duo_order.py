#!/usr/bin/env python3
"""
duo_order.py  -  does a DIFFERENT other keep the arrow alive?

Builds on duo.py from 0x-auth/self-referential-seed.

duo.py already shows: a different other cuts COLLAPSE 2.28x, an identical
other does nothing. But error falls to ~1e-16 in all three conditions, so
the other does not keep surprise alive. That is the open problem.

This asks a different question. Collapse only detects x -> 0. It cannot see
the PARKED case: large state, nothing moving. A system frozen at a big fixed
point scores fine on ||x|| and is just as dead.

So score ORDER instead - is the state still distinguishable from itself
across time.

MEMORY: streaming. No trajectory is stored. Two accumulators per seed,
O(1) in steps. 200 seeds x 40000 steps stores 400 floats, not 384 MB.
"""

import numpy as np
from scipy.stats import fisher_exact

RNG = np.random.default_rng
DIM, SEEDS, STEPS, MU = 6, 200, 40000, 0.01
BURN = 0.5


def mv(M, v):
    return np.einsum("sij,sj->si", M, v)


def make(seed, c, rho0=1.05):
    rng = RNG(seed)
    R = rng.normal(0, 1, (SEEDS, DIM, DIM))
    for i in range(SEEDS):
        R[i] *= rho0 / max(abs(np.linalg.eigvals(R[i])))
    WA = rng.normal(0, 0.3, (SEEDS, DIM, DIM))
    WB = rng.normal(0, 0.3, (SEEDS, DIM, DIM))
    x = rng.normal(0, 1, (SEEDS, DIM))
    return R, WA, WB, x


def run(R, WA, WB, x, c, mode, steps=STEPS):
    """
    mode: 'solo' | 'clone' | 'diff'

    Streaming ORDER, no trajectory stored:
        moved = sum of ||x_t - x_{t-1}|| over the second half
        scale = max ||x_t|| over the second half
        ORDER = moved / (steps_counted * scale)

    Reads as average per-step displacement, normalised by the system's own
    size. Scale-free, so parked-at-a-constant and collapsed-to-zero both
    score ~0, and only genuine movement scores high.
    """
    S, n = x.shape
    WA, WB, x = WA.copy(), WB.copy(), x.copy()
    if mode == "clone":
        WB = WA.copy()
    duo = mode != "solo"

    eA = np.full((S, n), 0.05)
    eB = np.full((S, n), 0.05)
    pvA = np.ones(S)
    pvB = np.ones(S)

    moved = np.zeros(S)
    scale = np.zeros(S)
    counted = 0
    half = steps // 2
    last_err = np.zeros(S)

    for t in range(steps):
        pA = mv(WA, x)
        pB = mv(WB, x) if duo else None
        drive = c * ((pA + pB) / 2 if duo else pA)
        xn = np.tanh(mv(R, x) + drive)

        dA = xn - pA
        mA = np.linalg.norm(dA, axis=1)
        WA = WA + eA[:, :, None] * dA[:, :, None] * x[:, None, :]
        eA = np.clip(eA * (1 + MU * np.sign(pvA - mA)[:, None]), 1e-6, 0.5)
        pvA = mA

        if duo:
            dB = xn - pB
            mB = np.linalg.norm(dB, axis=1)
            WB = WB + eB[:, :, None] * dB[:, :, None] * x[:, None, :]
            eB = np.clip(eB * (1 + MU * np.sign(pvB - mB)[:, None]), 1e-6, 0.5)
            pvB = mB

        if t >= half:
            moved += np.linalg.norm(xn - x, axis=1)
            scale = np.maximum(scale, np.linalg.norm(xn, axis=1))
            counted += 1

        x = xn
        last_err = mA * mA

    order = np.where(scale > 1e-12, moved / (counted * np.maximum(scale, 1e-12)), 0.0)
    xn_final = np.linalg.norm(x, axis=1)
    wdiff = np.linalg.norm(WA - WB, axis=(1, 2)) if duo else np.full(S, np.nan)
    return order, last_err, xn_final, wdiff


ALIVE = 1e-4          # ORDER above this = still moving


if __name__ == "__main__":
    print("=" * 88)
    print("  DUO, SCORED BY ORDER")
    print(f"  {SEEDS} seeds, {STEPS} steps, dim {DIM}   (streaming, O(1) memory)")
    print("=" * 88)

    results = {}
    for c in (0.4, 0.6, 0.8, 1.0):
        R, WA, WB, x = make(0, c)
        print(f"\n  c = {c}")
        print(f"  {'condition':<18}{'collapsed':>11}{'ORDER~0':>10}"
              f"{'median ORDER':>15}{'median err':>13}{'median |x|':>12}")
        print("  " + "-" * 79)
        row = {}
        for mode, label in (("solo", "solo"),
                            ("clone", "duo, clones"),
                            ("diff", "duo, different")):
            o, e, xn, wd = run(R, WA, WB, x, c, mode)
            collapsed = float((xn < 1e-6).mean() * 100)
            frozen = float((o < ALIVE).mean() * 100)
            print(f"  {label:<18}{collapsed:>10.1f}%{frozen:>9.1f}%"
                  f"{np.median(o):>15.3e}{np.median(e):>13.3e}{np.median(xn):>12.4f}")
            row[mode] = (collapsed, frozen, o, xn)
        results[c] = row

    print("\n" + "=" * 88)
    print("  COLLAPSE vs FROZEN  -  what the old measure missed")
    print("=" * 88)
    print(f"\n  {'c':>6}{'':4}{'solo coll':>11}{'solo froz':>11}"
          f"{'':4}{'diff coll':>11}{'diff froz':>11}")
    print("  " + "-" * 62)
    for c, row in results.items():
        s, d = row["solo"], row["diff"]
        print(f"  {c:>6}{'':4}{s[0]:>10.1f}%{s[1]:>10.1f}%"
              f"{'':4}{d[0]:>10.1f}%{d[1]:>10.1f}%")

    print("\n" + "=" * 88)
    print("  SIGNIFICANCE  -  solo vs different, at each c")
    print("=" * 88)
    print(f"\n  {'c':>6}{'solo frozen':>14}{'diff frozen':>14}"
          f"{'odds ratio':>13}{'p (Fisher)':>13}")
    print("  " + "-" * 60)
    for c, row in results.items():
        so, do = row["solo"][2], row["diff"][2]
        a = int((so < ALIVE).sum()); b = SEEDS - a
        cc = int((do < ALIVE).sum()); d = SEEDS - cc
        try:
            orat, p = fisher_exact([[a, b], [cc, d]])
        except Exception:
            orat, p = float("nan"), float("nan")
        print(f"  {c:>6}{a:>10} /{SEEDS:<3}{cc:>10} /{SEEDS:<3}"
              f"{orat:>13.3f}{p:>13.4f}")

    print("""
  READING IT

  'collapsed' is the old measure: x went to zero.
  'frozen'    is the new one: the state stopped moving, wherever it parked.

  If frozen > collapsed, the old measure was undercounting - systems
  reported as healthy had in fact stopped, just not at the origin.

  If difference lowers 'frozen' the way it lowers 'collapsed', then an
  other protects the arrow and not merely the magnitude, which is the
  stronger claim.

  If it does not, then difference keeps a system from vanishing but not
  from settling - and the open problem in the README stands exactly where
  it was.
""")
