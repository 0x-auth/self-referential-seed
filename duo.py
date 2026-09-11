#!/usr/bin/env python3
"""
duo.py -- can two vanish?

isolate.py measured the shunya trap: a lone organism can reach error zero
by becoming nothing, and cannot tell from inside which fixed point it hit.

Question: does an OTHER close that trap?

Setup. One shared state x, one shared law R. Both agents predict x, and BOTH
predictions feed back into the law:

    p_A = W_A x        p_B = W_B x
    x_next = tanh(R x + c * (p_A + p_B)/2)
    e_A = x_next - p_A      e_B = x_next - p_B     (each learns from its own)

The averaging keeps total feedback magnitude the same as the solo case, so
solo at coupling c is a fair control for duo at coupling c.

THE CONTROL THAT MATTERS: a clone duo, W_A == W_B at t=0. If two-ness alone
rescues, clones are rescued too. If only a DIFFERENT other rescues, the
difference is doing the work.

Measured:
  - collapse rate: fraction of runs ending at |x| < 1e-6 (the empty fixed point)
  - residual error: does the arrow of time terminate, or does the other keep it running
  - divergence |W_A - W_B|: do they stay two, or merge into one

RESULT (see README): the trap NARROWS but does not CLOSE. A different other
cuts the collapse rate 2.28x (6.33% -> 2.78%, n=900, Fisher p=0.0004), and
clones reproduce solo to every digit -- so it is difference, not number. But
the residual error still falls to ~1e-16 in all three conditions: the other
does not keep surprise alive. They co-settle while remaining different.
"""
import numpy as np

DIM, STEPS, SEEDS, MU = 6, 40000, 200, 0.02


def population(n, dim=DIM, rho=1.05, clone=False):
    R = np.zeros((n, dim, dim)); WA = np.zeros((n, dim, dim))
    WB = np.zeros((n, dim, dim)); x = np.zeros((n, dim))
    for s in range(n):
        g = np.random.default_rng(s)
        r = g.normal(0, 1, (dim, dim)) / np.sqrt(dim)
        R[s] = r * (rho / max(abs(np.linalg.eigvals(r)).max(), 1e-9))
        WA[s] = g.normal(0, 0.1, (dim, dim))
        WB[s] = WA[s].copy() if clone else np.random.default_rng(s + 100000).normal(0, 0.1, (dim, dim))
        x[s] = g.normal(0, 0.5, dim)
    return R, WA, WB, x


def mv(M, v):
    return np.einsum('sij,sj->si', M, v)


def run(R, WA, WB, x, c, duo=True, steps=STEPS):
    S, n = x.shape
    eA = np.full((S, n), 0.05); eB = np.full((S, n), 0.05)
    pvA = np.ones(S); pvB = np.ones(S)
    errA = np.zeros((S, steps))
    for t in range(steps):
        pA = mv(WA, x)
        pB = mv(WB, x) if duo else None
        drive = c * ((pA + pB) / 2 if duo else pA)
        xn = np.tanh(mv(R, x) + drive)

        dA = xn - pA
        mA = np.linalg.norm(dA, axis=1); errA[:, t] = mA
        WA = WA + eA[:, :, None] * dA[:, :, None] * x[:, None, :]
        eA = np.clip(eA * (1 + MU * np.sign(pvA - mA)[:, None]), 1e-6, 0.5); pvA = mA

        if duo:
            dB = xn - pB
            mB = np.linalg.norm(dB, axis=1)
            WB = WB + eB[:, :, None] * dB[:, :, None] * x[:, None, :]
            eB = np.clip(eB * (1 + MU * np.sign(pvB - mB)[:, None]), 1e-6, 0.5); pvB = mB
        x = xn
    return WA, WB, x, errA


def report(tag, WA, WB, x, errA, duo):
    nx = np.linalg.norm(x, axis=1)
    fin = errA[:, -2000:].mean(axis=1)
    collapsed = nx < 1e-6
    solved = fin < 1e-9
    div = (np.linalg.norm((WA - WB).reshape(len(x), -1), axis=1)
           if duo else np.full(len(x), np.nan))
    print(f"  {tag:<22}{collapsed.mean()*100:>12.1f} %{solved.mean()*100:>12.1f} %"
          f"{np.median(fin):>16.3e}{np.median(nx):>12.4f}"
          f"{(np.median(div) if duo else float('nan')):>14.4f}")
    return collapsed.mean(), solved.mean(), np.median(fin)


if __name__ == "__main__":
    print("=" * 92)
    print("CAN TWO VANISH?   shared state, shared law, both predictions feed back")
    print("=" * 92)
    print(f"  {SEEDS} seeds, {STEPS} steps each")
    print()
    print(f"  {'condition':<22}{'collapsed':>14}{'err<1e-9':>14}{'median err':>16}"
          f"{'median |x|':>12}{'|W_A-W_B|':>14}")
    print("  " + "-" * 90)

    summary = {}
    for c in [0.4, 0.6, 0.8, 1.0]:
        print(f"\n  c = {c}")
        R, WA, WB, x = population(SEEDS)
        a, b, cc, d = run(R, WA.copy(), WB.copy(), x.copy(), c, duo=False)
        s_solo = report("solo", a, b, cc, d, False)

        R, WA, WB, x = population(SEEDS, clone=True)
        a, b, cc, d = run(R, WA.copy(), WB.copy(), x.copy(), c, duo=True)
        s_clone = report("duo, clones", a, b, cc, d, True)

        R, WA, WB, x = population(SEEDS, clone=False)
        a, b, cc, d = run(R, WA.copy(), WB.copy(), x.copy(), c, duo=True)
        s_duo = report("duo, different", a, b, cc, d, True)

        summary[c] = (s_solo, s_clone, s_duo)

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  {'c':>6}{'solo collapse':>18}{'clone collapse':>18}{'different collapse':>22}")
    for c, (so, cl, du) in summary.items():
        print(f"  {c:>6.1f}{so[0]*100:>16.1f} %{cl[0]*100:>16.1f} %{du[0]*100:>20.1f} %")
    print()
    print(f"  {'c':>6}{'solo err':>18}{'clone err':>18}{'different err':>22}")
    for c, (so, cl, du) in summary.items():
        print(f"  {c:>6.1f}{so[2]:>18.3e}{cl[2]:>18.3e}{du[2]:>22.3e}")
    print()
    print("  If clone ~ solo and different differs, the rescue comes from")
    print("  DIFFERENCE, not from number. If all three match, an other changes nothing.")
