#!/usr/bin/env python3
"""sector.py -- which Lorentz sector does each fate live in?

Hypothesis (from 0x-auth/darmiyan-fs/sectors.py): the three fates are the
three Möbius/Lorentz sectors. Frozen = hyperbolic (attracting), exploring =
elliptic (neutral, on the unit circle). Tested three ways on alive.py's
population, with alive.py's own classifier:

  A. lambda1 of the state dynamics, W held fixed (Jacobian + QR)
  B. lambda1 of the full organism, learning included (nudged twin)
  C. self-model gap: how far W is from the law it lives in (R + cW)

    python3 sector.py            # c = 0.6, 200 seeds, 40000 steps
"""
import numpy as np
from scipy.stats import mannwhitneyu
from duo_order import make, mv, MU
from alive import classify, VAR_FLOOR

C, STEPS, TAIL, LY, D0, RENORM = 0.6, 40000, 8000, 4000, 1e-9, 10


def learn(R, W, x, eta, pv):
    p = mv(W, x); xn = np.tanh(mv(R, x) + C * p)
    d = xn - p; m = np.linalg.norm(d, axis=1)
    W = W + eta[:, :, None] * d[:, :, None] * x[:, None, :]
    eta = np.clip(eta * (1 + MU * np.sign(pv - m)[:, None]), 1e-6, 0.5)
    return W, xn, eta, m


def main():
    R, W, _, x = make(0, C); S, n = x.shape
    eta = np.full((S, n), 0.05); pv = np.ones(S)
    moved = np.zeros(S); scale = np.zeros(S); cnt = 0
    M2 = np.zeros((S, n, n)); mean = np.zeros((S, n)); tc = 0
    Q = np.tile(np.eye(n), (S, 1, 1)); lamA = np.zeros(S); lamB = np.zeros(S); kB = 0
    rng = np.random.default_rng(7)
    for t in range(STEPS):
        tail = t >= STEPS - LY
        if t == STEPS - LY:
            u = rng.normal(size=(S, n)); u /= np.linalg.norm(u, axis=1, keepdims=True)
            x2, W2, eta2, pv2 = x + D0 * u, W.copy(), eta.copy(), pv.copy()
        Wn, xn, eta, m = learn(R, W, x, eta, pv)
        if tail:
            J = (1 - xn**2)[:, :, None] * (R + C * W)                  # A: W frozen
            Q, Rq = np.linalg.qr(J @ Q)
            lamA += np.log(np.abs(np.diagonal(Rq, axis1=1, axis2=2))[:, 0] + 1e-300)
            W2, x2n, eta2, pv2 = learn(R, W2, x2, eta2, pv2)          # B: twin, learning on
            if (t - (STEPS - LY)) % RENORM == RENORM - 1:
                dist = np.sqrt(((x2n - xn)**2).sum(1) + ((W2 - Wn)**2).sum((1, 2)))
                lamB += np.log(dist / D0 + 1e-300); kB += 1
                f = D0 / np.maximum(dist, 1e-300)
                x2n = xn + (x2n - xn) * f[:, None]; W2 = Wn + (W2 - Wn) * f[:, None, None]
            x2 = x2n
        if t >= STEPS // 2:
            moved += np.linalg.norm(xn - x, axis=1)
            scale = np.maximum(scale, np.linalg.norm(xn, axis=1)); cnt += 1
        if t >= STEPS - TAIL:
            M2 += x[:, :, None] * x[:, None, :]; mean += x; tc += 1
        W, x, pv = Wn, xn, m
    lamA /= LY; lamB /= kB * RENORM
    order = np.where(scale > VAR_FLOOR, moved / (cnt * np.maximum(scale, VAR_FLOOR)), 0.0)
    mean /= tc; cov = M2 / tc - mean[:, :, None] * mean[:, None, :]
    er = np.zeros(S); tr = np.zeros(S)
    for i, cv in enumerate(cov):
        lam = np.abs(np.linalg.eigvalsh((cv + cv.T) / 2)); s1, s2 = lam.sum(), (lam**2).sum()
        er[i] = s1**2 / s2 if s2 > 1e-300 else 1.0; tr[i] = np.trace(cv)
    fr, cy, ex = classify(order, er, tr)
    law = R + C * W
    gap = np.linalg.norm(W - law, axis=(1, 2)) / np.linalg.norm(law, axis=(1, 2))

    print(f"c={C}, {S} seeds, {STEPS} steps\n")
    print(f"{'fate':>10} {'n':>4} | {'A: λ1 (W fixed)':>16} {'|λ1|<.01':>9} | {'B: λ1 (learning)':>17} {'|λ1|<.01':>9} | {'C: self-gap':>11}")
    for nm, msk in (("frozen", fr), ("cycling", cy), ("exploring", ex)):
        print(f"{nm:>10} {msk.sum():>4} | {np.median(lamA[msk]):>16.4f} {np.mean(np.abs(lamA[msk]) < .01)*100:>8.0f}% |"
              f" {np.median(lamB[msk]):>17.4f} {np.mean(np.abs(lamB[msk]) < .01)*100:>8.0f}% | {np.median(gap[msk]):>11.3f}")
    print()
    for lab, v in (("A", lamA), ("B", lamB), ("C", gap)):
        print(f"  {lab}: exploring vs frozen p = {mannwhitneyu(v[ex], v[fr]).pvalue:.1e}   "
              f"exploring vs cycling p = {mannwhitneyu(v[ex], v[cy]).pvalue:.1e}")


if __name__ == "__main__":
    main()
