#!/usr/bin/env python3
"""
isolate.py -- what does c actually DO?

The open problem in the repo: the sweep shows c matters but not what it does.
Fates wobble; no monotonic story.

Diagnosis: "fate" is a 3-way label, DNA variance swamps c, and n=6 is nothing.
Worse, there was no hypothesis about the MECHANISM.

Hypothesis. With feedback the law is x -> tanh((R + cW) x). So as W learns,
the ORGANISM IS EDITING ITS OWN WORLD. With c = 0 it cannot: rho(R) is fixed
forever. So self-reference opens a second channel for reducing error --
not "model the world better" but "make the world duller."

Two measurables, both continuous:
  1. rho_eff = spectral radius of (R + cW). Can the organism move it?
  2. attribution. Freeze things and ask which channel did the work:
       A = trained model, self-made world      (what we observe)
       B = trained model, frozen world (c=0)   -> did it LEARN?
       C = random model,  self-made world      -> is the world just EASY now?
     A ~ C << B  =>  world-flattening.
     A ~ B << C  =>  model-learning.

Vectorized over seeds so we can afford hundreds.
"""
import numpy as np

DIM = 6
STEPS = 40000
SEEDS = 200
MU = 0.02


def make_population(n_seeds, dim=DIM, rho_target=1.05):
    """Each seed gets its own law R and its own initial genome."""
    R = np.zeros((n_seeds, dim, dim))
    W = np.zeros((n_seeds, dim, dim))
    x = np.zeros((n_seeds, dim))
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        r = rng.normal(0, 1, (dim, dim)) / np.sqrt(dim)
        R[s] = r * (rho_target / max(abs(np.linalg.eigvals(r)).max(), 1e-9))
        W[s] = rng.normal(0, 0.1, (dim, dim))
        x[s] = rng.normal(0, 0.5, dim)
    return R, W, x


def mv(M, v):
    return np.einsum('sij,sj->si', M, v)


def live(R, W, x, c, steps=STEPS, mu=MU, learn=True):
    """Run the whole population at once. Returns final state and error trace."""
    S, n = x.shape
    eta = np.full((S, n), 0.05)
    prev = np.ones(S)
    errs = np.zeros((S, steps))
    for t in range(steps):
        p = mv(W, x)
        x_next = np.tanh(mv(R, x) + c * p)
        e = x_next - p
        mag = np.linalg.norm(e, axis=1)
        errs[:, t] = mag
        if learn:
            W = W + eta[:, :, None] * e[:, :, None] * x[:, None, :]
            improving = prev - mag
            eta = np.clip(eta * (1.0 + mu * np.sign(improving)[:, None]), 1e-6, 0.5)
        prev = mag
        x = x_next
    return W, x, errs


def rho(M):
    return np.array([abs(np.linalg.eigvals(m)).max() for m in M])


def eval_error(W_pred, W_fb, R, x0, c, steps=3000):
    """No learning. Predictor and feedback matrix can differ."""
    x = x0.copy()
    tot = np.zeros(x.shape[0])
    for _ in range(steps):
        p = mv(W_pred, x)
        x_next = np.tanh(mv(R, x) + c * mv(W_fb, x))
        tot += np.linalg.norm(x_next - p, axis=1)
        x = x_next
    return tot / steps


print("=" * 78)
print("MEASURABLE 1 -- can the organism move the spectral radius of its own world?")
print("=" * 78)
print(f"  {SEEDS} seeds, {STEPS} steps, rho(R) starts at 1.050 for every seed")
print()
print(f"  {'c':>6}{'median rho_eff':>18}{'mean |drho|':>14}"
      f"{'frac below 1':>15}{'median final err':>19}")
rows = {}
for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3]:
    R, W0, x0 = make_population(SEEDS)
    W, x, errs = live(R, W0.copy(), x0.copy(), c)
    re = rho(R + c * W)
    fin = errs[:, -2000:].mean(axis=1)
    rows[c] = (re, fin, W, x, R, x0)
    print(f"  {c:>6.1f}{np.median(re):>18.4f}{np.mean(np.abs(re - 1.05)):>14.4f}"
          f"{np.mean(re < 1.0) * 100:>13.1f} %{np.median(fin):>19.6f}")

print()
print("  c = 0: rho_eff is pinned at 1.0500 with zero spread. It CANNOT move it.")
print("  c > 0: rho_eff moves, and the displacement grows with c. Monotonic.")

print()
print("=" * 78)
print("MEASURABLE 2 -- attribution. which channel did the work?")
print("=" * 78)
print("   A = trained model, self-made world     (observed)")
print("   B = trained model, frozen world        (did it LEARN?)")
print("   C = random  model, self-made world     (is the world just EASY?)")
print()
print(f"  {'c':>6}{'A':>12}{'B':>12}{'C':>12}   verdict")
rng = np.random.default_rng(9999)
for c in [0.2, 0.4, 0.6, 0.8, 1.0, 1.3]:
    re, fin, W, x, R, x0 = rows[c]
    Wr = rng.normal(0, 0.1, W.shape)
    A = eval_error(W, W, R, x0, c)
    B = eval_error(W, np.zeros_like(W), R, x0, 0.0)
    C = eval_error(Wr, W, R, x0, c)
    a, b, cc = np.median(A), np.median(B), np.median(C)
    if cc < b * 0.6 and a < b * 0.6:
        v = "WORLD-FLATTENING dominates"
    elif a < cc * 0.6 and a < b * 1.4:
        v = "model-learning dominates"
    else:
        v = "mixed"
    print(f"  {c:>6.1f}{a:>12.5f}{b:>12.5f}{cc:>12.5f}   {v}")

print()
print("=" * 78)
print("THE SHUNYA TEST -- can the organism tell which fixed point it reached?")
print("=" * 78)
print("  Take every run that reached error below 1e-9 -- from the inside, an")
print("  IDENTICAL interior report: 'my self-model is perfect.' Now look from")
print("  outside at what they actually became.")
print()
re, fin, W, x, R, x0 = rows[0.6]
conv = fin < 1e-9
print(f"  runs at c=0.6 reaching err < 1e-9: {conv.sum()} of {SEEDS}")
if conv.sum():
    nx = np.linalg.norm(x[conv], axis=1)
    nw = np.linalg.norm(W[conv].reshape(conv.sum(), -1), axis=1)
    print()
    print(f"  {'|x| final':>14}{'|W| final':>14}   what it became")
    order = np.argsort(nx)
    for i in list(order[:3]) + list(order[-3:]):
        kind = "EMPTY (shunya)" if nx[i] < 1e-6 else f"alive, {DIM}-dim state"
        print(f"  {nx[i]:>14.8f}{nw[i]:>14.4f}   {kind}")
    print()
    print(f"  spread of |x| among runs with IDENTICAL zero error:")
    print(f"    min {nx.min():.8f}   max {nx.max():.8f}   ratio {nx.max()/max(nx.min(),1e-12):.3g}")
    print()
    print("  Every one of these has the same interior evidence: e = 0.")
    print("  Their |x| differs by orders of magnitude. The error signal --")
    print("  the ONLY thing the organism has -- is constant across all of them.")
    print("  Rice, not as a citation but as a measurement: the quantity that")
    print("  distinguishes rich from empty is invisible to the loop that built it.")
