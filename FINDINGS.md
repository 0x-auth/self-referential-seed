# Findings

Measured results, in the order established. Every negative is kept, and so is
every prediction that turned out wrong.

**Corrections from external review are folded in below and credited.**

---

## Correction: coverage was load-bearing too early

A reviewer caught this and is right. The coverage result — eff_rank vs
off-trajectory error, rho = -0.152, p = 0.032 — is r² ≈ 0.023, **two percent of
variance**. At least six hypotheses were tested in this repo; Bonferroni gives
0.05/6 = 0.0083, so **p = 0.032 does not survive correction**. It was promoted
to *first open problem* on the one correlation that landed under an uncorrected
0.05. Stated plainly rather than quietly fixed.

What does survive: among **unsettled runs only**, where the measure is
well-defined, rho = **-0.361, p = 2.4e-4**. That clears Bonferroni. So the
effect looks real in the subgroup where it can be measured and is
underpowered in the pooled sample. Needs more seeds either way.

Second caveat, also from the review: **coverage measures spread, not law.** A
slow random walk scores high on ORDER and high on coverage and has learned
nothing. See the ceiling result below for a direct measure of law-learning
that does not route through coverage.

---

## The ceiling, and what the organism actually achieved

The review's experiment 2: fit `W` by least squares to the trajectory the
organism actually saw, and test off-trajectory. Its two predicted branches were
batch ≈ online (coverage binding) or batch ≪ online (the online rule is the
culprit).

**Neither happened.** Plus one control the review did not name and which
matters more: the law is `tanh(Rx + cWx)`, nonlinear, while the model `W@x` is
linear. So there is a *ceiling* — the best linear predictor of this world,
fit with full knowledge of the whole state space.

300 seeds, 40000 steps, c = 0.6, dim 6. Feedback matrix held at the trained `W`
in every case, so only the predictor varies:

| predictor | off-trajectory | on-trajectory |
|---|---|---|
| online (the organism) | 0.84291 | 1.57e-16 |
| batch (least squares on visited pairs) | **4.56968** | 8.26e-14 |
| oracle (best linear, full state space) | **0.19265** | — |
| random (untrained) | 1.23602 | — |

**Verdict 1 — coverage is binding, emphatically.** Batch is **5.4x worse** than
online, not equal and not better. The visited data has eff_rank ~1.8 of 6, so
the least-squares solve is ill-conditioned: it fits the sliver to 8e-14 and
explodes everywhere else. The data is so degenerate that an *optimal* fit to it
is five times worse than the online rule. And the corollary is a surprise — the
rank-1 online update is acting as **implicit regularization**, protecting the
organism from its own data.

**Verdict 2 — there was a law, and it largely missed it.** The oracle reaches
0.193, far below random's 1.236, so `tanh` is not the obstacle and the earlier
headline was not measuring the activation function. Restated against the right
baseline:

> The organism closed **37.7%** of the achievable range (0.193 .. 1.236).

That replaces the old framing of "20% better than random," which compared
against noise instead of against what was possible. The negative stands and is
now measured properly.

*Note:* off-trajectory numbers shift with probe count and seed — random W reads
1.049 at 400 probes and 1.236 at 500. Compare within a run, not across.

---

## A tension in the founding premise

Also from the review, and the sharpest point in it. The README promises a seed
that builds itself **with no external trainer**. The fix proposed for low
coverage — a world with a drive of its own, an inhomogeneous or non-autonomous
`R` — *is* external structure. Adding it quietly would abandon the premise
while claiming to rescue it.

The honest statement is stronger than the rescue:

> **Pure self-reference does not reach coverage.** Left alone, a
> self-predicting system co-adapts into a sliver of its own world and stays
> there — 6.5% of runs both move and explore. This is a measured limit on the
> founding premise, not a missing feature.

Anything added to raise coverage should be labelled as what it is: external
structure, and a departure from the no-trainer claim.

---

## Death was undercounted by 5x

`duo.py` only detects `x -> 0`. A system frozen at a *large* state passes any
norm test and is just as dead. `duo_order.py` scores **ORDER** instead — mean
per-step displacement normalised by the system's own scale, streaming, O(1) in
steps.

| c | solo collapsed | solo frozen | different collapsed | different frozen |
|---|---|---|---|---|
| 0.4 | 7.5% | **34.0%** | 6.5% | 31.0% |
| 0.6 | 7.0% | **35.5%** | 8.0% | 34.5% |
| 0.8 | 6.0% | **36.0%** | 7.0% | 34.5% |
| 1.0 | 2.5% | **34.5%** | 4.5% | 36.0% |

About a third of all runs had stopped. The collapse measure was seeing a fifth
of it.

**Difference does not protect the arrow.** Fisher exact, solo vs different, on
frozen: p = 0.594, 0.917, 0.834, 0.834, odds ratios near 1.0. An other keeps a
system from *vanishing* (2.28x, p = 0.0004) and does nothing to keep it
*moving*. An independent run at 4k steps gives 0.453 / 0.752 / 0.753 / 0.916 —
different numbers, same conclusion.

---

## Movement is not exploration

ORDER has the mirror-image blind spot. A period-2 orbit flipping between `+x`
and `-x` along one axis scores **maximal** ORDER and visits **two points**.
Gate it on coverage (`alive.py`, c = 0.6, dim 6, 200 seeds):

| among ORDER-alive runs | share | what it is |
|---|---|---|
| eff_rank < 1.2 | 37.4% | a 2-cycle on a single axis |
| 1.2 – 2.5 | 52.7% | a short cycle |
| > 2.5 | **9.9%** | actually exploring |

Joint — moving **and** exploring — **13 / 200 = 6.5%**.

The trap runs both ways: for a truly frozen run the covariance is
floating-point noise spread evenly, so eff_rank comes out *high*. The frozen
group reads median 2.768 against the alive group's 1.858 — an artefact, not a
finding. **Always gate a rank on `trace(cov)`**; 27.4% of settled runs fail it.

Together the two measures recover the trichotomy: ORDER separates **settled**
from the rest, coverage separates **cycling** from **drifting**.

---

## What self-reference does

With feedback the law is `x -> tanh((R + cW) x)`, so as `W` learns the organism
is editing its own world. At `c = 0` it cannot: rho pinned at 1.0500 with
**zero spread across 200 seeds**. Every `c > 0` moves it, monotonically.

| c | median rho(R+cW) | B: own model, frozen world |
|---|---|---|
| 0.0 | 1.0500 | — |
| 0.2 | 1.2476 | 0.07153 |
| 0.6 | 1.6333 | 0.21937 |
| 1.0 | 2.0050 | 0.29645 |
| 1.3 | 2.2805 | 0.35444 |

The obvious hypothesis — that it would *flatten* its world to make itself easy
to predict — is wrong in direction. It destabilizes, 1.05 -> 2.28, and models
the harder world anyway. Attribution by freezing shows **co-adaptation**: the
model and world fit each other, and `B` rising with `c` is the cost.

---

## The shunya measurement

At `c = 0.6`, **134 of 200 runs reach error below 1e-9** — identical interior
evidence. From outside, final states span 0.0 to 2.19233970, a ratio of
2.19e12. The empty ones still carry a weight norm near 0.6: structure intact,
state gone.

The error signal — the only thing the organism has — is *constant* across all
134. Rice's theorem, as a measurement rather than a citation: the quantity
separating rich from empty is invisible to the loop that produced it.

---

## A record of wrong predictions

Kept deliberately. A retracted claim is worth more than a quiet deletion.

| predicted | observed |
|---|---|
| self-reference changes nothing (6 seeds, 3-way label) | clean monotonic effect at 200 seeds with a continuous measure — the null was a measurement failure |
| the organism flattens its world, rho < 1 | it destabilizes, rho 1.05 -> 2.28 |
| settled runs have eff_rank ~ 1 | median 1.79, max 4.71 |
| settled runs generalize worse | identical, p = 0.84 |
| error is the channel a law travels through | error predicts nothing; coverage does, weakly |
| an other keeps surprise alive | no effect on freezing at any c |
| coverage is the first open problem | true in the unsettled subgroup; the pooled result fails Bonferroni |
| batch fit would match or beat the online rule | 5.4x worse — the online rule is implicitly regularizing |
| "20% better than random" was the right framing | wrong baseline; 37.7% of the achievable range |
