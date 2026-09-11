# Findings

Measured results, in the order they were established. Every negative is kept.

---

## Death was undercounted by 5x

`duo.py` only detects `x -> 0`. A system frozen at a *large* state passes any
norm test and is just as dead. `duo_order.py` scores **ORDER** instead — mean
per-step displacement normalised by the system's own scale, streaming, O(1) in
steps — so parked and collapsed both score ~0 and only real movement scores
high.

| c | solo collapsed | solo frozen | different collapsed | different frozen |
|---|---|---|---|---|
| 0.4 | 7.5% | **34.0%** | 6.5% | 31.0% |
| 0.6 | 7.0% | **35.5%** | 8.0% | 34.5% |
| 0.8 | 6.0% | **36.0%** | 7.0% | 34.5% |
| 1.0 | 2.5% | **34.5%** | 4.5% | 36.0% |

About a third of all runs had stopped. The collapse measure was seeing a fifth
of it.

**And difference does not protect the arrow.** Fisher exact, solo vs different,
on frozen: p = 0.594, 0.917, 0.834, 0.834, odds ratios all near 1.0. An other
keeps a system from *vanishing* (2.28x, p = 0.0004) and does nothing at all to
keep it *moving*.

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
floating-point noise spread evenly across dimensions, so eff_rank comes out
*high*. The frozen group shows median eff_rank 2.768 against the alive group's
1.858 — an artefact, not a finding. Always gate a rank on `trace(cov)`;
27.4% of settled runs fail that gate.

Together the two measures recover the trichotomy: ORDER separates **settled**
from the rest, coverage separates **cycling** from **drifting**.

---

## Does intelligence arrive as a side effect? No.

`rank.py` stated a falsifiable prediction and it was falsified. Predicted: a
settled organism has `eff_rank ~ 1`, has learned a point not a law, and so
generalizes *worse* — making error the channel a law travels through.

Both halves false. Settled runs show median eff_rank 1.79 (not 1), zero error
does not even imply a static state — many track a moving orbit exactly — and
settled vs unsettled generalization is 0.8395 vs 0.8482, p = 0.84.
**Residual error predicts generalization not at all** (rho +0.045, p 0.53).

What survived is a different variable. **Coverage** predicts it: rho = -0.152,
p = 0.032. How much of the world got visited matters; how much error remained
does not. Those two had been conflated; the data separates them.

The headline, though:

| | off-trajectory error |
|---|---|
| best of 300 runs | 0.477 |
| median | 0.841 |
| **untrained random W** | **1.049** |

After 40,000 steps of near-perfect self-prediction, the median organism is
**20% better than chance** on states it never visited. It learned a slice, not
a law.

The mechanism is a design flaw, not a mystery: every update is rank-1 along the
visited state, and median visitation is 1.8 of 6 dimensions. It never sees
enough of its world to learn it — and self-reference makes this *worse*, since
co-adaptation lets it reshape the world toward the part it already occupies.

Which argues for a world with a drive of its own. Coverage is the one variable
that measured significant, and an external drive is the only thing on the table
that raises coverage without the organism choosing to.

---

## What self-reference does (earlier, and it holds)

With feedback the law is `x -> tanh((R + cW) x)`, so as `W` learns the organism
is editing its own world. At `c = 0` it cannot: rho is pinned at 1.0500 with
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
model and world fit each other, and `B` rising with `c` is the cost — higher
self-reference buys accuracy in a world of your own making and pays in
transferability.

---

## A record of wrong predictions

Kept deliberately. A retracted claim is worth more than a quiet deletion.

| predicted | observed |
|---|---|
| self-reference changes nothing (6 seeds, 3-way label) | clean monotonic effect at 200 seeds with a continuous measure — the null was a measurement failure |
| the organism flattens its world, rho < 1 | it destabilizes, rho 1.05 -> 2.28 |
| settled runs have eff_rank ~ 1 | median 1.79, max 4.71 |
| settled runs generalize worse | identical, p = 0.84 |
| error is the channel a law travels through | error predicts nothing; coverage does |
| an other keeps surprise alive | no effect on freezing at any c |
