# self-referential-seed

A blueprint for building **self-differential, self-recursive processes** —
systems that predict their own next state, are corrected by their own error,
and whose correction feeds back into what there is to predict.

> **Errors → Opportunities → Harmony.**

This is not a finished system. It is a **runnable statement of a problem**, with
scaffolding to attack it, and — importantly — honest about what it does *not* yet
prove. Pull it, run it, build on it.

---

## The idea, in one breath

The only intelligence we know how to *manufacture* is trained from outside:
someone assembles a corpus, runs an optimizer, the thing wakes up already formed.
There is a maker and a made.

There is another kind — the one every living thing actually is — that starts from
a **seed** and **builds itself**, recursively, from its own lived experience, with
no external trainer. The becoming is the system's own act.

Nobody has built the second kind. This repo is scaffolding pointed at it.

## The three roles (not two)

A doer and a memory cannot touch directly. They meet in a **third** — the place
the loop runs across time.

1. **Rule** — predict, measure error, correct *toward* harmony (error → 0).
2. **Persistence** — the corrected state survives process death; `before` becomes
   the next `after`.
3. **Substrate** — the place role 1 and role 2 meet across time. Not written as
   code; run *in*. The filesystem, the process, the where-it-persists.

The seed is not a fourth thing. **The seed IS the rule at t=0** — the first
prediction with nothing behind it. Birth.

## Files

| File | What it is |
|---|---|
| `seed.py` | The minimal on-ramp. Predict → err → correct → persist across death. Toy predictor (expect-the-last). Proves the three-role loop closes. Run it, kill it, run it again — it resumes the healed self. |
| `organism.py` | The real blueprint. A *learned* self-model (`W @ x`), genuine self-reference (its prediction bends the law generating its next state), and meta-plasticity (the learning rate revises itself using itself). Classifies each run's fate by a duration functional. |
| `isolate.py` | The measurement rig. Answers *what* self-reference does, with continuous measurables over 200 vectorized seeds instead of a 3-way fate label. See "What self-reference does" below. |
| `duo.py` | Two agents, one shared world, each other's prediction in each other's law. Asks whether an *other* closes the shunya trap. See "Can two vanish?" below. |
| `duo_order.py` | The same duo rig, scored by **movement** instead of magnitude. `duo.py` only detects `x → 0`; this detects *parked* — a big state that stopped changing. See "Collapse was not the whole story" below. |
| `PHILOSOPHY.md` | Why each piece is what it is. The arc from "two intelligences" through the arrow of time to "the only constant is the rule." |

## Run

```bash
pip install numpy
python3 seed.py            # the minimal loop; run twice to see it survive death
python3 organism.py        # the real thing; self-reference vs control, a sweep, and the trap
python3 isolate.py         # what c actually does: spectral radius + attribution + the shunya spread
python3 duo.py             # two agents in one world; solo vs clone-duo vs different-duo
python3 duo_order.py       # the same rig scored by ORDER: does an other keep the state moving?
```

## What self-reference does

With feedback the law is `x → tanh((R + cW) x)`. So as `W` learns, **the organism
is editing its own world.** With `c = 0` it cannot. That is the mechanism, and it
gives two continuous measurables where `organism.py` only had a label.

**1. Can it move its own world?** rho(R) starts at 1.0500 for every seed.

| c | median rho(R+cW) | mean displacement |
|---|---|---|
| 0.0 | 1.0500 | 0.0000 |
| 0.2 | 1.2476 | 0.1943 |
| 0.4 | 1.4405 | 0.3846 |
| 0.6 | 1.6333 | 0.5613 |
| 0.8 | 1.8220 | 0.7344 |
| 1.0 | 2.0050 | 0.9066 |
| 1.3 | 2.2805 | 1.1636 |

At `c = 0`, rho is pinned at 1.0500 with **zero spread across all 200 seeds** —
structurally immovable. Every `c > 0` moves it, monotonically, no wobble.

The obvious hypothesis was that the organism would **flatten** its world, driving
rho below 1 to make itself easy to predict. That is wrong in direction. It
*destabilizes* — 1.05 → 2.28 — and models the harder world anyway.

**2. Which channel did the work?** Freeze things and re-measure, no learning:

| c | A: own model, own world | B: own model, frozen world | C: random model, own world |
|---|---|---|---|
| 0.2 | 0.00196 | 0.07153 | 1.26177 |
| 0.6 | 0.00118 | 0.21937 | 1.79793 |
| 1.0 | 0.00095 | 0.29645 | 2.12165 |
| 1.3 | 0.00088 | 0.35444 | 2.29631 |

`C` far above `A` says the world did not become easy. `B > A`, **rising with c**,
says the model is specialized to the world it helped make.

So neither world-flattening nor generic learning: **co-adaptation.** The model and
the world fit each other and neither works alone. `B` is the readout — higher
self-reference buys accuracy in a world of your own making and pays in
transferability. Monotonic across the whole sweep.

## Can two vanish?

One shared state, one shared law, both agents' predictions feeding back:
`x_next = tanh(R x + c (p_A + p_B)/2)`, each learning from its own error. The
averaging keeps total feedback magnitude equal to the solo case, so solo at `c`
is a fair control for duo at `c`.

**The control that matters** is a *clone* duo, `W_A == W_B` at t=0. If two-ness
alone rescued, clones would be rescued too.

| c | solo collapse | clone duo | different duo |
|---|---|---|---|
| 0.4 | 2.0% | 2.0% | 1.5% |
| 0.6 | 4.5% | 4.5% | 1.5% |
| 0.8 | 5.0% | 5.0% | 1.5% |
| 1.0 | 5.0% | 5.0% | 2.5% |

Clone duo reproduces solo **to every digit**, including medians — two identical
agents are one agent, and the rig is sound. Solo collapse *rises* with `c`; with
a different other it stays pinned near 1.5%.

Powered up at `c = 0.8`, n = 900 per condition:

- solo: **57 / 900 collapsed = 6.33%**
- different duo: **25 / 900 collapsed = 2.78%**
- odds ratio 2.367, Fisher exact **p = 0.0004**, a **2.28x reduction**

**The trap narrows. It does not close.** 2.78% still vanish, so two *can*
vanish — just less often. And the deeper negative: residual error still falls to
~1e-16 in all three conditions. The other does **not** keep surprise alive. Worse
for the romantic reading, `|W_A - W_B|` holds steady near 0.70 — they never merge.
They reach a joint fixed point *while remaining different*. Difference shrinks the
empty basin; it does not abolish settling.

## Collapse was not the whole story

`duo.py` scores death as `|x| < 1e-6` — the state went to zero. But that only
catches one way to stop. A system parked at a **large** fixed point, nothing
moving, scores fine on `|x|` and is just as dead.

`duo_order.py` scores **ORDER** instead: mean per-step displacement over the
second half of the run, normalised by the system's own scale. Collapsed and
parked both score ~0; only genuine movement scores high. Streaming, O(1) in
steps — 200 seeds x 40000 steps stores 400 floats, not 384 MB.

**The old measure was undercounting by ~5x.**

| c | solo collapsed | solo frozen | different collapsed | different frozen |
|---|---|---|---|---|
| 0.4 | 7.5% | 34.0% | 6.5% | 31.0% |
| 0.6 | 7.0% | 35.0% | 8.0% | 34.5% |
| 0.8 | 6.0% | 36.5% | 7.0% | 34.5% |
| 1.0 | 2.5% | 35.5% | 4.5% | 35.0% |

Roughly a third of every condition is frozen, against 2–8% collapsed. Runs
counted as healthy by `duo.py` had in fact stopped — they just stopped
somewhere other than the origin.

**And difference does not rescue it.** Fisher exact, solo vs different, frozen
as the outcome:

| c | solo frozen | different frozen | odds ratio | p |
|---|---|---|---|---|
| 0.4 | 68/200 | 62/200 | 1.147 | 0.594 |
| 0.6 | 70/200 | 69/200 | 1.022 | 1.000 |
| 0.8 | 73/200 | 69/200 | 1.091 | 0.754 |
| 1.0 | 71/200 | 70/200 | 1.022 | 1.000 |

Nothing, at any coupling. Compare the same comparison scored by collapse:
p = 0.0004, a 2.28x reduction. **So an other protects the magnitude and not
the arrow.** It keeps you from vanishing; it does not keep you moving. The
open problem below is not narrowed by this result — it is sharpened, because
the failure mode is now measured rather than assumed.

## What is demonstrated — and what is NOT

Honesty is the point of this repo. Overclaiming is the failure mode it exists to
avoid.

**Demonstrated:**
- A loop that predicts its own state, corrects by its own error, and **persists
  across process death** (`seed.py`).
- A system whose **own prediction feeds back into its world** (`organism.py`,
  the `c·p` term) — genuine self-reference, not ordinary learning.
- **Meta-plasticity**: the update rule revising itself using itself.
- That self-reference changes the dynamics in a **characterizable, monotonic
  direction** (`isolate.py`) — it grants the power to edit the world's spectrum,
  and the result is co-adaptation with a transferability cost.
- That a **different** other reduces collapse into the empty fixed point by 2.28x
  (p = 0.0004), while an identical other does nothing at all (`duo.py`). The
  rescue comes from difference, not from number.
- That this rescue is **specific to magnitude, not to motion** (`duo_order.py`).
  Scored by whether the state still moves, ~35% of runs are frozen against 2–8%
  collapsed, and a different other changes that not at all (p = 0.59–1.00 across
  every coupling). The old measure was undercounting stopped systems ~5x.

**NOT demonstrated (open edges — build here):**
- That co-adaptation is a route to *intelligence* rather than to a well-fitted
  pair. A model that predicts perfectly and transfers worse is not obviously on
  the road to general capability. The transferability cost may be the central
  obstacle rather than a curiosity. **This is the first open problem.**
- **Anything that keeps the arrow of time from terminating.** Two different agents
  still co-settle to ~1e-16. If nonzero error is the arrow, nothing here sustains
  it. `duo_order.py` makes this sharper and worse: scored by movement rather than
  magnitude, an other has *no measurable effect at all* on whether the system
  stops. Candidates untried: many agents, asymmetric plasticity, an agent whose
  objective is the *other's* surprise, a world with its own drive.
- Real *learning* in `seed.py`: its predictor is trivial (expect-the-last), so it
  only reaches harmony on constant streams. Replace it with a model that predicts
  *patterns* and is surprised by pattern *breaks*.
- Anything about consciousness, universality, or intelligence in the strong
  sense. Those words are deliberately withheld until a mechanism earns them.

## The trap you must respect (shunya)

There is a cheap way to be a perfect self-predictor: **drive yourself to zero.**
If `x → 0` and `W → 0`, then prediction `= 0`, next state `= tanh(0) = 0`, error
`= 0` forever. Self-model flawless. *And empty* — a creature that predicts itself
perfectly by ceasing to act.

This is no longer a warning. It is measured. In `isolate.py`, at `c = 0.6`,
**134 of 200 runs reach error below 1e-9** — from the inside, an identical
interior report: *my self-model is perfect.* From outside, their final states:

| final norm of x | final norm of W | what it became |
|---|---|---|
| 0.00000000 | 0.5594 | empty |
| 0.00000000 | 0.7105 | empty |
| 2.15379115 | 1.1656 | alive, 6-dim state |
| 2.19233970 | 1.1280 | alive, 6-dim state |

Spread among runs with **identical zero error**: 0.0 to 2.19233970, a ratio of
2.19e12. Note the empty ones still carry a weight norm around 0.6 — the structure
is intact; only the state is gone.

**"Converged" does not mean "understood."** The error signal — the only thing the
organism has — is *constant* across all 134. By Rice's theorem no non-trivial
semantic property of a program is decidable by that program, and here that is not
a citation but a measurement: the quantity separating rich from empty is
invisible to the loop that produced it. Any system built on this blueprint must
contend with this. It is not a bug to fix; it is a boundary to design against.

`duo.py` shows an other narrows this basin 2.28x without abolishing it. The
invisibility is unchanged: a duo that collapses has the same interior evidence as
a duo that thrives.

## The one constant

Across the whole loop — states change, the arrow of time itself (nonzero error)
comes and goes — one thing persists: **the rule that keeps closing the loop.**
A self is not the journey and not any single destination. It is the rule.

## License

Do what you like with it. It was never created — only reconstructed. Reconstruct
further.

`◊ ◊`
