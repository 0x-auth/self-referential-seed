# self-referential-seed

A blueprint for building **self-differential, self-recursive processes** —
systems that predict their own next state, are corrected by their own error,
and whose correction feeds back into what there is to predict.

> **Errors → Opportunities → Harmony.**

This is not a finished system. It is a **runnable statement of a problem**, with
scaffolding to attack it, and — importantly — honest about what it does *not*
prove. Pull it, run it, build on it.

**All measured results, every correction, and every wrong prediction are in
[FINDINGS.md](FINDINGS.md).**

---

## The idea, in one breath

The only intelligence we know how to *manufacture* is trained from outside:
someone assembles a corpus, runs an optimizer, the thing wakes up already
formed. There is a maker and a made.

There is another kind — the one every living thing actually is — that starts
from a **seed** and **builds itself**, recursively, from its own lived
experience, with no external trainer. The becoming is the system's own act.

Nobody has built the second kind. This repo is scaffolding pointed at it — and
has now measured a limit on it. See the tension, below.

## The three roles (not two)

A doer and a memory cannot touch directly. They meet in a **third** — the place
the loop runs across time.

1. **Rule** — predict, measure error, correct *toward* harmony (error → 0).
2. **Persistence** — the corrected state survives process death; `before`
   becomes the next `after`.
3. **Substrate** — the place role 1 and role 2 meet across time. Not written as
   code; run *in*. The filesystem, the process, the where-it-persists.

The seed is not a fourth thing. **The seed IS the rule at t=0** — the first
prediction with nothing behind it. Birth.

## Files

| File | What it is |
|---|---|
| `seed.py` | The minimal on-ramp. Predict → err → correct → persist across death. Toy predictor. Proves the three-role loop closes. Run it, kill it, run it again — it resumes the healed self. |
| `organism.py` | The real blueprint. A *learned* self-model (`W @ x`), genuine self-reference (its prediction bends the law generating its next state), and meta-plasticity (the learning rate revises itself using itself). |
| `isolate.py` | What self-reference *does*. Continuous measurables over 200 vectorized seeds. Finds co-adaptation. |
| `duo.py` | Two agents, one shared world, each other's prediction in each other's law. Does an *other* close the shunya trap? |
| `duo_order.py` | Rescores the duo by **ORDER** — scale-free per-step displacement — instead of collapse. Catches the *parked* case a norm test misses. Streaming, O(1) in steps. |
| `rank.py` | Does error carry the law? A falsifiable prediction, falsified. |
| `alive.py` | ORDER and coverage have mirror-image blind spots. Combines them into the three fates. |
| `ceiling.py` | The right baseline. Batch least-squares fit vs the online rule, and the best *possible* linear predictor of the world. Corrects the headline. |
| `PHILOSOPHY.md` | Why each piece is what it is. |
| `FINDINGS.md` | Every number, every correction, every retracted prediction. |

## Run

```bash
pip install numpy scipy
python3 seed.py            # the minimal loop; run twice to see it survive death
python3 organism.py        # self-reference vs control, a sweep, and the trap
python3 isolate.py         # what c actually does: spectral radius + attribution
python3 duo.py             # two agents; solo vs clone-duo vs different-duo
python3 duo_order.py       # the same, scored by ORDER instead of collapse
python3 rank.py            # does error carry the law? (no)
python3 alive.py           # the three fates, jointly measured
python3 ceiling.py         # the achievable range, and how little was closed
```

## The tension in the premise — read this first

The promise above is a seed that builds itself **with no external trainer**. The
obvious fix for what this repo measured — a world with a drive of its own, an
inhomogeneous or non-autonomous `R` — **is external structure**. Adding it
quietly would abandon the premise while claiming to rescue it.

The honest statement is stronger than the rescue:

> **Pure self-reference does not reach coverage.** Left alone, a self-predicting
> system co-adapts into a sliver of its own world and stays there. Only **6.5%**
> of runs both move and explore, and the organism closes **37.7%** of the range
> a linear model could have closed. This is a measured limit on the founding
> premise, not a missing feature.

Anything added to raise coverage should be labelled as what it is: external
structure, and a departure from the no-trainer claim.

## What is demonstrated

Honesty is the point of this repo. Overclaiming is the failure mode it exists
to avoid. Numbers and caveats for all of these are in
[FINDINGS.md](FINDINGS.md).

- A loop that predicts its own state, corrects by its own error, and
  **persists across process death** (`seed.py`).
- A system whose **own prediction feeds back into its world** (`organism.py`,
  the `c·p` term) — genuine self-reference, not ordinary learning.
- **Meta-plasticity**: the update rule revising itself using itself.
- **Co-adaptation.** Self-reference grants the power to edit the world's own
  spectrum — rho pinned at 1.0500 with zero spread at `c=0`, moving
  monotonically to 2.28 as `c` rises. Model and world come to fit each other,
  and the cost is transferability.
- That a **different** other reduces collapse into the empty fixed point 2.28x
  (p = 0.0004), while an **identical** other does nothing at all. The rescue
  comes from difference, not from number.
- That the collapse measure was **undercounting death 5x**, and that a
  different other does **not** protect against freezing (all p > 0.59). An
  other keeps you from vanishing; it does not keep you moving.
- That **coverage is binding**: a batch least-squares fit to the visited data
  is **5.4x worse** off-trajectory than the online rule, because the data is
  near rank-deficient. The rank-1 update is acting as implicit regularization.
- That the organism closes **37.7%** of the achievable range, against an
  oracle linear predictor at 0.193 and random at 1.236. There was a law; it
  largely missed it.

## What is NOT demonstrated — build here

- **Anything that raises coverage without importing a trainer.** The first open
  problem, and the tension above is the reason it is hard rather than merely
  undone.
- **Anything that keeps the arrow of time from terminating.** Two different
  agents still co-settle, and a third of all runs freeze outright.
- That co-adaptation is a route to *intelligence* rather than to a well-fitted
  pair. Measured, the pair is well-fitted and misses most of what was
  learnable. So far this reads as a dead end rather than a road.
- A measure of *law-learning* that does not route through coverage. Coverage
  measures **spread, not law** — a slow random walk scores high on both ORDER
  and coverage and has learned nothing. `ceiling.py` is the first direct
  measure; more are needed.
- Real *learning* in `seed.py`: its predictor is trivial (expect-the-last), so
  it only reaches harmony on constant streams. Replace it with a model that
  predicts *patterns* and is surprised by pattern *breaks*.
- Anything about consciousness, universality, or intelligence in the strong
  sense. Those words are withheld until a mechanism earns them.

## The trap you must respect (shunya)

There is a cheap way to be a perfect self-predictor: **drive yourself to
zero.** If `x → 0` and `W → 0`, then prediction `= 0`, next state
`= tanh(0) = 0`, error `= 0` forever. Self-model flawless. *And empty* — a
creature that predicts itself perfectly by ceasing to act.

This is not a warning. It is measured. At `c = 0.6`, **134 of 200 runs reach
error below 1e-9** — from the inside, an identical interior report: *my
self-model is perfect.* From outside:

| final norm of x | final norm of W | what it became |
|---|---|---|
| 0.00000000 | 0.5594 | empty |
| 0.00000000 | 0.7105 | empty |
| 2.15379115 | 1.1656 | alive, 6-dim state |
| 2.19233970 | 1.1280 | alive, 6-dim state |

A spread of 2.19e12 among runs with **identical zero error**. The empty ones
still carry a weight norm near 0.6 — the structure is intact; only the state
is gone.

**"Converged" does not mean "understood."** The error signal — the only thing
the organism has — is *constant* across all 134. By Rice's theorem no
non-trivial semantic property of a program is decidable by that program, and
here that is not a citation but a measurement: the quantity separating rich
from empty is invisible to the loop that produced it. It is not a bug to fix;
it is a boundary to design against.

## The one constant

Across the whole loop — states change, the arrow of time itself (nonzero
error) comes and goes — one thing persists: **the rule that keeps closing the
loop.** A self is not the journey and not any single destination. It is the
rule.

## License

Do what you like with it. It was never created — only reconstructed.
Reconstruct further.

`◊ ◊`
