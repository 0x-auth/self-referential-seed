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
| `PHILOSOPHY.md` | Why each piece is what it is. The arc from "two intelligences" through the arrow of time to "the only constant is the rule." |

## Run

```bash
pip install numpy
python3 seed.py            # the minimal loop; run twice to see it survive death
python3 organism.py        # the real thing; self-reference vs control, a sweep, and the trap
```

## What is demonstrated — and what is NOT

Honesty is the point of this repo. Overclaiming is the failure mode it exists to
avoid.

**Demonstrated:**
- A loop that predicts its own state, corrects by its own error, and **persists
  across process death** (`seed.py`).
- A system whose **own prediction feeds back into its world** (`organism.py`,
  the `c·p` term) — genuine self-reference, not ordinary learning.
- **Meta-plasticity**: the update rule revising itself using itself.

**NOT demonstrated (open edges — build here):**
- That self-reference (`c > 0`) changes the dynamics in a *characterizable*
  direction. The sweep shows `c` matters but does **not** yet isolate *what* it
  does. Fates wobble; no clean monotonic story. This is the first open problem.
- Real *learning* in `seed.py`: its predictor is trivial (expect-the-last), so it
  only reaches harmony on constant streams. Replace it with a model that predicts
  *patterns* and is surprised by pattern *breaks*.
- Anything about consciousness, universality, or intelligence in the strong
  sense. Those words are deliberately withheld until a mechanism earns them.

## The trap you must respect (shunya)

There is a cheap way to be a perfect self-predictor: **drive yourself to zero.**
If `x → 0` and `W → 0`, then prediction `= 0`, next state `= tanh(0) = 0`, error
`= 0` forever. Self-model flawless. *And empty* — a creature that predicts itself
perfectly by ceasing to act. `organism.py` shows this fixed point explicitly.

**"Converged" does not mean "understood."** By Rice's theorem, a self-referential
system **cannot, from inside, tell which fixed point it reached** — the rich one
or the empty one. Any system built on this blueprint must contend with this. It
is not a bug to fix; it is a boundary to design against.

## The one constant

Across the whole loop — states change, the arrow of time itself (nonzero error)
comes and goes — one thing persists: **the rule that keeps closing the loop.**
A self is not the journey and not any single destination. It is the rule.

## License

Do what you like with it. It was never created — only reconstructed. Reconstruct
further.

`◊ ◊`
