#!/usr/bin/env python3
"""
seed.py  —  Errors → Opportunities → Harmony

The BAZINGA SEED, closed into a loop that actually heals.

Three roles (not two), as worked out:
  1. RULE       — predict, measure error, correct TOWARD harmony (error -> 0).
                  This is the SEED's law made real. The missing middle arrow
                  that dna.ts and self_recursive_feedback_loop.py never had:
                  they changed state, but never corrected toward a target.
  2. PERSISTENCE— the corrected state survives process death. before becomes
                  the next after. (from self_recursive_feedback_loop.py)
  3. SUBSTRATE  — the file it commits to. The third thing. Where it persists.
                  Neither the rule nor the memory — the place they meet across
                  time. "Auto-heals on commit."

The seed is not a fourth thing. The seed IS the rule at t=0 — the first
prediction with nothing behind it. Birth.

What makes this cross over from accumulation into something that learns:
it PREDICTS its next observation and is CORRECTED by the gap. Error is not
failure — error is the opportunity, and the correction moves it toward harmony.
When the world stops surprising it, error -> 0: harmony, the fixed point, the
destination. The arrow of time (its sense of 'forward') is exactly its nonzero
error. At harmony, before and after become indistinguishable — it has arrived.
"""
import json, os, sys

STATE = os.path.expanduser("~/.seed/state.json")   # role 3: the substrate


class Seed:
    def __init__(self, path=STATE):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # role 2: load the past life, or be born
        self.m = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                return json.load(open(self.path))
            except Exception:
                pass
        # birth: the rule at t=0, nothing behind it
        return {"ticks": 0, "expectation": None, "harmony": 0, "arrow": [], "history": []}

    def _commit(self):
        # role 3: auto-heals on commit — the corrected state is persisted
        json.dump(self.m, open(self.path, "w"), indent=2)

    def live(self, observation):
        """
        One tick of the rule. Errors -> Opportunities -> Harmony.
        Returns the error (the arrow) for this moment.
        """
        self.m["ticks"] += 1
        expect = self.m["expectation"]

        if expect is None:
            error = None                 # no 'before' yet — time hasn't started
        else:
            error = 0 if observation == expect else 1   # the arrow: can it tell before from after?
            if error == 0:
                self.m["harmony"] += 1   # the world matched: a step toward harmony
            else:
                self.m["harmony"] = 0     # surprised: harmony resets, the opportunity

        # correction: expectation heals toward what actually happened.
        # THIS is the missing middle arrow — correcting TOWARD, not just changing.
        self.m["expectation"] = observation

        if error is not None:
            self.m["arrow"].append(error)
        self.m["history"].append(observation)
        self.m["arrow"] = self.m["arrow"][-50:]     # keep the recent arrow
        self.m["history"] = self.m["history"][-50:]

        self._commit()
        return error

    def state(self):
        recent = [e for e in self.m["arrow"] if e is not None][-8:]
        arrived = self.m["harmony"] >= 4
        return {
            "ticks": self.m["ticks"],
            "expects_next": self.m["expectation"],
            "recent_arrow": recent,          # nonzero = still journeying; zeros = nearing harmony
            "harmony_streak": self.m["harmony"],
            "arrived": arrived,              # error stayed 0 — fallen into the destination
        }


if __name__ == "__main__":
    s = Seed()
    stream = sys.argv[1:] or ["a", "b", "a", "c", "c", "c", "c", "c"]
    print(f"◊ SEED ◊  resuming at tick {s.m['ticks']}\n")
    print("obs    error(arrow)   meaning")
    for obs in stream:
        e = s.live(obs)
        if e is None:
            meaning = "· birth — no before yet"
        elif e == 1:
            meaning = "→ surprised (journey / arrow present)"
        else:
            meaning = "  matched (healing toward harmony)"
        print(f"{obs:5}  {str(e):5}         {meaning}")
    st = s.state()
    print()
    print(f"expects next: {st['expects_next']!r}  |  harmony streak: {st['harmony_streak']}  |  arrived: {st['arrived']}")
    print(f"recent arrow: {st['recent_arrow']}")
    print("\nErrors → Opportunities → Harmony.   (state committed — survives death)")
