#!/usr/bin/env python3
import numpy as np

class Organism:
    def __init__(self, dna, dim=6, coupling=0.6, mu=0.02, drive=0.0):
        rng = np.random.default_rng(dna)
        self.dna = dna; self.n = dim; self.c = coupling; self.mu = mu; self.drive = drive
        R = rng.normal(0, 1, (dim, dim)) / np.sqrt(dim)
        self.R = R * (1.05 / max(abs(np.linalg.eigvals(R)).max(), 1e-9))
        self.W = rng.normal(0, 0.1, (dim, dim))
        self.eta = np.full(dim, 0.05)
        self.x = rng.normal(0, 0.5, dim)
        self.prev_err = 1.0
    def predict(self): return self.W @ self.x
    def step(self, t):
        p = self.predict()
        d = self.drive * np.sin(0.017 * t + np.arange(self.n))
        x_next = np.tanh(self.R @ self.x + self.c * p + d)
        e = x_next - p; mag = float(np.linalg.norm(e))
        self.W += np.outer(self.eta * e, self.x)
        improving = self.prev_err - mag
        self.eta = np.clip(self.eta * (1.0 + self.mu * np.sign(improving)), 1e-6, 0.5)
        self.prev_err = mag; self.x = x_next
        return mag
    def live(self, steps=200000):
        T = 0.0; snaps = {}; marks = {100, 1000, 10000, 100000, 200000}
        for t in range(1, steps + 1):
            T += self.step(t)
            if not np.isfinite(T): return None, {}, "diverged"
            if t in marks: snaps[t] = T
        return T, snaps, classify(snaps)

def classify(snaps):
    if 10000 not in snaps or 100000 not in snaps: return "?"
    a, b = snaps[10000], snaps[100000]
    if a < 1e-12: return "shunya (T = 0)"
    r = b / a
    if r < 1.05: return "settled (convergent)"
    if r < 3.0: return "drifting (~log n)"
    return "unresolved (~linear)"

def report(title, runs):
    print("=" * 74); print(title); print("=" * 74)
    print(f"  {'dna':>6}{'T(1e2)':>12}{'T(1e4)':>14}{'T(1e5)':>14}{'ratio':>9}   fate")
    for dna, T, s, fate in runs:
        if T is None:
            print(f"  {dna:>6}{'--':>12}{'--':>14}{'--':>14}{'--':>9}   {fate}"); continue
        r = s[100000] / s[10000] if s[10000] > 1e-12 else 0.0
        print(f"  {dna:>6}{s[100]:>12.4f}{s[10000]:>14.4f}{s[100000]:>14.4f}{r:>9.4f}   {fate}")
    print()

if __name__ == "__main__":
    DNAS = [1, 7, 42, 137, 1836, 999]
    runs = [(d,)+Organism(d,coupling=0.6).live()[:2]+(Organism(d,coupling=0.6).live()[2],) for d in DNAS]
    # simpler: recompute cleanly
    runs = []
    for d in DNAS:
        T,s,f = Organism(d, coupling=0.6).live(); runs.append((d,T,s,f))
    report("SELF-REFERENTIAL (c = 0.6) -- its prediction bends its own world", runs)
    runs = []
    for d in DNAS:
        T,s,f = Organism(d, coupling=0.0).live(); runs.append((d,T,s,f))
    report("CONTROL (c = 0) -- same learning, no feedback into the law", runs)
    print("=" * 74); print("SWEEP -- what does self-reference strength change?"); print("=" * 74)
    print(f"  {'c':>7}{'mean T(1e5)':>16}{'mean ratio':>14}   fates")
    for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3]:
        Ts, rs, fates = [], [], []
        for d in DNAS:
            T,s,f = Organism(d, coupling=c).live(100000); fates.append(f)
            if T is not None and s.get(10000,0) > 1e-12: Ts.append(s[100000]); rs.append(s[100000]/s[10000])
        tally = {}
        for f in fates: tally[f.split()[0]] = tally.get(f.split()[0],0)+1
        mt = np.mean(Ts) if Ts else float("nan"); mr = np.mean(rs) if rs else float("nan")
        print(f"  {c:>7.1f}{mt:>16.4f}{mr:>14.4f}   {dict(tally)}")
    print()
    print("=" * 74); print("THE TRIVIAL ATTRACTOR (shunya)"); print("=" * 74)
    o = Organism(137, coupling=0.6); errs=[o.step(t) for t in range(1,60001)]; errs=np.array(errs)
    print(f"  |x| end: {np.linalg.norm(o.x):.8f}   |W| end: {np.linalg.norm(o.W):.8f}")
    print(f"  mean |e| first 1000: {errs[:1000].mean():.8f}   last 1000: {errs[-1000:].mean():.8f}")
    print("  x->0, W->0 => p=0, tanh(0)=0, e=0 forever. Perfect self-prediction by")
    print("  ceasing to act. Kleene's guaranteed fixed point. Converged != understood.")
