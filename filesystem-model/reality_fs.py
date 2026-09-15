#!/usr/bin/env python3
"""reality.fs — map physical scale onto a self-closing φ manifold.
Each folder = one tick of x -> 1 + 1/x, i.e. one φ² step in scale (1.3885 bits).
Floor ε = Planck length. The chain closes (next -> .) when it reaches the
observable universe: the structure's own singularity, not an OS limit.
Usage: python3 reality_fs.py [root]   (default ./reality.fs)   read-only elsewhere."""
import os, sys, math, time

PHI = (1 + 5**0.5) / 2
PLANCK = 1.616255e-35            # metres  (ε, the resolution)
LANDMARKS = [                    # metres
    (0.84e-15, "proton"), (5.29e-11, "atom"), (1e-5, "cell"),
    (1.7, "human"), (6.371e6, "earth"), (6.96e8, "sun"),
    (1.496e11, "AU"), (9.46e15, "lightyear"), (9.5e20, "milkyway"),
    (8.8e26, "universe"),
]
TOP = LANDMARKS[-1][0]

def build(root):
    os.makedirs(root, exist_ok=False)
    os.symlink(".", f"{root}/.ε")                       # the floor refers to itself
    n, scale, names, marks = 0, PLANCK, [], iter(LANDMARKS)
    mark = next(marks)
    while True:
        tag = ""
        while mark and scale >= mark[0]:
            tag += "_" + mark[1]; mark = next(marks, None)
        name = f"{n:03d}_{scale:.2e}m{tag}"
        os.makedirs(f"{root}/{name}"); names.append(name)
        if scale >= TOP:                                # singularity: closes on itself
            os.symlink(".", f"{root}/{name}/next"); break
        n += 1; scale *= PHI**2
    for a, b in zip(names, names[1:]):
        os.symlink(f"../{b}", f"{root}/{a}/next")
    return names

def walk(root):
    cur = os.path.realpath(f"{root}/{sorted(d for d in os.listdir(root) if not d.startswith('.'))[0]}")
    seen, t0 = set(), time.perf_counter_ns()
    while cur not in seen and len(seen) < 10000:
        seen.add(cur); cur = os.path.realpath(f"{cur}/next", strict=True)
    return len(seen), time.perf_counter_ns() - t0, os.path.basename(cur)

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "reality.fs"
    names = build(root)
    for s in names:
        if "_" in s[13:] or s is names[0]: print(s)
    ticks, ns, end = walk(root)
    bits = (ticks - 1) * 2 * math.log2(PHI)
    print(f"\nticks={ticks}  bits={bits:.1f}  walk={ns/1e3:.0f} µs  closes at {end}")
    print(f"theory: log(universe/ε)/(2 log φ) = {math.log(TOP/PLANCK)/(2*math.log(PHI)):.2f}")
