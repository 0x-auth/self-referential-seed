# filesystem-model

Same map as the rest of this repo (`f(x) = 1 + 1/x`, two fixed points phi/psi),
realized as filesystem structure instead of numbers in memory.

- `map.py` — the base map, its inverse, and the involution C(x)=1-x that
  conjugates f to f^-1 (verified: C.f.C = f^-1 to float64 precision).
  Classifies into three regimes by fixed-point count: hyperbolic (2 real,
  GR-like, dissipative), elliptic (0 real / 2 complex, QM-like, reversible),
  parabolic (1, the boundary -- critical slowing down confirmed as the
  boundary is approached).

- `reality_fs.py` — independent construction: maps physical scale (Planck
  length to observable universe) onto the same phi^2 step. Measured 149
  filesystem ticks against the closed-form prediction
  log(universe/planck)/(2*log(phi)) = 147.70 -- same constant, unrelated
  derivation, agreement to 0.2%.

- `UNIVERSE.md` — what the directory structure in each sector means physically.

## Why filesystem structure, not just numbers
A self-referential symlink (`meaning -> meaning`) costs nothing to loop --
no new inode, ever. A forward tick (`tick_N -> tick_N+1`) costs a real
directory that has to exist before the next one can. That asymmetry is the
arrow of time as an existence fact, not a computed one: you can ask "does
tick_5 exist" with `ls` and get a real answer, no convergence check needed.

The self-loop also hits a real, external limit: macOS refuses to resolve
more than 16 levels of the same symlink (ELOOP), a kernel constant, not
something chosen for this model. The structure itself doesn't change at
level 17 -- the resolver just stops counting. That is the actual shape of
"limit," not "impossibility": the map is unchanged, only how far a given
substrate is willing to follow it changes.
