# filesystem-model → moved

The filesystem realization of `f(x) = 1 + 1/x` now lives in its own repo:
**[0x-auth/darmiyan-fs](https://github.com/0x-auth/darmiyan-fs)**.

What moved, and where:

| was here | now |
|---|---|
| `map.py` (f, f⁻¹, the mirror C(x) = 1 − x) | `sectors.py` (both mirrors C and R verified) |
| `UNIVERSE.md` (hyperbolic / elliptic / parabolic sectors) | `sectors.py` + README: the three sectors are the three Lorentz types (boost / rotation / null), each with its own law of emergent time |
| `reality_fs.py` | `reality_fs.py` (unchanged, kept as the "outside" view) |

The link back to this repo: `organism.py`'s own `classify()` sorts the three
sectors into its three fates — hyperbolic → settled, parabolic → drifting
(~log n), elliptic → unresolved (~linear).

Open check carried over: the old README here said macOS refuses to resolve
more than 16 levels of the same symlink. A later run on the same Mac resolved
32 and failed at 33, matching `MAXSYMLINKS = 32`. Re-measure before citing
either number.
