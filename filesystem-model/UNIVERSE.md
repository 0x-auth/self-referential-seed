# universe/ — a filesystem model with three physical sectors

One family of maps (Mobius transformations on a single real line), split by
how many real fixed points they have. That number alone determines the whole
character of the physics.

## spacetime/hyperbolic_sector/   — GR
f(x) = 1 + 1/x
Two REAL fixed points: phi=1.618... (attracts), psi=-0.618... (repels).
Orbits fall toward phi and never leave. Exponential convergence.
This is geodesic flow: matter falls into a well, dissipative, time has a
clear forward direction because the attractor is irreversibly "downhill."

## spacetime/elliptic_sector/     — QM
f(x) = -1/x
Fixed points are COMPLEX (+-i), off the real line entirely. Nothing to fall
into. Orbits rotate forever, exact period, fully reversible: apply f twice,
you're back exactly where you started. This is unitary evolution: norm-
preserving, periodic, no arrow of time built in.

## spacetime/boundary_sector/     — the seam
The map that sits exactly between the two: the moment when hyperbolic's two
real fixed points MERGE into one and are about to leave the real line and
turn into elliptic's complex pair. Orbits drift toward the single remaining
fixed point but at 1/n, not exponentially -- they approach forever and never
arrive. Neither falling in cleanly (hyperbolic) nor rotating forever
(elliptic): the transition itself.

## The symmetry that ties it together
C(t) = 1-t  and  R(t) = -1/t  are two DIFFERENT involutions, and yet both
satisfy the same identity:  X . f . X = f^-1  (verified to float64 precision,
see cpt scratch work). Neither is privileged as "the" time-reversal operator
-- reversal is a structural fact about f having two fixed points, not a
property of any one chosen mirror.

## What each file IS
A `.state` file's content is not a label for a state -- it IS the state, as
a single real number. The directory's read order IS the arrow of time for
that sector. There is no simulation layer: the filesystem's own ordering and
content ARE the physics, which is the whole point of building it this way
rather than describing it in prose.
