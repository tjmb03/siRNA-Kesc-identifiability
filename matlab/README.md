# MATLAB cross-check

An independent reimplementation of the structural identifiability result in
`src/structural.py`, for running under your own MATLAB licence. The Python and
MATLAB paths read the same parameter file (`../config/reference_parameters.json`)
so they cannot drift apart.

| File | Needs | Status |
|---|---|---|
| `symbolic_proof.m` | Symbolic Math Toolbox | written, not executed |
| `verify_invariance.m` | core MATLAB (`ode15s`) | written, not executed |
| `sensitivity_rank.m` | core MATLAB (`ode15s`) | written, not executed |
| `genssi_sirna.m` | GenSSI 2.0 | written, **API unverified** |

Nothing in this folder has been run. There was no MATLAB available where these
were written, so treat them as a translation to be checked rather than as a
passing test. The Python equivalents in `src/structural.py` have been executed
and their output is in `results/results.json`.

## What to expect

```
>> symbolic_proof
scaled equations
  u: a*u(t) + diff(u(t), t)
  v: b*v(t) - u(t) + diff(v(t), t)
  w: k_loss*w(t) - v(t) + diff(w(t), t)
lumped factors remaining after scaling: none
Emax rewrite residual: 0

>> verify_invariance
k_esc  0.1351 -> 0.5405  (4.0x)
escape fraction  10.0% -> 40.0%
max protein difference  ~1e-11
max plasma  difference  ~1e-13

>> sensitivity_rank
plasma + protein                                     free=9 rank=7 unresolved=2
plasma + protein + total tissue siRNA                free=9 rank=8 unresolved=1
the above, with k_degt fixed by a stability assay    free=8 rank=8 unresolved=0
```

The residual sizes in `verify_invariance` are integrator tolerance, not model
error: the invariance is exact, and any difference you see is the price of
`RelTol`. If MATLAB and Python disagree by more than a couple of orders of
magnitude there, something has gone wrong in the translation rather than in the
mathematics.

`sensitivity_rank` is the one to watch for numerical fragility, and it was
rewritten after an audit caught it. An earlier version read the rank off a
fixed `1e-7` cutoff on the singular values; that is not safe, because the
smallest singular value wanders over two orders of magnitude as the
finite-difference step varies, and the verdict flipped at `h = 1e-6`. Rank is
now taken at the largest *gap* in the spectrum, guarded so that a gap only
counts if it exceeds `1e3` — without that guard a full-rank matrix reports
spurious null directions. The verdict is stable for `h` in `[1e-6, 1e-4]` and
is wrong at `h = 1e-3`, where truncation error dominates.

The threshold-free evidence is the generator residual: applying the analytic
group generator `v` to the sensitivity matrix should give
`||S v|| / (||S|| ||v||)` at the integrator noise floor (~1e-9) for a genuine
null direction. Python reports 4e-9 and 1e-9 for the two generators on
plasma + protein, and 4e-2 for the alpha generator once tissue siRNA is
observed — which is the direction being resolved.
