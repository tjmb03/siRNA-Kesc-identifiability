"""
Practical identifiability of k_esc by profile likelihood.

The amplitude-share table in analysis.pole_table is a shape metric.  It says
where the signal is, not whether an experiment can see it.  This module does
the actual test: simulate data under a sampling design, profile k_esc, and read
off the confidence interval.

Estimated:  k_esc, Emax        (the confounded pair)
Fixed:      k_el, k_up, k_degt, k_loss, k_out, k_syn, EC50
            -- treated as independently measured, which is the most generous
            possible assumption.  Anything that cannot be resolved here cannot
            be resolved in a real fit either.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from model import Params, simulate

#: solver tolerances used inside the likelihood (thousands of solves)
FIT_TOL = dict(rtol=1e-7, atol=1e-10)

__all__ = ["DESIGNS", "simulate_dataset", "profile_k_esc",
           "interval_from_profile"]

#: chi-square 95 % cutoff, 1 degree of freedom
DELTA_95 = 3.841458820694124

MONTHLY = [0.0, 28.0, 56.0, 84.0, 112.0, 140.0, 168.0]
EARLY = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 14.0]

DESIGNS = {
    "monthly_protein": {
        "label": "Protein, monthly",
        "protein_days": MONTHLY,
        "tissue_days": [],
        "note": "the usual in vivo PD schedule",
    },
    "dense_early_protein": {
        "label": "Protein, dense early + monthly",
        "protein_days": sorted(set(EARLY + MONTHLY)),
        "tissue_days": [],
        "note": "same readout, earlier sampling",
    },
    "protein_plus_tissue": {
        "label": "Protein + total tissue siRNA, dense early",
        "protein_days": sorted(set(EARLY + MONTHLY)),
        "tissue_days": [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0],
        "note": "adds the observation that opens the tissue pole",
    },
}


def _predict(p: Params, protein_days, tissue_days):
    days = sorted(set(list(protein_days) + list(tissue_days)))
    t = np.array(days, dtype=float)
    # solve_ivp needs a strictly increasing grid starting at/after t0
    _, y = simulate(p, t_eval=t, **FIT_TOL)
    idx = {d: i for i, d in enumerate(days)}
    prot = np.array([y["P"][idx[d]] / p.P0 for d in protein_days])
    tis = np.array([y["A_t"][idx[d]] for d in tissue_days])
    return prot, tis


def simulate_dataset(truth: Params, design: dict, cv_protein: float,
                     cv_tissue: float, seed: int):
    """Log-normal observation error, fixed seed."""
    rng = np.random.default_rng(seed)
    prot, tis = _predict(truth, design["protein_days"], design["tissue_days"])
    obs_p = prot * np.exp(rng.normal(0.0, cv_protein, size=prot.shape))
    obs_t = (tis * np.exp(rng.normal(0.0, cv_tissue, size=tis.shape))
             if len(tis) else tis)
    return {"protein": obs_p, "tissue": obs_t}


def _neg2ll(p: Params, design, data, cv_protein, cv_tissue) -> float:
    prot, tis = _predict(p, design["protein_days"], design["tissue_days"])
    if np.any(prot <= 0):
        return 1e12
    r = np.sum(((np.log(data["protein"]) - np.log(prot)) / cv_protein) ** 2)
    if len(tis):
        tis = np.clip(tis, 1e-300, None)
        r += np.sum(((np.log(data["tissue"]) - np.log(tis)) / cv_tissue) ** 2)
    return float(r)


#: nuisance parameters that may be re-optimised at each profile point,
#: with (lower, upper) bounds
NUISANCE_BOUNDS = {"Emax": (1e-3, 1.0),
                   "k_degt": (1e-3, 20.0),
                   "EC50": (1e-4, 10.0)}


def _profile_point(truth: Params, k_esc: float, free: tuple[str, ...],
                   design, data, cv_protein, cv_tissue, start=None):
    """Minimise -2 log L over the nuisance parameters at a fixed k_esc.

    `start` is the solution from the neighbouring grid point (continuation);
    warm-starting keeps the sweep to a few thousand ODE solves instead of tens
    of thousands.
    """
    base = truth.with_(k_esc=float(k_esc))
    if not free:
        return _neg2ll(base, design, data, cv_protein, cv_tissue), {}

    if len(free) == 1:
        name = free[0]
        lo, hi = NUISANCE_BOUNDS[name]
        f = lambda v: _neg2ll(base.with_(**{name: float(v)}), design, data,
                              cv_protein, cv_tissue)
        r = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-6})
        return float(r.fun), {name: float(r.x)}

    from scipy.optimize import minimize
    lo = np.array([NUISANCE_BOUNDS[n][0] for n in free])
    hi = np.array([NUISANCE_BOUNDS[n][1] for n in free])

    def obj(u):                       # box constraint by clipping + penalty
        v = np.clip(u, lo, hi)
        pen = 1e6 * float(np.sum((u - v) ** 2))
        return _neg2ll(base.with_(**dict(zip(free, map(float, v)))),
                       design, data, cv_protein, cv_tissue) + pen

    x0 = (np.asarray(start, dtype=float) if start is not None
          else np.array([getattr(truth, n) for n in free]))
    r = minimize(obj, np.clip(x0, lo, hi), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 400})
    v = np.clip(r.x, lo, hi)
    return float(r.fun), dict(zip(free, map(float, v)))


def profile_k_esc(truth: Params, design: dict, cv_protein=0.10, cv_tissue=0.15,
                  seed=20260915, k_esc_grid=None,
                  free: tuple[str, ...] = ("Emax",)) -> dict:
    """Profile -2 log L over k_esc, re-optimising the nuisance parameters in
    `free` at each grid point.

    Returns the profile, the 95 % interval (delta = 3.84, 1 d.f.) and the
    fold-width of that interval.
    """
    if k_esc_grid is None:
        k_esc_grid = np.logspace(np.log10(truth.k_esc / 12),
                                 np.log10(truth.k_esc * 12), 31)
    k_esc_grid = np.sort(np.asarray(k_esc_grid, dtype=float))
    free = tuple(free)
    data = simulate_dataset(truth, design, cv_protein, cv_tissue, seed)

    # continuation sweep outward from the grid point nearest the truth
    i0 = int(np.argmin(np.abs(k_esc_grid - truth.k_esc)))
    out: dict[int, dict] = {}
    for direction in (+1, -1):
        start = None
        i = i0
        while 0 <= i < len(k_esc_grid):
            val, hat = _profile_point(truth, k_esc_grid[i], free, design, data,
                                      cv_protein, cv_tissue, start=start)
            out[i] = {"k_esc": float(k_esc_grid[i]), "neg2ll": val,
                      **{f"{k}_hat": v for k, v in hat.items()}}
            start = [hat[n] for n in free] if hat else None
            i += direction
    prof = [out[i] for i in sorted(out)]

    best = min(prof, key=lambda d: d["neg2ll"])
    for d in prof:
        d["delta"] = d["neg2ll"] - best["neg2ll"]

    iv = interval_from_profile(prof)
    lo, hi, at_edge = iv["ci95"][0], iv["ci95"][1], iv["interval_hits_grid_edge"]

    return {
        "design": design["label"],
        "note": design["note"],
        "free": list(free),
        "n_protein": len(design["protein_days"]),
        "n_tissue": len(design["tissue_days"]),
        "first_sample_day": float(min(
            [d for d in list(design["protein_days"]) + list(design["tissue_days"])
             if d > 0])),
        "truth_k_esc": truth.k_esc,
        "k_esc_hat": best["k_esc"],
        "ci95": [float(lo), float(hi)],
        "fold_width": iv["fold_width"],
        "interval_hits_grid_edge": at_edge,
        "ci_method": iv["ci_method"],
        "profile": prof,
    }


def interval_from_profile(prof: list[dict]) -> dict:
    """95 % interval from a profile, by interpolating the delta = 3.84 crossing.

    Reading the interval off the grid points instead understates it by up to a
    grid step -- about 18 % on a 31-point grid spanning 144-fold -- and it
    understates in the optimistic direction, making k_esc look better resolved
    than it is.  The crossing is interpolated linearly in (log k_esc, delta).

    Where the profile never rises above the cutoff on one side, that side is
    edge-limited: the reported bound is the end of the grid and the fold width
    is a lower bound, flagged by interval_hits_grid_edge.
    """
    k = np.array([d["k_esc"] for d in prof], dtype=float)
    dl = np.array([d["delta"] for d in prof], dtype=float)
    order = np.argsort(k)
    k, dl = k[order], dl[order]
    i0 = int(np.argmin(dl))

    def crossing(idx_range) -> float | None:
        for a, b in idx_range:
            if (dl[a] - DELTA_95) * (dl[b] - DELTA_95) < 0:
                f = (DELTA_95 - dl[a]) / (dl[b] - dl[a])
                return float(np.exp(np.log(k[a]) + f * (np.log(k[b]) - np.log(k[a]))))
        return None

    lo = crossing([(i, i - 1) for i in range(i0, 0, -1)])
    hi = crossing([(i, i + 1) for i in range(i0, len(k) - 1)])
    at_edge = lo is None or hi is None
    lo = float(k[0]) if lo is None else lo
    hi = float(k[-1]) if hi is None else hi
    return {"ci95": [lo, hi], "fold_width": float(hi / lo),
            "interval_hits_grid_edge": bool(at_edge),
            "ci_method": "grid endpoint (lower bound)" if at_edge
                         else "interpolated crossing of delta = 3.84"}
