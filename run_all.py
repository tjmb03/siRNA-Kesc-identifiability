#!/usr/bin/env python3
"""
Compute every number that appears in the document, write results.json, draw the
figures, then build the PDF from that file.

    python run_all.py              # full run (~3 min, profile likelihood dominates)
    python run_all.py --quick      # coarser profile grid, for a smoke test
    python run_all.py --no-pdf     # analysis and figures only

Nothing downstream of this script contains a hard-coded scientific number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import load_reference, nadir, simulate            # noqa: E402
import analysis as A                                          # noqa: E402
import structural as St                                       # noqa: E402
import figures as F                                           # noqa: E402
from identifiability import (DESIGNS, profile_k_esc,          # noqa: E402
                             interval_from_profile)

CONFIG = ROOT / "config" / "reference_parameters.json"


class NpEncoder(json.JSONEncoder):
    """numpy scalars leak in from scipy; make them JSON."""
    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
#: the profile sweeps dominate runtime (thousands of stiff ODE solves), so each
#: (scenario, design) is cached on completion.  Delete this file to force
#: recomputation; the key includes every setting that changes the answer.
PROFILE_CACHE = RESULTS / "profiles_cache.json"

SCENARIOS = [
    (("Emax",), "F - k_degt known from a stability assay"),
    (("Emax", "k_degt"), "G - k_degt unknown"),
]


def _key(free, design_key, n_grid, an) -> str:
    return "|".join([",".join(free), design_key, str(n_grid), str(an["seed"]),
                     str(an["noise_cv_protein"]), str(an["noise_cv_tissue"])])


def compute_profiles(p, an, n_grid: int, refresh: bool = False) -> list[dict]:
    """Run (or load) the profile sweeps, caching each one as it finishes."""
    cache = {} if refresh or not PROFILE_CACHE.exists() else json.loads(
        PROFILE_CACHE.read_text())
    grid = np.logspace(np.log10(p.k_esc / 12), np.log10(p.k_esc * 12), n_grid)

    blocks = []
    for free, title in SCENARIOS:
        block = {"title": title, "free": list(free), "designs": []}
        for design_key, d in DESIGNS.items():
            k = _key(free, design_key, n_grid, an)
            if k in cache:
                r = cache[k]
                # the sweep is what is expensive and what is cached; the
                # interval is cheap, so recompute it rather than trusting a
                # value that may predate a change in how it is derived
                r.update(interval_from_profile(r["profile"]))
                tag = "cached"
            else:
                r = profile_k_esc(p, d, cv_protein=an["noise_cv_protein"],
                                  cv_tissue=an["noise_cv_tissue"],
                                  seed=an["seed"], k_esc_grid=grid, free=free)
                r["key"] = design_key
                cache[k] = r
                PROFILE_CACHE.write_text(json.dumps(cache, cls=NpEncoder))
                tag = "computed"
            block["designs"].append(r)
            print(f"    {title[0]} {r['design']:<42s} "
                  f"{'>' if r['interval_hits_grid_edge'] else ' '}"
                  f"{r['fold_width']:7.1f}-fold  ({tag})")
        blocks.append(block)
    return blocks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="coarse profile grid, for a smoke test")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--profiles-only", action="store_true",
                    help="compute and cache the profile sweeps, then stop")
    ap.add_argument("--refresh-profiles", action="store_true",
                    help="ignore the cache and recompute")
    args = ap.parse_args(argv)

    cfg = json.loads(CONFIG.read_text())
    an = cfg["analysis"]
    p = load_reference(CONFIG)
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    t0 = time.time()

    res: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "parameters": p.to_dict(),
        "t_end_days": an["t_end_days"],
        "low_dose_factor": an["low_dose_factor"],
        "truth_k_esc": p.k_esc,
        "derived": {
            "baseline_protein": p.P0,
            "uptake_fraction": p.uptake_fraction,
            "escape_fraction": p.escape_fraction,
            "phi": p.phi,
            "observable_gain": p.observable_gain,
            "nadir_reference_dose": nadir(p),
        },
    }

    print("· pole decomposition")
    res["pole_table"] = A.pole_table(p)
    res["sampling_leverage"] = {
        f"{d:g}": A.sampling_leverage(p, d) for d in (0.25, 1.0, 28.0)}

    print("· saturation / phi validity")
    res["saturation"] = A.saturation_diagnostic(p)
    res["phi_validity"] = A.phi_validity_window(p)

    print("· pole-assignment ambiguity")
    res["pole_swap"] = A.pole_swap_check(p)

    print("· structural identifiability")
    res["structural"] = {
        "factorisation": St.symbolic_risc_factorisation(),
        "invariance": St.verify_invariance(p),
        "ladder": St.identifiability_ladder(p),
    }

    print("· degeneracy sets")
    m = an["k_esc_manifold"]
    grid = np.linspace(m["low"], m["high"], m["n_points"])
    members = A.iso_nadir_set(p, grid)
    res["manifold"] = {
        "kind": "iso-nadir set (one scalar constraint), as walked in the note",
        "n_members": len(members),
        "k_esc": [q.k_esc for q in members],
        "Emax": [q.Emax for q in members],
        "fold_range_k_esc": max(q.k_esc for q in members) / min(q.k_esc for q in members),
        "target_nadir": nadir(p),
    }
    lin = A.linear_degeneracy_set(p, np.linspace(0.02, p.k_esc + p.k_degt - 0.02, 40))
    res["linear_degeneracy_set"] = {
        "kind": "exact flat manifold of the linearised model "
                "(constant gain and constant tissue pole)",
        "n_members": len(lin),
        "k_esc_range": [min(q.k_esc for q in lin), max(q.k_esc for q in lin)],
        "note": "a different object from the iso-nadir set above",
    }

    print("· decision verdicts")
    res["verdicts_reference_dose"] = A.decision_verdicts(
        members, 1.0, an["nadir_thresholds"])
    res["verdicts_low_dose"] = A.decision_verdicts(
        members, an["low_dose_factor"], an["nadir_thresholds"])

    print("· profile likelihood (slow, cached)")
    res["profiles"] = compute_profiles(p, an, n_grid=15 if args.quick else 31,
                                       refresh=args.refresh_profiles)
    if args.profiles_only:
        print("· profiles cached; stopping")
        return 0

    print("· figures")
    F.fig_model_schematic(p, res, FIGS / "fig0_model.png")
    F.fig_manifold(p, members, res, FIGS / "fig1_manifold.png")
    F.fig_invariance(p, res, FIGS / "fig5_invariance.png")
    F.fig_profiles(res, FIGS / "fig2_profiles.png")
    F.fig_pole_swap(p, res, FIGS / "fig3_pole_swap.png")
    F.fig_phi_window(p, res, FIGS / "fig4_phi_window.png")

    res["runtime_seconds"] = round(time.time() - t0, 1)
    (RESULTS / "results.json").write_text(json.dumps(res, indent=1, cls=NpEncoder))
    print(f"· wrote {RESULTS/'results.json'}  ({res['runtime_seconds']} s)")

    if not args.no_pdf:
        import build_pdf
        build_pdf.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
