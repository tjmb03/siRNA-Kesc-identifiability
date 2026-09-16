"""
Structural and decision analysis of the minimal siRNA model.

Everything reported in the accompanying document is computed here.  Nothing is
asserted that is not returned by one of these functions.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from model import Params, simulate, nadir, linear_protein, pole_residues

__all__ = ["pole_table", "saturation_diagnostic", "iso_nadir_set",
           "linear_degeneracy_set", "decision_verdicts", "pole_swap_check",
           "sampling_leverage"]


# ---------------------------------------------------------------------------
# 1. where the amplitude sits
# ---------------------------------------------------------------------------
def pole_table(p: Params) -> list[dict]:
    """Amplitude share of each process in the linear-regime protein response.

    'Share' = |residue_i| / sum_j |residue_j|, i.e. the share of the summed
    exponential amplitudes.  This is a shape metric only: it says nothing about
    whether a sampling schedule can see the process (see sampling_leverage).
    """
    names, poles, res = pole_residues(p)
    share = np.abs(res) / np.abs(res).sum()
    label = {"plasma": "Plasma (k_el + k_up)",
             "tissue": "Tissue (k_degt + k_esc)",
             "risc": "RISC decay (k_loss)",
             "protein": "Protein turnover (k_out)"}
    rows = [{"process": n, "label": label[n], "rate_per_day": float(poles[i]),
             "time_constant_days": float(1.0 / poles[i]),
             "residue": float(res[i]), "amplitude_share": float(share[i])}
            for i, n in enumerate(names)]
    return sorted(rows, key=lambda r: -r["rate_per_day"])


def sampling_leverage(p: Params, first_sample_day: float) -> list[dict]:
    """Fraction of each exponential still alive at the first post-dose sample.

    exp(-pole * t1).  A process whose term has decayed to ~0 before the first
    observation contributes no information regardless of its amplitude share.
    """
    names, poles, _ = pole_residues(p)
    return [{"process": n,
             "surviving_fraction": float(np.exp(-poles[i] * first_sample_day))}
            for i, n in enumerate(names)]


def saturation_diagnostic(p: Params) -> dict:
    """Is the reference dose inside the linear range where phi is defined?

    phi = Emax*k_esc/EC50 comes from Emax*A_r/(EC50+A_r) ~ (Emax/EC50)*A_r,
    which requires A_r << EC50.  This checks that assumption instead of
    assuming it.
    """
    _, y = simulate(p)
    ar_max = float(y["A_r"].max())
    occupancy = ar_max / (p.EC50 + ar_max)
    t = np.linspace(0.0, 180.0, 2000)
    lin = 1.0 + linear_protein(p, t)          # fractional protein, linear approx
    _, yfull = simulate(p, t_eval=t)
    full = yfull["P"] / p.P0
    return {
        "A_r_max": ar_max,
        "EC50": p.EC50,
        "A_r_max_over_EC50": ar_max / p.EC50,
        "peak_fractional_occupancy": float(occupancy),
        "nadir_full_model": float(full.min()),
        "nadir_linear_approx": float(lin.min()),
        "linear_approximation_error_pp": float(100 * (lin.min() - full.min())),
    }


# ---------------------------------------------------------------------------
# 2. the two degeneracy sets -- which are not the same object
# ---------------------------------------------------------------------------
def iso_nadir_set(p: Params, k_esc_values, target_nadir: float | None = None,
                  emax_max: float = 1.0) -> list[Params]:
    """Parameter sets that reproduce the observed NADIR only.

    One scalar constraint.  Emax is solved for at each k_esc.  This is the set
    walked in the original note; it is a codimension-1 slice, not the
    likelihood-flat manifold.
    """
    target = nadir(p) if target_nadir is None else target_nadir
    out = []
    for ke in np.atleast_1d(k_esc_values):
        f = lambda e: nadir(p.with_(k_esc=float(ke), Emax=float(e))) - target
        lo, hi = 1e-4, emax_max
        if f(lo) * f(hi) > 0:
            continue                      # no admissible Emax <= 1
        e = brentq(f, lo, hi, xtol=1e-10)
        out.append(p.with_(k_esc=float(ke), Emax=float(e)))
    return out


def linear_degeneracy_set(p: Params, k_esc_values) -> list[Params]:
    """The exact flat manifold of the LINEARISED model.

    Two constraints: hold the observable gain (k_out*k_up*Emax*k_esc/EC50) and
    the tissue pole (k_degt + k_esc) fixed.  k_degt absorbs the change in k_esc
    and Emax rescales to hold the gain.  Every point gives a bit-for-bit
    identical linear protein curve.
    """
    tissue_pole = p.k_esc + p.k_degt
    target_gain = p.observable_gain
    out = []
    for ke in np.atleast_1d(k_esc_values):
        ke = float(ke)
        if not (0 < ke < tissue_pole):
            continue                      # k_degt would go negative
        emax = target_gain * p.EC50 / (p.dose * p.k_out * p.k_up * ke)
        if emax > 1.0:
            continue
        out.append(p.with_(k_esc=ke, k_degt=tissue_pole - ke, Emax=emax))
    return out


# ---------------------------------------------------------------------------
# 3. does the degeneracy change a decision?
# ---------------------------------------------------------------------------
def decision_verdicts(members: list[Params], dose_factor: float,
                      thresholds=(0.50, 0.70)) -> dict:
    """Re-run an untested prediction on every point of a degeneracy set and
    tabulate the spread of verdicts.

    A 'PASS' means the nadir is at or below the threshold fraction of baseline,
    i.e. the compound achieves the required depth of suppression.
    """
    nadirs = [nadir(m.with_(dose=m.dose * dose_factor)) for m in members]
    out = {"dose_factor": dose_factor,
           "n_members": len(members),
           "nadir_min": float(min(nadirs)),
           "nadir_max": float(max(nadirs)),
           "nadirs": [float(x) for x in nadirs],
           "rules": []}
    for thr in thresholds:
        verdicts = ["PASS" if x <= thr else "FAIL" for x in nadirs]
        n_pass = verdicts.count("PASS")
        out["rules"].append({
            "threshold": thr,
            "n_pass": n_pass,
            "n_fail": len(verdicts) - n_pass,
            "unanimous": n_pass in (0, len(verdicts)),
            "verdicts": verdicts,
        })
    return out


# ---------------------------------------------------------------------------
# 4. the pole-assignment (flip-flop) ambiguity
# ---------------------------------------------------------------------------
def phi_validity_window(p: Params, dose_factors=None) -> dict:
    """Dose range over which phi = Emax*k_esc/EC50 is a valid description.

    phi lives in the linear approximation.  For each dose this compares the
    depth of silencing actually achieved against the error the linear
    approximation makes, and reports the deepest suppression reachable while
    the approximation still holds to 5 % relative.

    The structural point: deep silencing requires operating near the Emax
    ceiling, and the linear approximation requires staying far below it.  Those
    two requirements are in direct conflict, so the window where phi is valid
    may sit entirely below the assay noise floor.
    """
    if dose_factors is None:
        dose_factors = np.logspace(-4, 1, 40)
    t = np.linspace(0.0, 180.0, 2000)

    def _point(f: float) -> dict:
        q = p.with_(dose=p.dose * float(f))
        _, y = simulate(q, t_eval=t, rtol=1e-11, atol=1e-14)
        supp_full = 1.0 - float((y["P"] / q.P0).min())
        supp_lin = 1.0 - float((1.0 + linear_protein(q, t)).min())
        return {"dose_factor": float(f),
                "A_r_max_over_EC50": float(y["A_r"].max() / q.EC50),
                "suppression_full": supp_full,
                "suppression_linear": supp_lin,
                "relative_error": abs(supp_lin - supp_full) / max(supp_full, 1e-12)}

    rows = [_point(f) for f in dose_factors]
    tol = 0.05

    # The crossing is bracketed and solved for, not read off the grid: the log
    # grid steps by ~30 % in dose, so taking the largest grid point inside
    # tolerance understates the answer by roughly that much.
    lo = max((r["dose_factor"] for r in rows if r["relative_error"] <= tol),
             default=None)
    hi = min((r["dose_factor"] for r in rows if r["relative_error"] > tol),
             default=None)
    if lo is None or hi is None:
        crossing, supp_at_crossing = float("nan"), float("nan")
    else:
        crossing = brentq(lambda f: _point(f)["relative_error"] - tol, lo, hi,
                          xtol=1e-10)
        supp_at_crossing = _point(crossing)["suppression_full"]

    return {"curve": rows,
            "tolerance": tol,
            "dose_factor_at_tolerance": float(crossing),
            "max_suppression_within_tolerance": float(supp_at_crossing),
            "grid_only_estimate":
                float(max((r["suppression_full"] for r in rows
                           if r["relative_error"] <= tol), default=0.0)),
            "reference_dose_relative_error": float(_point(1.0)["relative_error"])}


# ---------------------------------------------------------------------------
# 4. the pole-assignment (flip-flop) ambiguity
# ---------------------------------------------------------------------------
def pole_swap_check(p: Params) -> dict:
    """Swap k_loss and k_out, then refit, and compare the protein curves.

    The linear transfer function is  gain / prod(s + p_i).  The denominator is
    symmetric under permutation of its roots and the numerator is a single
    number, so any reassignment of poles to compartments can be absorbed by
    rescaling the lumped gain.  From protein data alone you recover the SET of
    time constants, not which compartment each belongs to -- which means the
    50-day time constant being RISC persistence rather than slow target
    turnover is a modelling assumption, not an inference from the data.

    Two tests:
      * linear regime  -- analytic, exact, no fitting
      * full model     -- refit Emax and EC50 of the swapped assignment to the
                          reference protein curve and report the residual
    """
    t = np.linspace(0.0, 180.0, 3000)

    # (a) exact linear check: rescale the gain, compare analytically
    sw = p.with_(k_loss=p.k_out, k_out=p.k_loss)
    sw = sw.with_(k_syn=sw.k_out)                       # hold baseline at 1
    sw_lin = sw.with_(EC50=sw.EC50 * sw.observable_gain / p.observable_gain)
    lin_err = float(np.abs(linear_protein(sw_lin, t) - linear_protein(p, t)).max()
                    / np.abs(linear_protein(p, t)).max())

    # (b) full model: best achievable fit of the swapped assignment
    _, yref = simulate(p, t_eval=t)
    ref = yref["P"] / p.P0

    def rmse(theta):
        emax, log_ec = theta
        emax = float(np.clip(emax, 1e-3, 1.0))
        q = sw.with_(Emax=emax, EC50=float(np.exp(log_ec)))
        _, y = simulate(q, t_eval=t)
        return float(np.sqrt(np.mean((y["P"] / q.P0 - ref) ** 2)))

    from scipy.optimize import minimize
    best = min((minimize(rmse, x0, method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 600})
                for x0 in ([0.9, np.log(p.EC50)], [0.6, np.log(p.EC50 * 10)])),
               key=lambda r: r.fun)

    return {
        "reference": {"tau_risc_days": 1 / p.k_loss,
                      "tau_protein_days": 1 / p.k_out},
        "swapped": {"tau_risc_days": 1 / sw.k_loss,
                    "tau_protein_days": 1 / sw.k_out},
        "linear_regime_max_relative_difference": lin_err,
        "full_model_best_fit_rmse_pp": float(100 * best.fun),
        "full_model_refit": {"Emax": float(np.clip(best.x[0], 1e-3, 1.0)),
                             "EC50": float(np.exp(best.x[1]))},
    }
