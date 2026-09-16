"""
Structural identifiability of the minimal siRNA model.

GenSSI / SIAN return a verdict.  This returns the degeneracy itself: the exact
group of parameter transformations that leave the observables invariant, proved
symbolically and then verified against the full nonlinear integrator.

The argument, in three steps:

 1. The upstream subsystem is linear, so it can be solved in closed form.  Doing
    so shows

        A_t(t) = D * k_up * g(t; a, b)
        A_r(t) = D * k_up * k_esc * f(t; a, b, k_loss)

    with a = k_el + k_up and b = k_esc + k_degt.  Every parameter enters the
    RISC pool either as a pole or through the single product D*k_up*k_esc.

 2. The Emax term is homogeneous of degree zero in (A_r, EC50):

        Emax * A_r / (EC50 + A_r)  =  Emax * (psi*f) / (1 + psi*f),
        psi = D * k_up * k_esc / EC50

    so the protein equation sees the nine parameters only through the seven
    quantities  {k_syn, k_out, k_loss, Emax, a, b, psi}.

 3. Therefore the level sets of that map are a two-parameter group.  With
    alpha, beta > 0:

        k_up   -> alpha*k_up        k_el   -> a - alpha*k_up
        k_esc  -> beta*k_esc        k_degt -> b - beta*k_esc
        EC50   -> alpha*beta*EC50

    leaves plasma and protein pointwise unchanged.  k_esc is not structurally
    identifiable from (A_p, P) at any sampling density, with any precision.

Observing total tissue siRNA identifies k_up and collapses the group to one
dimension; adding k_degt from a stability assay collapses it to nothing.  That
ladder is what `identifiability_ladder` measures numerically.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from model import Params, simulate

__all__ = ["symbolic_risc_factorisation", "invariance_group",
           "verify_invariance", "identifiability_ladder"]


# ---------------------------------------------------------------------------
# 1. closed form for the linear subsystem
# ---------------------------------------------------------------------------
def symbolic_risc_factorisation() -> dict:
    """Prove the factorisation by substitution rather than by inverting the
    Laplace transform (which sympy can do, slowly, with no extra rigour).

    Substituting the scaled states

        u = A_p / D,   v = A_t / (D*k_up),   w = A_r / (D*k_up*k_esc)

    into the upstream ODEs must leave a system whose coefficients are the three
    poles and nothing else.  Each residual below is simplified to zero
    symbolically; that is the proof.
    """
    t = sp.symbols("t", nonnegative=True)
    D, k_el, k_up, k_esc, k_degt, k_loss, Emax, EC50 = sp.symbols(
        "D k_el k_up k_esc k_degt k_loss Emax EC50", positive=True)
    a, b = k_el + k_up, k_esc + k_degt

    u, v, w = (sp.Function(n)(t) for n in ("u", "v", "w"))
    A_p, A_t, A_r = D * u, D * k_up * v, D * k_up * k_esc * w

    residuals = {
        "dA_p/dt + (k_el+k_up)A_p": sp.diff(A_p, t) + a * A_p,
        "dA_t/dt - k_up*A_p + (k_esc+k_degt)A_t":
            sp.diff(A_t, t) - k_up * A_p + b * A_t,
        "dA_r/dt - k_esc*A_t + k_loss*A_r":
            sp.diff(A_r, t) - k_esc * A_t + k_loss * A_r,
    }
    # after dividing out the lumped prefactor, each residual must be the scaled
    # equation, carrying only poles
    scaled = {
        "u": sp.simplify(residuals["dA_p/dt + (k_el+k_up)A_p"] / D),
        "v": sp.simplify(residuals["dA_t/dt - k_up*A_p + (k_esc+k_degt)A_t"]
                         / (D * k_up)),
        "w": sp.simplify(residuals["dA_r/dt - k_esc*A_t + k_loss*A_r"]
                         / (D * k_up * k_esc)),
    }
    # re-express in terms of the poles: k_el = a - k_up, k_degt = b - k_esc.
    # If the factorisation is right, D, k_up and k_esc must all cancel.
    A, B = sp.symbols("a b", positive=True)
    in_poles = {k: sp.simplify(sp.expand(
        expr.subs({k_el: A - k_up, k_degt: B - k_esc})))
        for k, expr in scaled.items()}
    lumped = {D, k_up, k_esc}
    carries_lump = {k: sorted(str(x) for x in expr.free_symbols & lumped)
                    for k, expr in in_poles.items()}

    psi = sp.symbols("psi", positive=True)
    emax_term = Emax * A_r / (EC50 + A_r)
    rewritten = sp.simplify(
        emax_term.subs(EC50, D * k_up * k_esc / psi) - Emax * psi * w / (1 + psi * w))

    return {
        "scaled_equations": {k: str(sp.expand(v)) for k, v in in_poles.items()},
        "lumped_parameters_remaining_after_scaling": carries_lump,
        "factorisation_holds": all(not v for v in carries_lump.values()),
        "emax_term_rewrite_residual": str(rewritten),
        "emax_rewrite_holds": rewritten == 0,
        "identifiable_functions": [
            "k_el + k_up", "k_esc + k_degt", "k_loss", "k_out", "k_syn",
            "Emax", "psi = D*k_up*k_esc/EC50"],
        "conclusion": (
            "nine parameters enter the observables through seven functions, so "
            "the degeneracy is two-dimensional and k_esc lies on it"),
    }


# ---------------------------------------------------------------------------
# 2. the invariance group
# ---------------------------------------------------------------------------
def invariance_group(p: Params, alpha: float, beta: float) -> Params:
    """Apply the two-parameter transformation that leaves (A_p, P) fixed.

    Raises if the transform would push a rate constant negative, which bounds
    the group to alpha < (k_el+k_up)/k_up and beta < (k_esc+k_degt)/k_esc.
    """
    a, b = p.k_el + p.k_up, p.k_esc + p.k_degt
    k_up, k_esc = alpha * p.k_up, beta * p.k_esc
    k_el, k_degt = a - k_up, b - k_esc
    if k_el <= 0 or k_degt <= 0:
        raise ValueError(
            f"alpha must be < {a/p.k_up:.3f} and beta < {b/p.k_esc:.3f}")
    return p.with_(k_up=k_up, k_el=k_el, k_esc=k_esc, k_degt=k_degt,
                   EC50=alpha * beta * p.EC50)


def verify_invariance(p: Params, alpha: float = 1.15, beta: float = 4.0,
                      t_end: float = 180.0) -> dict:
    """Integrate the full nonlinear model under the transformation and compare.

    This is the check that matters: the proof is a statement about the ODEs,
    and this confirms the integrator agrees with it.
    """
    q = invariance_group(p, alpha, beta)
    t = np.linspace(0.0, t_end, 1200)
    _, y0 = simulate(p, t_eval=t, rtol=1e-11, atol=1e-14)
    _, y1 = simulate(q, t_eval=t, rtol=1e-11, atol=1e-14)

    dP = float(np.abs(y1["P"] / q.P0 - y0["P"] / p.P0).max())
    dAp = float(np.abs(y1["A_p"] - y0["A_p"]).max() / p.dose)
    dAt = float(y1["A_t"].max() / y0["A_t"].max())
    return {
        "alpha": alpha, "beta": beta,
        "original": {"k_el": p.k_el, "k_up": p.k_up, "k_esc": p.k_esc,
                     "k_degt": p.k_degt, "EC50": p.EC50,
                     "escape_fraction": p.escape_fraction,
                     "uptake_fraction": p.uptake_fraction},
        "transformed": {"k_el": q.k_el, "k_up": q.k_up, "k_esc": q.k_esc,
                        "k_degt": q.k_degt, "EC50": q.EC50,
                        "escape_fraction": q.escape_fraction,
                        "uptake_fraction": q.uptake_fraction},
        "k_esc_ratio": q.k_esc / p.k_esc,
        "max_difference_protein": dP,
        "max_difference_plasma": dAp,
        "tissue_peak_ratio": dAt,
        "invariants": {"k_el+k_up": q.k_el + q.k_up,
                       "k_esc+k_degt": q.k_esc + q.k_degt,
                       "psi = D*k_up*k_esc/EC50":
                           q.dose * q.k_up * q.k_esc / q.EC50},
    }


# ---------------------------------------------------------------------------
# 3. numerical rank of the sensitivity map
# ---------------------------------------------------------------------------
_ALL = ("k_el", "k_up", "k_esc", "k_degt", "k_loss", "k_out", "k_syn",
        "Emax", "EC50")


def _sensitivity_matrix(p: Params, windows: dict[str, np.ndarray],
                        free: tuple[str, ...], h: float = 1e-5):
    """Relative sensitivities d log y / d log theta by central differences.

    Each observable gets its own sampling window.  Plasma is only sampled while
    it is above the limit of quantification: past ~1 day A_p has underflowed to
    numerical zero, and relative sensitivities computed there are integrator
    noise that swamps the SVD.
    """
    def obs(q: Params):
        v = []
        for name, t in windows.items():
            y = simulate(q, t_eval=t, rtol=1e-12, atol=1e-16)[1][name]
            v.append(y)
        return v

    base = obs(p)
    cols = []
    for name in free:
        v = getattr(p, name)
        up, dn = obs(p.with_(**{name: v * (1 + h)})), obs(p.with_(**{name: v * (1 - h)}))
        col = [(u - d) / (2 * h) / b for u, d, b in zip(up, dn, base)]
        cols.append(np.concatenate(col))
    return np.column_stack(cols)


def _generators(p: Params) -> dict[str, np.ndarray]:
    """The two group generators in log-parameter coordinates.

    Differentiating the transformation at alpha = beta = 1 gives
        alpha:  dlog k_up = 1, dlog k_el = -k_up/k_el, dlog EC50 = 1
        beta:   dlog k_esc = 1, dlog k_degt = -k_esc/k_degt, dlog EC50 = 1
    """
    idx = {n: i for i, n in enumerate(_ALL)}
    out = {}
    for name, (a, b) in {"alpha": ("k_up", "k_el"),
                         "beta": ("k_esc", "k_degt")}.items():
        v = np.zeros(len(_ALL))
        v[idx[a]] = 1.0
        v[idx[b]] = -getattr(p, a) / getattr(p, b)
        v[idx["EC50"]] = 1.0
        out[name] = v
    return out


def identifiability_ladder(p: Params, tol: float = 1e-7,
                           gap_threshold: float = 1e3) -> dict:
    """Rank of the sensitivity map under each experimental design.

    Rank is taken at the largest gap in the singular value spectrum rather than
    at a fixed cutoff.  A fixed cutoff is not safe here: the smallest singular
    value moves by two orders of magnitude as the finite-difference step varies
    between 1e-3 and 1e-7, because central differences trade truncation error
    against integrator roundoff.  The gap does not move -- it sits after the
    same index across that whole range.

    As a check that does not depend on any threshold, the analytic generators
    are applied to the sensitivity matrix directly: a genuine null direction
    gives ||S v|| / (||S|| ||v||) at the integrator noise floor.

    Valid over finite-difference steps h in roughly [1e-6, 1e-4]; outside that
    the verdict is not trustworthy and the generator residuals are the evidence
    to rely on.
    """
    # each readout sampled only where it is actually quantifiable
    plasma_t = np.linspace(0.02, 1.0, 25)
    tissue_t = np.linspace(0.05, 7.0, 25)
    protein_t = np.concatenate([np.linspace(0.25, 7.0, 20),
                                np.linspace(10.0, 180.0, 30)])
    W_pp = {"A_p": plasma_t, "P": protein_t}
    W_ppt = {"A_p": plasma_t, "A_t": tissue_t, "P": protein_t}
    designs = [
        {"label": "plasma + protein", "windows": W_pp, "free": _ALL},
        {"label": "plasma + protein + total tissue siRNA",
         "windows": W_ppt, "free": _ALL},
        {"label": "the above, with k_degt fixed by a stability assay",
         "windows": W_ppt, "free": tuple(x for x in _ALL if x != "k_degt")},
    ]
    out = []
    for d in designs:
        S = _sensitivity_matrix(p, d["windows"], d["free"])
        sv = np.linalg.svd(S, compute_uv=False)
        sv = sv / sv[0]
        ratios = sv[:-1] / sv[1:]
        # A gap only means a rank deficiency if it is large.  A full-rank
        # matrix still has a largest ratio -- here around 4 -- and reading that
        # as a null space would invent degeneracies that are not there.
        gap = float(ratios.max())
        rank = int(np.argmax(ratios) + 1) if gap > gap_threshold else len(sv)
        _, _, Vt = np.linalg.svd(S)
        null = Vt[rank:]

        norm_S = float(np.linalg.norm(S, 2))
        gen_resid = {}
        if len(d["free"]) == len(_ALL):
            for name, v in _generators(p).items():
                gen_resid[name] = float(
                    np.linalg.norm(S @ v) / (norm_S * np.linalg.norm(v)))

        out.append({
            "design": d["label"],
            "n_free": len(d["free"]),
            "rank": rank,
            "unresolved_directions": len(d["free"]) - rank,
            "spectrum_gap": gap,
            "gap_exceeds_threshold": bool(gap > gap_threshold),
            "rank_by_fixed_tolerance": int((sv > tol).sum()),
            "singular_values_normalised": [float(x) for x in sv],
            "generator_residuals": gen_resid,
            "null_space": [
                {n: round(float(v), 3) for n, v in zip(d["free"], vec)
                 if abs(v) > 0.05}
                for vec in null],
        })
    return {"tolerance": tol,
            "n_timepoints": {"plasma": len(plasma_t), "tissue": len(tissue_t),
                             "protein": len(protein_t)},
            "designs": out}
