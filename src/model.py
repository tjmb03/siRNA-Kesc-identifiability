"""
Minimal siRNA PK/PD model.

    plasma  ->  tissue (endosomal)  ->  RISC-loaded  ->  target protein

Four states.  Plasma and protein are measurable; the RISC pool is not; total
tissue siRNA is measurable in principle (stem-loop RT-PCR) and the question of
whether you measure it is the crux of the identifiability argument.

    dA_p/dt = -(k_el + k_up) * A_p
    dA_t/dt =  k_up * A_p - (k_esc + k_degt) * A_t
    dA_r/dt =  k_esc * A_t - k_loss * A_r
    dP/dt   =  k_syn * (1 - Emax * A_r / (EC50 + A_r)) - k_out * P

The bracket is an Emax term multiplying a synthesis rate: an indirect-response
structure, not Michaelis-Menten kinetics.

All parameters are illustrative.  The transferable content is the structure of
the argument, not the numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["Params", "load_reference", "simulate", "linear_protein",
           "nadir", "pole_residues", "POLE_NAMES"]

POLE_NAMES = ("plasma", "tissue", "risc", "protein")


@dataclass(frozen=True)
class Params:
    """Rate constants (1/day) and PD parameters.  Concentrations are in
    arbitrary but consistent units; protein is normalised so baseline = 1."""
    k_el: float
    k_up: float
    k_esc: float
    k_degt: float
    k_loss: float
    k_out: float
    k_syn: float
    Emax: float
    EC50: float
    dose: float

    # ---- derived quantities -------------------------------------------------
    @property
    def P0(self) -> float:
        """Baseline protein, k_syn / k_out."""
        return self.k_syn / self.k_out

    @property
    def uptake_fraction(self) -> float:
        """Fraction of dose delivered to tissue rather than cleared."""
        return self.k_up / (self.k_el + self.k_up)

    @property
    def escape_fraction(self) -> float:
        """Fraction of tissue siRNA that escapes rather than being degraded."""
        return self.k_esc / (self.k_esc + self.k_degt)

    @property
    def phi(self) -> float:
        """phi = Emax * k_esc / EC50 -- 'total silencing per unit dose' as
        defined in the original note."""
        return self.Emax * self.k_esc / self.EC50

    @property
    def observable_gain(self) -> float:
        """The quantity the protein time course actually pins down.

        Working in fractional suppression dP/P0 removes k_syn, leaving

            gain = D * k_out * k_up * Emax * k_esc / EC50

        so uptake (k_up) and protein turnover (k_out) are folded in alongside
        escape and potency.  phi alone is NOT the observable.
        """
        return self.dose * self.k_out * self.k_up * self.phi

    @property
    def poles(self) -> dict[str, float]:
        """Eigenvalue magnitudes of the linearised cascade, 1/day."""
        return {
            "plasma": self.k_el + self.k_up,
            "tissue": self.k_esc + self.k_degt,
            "risc": self.k_loss,
            "protein": self.k_out,
        }

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def load_reference(path: str | Path) -> Params:
    with open(path) as fh:
        cfg = json.load(fh)
    return Params(**{k: float(v) for k, v in cfg["parameters"].items()})


# ---------------------------------------------------------------------------
# full (nonlinear) model
# ---------------------------------------------------------------------------
def _rhs(t, y, p: Params):
    A_p, A_t, A_r, P = y
    inhibition = p.Emax * A_r / (p.EC50 + A_r)
    return [
        -(p.k_el + p.k_up) * A_p,
        p.k_up * A_p - (p.k_esc + p.k_degt) * A_t,
        p.k_esc * A_t - p.k_loss * A_r,
        p.k_syn * (1.0 - inhibition) - p.k_out * P,
    ]


def simulate(p: Params, t_eval=None, t_end: float = 180.0, n: int = 2000,
             rtol: float = 1e-9, atol: float = 1e-12):
    """Integrate the full model after a bolus dose into plasma at t = 0.

    Stiff: the fastest pole (plasma, ~0.08 d) and the slowest (RISC, 50 d)
    differ by ~600x, so an implicit method is used rather than an algebraic
    reduction of the fast compartment.  Tolerances are loosened inside the
    likelihood (see identifiability.FIT_TOL) where thousands of solves are
    needed; reporting runs use the defaults.
    """
    if t_eval is None:
        t_eval = np.linspace(0.0, t_end, n)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
    y0 = [p.dose, 0.0, 0.0, p.P0]
    sol = solve_ivp(_rhs, (0.0, float(t_eval[-1])), y0, t_eval=t_eval,
                    args=(p,), method="LSODA", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.t, dict(zip(("A_p", "A_t", "A_r", "P"), sol.y))


def nadir(p: Params, t_end: float = 180.0, n: int = 4000) -> float:
    """Minimum protein as a fraction of baseline."""
    _, y = simulate(p, t_end=t_end, n=n)
    return float(y["P"].min() / p.P0)


# ---------------------------------------------------------------------------
# linear-regime analytic solution
# ---------------------------------------------------------------------------
def pole_residues(p: Params):
    """Partial-fraction residues of 1 / prod(s + p_i).

    Returns (names, poles, residues).  The time-domain response is

        sum_i  residue_i * exp(-pole_i * t)

    so |residue_i| / sum|residue| is the amplitude share of each process --
    the metric reported as 'share of the response' in the original note.
    """
    names = list(POLE_NAMES)
    pv = np.array([p.poles[k] for k in names], dtype=float)
    if len(set(np.round(pv, 12))) != len(pv):
        raise ValueError("repeated poles; partial fractions not distinct")
    res = np.array([1.0 / np.prod(np.delete(pv, i) - pv[i]) for i in range(len(pv))])
    return names, pv, res


def linear_protein(p: Params, t):
    """Fractional protein deviation dP(t)/P0 in the linear approximation,
    where Emax*A_r/(EC50+A_r) ~ (Emax/EC50)*A_r.

    Inverse Laplace of  -gain / prod(s + p_i).
    """
    t = np.asarray(t, dtype=float)
    _, pv, res = pole_residues(p)
    return -p.observable_gain * np.sum(
        res[:, None] * np.exp(-pv[:, None] * t[None, :]), axis=0)
