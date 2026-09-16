"""
Tests.  The point of these is that the claims in the document are checkable,
not that the code runs.

    python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import load_reference, simulate, linear_protein, pole_residues  # noqa: E402
import analysis as A                                                       # noqa: E402

CONFIG = ROOT / "config" / "reference_parameters.json"


@pytest.fixture(scope="module")
def p():
    return load_reference(CONFIG)


def test_plasma_is_exactly_monoexponential(p):
    """A_p decouples; check the integrator against the closed form."""
    t = np.linspace(0, 5, 60)
    _, y = simulate(p, t_eval=t)
    exact = p.dose * np.exp(-(p.k_el + p.k_up) * t)
    # absolute scale is the dose; beyond ~2 d the exact value underflows to
    # numerical zero and a relative comparison stops being meaningful
    assert np.abs(y["A_p"] - exact).max() < 1e-8 * p.dose
    alive = exact > 1e-6 * p.dose
    assert np.abs(y["A_p"][alive] / exact[alive] - 1).max() < 1e-6


def test_linear_solution_matches_integrator_at_low_dose(p):
    """The Laplace-derived expression is the low-dose limit of the full model."""
    q = p.with_(dose=p.dose * 1e-4)
    t = np.linspace(0, 120, 400)
    _, y = simulate(q, t_eval=t)
    numeric = y["P"] / q.P0 - 1.0
    analytic = linear_protein(q, t)
    scale = np.abs(numeric).max()
    assert np.abs(numeric - analytic).max() / scale < 1e-3


def test_residue_shares_are_a_partition(p):
    _, _, res = pole_residues(p)
    share = np.abs(res) / np.abs(res).sum()
    assert pytest.approx(1.0, abs=1e-12) == share.sum()


def test_pole_assignment_is_exactly_degenerate_in_linear_regime(p):
    """Swapping k_loss and k_out is unobservable once the gain is rescaled."""
    r = A.pole_swap_check(p)
    assert r["linear_regime_max_relative_difference"] < 1e-12


def test_iso_nadir_members_all_reproduce_the_nadir(p):
    from model import nadir
    target = nadir(p)
    members = A.iso_nadir_set(p, np.linspace(0.12, 1.8, 6))
    assert len(members) >= 4
    for m in members:
        assert abs(nadir(m) - target) < 1e-6


def test_linear_degeneracy_set_is_exactly_flat(p):
    """Every member gives a bit-for-bit identical linear protein curve."""
    t = np.linspace(0, 180, 500)
    members = A.linear_degeneracy_set(p, np.linspace(0.05, 1.2, 8))
    assert len(members) >= 4
    ref = linear_protein(members[0], t)
    for m in members[1:]:
        assert np.abs(linear_protein(m, t) - ref).max() / np.abs(ref).max() < 1e-10


def test_reference_dose_is_outside_the_linear_range(p):
    """Guards the section 6 claim: phi's validity window does not contain the
    dose that produces the worked example."""
    sat = A.saturation_diagnostic(p)
    assert sat["A_r_max_over_EC50"] > 1.0


@pytest.mark.skipif(not (ROOT / "results" / "results.json").exists(),
                    reason="run run_all.py first")
def test_results_file_is_clean():
    R = json.loads((ROOT / "results" / "results.json").read_text())
    flat = json.dumps(R)
    assert "NaN" not in flat and "Infinity" not in flat
    for key in ("pole_table", "manifold", "verdicts_low_dose", "profiles",
                "phi_validity", "pole_swap"):
        assert key in R


@pytest.mark.skipif(not (ROOT / "results" / "results.json").exists(),
                    reason="run run_all.py first")
def test_document_has_no_hard_coded_science():
    """The builder may contain layout numbers, but no results should be typed
    into it: every reported quantity must come through the results dict."""
    src = (ROOT / "src" / "build_pdf.py").read_text()
    body = src.split("def main(", 1)[1]          # skip the style definitions
    for forbidden in ("13.5", "36.7", "49.9", "0.74", "13.3", "1.52"):
        assert forbidden not in body, f"{forbidden!r} is typed into build_pdf.py"


# --- structural identifiability -------------------------------------------
def test_symbolic_factorisation_holds():
    """D, k_up and k_esc must cancel out of the scaled upstream system, and the
    Emax term must reduce exactly to a function of psi."""
    import structural as St
    r = St.symbolic_risc_factorisation()
    assert r["factorisation_holds"]
    assert r["emax_rewrite_holds"]


def test_invariance_group_leaves_observables_fixed(p):
    """A four-fold change in k_esc that the data cannot see."""
    import structural as St
    r = St.verify_invariance(p, alpha=1.15, beta=4.0)
    assert r["k_esc_ratio"] == pytest.approx(4.0)
    assert r["max_difference_protein"] < 1e-8
    assert r["max_difference_plasma"] < 1e-8
    # and it really is a different delivery story
    assert r["transformed"]["escape_fraction"] > 3 * r["original"]["escape_fraction"]


def test_identifiability_ladder_is_two_one_zero(p):
    """Each added observation should close exactly one degenerate direction."""
    import structural as St
    L = St.identifiability_ladder(p)
    assert [d["unresolved_directions"] for d in L["designs"]] == [2, 1, 0]
    # the rank verdict is only meaningful if the spectrum gap is decisive
    assert L["designs"][0]["spectrum_gap"] > 1e4
    assert L["designs"][1]["spectrum_gap"] > 1e4
    assert not L["designs"][2]["gap_exceeds_threshold"], "design 3 should be full rank"
    # threshold-free evidence: the analytic generators are annihilated where
    # they should be, and not where they should not be
    g1 = L["designs"][0]["generator_residuals"]
    g2 = L["designs"][1]["generator_residuals"]
    assert g1["alpha"] < 1e-6 and g1["beta"] < 1e-6
    assert g2["beta"] < 1e-6, "beta stays degenerate even with tissue observed"
    assert g2["alpha"] > 1e-3, "observing tissue must resolve the alpha direction"


def test_phi_window_crossing_is_solved_not_gridded(p):
    """The reported window must come from a bracketed root, not the coarse
    log grid, which understates it by roughly one grid step."""
    w = A.phi_validity_window(p)
    assert w["max_suppression_within_tolerance"] > w["grid_only_estimate"]
    q = p.with_(dose=p.dose * w["dose_factor_at_tolerance"])
    import numpy as np
    from model import simulate, linear_protein
    t = np.linspace(0, 180, 2000)
    full = 1 - (simulate(q, t_eval=t, rtol=1e-11)[1]["P"] / q.P0).min()
    lin = 1 - (1 + linear_protein(q, t)).min()
    assert abs(abs(lin - full) / full - w["tolerance"]) < 1e-4



# --- README ----------------------------------------------------------------
@pytest.mark.skipif(not (ROOT / "results" / "results.json").exists(),
                    reason="run run_all.py first")
def test_readme_images_exist_and_numbers_match():
    """Every image the README embeds must be a file the pipeline produces, and
    the figures it quotes must still be the computed ones."""
    import re
    readme = (ROOT / "README.md").read_text()
    for rel in set(re.findall(r"\((results/figures/[\w.]+\.png)\)", readme)):
        assert (ROOT / rel).exists(), f"README embeds a missing image: {rel}"

    R = json.loads((ROOT / "results" / "results.json").read_text())
    folds = {d["key"]: d for blk in R["profiles"] for d in blk["designs"]
             if blk["free"] == ["Emax"]}
    for key, quoted in (("monthly_protein", "11.3-fold"),
                        ("dense_early_protein", "2.3-fold"),
                        ("protein_plus_tissue", "1.6-fold")):
        assert quoted in readme, f"README no longer quotes {quoted}"
        assert f"{folds[key]['fold_width']:.1f}-fold" == quoted, (
            f"{key}: computed {folds[key]['fold_width']:.1f}, README says {quoted}")

    supp = 100 * R["phi_validity"]["max_suppression_within_tolerance"]
    assert f"{supp:.0f} % of baseline" in readme, (
        f"README phi window is stale; computed {supp:.2f} %")
    ladder = [d["unresolved_directions"] for d in R["structural"]["ladder"]["designs"]]
    assert ladder == [2, 1, 0]
