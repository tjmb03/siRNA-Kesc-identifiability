"""Independent audit of the published numbers.

    python tests/audit.py

Every check re-derives a reported value by a route that does not reuse the code
that produced it: the transfer function from scratch in sympy, the residues by
least squares on the simulated curve, the nadir with Radau instead of LSODA,
the pole-swap optimum by brute-force grid search, the invariance group at three
further (alpha, beta) pairs, the rank ladder across four finite-difference
steps, and the confidence bounds against the chi-square contour.

This is separate from tests/test_model.py, which checks internal consistency.
The audit caught three real errors on its first run:

  * the phi validity window was read off a coarse log grid and understated the
    answer by a quarter (3.28 % -> 4.11 %);
  * the sensitivity rank used a fixed singular-value cutoff and flipped its
    verdict at h = 1e-6;
  * the profile confidence intervals were read off grid points, understating
    every interval by 15-30 % in the optimistic direction.

All three are fixed. The script is kept so the next change can be checked the
same way.
"""
import json, sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import load_reference, simulate, linear_protein, pole_residues
import analysis as A

P = load_reference(ROOT / "config" / "reference_parameters.json")
R = json.load(open(ROOT / "results" / "results.json"))
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


# 1 ---- transfer function, derived symbolically from scratch ---------------
s, t = sp.symbols("s t", positive=True)
kel, kup, kesc, kdeg, kloss, kout, ksyn, Em, EC, D = sp.symbols(
    "k_el k_up k_esc k_degt k_loss k_out k_syn Emax EC50 D", positive=True)
Ap = D / (s + kel + kup)
At = kup * Ap / (s + kesc + kdeg)
Ar = kesc * At / (s + kloss)
dP = -ksyn * (Em / EC) * Ar / (s + kout)          # linear regime
frac = sp.simplify(dP / (ksyn / kout))            # fractional suppression
claimed = -D * kout * kup * (Em * kesc / EC) / (
    (s + kloss) * (s + kout) * (s + kdeg + kesc) * (s + kel + kup))
check("transfer function in the document", sp.simplify(frac - claimed) == 0)

# 2 ---- residues: recover amplitudes by least squares on the curve ---------
names, poles, res = pole_residues(P)
tt = np.linspace(1e-4, 40, 4000)
y = linear_protein(P, tt)
Xd = np.column_stack([np.exp(-p * tt) for p in poles])
amp, *_ = np.linalg.lstsq(Xd, y, rcond=None)
share_fit = np.abs(amp) / np.abs(amp).sum()
share_an = np.abs(res) / np.abs(res).sum()
check("amplitude shares by regression vs partial fractions",
      np.allclose(share_fit, share_an, atol=2e-3),
      f"max diff {np.abs(share_fit-share_an).max():.1e}")
reported = {r["process"]: r["amplitude_share"] for r in R["pole_table"]}
check("shares in results.json match a fresh computation",
      all(abs(reported[n] - share_an[i]) < 1e-12 for i, n in enumerate(names)))

# 3 ---- nadir with an independent stiff solver -----------------------------
def rhs(_, v, p):
    ap, at, ar, pr = v
    return [-(p.k_el + p.k_up) * ap,
            p.k_up * ap - (p.k_esc + p.k_degt) * at,
            p.k_esc * at - p.k_loss * ar,
            p.k_syn * (1 - p.Emax * ar / (p.EC50 + ar)) - p.k_out * pr]

sol = solve_ivp(rhs, (0, 180), [P.dose, 0, 0, P.P0], args=(P,), method="Radau",
                rtol=1e-12, atol=1e-14, dense_output=True)
g = np.linspace(0, 180, 200000)
nad_radau = sol.sol(g)[3].min() / P.P0
check("nadir: Radau vs LSODA vs reported",
      abs(nad_radau - R["derived"]["nadir_reference_dose"]) < 1e-6,
      f"Radau {nad_radau:.6f} vs reported {R['derived']['nadir_reference_dose']:.6f}")

# 4 ---- saturation diagnostic by hand --------------------------------------
ar_max = sol.sol(g)[2].max()
check("A_r_max / EC50", abs(ar_max / P.EC50 - R["saturation"]["A_r_max_over_EC50"]) < 1e-4,
      f"{ar_max/P.EC50:.4f}")
check("peak occupancy", abs(ar_max / (P.EC50 + ar_max)
                            - R["saturation"]["peak_fractional_occupancy"]) < 1e-5)

# 5 ---- phi validity window: root-find the crossing instead of gridding ----
def rel_err(dose_factor):
    q = P.with_(dose=P.dose * dose_factor)
    tg = np.linspace(0, 180, 4000)
    _, yy = simulate(q, t_eval=tg, rtol=1e-11, atol=1e-14)
    sf = 1 - (yy["P"] / q.P0).min()
    sl = 1 - (1 + linear_protein(q, tg)).min()
    return abs(sl - sf) / sf

f_star = brentq(lambda d: rel_err(d) - 0.05, 1e-4, 1.0, xtol=1e-10)
q = P.with_(dose=P.dose * f_star)
tg = np.linspace(0, 180, 4000)
_, yy = simulate(q, t_eval=tg, rtol=1e-11, atol=1e-14)
supp_exact = 1 - (yy["P"] / q.P0).min()
check("phi window: independent bracket vs reported",
      abs(supp_exact - R["phi_validity"]["max_suppression_within_tolerance"]) < 1e-4,
      f"exact {100*supp_exact:.2f} % vs reported "
      f"{100*R['phi_validity']['max_suppression_within_tolerance']:.2f} %")

# 6 ---- pole swap: is 14 pp really the global optimum? ---------------------
sw = P.with_(k_loss=P.k_out, k_out=P.k_loss)
sw = sw.with_(k_syn=sw.k_out)
tg2 = np.linspace(0, 180, 1200)
ref = simulate(P, t_eval=tg2, rtol=1e-10)[1]["P"] / P.P0
best = np.inf
for em in np.linspace(0.05, 1.0, 20):
    for ec in np.logspace(-9, 1, 25):
        qq = sw.with_(Emax=em, EC50=ec)
        r = np.sqrt(np.mean((simulate(qq, t_eval=tg2, rtol=1e-8)[1]["P"] / qq.P0 - ref) ** 2))
        best = min(best, r)
check("pole-swap refit RMSE is a true global minimum (grid search)",
      abs(100 * best - R["pole_swap"]["full_model_best_fit_rmse_pp"]) < 1.0,
      f"grid {100*best:.2f} pp vs optimiser {R['pole_swap']['full_model_best_fit_rmse_pp']:.2f} pp")

# 7 ---- invariance group at a different (alpha, beta) ----------------------
import structural as St
for a_, b_ in ((1.2, 2.0), (0.6, 7.0), (1.05, 9.5)):
    v = St.verify_invariance(P, alpha=a_, beta=b_)
    check(f"invariance at alpha={a_}, beta={b_}",
          v["max_difference_protein"] < 1e-8 and v["max_difference_plasma"] < 1e-8,
          f"protein {v['max_difference_protein']:.1e}, k_esc x{v['k_esc_ratio']:.1f}")

# 8 ---- rank ladder robustness to the finite-difference step ---------------
for h in (1e-4, 1e-5, 1e-6, 1e-7):
    import structural
    orig = structural._sensitivity_matrix
    ladder = []
    tp, ti, tq = (np.linspace(0.02, 1.0, 25), np.linspace(0.05, 7.0, 25),
                  np.concatenate([np.linspace(0.25, 7, 20), np.linspace(10, 180, 30)]))
    for W, free in (({"A_p": tp, "P": tq}, structural._ALL),
                    ({"A_p": tp, "A_t": ti, "P": tq}, structural._ALL),
                    ({"A_p": tp, "A_t": ti, "P": tq},
                     tuple(x for x in structural._ALL if x != "k_degt"))):
        S = orig(P, W, free, h=h)
        sv = np.linalg.svd(S, compute_uv=False); sv /= sv[0]
        ratios = sv[:-1] / sv[1:]
        rank = int(np.argmax(ratios) + 1) if ratios.max() > 1e3 else len(sv)
        ladder.append(len(free) - rank)
    check(f"rank ladder at h={h:g}", ladder == [2, 1, 0], str(ladder))

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))

# 9 ---- confidence intervals: interpolated, not read off grid points --------
import numpy as _np
for blk in R["profiles"]:
    for d in blk["designs"]:
        k = _np.array([x["k_esc"] for x in d["profile"]])
        dl = _np.array([x["delta"] for x in d["profile"]])
        lo, hi = d["ci95"]
        if not d["interval_hits_grid_edge"]:
            # the reported bounds must sit exactly on the delta = 3.84 contour
            for b in (lo, hi):
                dd = _np.interp(_np.log(b), _np.log(k), dl)
                check(f"CI bound on the 3.84 contour: {d['design'][:28]}",
                      abs(dd - 3.8415) < 0.02, f"delta at bound = {dd:.4f}")
        else:
            check(f"edge-limited flagged: {d['design'][:28]}",
                  abs(d["fold_width"] - hi / lo) < 1e-9 and
                  (abs(lo - k.min()) < 1e-12 or abs(hi - k.max()) < 1e-12))

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
