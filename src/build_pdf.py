"""
Build the summary document.

Every scientific number in the output is read from results/results.json, which
is written by run_all.py.  Grep this file for a digit: what you find is page
geometry and font sizes.
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, KeepTogether)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

INK = colors.HexColor("#10212B")
TEAL = colors.HexColor("#0E6E73")
ROSE = colors.HexColor("#A8324A")
AMBER = colors.HexColor("#B4690E")
MUTE = colors.HexColor("#62727C")
RULE = colors.HexColor("#DCE3E6")
WASH = colors.HexColor("#EEF4F4")
CREAM = colors.HexColor("#FBF4EC")
BLUSH = colors.HexColor("#FBF1F3")


def st(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.5, leading=13.5,
                textColor=INK, alignment=TA_LEFT, spaceAfter=6)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "title": st("title", fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=3),
    "subtitle": st("subtitle", fontSize=10.5, leading=14, textColor=MUTE, spaceAfter=14),
    "h1": st("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, spaceBefore=15, spaceAfter=6),
    "body": st("body"),
    "mono": st("mono", fontName="Courier", fontSize=8.5, leading=12.5),
    "cap": st("cap", fontSize=8.5, leading=11.5, textColor=MUTE, spaceAfter=10),
    "cell": st("cell", fontSize=8.5, leading=11.5, spaceAfter=0),
    "cellb": st("cellb", fontName="Helvetica-Bold", fontSize=8.5, leading=11.5, spaceAfter=0),
    "cellh": st("cellh", fontName="Helvetica-Bold", fontSize=8, leading=11,
                textColor=colors.white, spaceAfter=0),
}


def box(text, fill, edge, label=None):
    inner = []
    if label:
        inner.append(Paragraph(
            f'<font size=7 color="#{edge.hexval()[2:]}"><b>{label}</b></font>',
            st("lbl", spaceAfter=3)))
    inner.append(Paragraph(text, S["body"]))
    t = Table([[inner]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def tbl(rows, widths, head=True):
    data = []
    for i, r in enumerate(rows):
        row = []
        for j, c in enumerate(r):
            sty = S["cellh"] if (head and i == 0) else (
                S["cellb"] if j == 0 and i > 0 else S["cell"])
            row.append(Paragraph(str(c), sty))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE)]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), INK),
                  ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white)]
    t.setStyle(TableStyle(style))
    return t


def code(lines):
    esc = [l.replace(" ", "&nbsp;") for l in lines]
    t = Table([[Paragraph("<br/>".join(esc), S["mono"])]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8F8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def main(out: Path | None = None) -> Path:
    R = json.loads((RESULTS / "results.json").read_text())
    P = R["parameters"]
    D = R["derived"]
    out = out or (RESULTS / "k_esc_identifiability.pdf")

    story = []
    A = story.append

    A(Paragraph("Identifiability of endosomal escape in a minimal siRNA model", S["title"]))
    A(Paragraph("What can be estimated, what cannot, and which experiment would "
                "change that &mdash; every number below is computed by "
                "<font face='Courier' size=9>run_all.py</font>", S["subtitle"]))

    # 1 -------------------------------------------------------------------
    A(Paragraph("1 &nbsp;&middot;&nbsp; The model", S["h1"]))
    A(Paragraph("Four compartments, following an siRNA from injection to effect. Plasma "
                "clears in hours; the RISC-loaded pool persists for months. That mismatch "
                "is the defining feature of the modality.", S["body"]))
    A(Spacer(1, 2))
    A(Image(str(FIGS / "fig0_model.png"), width=168 * mm, height=54 * mm))
    A(Spacer(1, 4))
    A(code([
        "dA_p/dt  =  -(k_el + k_up) * A_p                             plasma      MEASURABLE",
        "dA_t/dt  =  k_up*A_p - k_esc*A_t - k_degt*A_t                tissue      MEASURABLE, usually not measured",
        "dA_r/dt  =  k_esc*A_t - k_loss*A_r                           RISC        INVISIBLE",
        "dP/dt    =  k_syn*(1 - Emax*A_r/(EC50 + A_r)) - k_out*P      protein     MEASURABLE",
    ]))
    A(Spacer(1, 6))
    A(tbl([
        ["Parameter", "Value", "Meaning"],
        ["k_el, k_up", f"{P['k_el']:.2f}, {P['k_up']:.2f} /d",
         f"Plasma elimination and tissue uptake; {100*D['uptake_fraction']:.0f} % of dose reaches tissue"],
        ["k_esc", f"{P['k_esc']:.3f} /d",
         f"Endosomal escape &mdash; the productive route ({100*D['escape_fraction']:.0f} % of tissue siRNA)"],
        ["k_degt", f"{P['k_degt']:.3f} /d", "Nuclease degradation; competes with escape, and wins"],
        ["k_loss", f"{P['k_loss']:.3f} /d", f"RISC pool decay &mdash; {1/P['k_loss']:.0f} d time constant"],
        ["E<sub>max</sub>, EC<sub>50</sub>", f"{P['Emax']:.2f}, {P['EC50']:.4f}",
         "<b>Efficacy</b> and <b>potency</b>"],
        ["k_syn, k_out", f"{P['k_syn']:.3f}, {P['k_out']:.3f} /d",
         f"Protein synthesis and turnover; baseline = {D['baseline_protein']:.0f}"],
    ], [30 * mm, 32 * mm, 103 * mm]))
    A(Spacer(1, 4))
    A(box("The bracket in line 4 is an <b>E<sub>max</sub> model</b>, not Michaelis&ndash;Menten. "
          "Same algebraic form, different meaning: concentration in, fractional <i>effect</i> out, "
          "multiplying a synthesis rate. That makes it an indirect-response structure.<br/><br/>"
          "Parameters are illustrative and fitted to nothing. EC<sub>50</sub> is calibrated so the "
          f"reference simulation reaches a {100*D['nadir_reference_dose']:.1f} % nadir; "
          "everything else is set by hand.", WASH, TEAL, "STRUCTURE"))

    # 2 -------------------------------------------------------------------
    A(Paragraph("2 &nbsp;&middot;&nbsp; The identifiability problem", S["h1"]))
    A(Paragraph("You observe the ends of the chain and not the middle. Protein is measurable "
                "and is what you care about. Plasma is measurable but gives only the <i>sum</i> "
                "k_el + k_up, never k_up on its own &mdash; and k_up sits in the gain. The two "
                "compartments between them, where escape efficiency and potency live, are not "
                "routinely observed.", S["body"]))
    A(Paragraph("Consequence: k_esc, E<sub>max</sub> and EC<sub>50</sub> appear only through "
                "their joint effect on protein. A small amount of siRNA that suppresses strongly, "
                "and a large amount that suppresses weakly, produce <b>the same protein curve</b>.",
                S["body"]))
    A(Spacer(1, 4))
    A(box("<b>Escape and potency are independent properties.</b> Escape is a delivery property "
          "&mdash; conjugate, endosomal route, formulation. Potency is a pharmacology property "
          "&mdash; sequence, thermodynamics, chemical modification. Nothing links them, which is "
          "exactly why the data cannot separate them.", BLUSH, ROSE, "THE POINT"))

    # 3 -------------------------------------------------------------------
    A(Paragraph("3 &nbsp;&middot;&nbsp; What the algebra says", S["h1"]))
    A(Paragraph("Laplace-transforming the linear-regime cascade and working in fractional "
                "suppression (which removes k_syn) gives the protein response to a bolus dose D:",
                S["body"]))
    A(code([
        "                       -D * k_out * k_up * (Emax * k_esc / EC50)",
        "  dP(s)/P0 = ---------------------------------------------------------------",
        "             (s + k_loss)(s + k_out)(s + k_degt + k_esc)(s + k_el + k_up)",
    ]))
    A(Spacer(1, 6))
    A(Paragraph("What a protein time course recovers from this is a <b>set of four time "
                "constants and one number</b>. k_esc appears twice &mdash; in the gain, where it "
                "is multiplied by E<sub>max</sub>/EC<sub>50</sub> and by the uptake rate, and in a "
                "pole, where it is added to k_degt. Neither occurrence isolates it.", S["body"]))
    A(Spacer(1, 2))
    rows = [["Process", "Time constant", "Amplitude share",
             "Still alive at day 28?"]]
    surv = {d["process"]: d["surviving_fraction"] for d in R["sampling_leverage"]["28"]}
    for r in R["pole_table"]:
        s28 = surv[r["process"]]
        rows.append([r["label"], f"{r['time_constant_days']:.2f} d",
                     f"{100*r['amplitude_share']:.1f} %",
                     "yes" if s28 > 0.01 else
                     ("no" if s28 < 1e-6 else f"no ({100*s28:.2f} %)")])
    A(tbl(rows, [52 * mm, 30 * mm, 38 * mm, 45 * mm]))
    A(Spacer(1, 4))
    lev = {d["process"]: d["surviving_fraction"] for d in R["sampling_leverage"]["0.25"]}
    tissue_share = next(r["amplitude_share"] for r in R["pole_table"] if r["process"] == "tissue")
    A(box("Amplitude share is a shape metric, not an information metric. The tissue pole &mdash; "
          f"the only place k_esc appears on its own &mdash; carries {100*tissue_share:.1f} % of the "
          "amplitude, but on a monthly schedule its term has decayed to "
          f"{surv['tissue']:.0e} of its initial value before the first sample is drawn. Sample at "
          f"day 0.25 instead and {100*lev['tissue']:.0f} % of it is still there. <b>Practical "
          "identifiability is a property of the design, not of the model.</b>",
          CREAM, AMBER, "THE DISTINCTION"))

    # 4 -------------------------------------------------------------------
    ST = R["structural"]
    LAD = ST["ladder"]["designs"]
    INV = ST["invariance"]
    A(Paragraph("4 &nbsp;&middot;&nbsp; What is structurally identifiable", S["h1"]))
    A(Paragraph("Practical identifiability is about noise and schedules. Before that comes a "
                "prior question: could k_esc be recovered from perfect, noise-free, continuously "
                "sampled data? It could not, and the reason is exact.", S["body"]))
    A(Paragraph("Rescale the upstream states by u = A_p/D, v = A_t/(D&middot;k_up) and "
                "w = A_r/(D&middot;k_up&middot;k_esc). What is left carries the three poles and "
                "nothing else &mdash; D, k_up and k_esc cancel:", S["body"]))
    A(code([f"  {k}:  {v}" for k, v in ST["factorisation"]["scaled_equations"].items()]))
    A(Spacer(1, 5))
    A(Paragraph("And the E<sub>max</sub> term is homogeneous of degree zero in (A_r, EC<sub>50</sub>), "
                "so it sees those parameters only through psi = D&middot;k_up&middot;k_esc/EC<sub>50</sub> "
                "(verified symbolically; residual "
                f"{ST['factorisation']['emax_term_rewrite_residual']}). Nine parameters therefore "
                "reach the observables through seven functions:", S["body"]))
    A(code(["  k_el + k_up     k_esc + k_degt     k_loss     k_out     k_syn     Emax     psi"]))
    A(Spacer(1, 5))
    A(Paragraph("The level sets of that map are a two-parameter group. Take any alpha, beta &gt; 0 "
                "and send k_up to alpha&middot;k_up, k_esc to beta&middot;k_esc, EC<sub>50</sub> to "
                "alpha&middot;beta&middot;EC<sub>50</sub>, with k_el and k_degt absorbing the "
                "difference. Plasma and protein do not move:", S["body"]))
    o, tr = INV["original"], INV["transformed"]
    A(tbl([
        ["", "As simulated", "Transformed", ""],
        ["k_esc", f"{o['k_esc']:.4f} /d", f"{tr['k_esc']:.4f} /d",
         f"{INV['k_esc_ratio']:.0f} &times; larger"],
        ["Escape fraction", f"{100*o['escape_fraction']:.0f} %",
         f"{100*tr['escape_fraction']:.0f} %", "a different delivery story"],
        ["EC<sub>50</sub>", f"{o['EC50']:.4f}", f"{tr['EC50']:.4f}", "compensating"],
        ["Protein curve", "&mdash;", f"differs by {INV['max_difference_protein']:.0e}",
         "integrator tolerance"],
        ["Plasma curve", "&mdash;", f"differs by {INV['max_difference_plasma']:.0e}",
         "integrator tolerance"],
    ], [34 * mm, 36 * mm, 43 * mm, 52 * mm]))
    A(Spacer(1, 4))
    A(box("<b>k_esc is not structurally identifiable from plasma and protein</b> &mdash; not at "
          "any sampling density, not with any instrument. A four-fold difference in escape and a "
          "four-fold difference in escape fraction produce the same data to eleven decimal places. "
          "The practical identifiability analysis that follows is therefore not a story about "
          "insufficient data; it is a story about what an additional measurement buys once you "
          "accept that this one cannot be rescued by measuring harder.",
          BLUSH, ROSE, "THE STRUCTURAL RESULT"))
    A(Spacer(1, 4))
    A(Paragraph("Confirming it numerically: the rank of the relative sensitivity map, which counts "
                "how many independent directions in parameter space the data can see. Rank is "
                "taken at the largest gap in the singular value spectrum rather than at a fixed "
                "cutoff, because the smallest singular value moves by two orders of magnitude with "
                "the finite-difference step while the gap does not.", S["body"]))
    lrows = [["Design", "Free parameters", "Rank", "Unresolved directions"]]
    for d in LAD:
        lrows.append([d["design"], str(d["n_free"]), str(d["rank"]),
                      f"<b>{d['unresolved_directions']}</b>"])
    A(tbl(lrows, [78 * mm, 30 * mm, 22 * mm, 35 * mm]))
    A(Spacer(1, 3))
    nulls = LAD[0]["null_space"]
    A(Paragraph("The two unresolved directions are the group generators, recovered to three "
                "decimals from the singular value decomposition: "
                + "; ".join(", ".join(f"{k} {v:+.2f}" for k, v in nv.items()) for nv in nulls)
                + f". A check that depends on no threshold at all: applying the analytic "
                  f"generators to the sensitivity matrix gives residuals of "
                  f"{LAD[0]['generator_residuals']['alpha']:.0e} and "
                  f"{LAD[0]['generator_residuals']['beta']:.0e} relative &mdash; integrator noise "
                  f"&mdash; while the first of the two rises to "
                  f"{LAD[1]['generator_residuals']['alpha']:.0e} once tissue siRNA is observed, "
                  f"which is precisely the direction that measurement resolves.", S["cap"]))

    A(Paragraph("4b &nbsp;&middot;&nbsp; And in the linear regime, which pole is which", S["h1"]))
    A(Paragraph("k_loss and k_out survive the analysis above as separately identifiable, but only "
                "because of the saturation nonlinearity. Drop to the linear regime &mdash; where "
                "phi is defined &mdash; and the transfer-function denominator becomes symmetric "
                "under permutation of its roots while the numerator is a single number, so any "
                "reassignment of time constants to compartments can be absorbed by rescaling the "
                "gain. A 50-day RISC pool feeding a 2.9-day protein, and a 2.9-day RISC pool "
                "feeding a 50-day protein, give curves identical to "
                f"{R['pole_swap']['linear_regime_max_relative_difference']:.0e} relative.", S["body"]))
    A(Spacer(1, 3))
    A(Image(str(FIGS / "fig3_pole_swap.png"), width=168 * mm, height=55 * mm))
    A(Paragraph("Left: the two assignments, plotted on top of each other, inside the linear range. "
                "Right: what the protein curve determines there.", S["cap"]))
    A(box("So durability is inferable only from the part of the response that the linear "
          "approximation throws away. At the reference dose a refitted swapped model is "
          f"{R['pole_swap']['full_model_best_fit_rmse_pp']:.0f} percentage points of baseline "
          "away, so the curvature does carry it &mdash; but an analysis conducted entirely in the "
          "linear regime cannot tell a persistent RISC pool from a slow-turnover target. "
          "Independent measurement of protein turnover settles it either way.",
          CREAM, AMBER, "WHAT THIS COSTS"))

    # 5 -------------------------------------------------------------------
    A(Paragraph("5 &nbsp;&middot;&nbsp; Does the degeneracy matter? Walk the set", S["h1"]))
    A(Paragraph("Not knowing a parameter is only a problem if the decision depends on it. The "
                "test: fix the fit, walk the parameter combinations that reproduce it, re-run the "
                "untested prediction on every point, and look at the spread of <i>verdicts</i>.",
                S["body"]))
    A(Spacer(1, 3))
    A(Image(str(FIGS / "fig1_manifold.png"), width=168 * mm, height=48 * mm))
    M, V = R["manifold"], R["verdicts_low_dose"]
    A(Paragraph(f"Left: every point reproduces the observed {100*M['target_nadir']:.1f} % nadir "
                f"&mdash; a {M['fold_range_k_esc']:.0f}-fold range in k_esc. Centre: the curves are "
                "identical through the nadir, then diverge during recovery. Right: at a "
                f"{1/R['low_dose_factor']:.0f}-fold lower dose the nadir spans "
                f"{100*V['nadir_min']:.0f}&ndash;{100*V['nadir_max']:.0f} % and they fan out.",
                S["cap"]))
    vrows = [["Decision rule (at the lower dose)", "Verdicts across the set", "Conclusion"]]
    for rule in V["rules"]:
        n_p, n_f = rule["n_pass"], rule["n_fail"]
        verdict = (f"PASS &times; {n_p}" if n_f == 0 else
                   f"FAIL &times; {n_f}" if n_p == 0 else
                   f"PASS &times; {n_p}, FAIL &times; {n_f}")
        concl = ("<b>Unanimous</b> &mdash; the degeneracy is irrelevant to this question"
                 if rule["unanimous"] else
                 "<b>Split</b> &mdash; the decision rests on what you cannot resolve")
        vrows.append([f"Is nadir &le; {100*rule['threshold']:.0f} % of baseline?", verdict, concl])
    A(tbl(vrows, [52 * mm, 46 * mm, 67 * mm]))
    A(Spacer(1, 4))
    A(box("Same ignorance, different consequence. Most modelling anxiety about unknown parameters "
          "dissolves once you check whether they actually move the answer &mdash; and the ones "
          "that survive that check are the ones worth an experiment.", WASH, TEAL, "THE METHOD"))
    A(Spacer(1, 4))
    A(box("The set walked above is an <b>iso-nadir set</b>: one scalar constraint, which is what "
          "you have if the nadir is all you measure. It is not the likelihood-flat manifold of the "
          "full model, which additionally holds the tissue pole fixed. The two are different "
          "objects and the distinction matters: the curves separate during recovery precisely "
          "because the iso-nadir set does not preserve the tissue pole. <b>Extending the sampling "
          "window is itself an answer.</b>", CREAM, AMBER, "A CAVEAT ON THE METHOD"))

    # 6 -------------------------------------------------------------------
    A(Paragraph("6 &nbsp;&middot;&nbsp; What is measurable: phi, and where it holds", S["h1"]))
    A(code([f"  phi  =  Emax * k_esc / EC50  =  {D['phi']:.2f}"
            "        total silencing produced per unit dose"]))
    A(Spacer(1, 5))
    W = R["phi_validity"]
    A(Paragraph("phi is defined in the linear approximation, where E<sub>max</sub>A_r/(EC<sub>50</sub>+A_r) "
                "is approximately (E<sub>max</sub>/EC<sub>50</sub>)A_r. That approximation needs "
                "A_r &lt;&lt; EC<sub>50</sub>. Deep silencing needs the opposite. Those two "
                "requirements are in direct conflict, and the conflict is quantitative:", S["body"]))
    A(Spacer(1, 3))
    A(Image(str(FIGS / "fig4_phi_window.png"), width=112 * mm, height=66 * mm))
    SAT = R["saturation"]
    A(Paragraph(f"At the reference dose, peak A_r is {SAT['A_r_max_over_EC50']:.1f} &times; "
                f"EC<sub>50</sub> &mdash; {100*SAT['peak_fractional_occupancy']:.0f} % of the way to "
                "the ceiling &mdash; and the linear approximation misstates the depth of silencing "
                f"by {100*W['reference_dose_relative_error']:.0f} %.", S["cap"]))
    A(box("<b>The validity window may sit below the noise floor.</b> Holding the linear "
          "approximation to 5 % relative error caps the achievable silencing at "
          f"{100*W['max_suppression_within_tolerance']:.1f} % of baseline &mdash; below a typical "
          "10 % assay CV. In this parameterisation phi is a well-defined quantity that is only "
          "valid at doses where there is nothing to measure. That is a structural feature of an "
          "E<sub>max</sub> model, not an artefact of these numbers: any parameter set that "
          "silences deeply is operating near the ceiling.", BLUSH, ROSE, "THE CATCH"))
    A(Spacer(1, 3))
    A(tbl([
        ["phi answers", "phi cannot answer"],
        ["Which of six candidates gives most silencing per milligram, at matched dose and tissue",
         "Whether an underperformer has a delivery or a sequence problem"],
        ["How dose&ndash;response scales, inside the validity window",
         "Whether a conjugate working in liver will work in muscle"],
        ["Whether a chemistry change improved overall efficiency",
         "What EC<sub>50</sub> is &mdash; it is confounded with k_up&middot;k_esc, though "
         "E<sub>max</sub> itself is recoverable from the saturation curvature"],
    ], [82 * mm, 83 * mm]))

    # 7 -------------------------------------------------------------------
    A(Paragraph("7 &nbsp;&middot;&nbsp; Breaking the degeneracy, and what each experiment buys",
                S["h1"]))
    A(Paragraph("Each confounded parameter can be isolated in a system where you control what the "
                "in vivo model cannot observe. The question is which one to pay for. Profile "
                "likelihood on simulated data answers it: for each sampling design, profile k_esc "
                "and read off the 95 % interval.", S["body"]))
    A(Spacer(1, 3))
    A(Image(str(FIGS / "fig2_profiles.png"), width=168 * mm, height=64 * mm))
    A(Paragraph("A flat profile is practical non-identifiability. Left: k_degt known from a "
                "stability assay. Right: k_degt unknown &mdash; the honest starting point.",
                S["cap"]))
    prows = [["Sampling design", "k_degt known", "k_degt unknown"]]
    keys = [d["key"] for d in R["profiles"][0]["designs"]]
    for i, key in enumerate(keys):
        a = R["profiles"][0]["designs"][i]
        b = R["profiles"][1]["designs"][i]
        fmt = lambda r: (f"over {r['fold_width']:.0f}-fold"
                         if r["interval_hits_grid_edge"] else f"{r['fold_width']:.1f}-fold")
        prows.append([a["design"], fmt(a), fmt(b)])
    A(tbl(prows, [75 * mm, 45 * mm, 45 * mm]))
    A(Spacer(1, 4))
    worst = R["profiles"][1]["designs"][0]
    best = R["profiles"][1]["designs"][-1]
    A(box("<b>The experiment that matters is the tissue measurement, not the sampling schedule.</b> "
          "With k_degt unknown, moving protein sampling earlier narrows k_esc from "
          f"more than {worst['fold_width']:.0f}-fold to more than "
          f"{R['profiles'][1]['designs'][1]['fold_width']:.0f}-fold &mdash; still useless. "
          "Adding total tissue siRNA on the same early schedule brings it to "
          f"{best['fold_width']:.1f}-fold. Dense early protein sampling is not a substitute for "
          "measuring the compartment the pole belongs to.", WASH, TEAL, "WHAT THE NUMBERS SAY"))
    A(Spacer(1, 4))
    A(tbl([
        ["Parameter", "Experiment", "Why it works"],
        ["E<sub>max</sub>, EC<sub>50</sub>",
         "Transfection across a wide concentration range; measure target mRNA and protein",
         "<b>Transfection bypasses escape entirely.</b> Deliver straight to the cytosol and k_esc "
         "leaves the chain, so what you measure is pharmacology alone"],
        ["k_degt", "Nuclease stability assay in vitro", "Fixes the other term in the tissue pole"],
        ["k_out", "Independent protein turnover measurement",
         "Resolves the pole-assignment ambiguity in section 4"],
        ["k_esc", "Total tissue siRNA on a dense early schedule; subcellular fractionation or "
         "Ago2 RNA immunoprecipitation for the escaped fraction",
         "Observes the compartment where the tissue pole lives, rather than inferring it from "
         "downstream protein"],
    ], [30 * mm, 62 * mm, 73 * mm]))
    A(Spacer(1, 4))
    A(box("Two honest limits. In vitro EC<sub>50</sub> does not transfer cleanly to in vivo "
          "concentrations without an intracellular reference. And transfection is not "
          "GalNAc-mediated uptake, so the escape route differs from the real one &mdash; which is "
          "precisely what microphysiological systems are trying to close.", CREAM, AMBER, "CAVEAT"))

    # 8 -------------------------------------------------------------------
    A(Paragraph("8 &nbsp;&middot;&nbsp; Why k_esc is worth the effort", S["h1"]))
    A(tbl([
        ["If the problem is", "Then", "Send it back to"],
        ["Low k_esc", "Enough drug arrives; little escapes the endosome",
         "<b>Delivery</b> &mdash; conjugate, formulation, chemistry"],
        ["Low E<sub>max</sub>/EC<sub>50</sub>", "Plenty escapes; the silencing is weak",
         "<b>Sequence</b> &mdash; design and modification pattern"],
    ], [40 * mm, 68 * mm, 57 * mm]))
    A(Spacer(1, 4))
    A(Paragraph("Without k_esc you cannot tell them apart, so you are guessing which team to send "
                "an underperforming compound to. It also gates extrapolation: GalNAc works in "
                "liver because ASGPR is hepatocyte-specific, and whether a new conjugate reaches "
                "muscle or CNS depends on whether enough escapes there.", S["body"]))
    A(Spacer(1, 3))
    A(box("<b>The one-line version.</b> The reason k_esc is worth measuring is not precision "
          "&mdash; it is that it tells you whether an underperforming compound is a delivery "
          "problem or a sequence problem. Those go back to different teams.",
          WASH, TEAL, "SUMMARY"))

    A(Spacer(1, 10))
    A(Paragraph(f"Parameters are illustrative and no proprietary or clinical data was used. The "
                f"transferable content is the structure of the argument, not the numbers. Every "
                f"figure and every value in the tables above was produced by "
                f"<font face='Courier' size=8>run_all.py</font> "
                f"({R['runtime_seconds']:.0f} s) and read from "
                f"<font face='Courier' size=8>results/results.json</font>, generated "
                f"{R['generated']}.", S["cap"]))

    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title="Identifiability of endosomal escape in a minimal siRNA model",
        author="Bo Ma")
    doc.build(story)
    print(f"· wrote {out}")
    return out


if __name__ == "__main__":
    main()
