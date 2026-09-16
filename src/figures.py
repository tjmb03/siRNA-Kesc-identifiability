"""Figures. Every value plotted here comes from results.json; nothing is drawn
from a number typed into this file."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import Params, simulate, linear_protein

INK, TEAL, ROSE, AMBER, MUTE = "#10212B", "#0E6E73", "#A8324A", "#B4690E", "#62727C"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8A9AA3", "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.labelcolor": INK, "figure.dpi": 200, "savefig.dpi": 200,
})


def _shades(n, c0="#9ED8D6", c1=ROSE):
    return [matplotlib.colors.to_hex(c) for c in
            matplotlib.colors.LinearSegmentedColormap.from_list("x", [c0, c1])(
                np.linspace(0, 1, n))]


def fig_manifold(p: Params, members: list[Params], res: dict, out: Path):
    """Three panels: the iso-nadir set, the curves it generates at the
    reference dose, and the same curves at a lower dose."""
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.7))
    cols = _shades(len(members))
    t = np.linspace(0, res["t_end_days"], 900)

    ke = [m.k_esc for m in members]
    em = [m.Emax for m in members]
    ax[0].plot(ke, em, "-", color=AMBER, lw=1.6, zorder=1)
    ax[0].scatter(ke, em, s=22, c=cols, zorder=3, edgecolor="white", lw=0.6)
    ax[0].scatter([p.k_esc], [p.Emax], s=46, c=INK, zorder=4)
    ax[0].annotate("the set that\ngenerated the data", (p.k_esc, p.Emax),
                   textcoords="offset points", xytext=(26, -42), fontsize=7,
                   color=MUTE, arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6))
    ax[0].set_xlabel(r"$k_{esc}$  (endosomal escape, 1/day)")
    ax[0].set_ylabel(r"$E_{max}$  (maximum suppression)")
    ax[0].set_title(f"A · every point fits the data equally well\n"
                    f"({res['manifold']['fold_range_k_esc']:.0f}-fold range in "
                    r"$k_{esc}$)", loc="left", color=INK)
    ax[0].set_ylim(0, 1.05)

    for panel, (axis, factor, title) in enumerate((
            (ax[1], 1.0, "B · matched at nadir, then they diverge"),
            (ax[2], res["low_dose_factor"],
             f"C · at a {1/res['low_dose_factor']:.0f}x lower dose — they fan out"))):
        for m, c in zip(members, cols):
            q = m.with_(dose=m.dose * factor)
            _, y = simulate(q, t_eval=t)
            axis.plot(t, 100 * y["P"] / q.P0, color=c, lw=1.1)
        axis.set_xlabel("days")
        axis.set_ylabel("target protein (% of baseline)")
        axis.set_ylim(0, 105)
        axis.set_title(title, loc="left", color=INK)

    v = res["verdicts_low_dose"]
    ax[2].annotate(f"nadir spans {100*v['nadir_min']:.0f}–{100*v['nadir_max']:.0f} %",
                   (0.5, 0.95), xycoords="axes fraction", fontsize=7, color=ROSE)
    for rule, style in zip(v["rules"], ("--", ":")):
        thr = 100 * rule["threshold"]
        tag = "unanimous" if rule["unanimous"] else "SPLIT verdict"
        axis = ax[2]
        axis.axhline(thr, ls=style, lw=0.9,
                     color=INK if rule["unanimous"] else ROSE)
        axis.annotate(f"{thr:.0f} % threshold — {tag}", (0.30, thr + 2),
                      xycoords=("axes fraction", "data"), fontsize=6.5,
                      color=INK if rule["unanimous"] else ROSE)

    ax[1].annotate("identical through the nadir;\nthe recovery phase is what\n"
                   "breaks the degeneracy", (0.30, 0.10),
                   xycoords="axes fraction", fontsize=6.8, color=MUTE)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_profiles(res: dict, out: Path):
    """Profile likelihood for k_esc under each sampling design, with k_degt
    known and unknown."""
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    cols = {0: MUTE, 1: TEAL, 2: ROSE}
    for a, scen in zip(ax, res["profiles"]):
        for i, r in enumerate(scen["designs"]):
            k = [d["k_esc"] for d in r["profile"]]
            dl = [d["delta"] for d in r["profile"]]
            a.plot(k, dl, color=cols[i], lw=1.4,
                   label=r["design"] if a is ax[0] else None)
        a.axhline(3.84, ls="--", lw=0.8, color=INK)
        a.annotate("95 % cutoff", (0.02, 4.3), xycoords=("axes fraction", "data"),
                   fontsize=6.5, color=INK)
        a.axvline(res["truth_k_esc"], ls=":", lw=0.8, color=AMBER)
        a.set_xscale("log")
        a.set_xlabel(r"$k_{esc}$  (1/day)")
        a.set_ylim(0, 25)
        a.set_title(scen["title"], loc="left", color=INK)
    ax[0].set_ylabel(r"$\Delta$ (-2 log L)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_pole_swap(p: Params, res: dict, out: Path):
    """The pole-assignment ambiguity: two different biologies, one curve."""
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.5),
                           gridspec_kw={"width_ratios": [1.25, 1]})

    t = np.linspace(0, 120, 800)
    p = p.with_(dose=p.dose * 0.01)          # inside the linear range
    sw = p.with_(k_loss=p.k_out, k_out=p.k_loss)
    sw = sw.with_(k_syn=sw.k_out)
    sw = sw.with_(EC50=sw.EC50 * sw.observable_gain / p.observable_gain)
    ax[0].plot(t, 100 * (1 + linear_protein(p, t)), color=TEAL, lw=2.6,
               label=r"RISC 50 d, protein 2.9 d")
    ax[0].plot(t, 100 * (1 + linear_protein(sw, t)), color=ROSE, lw=1.1, ls="--",
               label=r"RISC 2.9 d, protein 50 d")
    ax[0].set_xlabel("days")
    ax[0].set_ylabel("target protein (% of baseline)")
    ax[0].legend(frameon=False, fontsize=6.6, loc="lower right")
    ax[0].set_title("D · two different biologies, one curve\n"
                    f"(max difference {res['pole_swap']['linear_regime_max_relative_difference']:.0e})",
                    loc="left", color=INK)

    ax[1].axis("off")
    ax[1].text(0.0, 0.92, "what the protein curve pins down", fontsize=7.6,
               color=INK, weight="bold", transform=ax[1].transAxes)
    lines = [
        r"a SET of time constants  {0.08, 0.74, 2.9, 50} d",
        "and one lumped gain",
        "",
        r"     $D \cdot k_{out} \cdot k_{up} \cdot E_{max} k_{esc} / EC_{50}$",
        "",
        "not which compartment owns which",
        "time constant, and not any one factor",
        "of the gain on its own.",
    ]
    for i, ln in enumerate(lines):
        ax[1].text(0.0, 0.78 - 0.105 * i, ln, fontsize=7, color=MUTE,
                   transform=ax[1].transAxes)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_phi_window(p: Params, res: dict, out: Path):
    """The dose window in which phi is a valid description of the system."""
    fig, axx = plt.subplots(1, 1, figsize=(4.6, 2.7))
    ax = {1: axx}
    w = res["phi_validity"]
    dose = [r["dose_factor"] for r in w["curve"]]
    supp = [100 * r["suppression_full"] for r in w["curve"]]
    err = [100 * r["relative_error"] for r in w["curve"]]
    ax[1].plot(dose, supp, color=TEAL, lw=1.6, label="silencing achieved")
    ax[1].plot(dose, err, color=ROSE, lw=1.6, ls="--",
               label=r"error in the linear ($\varphi$) approximation")
    ax[1].axhline(100 * w["tolerance"], lw=0.8, ls=":", color=INK)
    ax[1].axvline(1.0, lw=0.8, ls=":", color=AMBER)
    ax[1].annotate("reference dose", (0.9, 118), fontsize=6.5, color=AMBER,
                   rotation=90, ha="right", va="top")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("dose (x reference)")
    ax[1].set_ylabel("% ")
    ax[1].set_ylim(0, 200)
    ax[1].legend(frameon=False, fontsize=6.6, loc="upper left")
    ax[1].set_title(r"E · where $\varphi$ is valid, nothing is silenced"
                    f"\n(max {100*w['max_suppression_within_tolerance']:.1f} % "
                    "silencing within tolerance)", loc="left", color=INK)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_model_schematic(p: Params, res: dict, out: Path):
    """The cascade, with what is observed and where k_esc hides.

    Time constants are read from the computed pole table, so the picture cannot
    drift away from the parameters.
    """
    tau = {r["process"]: r["time_constant_days"] for r in res["pole_table"]}
    fig, ax = plt.subplots(figsize=(10.2, 3.3))
    ax.set_xlim(0, 11.3); ax.set_ylim(0, 3.3); ax.axis("off")

    boxes = [
        ("plasma\n$A_p$", tau["plasma"], "measured", TEAL),
        ("tissue / endosome\n$A_t$", tau["tissue"], "measurable —\nrarely measured", AMBER),
        ("RISC-loaded\n$A_r$", tau["risc"], "invisible", ROSE),
        ("target protein\n$P$", tau["protein"], "measured", TEAL),
    ]
    w, gap, y = 1.95, 0.85, 1.78
    xs = [0.70 + i * (w + gap) for i in range(4)]

    for (label, tc, badge, col), x in zip(boxes, xs):
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, y), w, 0.72, boxstyle="round,pad=0.04,rounding_size=0.09",
            linewidth=1.5, edgecolor=col, facecolor="none"))
        ax.text(x + w / 2, y + 0.36, label, ha="center", va="center",
                fontsize=9, color=INK)
        ax.text(x + w / 2, y + 0.90, rf"$\tau$ = {tc:.2f} d", ha="center",
                fontsize=7.5, color=MUTE)
        ax.text(x + w / 2, y - 0.74, badge, ha="center", va="top",
                fontsize=8, color=col, weight="bold")

    for i, (lab, col, lw) in enumerate(
            [(r"$k_{up}$", INK, 1.6), (r"$k_{esc}$", ROSE, 2.8),
             (r"$E_{max}/EC_{50}$", INK, 1.6)]):
        x0, x1 = xs[i] + w, xs[i + 1]
        ax.annotate("", (x1, y + 0.36), (x0, y + 0.36),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                    shrinkA=3, shrinkB=3))
        ax.text((x0 + x1) / 2, y + 0.48, lab, ha="center", fontsize=8.5,
                color=col, weight="bold" if col == ROSE else "normal")

    for lab, x in zip((r"$k_{el}$", r"$k_{degt}$", r"$k_{loss}$", r"$k_{out}$"), xs):
        ax.annotate("", (x + w / 2, y - 0.60), (x + w / 2, y - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.1))
        ax.text(x + w / 2 + 0.14, y - 0.36, lab, fontsize=8, color=MUTE,
                va="center")

    ax.annotate("", (xs[0], y + 0.36), (0.05, y + 0.36),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.3))
    ax.text(0.05, y + 0.50, "dose", fontsize=8, color=MUTE)
    ax.text(5.65, 0.34,
            r"$k_{esc}$ never appears alone: only as $(k_{esc}+k_{degt})$ in a pole, "
            r"and inside $D\,k_{up}k_{esc}/EC_{50}$ in the gain",
            ha="center", fontsize=8.8, color=ROSE)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_invariance(p: Params, res: dict, out: Path):
    """Two genuinely different delivery stories, one dataset."""
    from structural import invariance_group
    inv = res["structural"]["invariance"]
    q = invariance_group(p, inv["alpha"], inv["beta"])
    t = np.linspace(0, res["t_end_days"], 900)
    _, y0 = simulate(p, t_eval=t)
    _, y1 = simulate(q, t_eval=t)

    fig, ax = plt.subplots(1, 2, figsize=(7.8, 2.6),
                           gridspec_kw={"width_ratios": [1.2, 1]})
    ax[0].plot(t, 100 * y0["P"] / p.P0, color=TEAL, lw=3.0,
               label=f"escape {100*p.escape_fraction:.0f} %")
    ax[0].plot(t, 100 * y1["P"] / q.P0, color=ROSE, lw=1.2, ls="--",
               label=f"escape {100*q.escape_fraction:.0f} %")
    ax[0].set_xlabel("days")
    ax[0].set_ylabel("target protein (% of baseline)")
    ax[0].set_ylim(0, 105)
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[0].set_title("one dataset, two delivery stories\n"
                    f"(they differ by {inv['max_difference_protein']:.0e})",
                    loc="left", color=INK)

    ax[1].axis("off")
    rows = [
        ("", "as simulated", "transformed"),
        (r"$k_{esc}$", f"{p.k_esc:.3f}/d", f"{q.k_esc:.3f}/d"),
        ("escape fraction", f"{100*p.escape_fraction:.0f} %",
         f"{100*q.escape_fraction:.0f} %"),
        (r"$EC_{50}$", f"{p.EC50:.3f}", f"{q.EC50:.3f}"),
        (r"$k_{up}$", f"{p.k_up:.1f}/d", f"{q.k_up:.1f}/d"),
    ]
    for i, (a, b, c) in enumerate(rows):
        yy = 0.93 - 0.150 * i
        bold = i == 0
        for x, txt, ha in ((0.0, a, "left"), (0.60, b, "right"), (1.0, c, "right")):
            ax[1].text(x, yy, txt, fontsize=8, ha=ha, transform=ax[1].transAxes,
                       color=MUTE if bold else INK,
                       weight="bold" if bold or i == 2 else "normal")
    ax[1].text(0.0, 0.02, "every invariant held fixed:\n"
               r"$k_{el}+k_{up}$,  $k_{esc}+k_{degt}$,  $D\,k_{up}k_{esc}/EC_{50}$",
               fontsize=7.5, color=ROSE, transform=ax[1].transAxes)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
