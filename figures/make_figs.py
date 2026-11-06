"""English, IEEE-sized figures for the TLT submission.

Usage: python3 make_figs.py <data_dir> <out_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

COL1, COL2 = 3.5, 7.16          # IEEE single / double column widths (inches)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
C = {"expert": "#0072B2", "same": "#E69F00", "cross": "#8C8C8C", "reversed": "#CC79A7",
     "learned": "#009E73", "prior": "#56B4E9", "llm": "#D55E00", "llama": "#CC79A7", "order": "#000000"}
SRC = {"expert": ("Expert (correct)", C["expert"]),
       "theme_faux": ("Wrong, same topic", C["same"]),
       "hors_theme_faux": ("Wrong, cross-topic", C["cross"])}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


def p(*parts):
    return os.path.join(D, *parts)


# ---------------------------------------------------------------- Fig. 1: study overview
def fig_overview():
    fig, ax = plt.subplots(figsize=(COL2, 2.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, title, sub, lw=0.8, fc="white", bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=fc, ec="#333333", lw=lw))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=7.3,
                fontweight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.3, color="#444444")

    srcs = [("Expert graph", "978 prerequisites (Junyi)"), ("LLM graphs", "Qwen2.5-7B, Llama-3.1-8B"),
            ("Random controls", "same-topic, cross-topic"), ("Learned graphs", "edge weights learned by KT")]
    ys = [23, 16, 9, 2]
    for (t, s), y in zip(srcs, ys):
        box(1, y, 22, 5.5, t, s)
    ax.text(12, 29.6, "Graph sources", ha="center", fontsize=7, color="#444444")

    ax.add_patch(FancyBboxPatch((37, 8.5), 24, 13, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc="#F4F4F4", ec="#333333", lw=0.8))
    ax.text(49, 19.3, "Graph properties", ha="center", va="center", fontsize=7.3, fontweight="bold")
    for i, t in enumerate(["coverage", "edge correctness", "topical proximity", "edge direction (simulation)"]):
        ax.text(49, 16.3 - i * 2.2, t, ha="center", va="center", fontsize=6.5)
    ax.text(49, 29.6, "Varied or measured", ha="center", fontsize=7, color="#444444")
    for y in ys:
        ax.annotate("", xy=(36.6, 15), xytext=(23.4, y + 2.75),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555555", mutation_scale=7))

    box(75, 17, 24, 8, "Structural fidelity", "vs expert prerequisites:\nROC AUC, AUPRC, top-K", fc="#E8F1F8")
    box(75, 4, 24, 8, "Predictive utility", "first-attempt AUC gain,\nPrereqKT and GKT, sparse data", fc="#FDF1E0")
    ax.text(87, 29.6, "Two separate evaluations", ha="center", fontsize=7, color="#444444")
    for y in (21, 8):
        ax.annotate("", xy=(74.6, y), xytext=(61.4, 15),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555555", mutation_scale=7))
    ax.annotate("", xy=(87, 12.6), xytext=(87, 16.4),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="#B00000", ls="--", mutation_scale=7))
    ax.text(88.2, 14.5, "do they agree?", fontsize=6.3, color="#B00000", va="center", style="italic")
    save(fig, "fig1_overview")


# ---------------------------------------------------------------- Fig. 2: LLM per-item top-k
def fig_llm_topk():
    d = pd.read_csv(p("70705a39-resultats_etape16", "rq2_deux_llm_benefice.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 2.1), sharey=False)
    for ax, frac, ttl in zip(axs, [0.05, 0.01], ["5 % of learners", "1 % of learners"]):
        s = d[d.fraction == frac].set_index("graphe")
        base, exp = s.loc["aucun", "AUC_1re"], s.loc["expert", "AUC_1re"]
        ks = [1, 2, 3, 5]
        for key, lab, col, mk in [("Qwen", "Qwen2.5-7B", C["llm"], "o"), ("Llama", "Llama-3.1-8B", C["llama"], "s"),
                                  ("theme", "Random same-topic", C["cross"], "^")]:
            ax.plot(ks, [s.loc[f"{key}_k{k}", "AUC_1re"] for k in ks], marker=mk, ms=3, color=col, label=lab, lw=1.1)
        ax.axhline(exp, color=C["expert"], lw=0.9, ls="-")
        ax.axhline(base, color="black", lw=0.8, ls=":")
        ax.text(0.8, exp, "expert", color=C["expert"], fontsize=6, va="bottom")
        ax.text(0.8, base, "no graph", color="black", fontsize=6, va="bottom")
        ax.set_xticks(ks); ax.set_xlim(0.7, 5.3)
        ax.set_title(ttl)
    axs[0].set_ylabel("First-attempt AUC")
    fig.supxlabel("Prerequisites kept per item (k)", fontsize=7.5, y=0.13)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, handlelength=1.4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1))
    save(fig, "fig2_llm_topk")


# ---------------------------------------------------------------- Fig. 3: factorial design (Junyi, PrereqKT)
FACT = pd.DataFrame([
    (0.05, 25, "expert", 0.0034, 0.0012, 0.0056), (0.05, 25, "theme_faux", 0.0010, -0.0008, 0.0027),
    (0.05, 25, "hors_theme_faux", -0.0003, -0.0031, 0.0022), (0.05, 50, "expert", 0.0078, 0.0054, 0.0101),
    (0.05, 50, "theme_faux", 0.0068, 0.0048, 0.0088), (0.05, 50, "hors_theme_faux", 0.0041, 0.0024, 0.0059),
    (0.05, 100, "expert", 0.0136, 0.0107, 0.0163), (0.05, 100, "theme_faux", 0.0110, 0.0083, 0.0138),
    (0.05, 100, "hors_theme_faux", 0.0089, 0.0071, 0.0109), (0.01, 25, "expert", 0.0088, 0.0046, 0.0124),
    (0.01, 25, "theme_faux", 0.0101, 0.0064, 0.0138), (0.01, 25, "hors_theme_faux", 0.0064, 0.0040, 0.0086),
    (0.01, 50, "expert", 0.0170, 0.0115, 0.0220), (0.01, 50, "theme_faux", 0.0154, 0.0103, 0.0201),
    (0.01, 50, "hors_theme_faux", 0.0108, 0.0063, 0.0151), (0.01, 100, "expert", 0.0272, 0.0225, 0.0314),
    (0.01, 100, "theme_faux", 0.0225, 0.0165, 0.0277), (0.01, 100, "hors_theme_faux", 0.0170, 0.0131, 0.0212)],
    columns=["fraction", "couv", "source", "gain", "lo", "hi"])


def fig_factorial():
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 1.85), sharey=True)
    for ax, frac, ttl in zip(axs, [0.05, 0.01], ["5 % of learners", "1 % of learners"]):
        s = FACT[FACT.fraction == frac]
        for j, (src, (lab, col)) in enumerate(SRC.items()):
            r = s[s.source == src].sort_values("couv")
            x = np.arange(3) + (j - 1) * 0.09
            ax.errorbar(x, r.gain, yerr=[r.gain - r.lo, r.hi - r.gain], color=col, marker="o", ms=3,
                        capsize=1.5, elinewidth=0.7, lw=1.2, label=lab)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(["25 %", "50 %", "100 %"])
        ax.set_title(ttl); ax.set_xlabel("Coverage level")
    axs[0].set_ylabel("Gain in first-attempt AUC\nover no graph")
    axs[0].legend(loc="upper left", handlelength=1.4)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig3_factorial")


# ---------------------------------------------------------------- Fig. 4: PrereqKT vs GKT
def fig_architecture():
    d = pd.read_csv(p("472f69d9-rq2_gkt_synthese.csv"))
    conds = [("expert_c25", "Expert\n25 %"), ("expert_c100", "Expert\n100 %"),
             ("theme_faux_c100", "Same-\ntopic"), ("hors_theme_faux_c100", "Cross-\ntopic")]
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 1.95), sharey=True)
    for ax, frac, ttl in zip(axs, [0.05, 0.01], ["5 % of learners", "1 % of learners"]):
        for j, (mod, col) in enumerate([("PrereqKT", C["expert"]), ("GKT", C["llm"])]):
            s = d[(d.modele == mod) & (d.fraction == frac)].set_index("condition")
            g = np.array([s.loc[c, "gain_vs_aucun"] for c, _ in conds])
            lo = np.array([s.loc[c, "IC95_bas"] for c, _ in conds]); hi = np.array([s.loc[c, "IC95_haut"] for c, _ in conds])
            x = np.arange(len(conds)) + (j - 0.5) * 0.38
            ax.bar(x, g, width=0.36, color=col, label=mod, alpha=0.9)
            ax.errorbar(x, g, yerr=[g - lo, hi - g], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([l for _, l in conds], fontsize=6.3)
        ax.set_title(ttl)
    axs[0].set_ylabel("Gain in first-attempt AUC\nover no graph")
    axs[0].legend(loc="lower left", handlelength=1.2)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig4_architecture")


# ---------------------------------------------------------------- Fig. 5: learned graphs (recovery + sensitivity)
def fig_learned():
    q = pd.read_csv(p("012cd778-resultats_etape15_1", "rq3_qualite_fleches.csv"))
    q = q[(q.fraction == 1.0) & (q.mesure == "AUC")]
    s18 = pd.read_csv(p("..", "r18", "rq3_sensibilite_synthese.csv")).set_index("config")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.15), gridspec_kw=dict(width_ratios=[1.05, 1.25]))
    srcs = [("ordre_passage", "Practice order", C["order"]), ("score_llm", "Qwen score", C["llm"]),
            ("appris_prior_llm", "Learned, LLM prior", C["prior"]), ("appris_sans_prior", "Learned, no prior", C["learned"])]
    groups = [("toutes", "All pairs"), ("ordre_impose", "Imposed\norder"), ("ordre_variable", "Variable\norder")]
    for j, (src, lab, col) in enumerate(srcs):
        r = q[q.source == src].set_index("paires")
        m = np.array([r.loc[g, "moyenne"] for g, _ in groups])
        lo = np.array([r.loc[g, "IC95_bas"] for g, _ in groups]); hi = np.array([r.loc[g, "IC95_haut"] for g, _ in groups])
        x = np.arange(3) + (j - 1.5) * 0.2
        a.bar(x, m - 0.5, bottom=0.5, width=0.19, color=col, label=lab)
        a.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="black", elinewidth=0.6, capsize=1)
    a.axhline(0.5, color="black", lw=0.6, ls=":")
    a.set_ylim(0.4, 1.0); a.set_xticks(range(3)); a.set_xticklabels([l for _, l in groups], fontsize=6.5)
    a.set_ylabel("Recovery of expert prerequisites\n(ROC AUC, all learners)")
    a.set_title("(a) Recovery by source", loc="left")
    a.legend(loc="upper center", ncol=2, handlelength=1.0, columnspacing=0.8, fontsize=6.3)

    order = ["l1=0.1_acyc=0.0", "l1=0.1_acyc=0.01", "l1=0.01_acyc=0.0", "l1=0.1_acyc=0.1", "l1=0.01_acyc=0.01",
             "l1=0.0_acyc=0.01", "candidats=area", "l1=0.01_acyc=0.1", "l1=0.0_acyc=0.1", "l1=0.0_acyc=0.0", "depart=0.5"]

    def lab(c):
        if c == "depart=0.5":
            return "initial weights 0.5"
        if c == "candidats=area":
            return "area candidates"
        l1, ac = c.replace("l1=", "").split("_acyc=")
        txt = f"λ₁={float(l1):g}, λ_acyc={float(ac):g}"
        if c == "l1=0.01_acyc=0.01":
            txt += " (main)"
        return txt
    best = s18["AUC_valid_mean"].idxmax()
    for i, c in enumerate(order):
        col = C["expert"] if c == best else "#7A7A7A"
        b.errorbar(s18.loc[c, "AUC_mean"], i, xerr=s18.loc[c, "AUC_std"], fmt="o", ms=3.2, color=col,
                   elinewidth=0.8, capsize=1.2)
    b.set_yticks(range(len(order)))
    b.set_yticklabels([lab(c) + ("  ← best KT validation" if c == best else "") for c in order], fontsize=6.2)
    b.set_ylim(len(order) - 0.4, -1.6)
    for v, t, ls in [(0.5, "chance", ":"), (0.695, "Qwen", "--"), (0.803, "practice order", "-")]:
        b.axvline(v, color="black", lw=0.6, ls=ls)
        right = t == "practice order"
        b.text(v + (0.006 if right else -0.006), -0.9, t, fontsize=5.8, va="bottom", ha="left" if right else "right")
    b.set_xlim(0.32, 0.93)
    b.set_xlabel("Recovery ROC AUC (mean ± SD, 3 seeds)")
    b.set_title("(b) Sensitivity to learning settings (no prior)", loc="left")
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig5_learned")


# ---------------------------------------------------------------- Fig. 6: DBE-KT22 factorial
def fig_dbe():
    d = pd.read_csv(p("d28c9411-resultats_etape17b", "dbe_synthese.csv"))
    d = d[d.modele == "PrereqKT"]
    lab = {"expert": ("Reference relations", C["expert"]), "theme_faux": ("Wrong, same quiz*", C["same"]),
           "hors_theme_faux": ("Wrong, cross-quiz", C["cross"])}
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 1.8), sharey=True)
    for ax, frac, ttl in zip(axs, [0.2, 0.05], ["20 % of students", "5 % of students"]):
        for j, (src, (l, col)) in enumerate(lab.items()):
            r = d[(d.fraction == frac) & d.condition.str.startswith(src + "_c")].copy()
            r["c"] = r.condition.str.extract(r"_c(\d+)").astype(int)
            r = r.sort_values("c")
            x = np.arange(3) + (j - 1) * 0.09
            ax.errorbar(x, r.gain_vs_aucun, yerr=[r.gain_vs_aucun - r.IC95_bas, r.IC95_haut - r.gain_vs_aucun],
                        color=col, marker="o", ms=3, capsize=1.5, elinewidth=0.7, lw=1.2, label=l)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(["25 %", "50 %", "100 %"])
        ax.set_title(ttl); ax.set_xlabel("Coverage level")
    axs[0].set_ylabel("Gain in first-attempt AUC\nover no graph")
    axs[0].legend(loc="lower left", handlelength=1.4, fontsize=6.2)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig6_dbe")


# ---------------------------------------------------------------- Fig. 7: simulation
def fig_simulation():
    d = pd.read_csv(p("627cfc85-resultats_etape17c_1", "sim_synthese.csv"))
    r = pd.read_csv(p("627cfc85-resultats_etape17c_1", "sim_recuperation.csv"))
    scen = [("S1", "S1: strong\nprerequisites"), ("S2", "S2: weak\nprerequisites"), ("S3", "S3: none,\nimposed order")]
    conds = [("expert_c100", "True prerequisites", C["expert"]), ("theme_faux_c100", "Wrong, same topic", C["same"]),
             ("hors_theme_faux_c100", "Wrong, cross-topic", C["cross"]), ("inverse_c100", "Reversed", C["reversed"]),
             ("appris_sans_prior", "Learned", C["learned"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.0))
    for j, (cnd, lab, col) in enumerate(conds):
        s = d[(d.fraction == 0.2) & (d.condition == cnd)].set_index("scenario")
        g = np.array([s.loc[k, "gain_vs_aucun"] for k, _ in scen])
        lo = np.array([s.loc[k, "IC95_bas"] for k, _ in scen]); hi = np.array([s.loc[k, "IC95_haut"] for k, _ in scen])
        x = np.arange(3) + (j - 2) * 0.16
        a.bar(x, g, width=0.15, color=col, label=lab)
        a.errorbar(x, g, yerr=[g - lo, hi - g], fmt="none", ecolor="black", elinewidth=0.5, capsize=0.8)
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(range(3)); a.set_xticklabels([l for _, l in scen])
    a.set_ylabel("Gain in first-attempt AUC\nover no graph (20 % of learners)")
    a.set_title("(a) Predictive utility", loc="left")
    a.legend(loc="upper right", handlelength=1.0, fontsize=6.2)

    meths = [("pratique avant", "Practice-before", C["expert"]), ("ordre de passage", "Practice order", C["order"]),
             ("co-réussite", "Co-occurrence lift", C["same"]), ("appris", "Learned weights", C["learned"])]
    for j, (key, lab, col) in enumerate(meths):
        s = r[(r.fraction == 1.0) & r.methode.str.startswith(key)].set_index("scenario")
        m = np.array([s.loc[k, "AUC_vs_vrais_prerequis"] for k, _ in scen])
        lo = np.array([s.loc[k, "IC95_bas"] for k, _ in scen]); hi = np.array([s.loc[k, "IC95_haut"] for k, _ in scen])
        x = np.arange(3) + (j - 1.5) * 0.2
        b.bar(x, m - 0.5, bottom=0.5, width=0.19, color=col, label=lab)
        b.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="black", elinewidth=0.5, capsize=0.8)
    b.axhline(0.5, color="black", lw=0.6, ls=":")
    b.set_ylim(0.2, 0.95)
    b.set_xticks(range(3)); b.set_xticklabels([l for _, l in scen])
    b.set_ylabel("Recovery of true prerequisites\n(ROC AUC, all learners)")
    b.set_title("(b) Structural recovery", loc="left")
    b.legend(loc="upper right", handlelength=1.0, fontsize=6.2, ncol=2, columnspacing=0.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig7_simulation")


if __name__ == "__main__":
    for f in [fig_overview, fig_llm_topk, fig_factorial, fig_architecture, fig_learned, fig_dbe, fig_simulation]:
        f()
