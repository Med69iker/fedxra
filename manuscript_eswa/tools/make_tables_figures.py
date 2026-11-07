"""Generate every table (tables/*.tex) and figure (figures/*.pdf) of the manuscript from the
frozen evidence run (run 7, seeds 5000+; tools/common.py). Never edit the outputs by hand: change this script and rerun.

    python tools/evidence.py verify && python tools/make_tables_figures.py

All 17 rules appear in every table (no selective reporting). Attack success is reported as EXCESS
success: the success indicator minus the same rule's rate of the same rank shift without attack,
in the same cell (dataset x setting x seed).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import CODE as ROOT, HERE, MAIN, NAMES, PLAIN, REF, RUN, SEED0

TAB, FIG = HERE / "tables", HERE / "figures"
sys.path.insert(0, str(ROOT))

from analysis import figures as F  # noqa: E402
from analysis.stats import nemenyi_cd  # noqa: E402

SHOWN_FIG = ["psnr_v3_fit", "psnr_v3", "psnr_v3_unif", "geomedian", "multikrum", "trimmed_mean_0.3", "median",
             "fedavg"]
DATASETS = {"gmsc": "GMSC", "taiwan": "Taiwan", "german": "German", "homecredit": "Home Credit",
            "lendingclub": "LendingClub"}
BLOCK = ["dataset", "model", "scheme", "scheme_param", "K", "attack", "attack_frac", "attack_param"]
UNIT = ["dataset", "model", "scheme", "scheme_param", "K", "seed"]
CELL = UNIT + ["aggregator"]


def load(exp: str) -> pd.DataFrame:
    hits = glob.glob(str(RUN / "*" / "results" / f"*_{exp}" / "results.parquet"))
    assert len(hits) == 1, hits
    d = pd.read_parquet(hits[0])
    assert d.seed.min() >= SEED0
    d["attack_param"] = d.attack_param.astype(str)
    # the strong white-box coalition that also claims n = 1000 is reported as its own attack
    d.loc[(d.attack == "whitebox_strong") & (d.attack_param != "nan"), "attack"] = "wbs_claimn"
    return d


def write(name: str, tex: str, resize: bool = True) -> None:
    if resize and r"\resizebox" not in tex:
        tex = tex.replace(r"\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
        tex = tex.replace(r"\end{tabular}", r"\end{tabular}}", 1)
    (TAB / f"{name}.tex").write_text(tex, encoding="utf-8")
    print("table", name)


def with_excess(d: pd.DataFrame) -> pd.DataFrame:
    nn = d[d.attack == "none"].assign(null=lambda x: (x.target_rank_shift >= 2).astype(float),
                                      nullp=lambda x: (x.target_rank_shift <= -2).astype(float))
    d = d.join(nn.set_index(CELL)[["null", "nullp"]], on=CELL)
    d["excess"] = np.where(d.attack == "feature_promotion", d.attack_success - d.nullp, d.attack_success - d["null"])
    return d


def avg_ranks(d: pd.DataFrame, aggs=MAIN) -> tuple[pd.Series, int]:
    t = d[d.aggregator.isin(aggs)].groupby(BLOCK + ["aggregator"], dropna=False).rmse_norm.mean().unstack()
    return t.rank(axis=1).mean(), len(t)


def fmt_col(v: pd.Series, decimals: int, lower=True, bold_ties: float = 0.0) -> list[str]:
    best = v.min() if lower else v.max()
    out = []
    for x in v:
        if not np.isfinite(x):
            out.append("--")
        elif abs(x - best) <= bold_ties + 1e-12:
            out.append(rf"\textbf{{{x:.{decimals}f}}}")
        else:
            out.append(f"{x:.{decimals}f}")
    return out


def rows_table(cols: dict[str, pd.Series], decimals, lower=True) -> list[str]:
    cells = {c: fmt_col(v.reindex(MAIN), decimals if isinstance(decimals, int) else decimals[c],
                        lower if isinstance(lower, bool) else lower[c]) for c, v in cols.items()}
    return [NAMES[a] + " & " + " & ".join(cells[c][i] for c in cols) + r" \\" for i, a in enumerate(MAIN)]


# ------------------------------------------------------------------ datasets (unchanged data, same cap)
def table_datasets() -> None:
    from powersnr.runner import _load
    rows = []
    for key, label in DATASETS.items():
        ds = _load(key, 60000)
        rows.append((label, len(ds.y), ds.X.shape[1], 100 * float(np.mean(ds.y)), ds.protected_feature or "--"))
    body = "\n".join(rf"{n} & {r:,} & {f} & {p:.1f} & \texttt{{{t.replace('_', chr(92) + '_')}}} \\"
                     for n, r, f, p, t in rows)
    write("tab_datasets", r"""\begin{table}[!htbp]
\centering
\caption{Credit datasets (after the 60{,}000-row cap used in all experiments).}
\label{tab:datasets}
\begin{tabular}{lrrrl}
\toprule
Dataset & Rows & Features & Default rate (\%) & Protected feature \\
\midrule
""" + body.replace(",", "{,}") + r"""
\bottomrule
\end{tabular}
\end{table}
""", resize=False)


# ------------------------------------------------------------------ ranks: all 17 rules x all experiments
EXPS = [("E6", "Main"), ("E9", "Quality"), ("E10", "Minority"), ("E11", "Models"), ("E1", "Fallacy"),
        ("E5", "Skew"), ("E7", "Strength"), ("E3", "Fraction"), ("E4", r"$K$ sweep"), ("E2", r"$K{=}3$")]


SUMMARY = ["E9", "E10", "E2", "E5", "E3", "E4", "E6", "E7"]


def table_ranks() -> dict:
    cols, cds = {}, {}
    for e, _ in EXPS:
        r, nb = avg_ranks(load(e))
        cols[e] = r
        cds[e] = nemenyi_cd(len(MAIN), nb)
    summ = pd.DataFrame({e: cols[e] for e in SUMMARY})
    cols["Mean"], cols["Worst"] = summ.mean(axis=1), summ.max(axis=1)
    lines = rows_table(cols, 2)
    cd_line = (r"\midrule" + "\n" + r"Nemenyi CD (0.05) & " + " & ".join(f"{cds[e]:.2f}" for e, _ in EXPS)
               + r" & & \\")
    head = (" & ".join(rf"\makecell{{{e}\\{lab}}}" for e, lab in EXPS)
            + r" & \makecell{Mean\\(8 exp.)} & \makecell{Worst\\(8 exp.)}")
    write("tab_ranks", r"""\begin{table*}[!htbp]
\centering
\caption{Average rank of the explanation error (normalized RMSE to the centralized oracle; 1 = best of the
17 rules). Ranks are computed within each dataset $\times$ setting $\times$ attack block and then averaged.
Best per column in bold. Two rules whose ranks differ by less than the critical difference (last row) are not
significantly different (Friedman--Nemenyi, 0.05). Mean and Worst: mean and largest average rank over the
eight experiments on honest unreliability and attacks (""" + ", ".join(SUMMARY) + r"""; all except E1, E8 and E11).}
\label{tab:ranks}
\small
\begin{tabular}{l""" + "r" * (len(EXPS) + 2) + r"""}
\toprule
Rule & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + "\n" + cd_line + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    return cols


# ------------------------------------------------------------------ main benchmark per dataset (+ Holm Wilcoxon)
def holm(pvals: dict) -> dict:
    names = sorted(pvals, key=pvals.get)
    out, prev, m = {}, 0.0, len(names)
    for i, a in enumerate(names):
        prev = max(prev, min(1.0, (m - i) * pvals[a]))
        out[a] = prev
    return out


def table_e6() -> None:
    d = load("E6")
    cols = {}
    marks = {}
    for ds, lab in DATASETS.items():
        g = d[d.dataset == ds]
        cols[lab] = avg_ranks(g)[0]
        u = g.groupby(["seed", "aggregator"]).rmse_norm.mean().unstack()
        p = {a: stats.wilcoxon(u[REF] - u[a]).pvalue for a in MAIN if a != REF}
        adj = holm(p)
        marks[lab] = {a: ("$^{+}$" if (u[REF] - u[a]).median() < 0 else "$^{-}$") if adj[a] < 0.05 else ""
                      for a in adj}
    cols["All"] = avg_ranks(d)[0]
    cells = {c: fmt_col(v.reindex(MAIN), 2) for c, v in cols.items()}
    lines = []
    for i, a in enumerate(MAIN):
        row = [cells[c][i] + (marks[c].get(a, "") if c in marks else "") for c in cols]
        lines.append(NAMES[a] + " & " + " & ".join(row) + r" \\")
    write("tab_e6", r"""\begin{table}[!htbp]
\centering
\caption{Main benchmark (E6, five datasets, ten banks, """ + str(d.attack.nunique() - 1) + r""" attack conditions and the attack-free condition): average
rank per dataset (1 = best of 17). $^{+}$ / $^{-}$: FedXRA is significantly more / less accurate than this
rule on that dataset (Wilcoxon signed-rank test on per-seed mean normalized RMSE, Holm correction over the 16
comparisons, 0.05).}
\label{tab:e6}
\small
\begin{tabular}{l""" + "r" * len(cols) + r"""}
\toprule
Rule & """ + " & ".join(cols) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------ main benchmark: every attack, every rule
TARGETED = [("whitebox_strong", "Strong WB"), ("wbs_claimn", r"Strong WB, $n$"), ("whitebox", "White-box"),("alie", "ALIE"), ("collusion", "Collusion"),
            ("collusion_jitter", "Jittered"), ("mimic", "Mimicry"), ("feature_hiding", "Hiding"),
            ("adaptive_psnr", "Energy infl."), ("feature_promotion", "Promotion")]
UNTARGETED = [("none", "No attack"), ("input_noise", "Input noise"), ("label_flip", "Label flip"),
              ("frozen", "Frozen"), ("zero", "Zero"), ("scaling", "Scaling"), ("minmax", "Min-max")]


def table_e6_attacks() -> None:
    d = with_excess(load("E6"))
    nn = d[d.attack == "none"]
    cols = {"Null": nn.groupby("aggregator")["null"].mean()}
    for a, lab in TARGETED:
        cols[lab] = d[d.attack == a].groupby("aggregator").excess.mean()
    lines = rows_table(cols, 2)
    write("tab_e6_targeted", r"""\begin{table*}[!htbp]
\centering
\caption{Main benchmark (E6, 20\% attackers): \emph{excess} success of the targeted attacks, i.e.\ the share
of runs in which the protected feature lost at least two ranks (gained, for promotion) minus the share of
attack-free runs of the same rule and cell in which this happened anyway. ``Null'' gives that attack-free rate
(for losing two ranks). Lower is better; best per column in bold. ``Strong WB'': the rank-aware white-box
coalition, which picks against each rule the most damaging of its report variants, including reports placed just
inside the rule's trimming or screening thresholds; ``Strong WB, $n$'': the same coalition also claims $n=1000$
examples per bank; ``White-box'': the coalition chooses, against each rule separately, the most damaging of 24
report variants; ``Energy infl.'': the reliability-inflation
attacker (called adaptive in earlier versions), which is calibrated on SNR-type weights, not on each rule.}
\label{tab:e6targeted}
\small
\begin{tabular}{l""" + "r" * len(cols) + r"""}
\toprule
Rule & """ + " & ".join(cols) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    cols = {lab: 1e3 * d[d.attack == a].groupby("aggregator").rmse_norm.mean() for a, lab in UNTARGETED}
    cols["Kendall $\\tau$"] = nn.groupby("aggregator").kendall_tau.mean()
    cols["Raw RMSE"] = 1e3 * nn.groupby("aggregator").rmse_raw.mean()
    for a, lab in [("frozen", "$w$ frozen"), ("zero", "$w$ zero")]:
        cols[lab] = d[d.attack == a].groupby("aggregator").w_attacker.mean()
    dec = {c: 1 for c in cols} | {"Kendall $\\tau$": 3, "Raw RMSE": 2, "$w$ frozen": 3, "$w$ zero": 3}
    low = {c: True for c in cols} | {"Kendall $\\tau$": False}
    lines = rows_table(cols, dec, low)
    write("tab_e6_untargeted", r"""\begin{table*}[!htbp]
\centering
\caption{Main benchmark (E6, 20\% attackers): explanation error under the untargeted attacks
(normalized RMSE $\times 10^3$), rank agreement (Kendall $\tau$, higher is better) and raw-scale RMSE
($\times 10^3$) without attack, and weight share captured by frozen and dead (zero) attackers (nominal share
0.20; ``--'' for rules that do not weight clients). Best per column in bold.}
\label{tab:e6untargeted}
\small
\begin{tabular}{l""" + "r" * len(cols) + r"""}
\toprule
Rule & """ + " & ".join(cols) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


# ------------------------------------------------------------------ attack strength (E7): coalitions by fraction
def table_e7() -> None:
    d = with_excess(load("E7"))
    cols = {}
    for a, lab in [("whitebox_strong", "Strong WB"), ("whitebox", "White-box"), ("collusion", "Collusion"),
                   ("collusion_jitter", "Jittered"), ("mimic", "Mimicry"), ("alie", "ALIE")]:
        for fr in [0.1, 0.2, 0.3]:
            cols[f"{lab} {int(fr * 100)}\\%"] = d[(d.attack == a) & np.isclose(d.attack_frac, fr)]\
                .groupby("aggregator").excess.mean()
    lines = rows_table(cols, 2)
    groups = " & ".join(rf"\multicolumn{{3}}{{c}}{{{g}}}" for g in ["Strong white-box", "White-box", "Collusion",
                                                                     "Jittered collusion", "Mimicry", "ALIE"])
    rules = "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(6))
    write("tab_e7", r"""\begin{table*}[!htbp]
\centering
\caption{Coalition attacks of increasing size (E7, GMSC and Taiwan, ten banks): excess success (success minus
the attack-free rate of the same rule and cell; lower is better) for 10\%, 20\% and 30\% attackers. Collusion
and mimicry are averaged over their strength parameters. Best per column in bold.}
\label{tab:e7}
\small
\begin{tabular}{l""" + "r" * len(cols) + r"""}
\toprule
 & """ + groups + r""" \\
""" + rules + "\n" + r"""Rule & """ + " & ".join(["10\\%", "20\\%", "30\\%"] * 6) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


# ------------------------------------------------------------------ heterogeneous quality (E9)
def table_e9() -> None:
    d = with_excess(load("E9"))
    conds = [("none", "nan", "No attack"), ("input_noise", "0.5", r"Noise $0.5$"), ("input_noise", "1.0", r"Noise $1$"),
             ("input_noise", "40.0", r"Noise $40$"), ("label_flip", "0.2", r"Flip $0.2$"),
             ("label_flip", "0.4", r"Flip $0.4$"), ("frozen", "nan", "Frozen"), ("feature_hiding", "nan", "Hiding"),
             ("adaptive_psnr", "10.0", "Energy infl."), ("alie", "nan", "ALIE"), ("minmax", "nan", "Min-max"),
             ("whitebox", "nan", "White-box")]
    cols = {}
    for a, p, lab in conds:
        g = d[(d.attack == a) & (d.attack_param == p)]
        assert len(g), (a, p)
        cols[lab] = 1e3 * g.groupby("aggregator").rmse_norm.mean()
    cols["Rank"] = avg_ranks(d)[0]
    cols["Ex.\\ strong WB"] = d[d.attack == "whitebox_strong"].groupby("aggregator").excess.mean()
    cols["Ex.\\ strong WB, $n$"] = d[d.attack == "wbs_claimn"].groupby("aggregator").excess.mean()
    dec = {c: 1 for c in cols} | {"Rank": 2, "Ex.\\ strong WB": 2, "Ex.\\ strong WB, $n$": 2}
    lines = rows_table(cols, dec)
    write("tab_e9", r"""\begin{table*}[!htbp]
\centering
\caption{Banks of heterogeneous quality (E9: log-normal bank sizes, 30\% of the banks trained on noisy or
mislabelled data, $K\in\{10,20\}$, GMSC, Taiwan and Home Credit): normalized RMSE ($\times 10^3$) per condition,
average rank over all conditions, and excess success of the strong white-box coalition without and with the
claim of $n=1000$ examples per bank. Best per column in bold.}
\label{tab:e9}
\small
\begin{tabular}{l""" + "r" * len(cols) + r"""}
\toprule
Rule & """ + " & ".join(cols) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


# ------------------------------------------------------------------ ablation (E8) at K = 10 and K = 3
# block 1: from FedXRA, what changed relative to the earlier n-weight design; block 2: components, removed from
# the shared n-weight base (FedXRA-n); block 3: references
ABL = [("psnr_v3_fit", r"\textbf{FedXRA (full)}"),
       ("psnr_v3_fit_nweights_dd1", r"-- fitted-variance weights (capped $n$-weights)"),
       ("psnr_v3_fit_dd4", r"-- strict duplicate threshold (ratio 4)"),
       ("psnr_v3", r"-- both (= FedXRA-n)"), ("psnr_v3_unif", r"-- both, equal weights (= FedXRA-U)"),
       ("psnr_v3_no_dedup", "FedXRA-n -- duplicate grouping"), ("psnr_v3_no_screens", "FedXRA-n -- magnitude screen"),
       ("psnr_v3_cap3", r"FedXRA-n, weight cap $3\times$ median"),
       ("psnr_v3_trim0.2", r"FedXRA-n, trimming $\beta=0.2$"),
       ("psnr_v3_mean_stage", "FedXRA-n, weighted mean stage"), ("psnr_v3_median_stage", "FedXRA-n, weighted median stage"),
       ("psnr_v3_huber_stage", "FedXRA-n, Huber stage"), ("psnr_v3_profile_screen", "FedXRA-n + profile screen"),
       ("psnr_v3_run5", "FedXRA-n, earlier design (Huber stage, profile screen, cap 3)"),
       ("psnr_screen", "Power-SNR"), ("geomedian", "Geometric median"), ("trimmed_mean", r"Trimmed mean ($\beta=0.2$)"),
       ("trimmed_mean_0.3", r"Trimmed mean ($\beta=0.3$)"), ("fedavg", "FedAvg")]


def table_ablation() -> None:
    d = with_excess(load("E8"))
    aggs = [a for a, _ in ABL]
    lines = []
    for a, lab in ABL:
        cells = []
        for K in [10, 3]:
            g = d[d.K == K]
            r = avg_ranks(g, aggs)[0]
            cells.append(f"{r[a]:.2f}")
            cells.append(f"{1e3 * g[(g.attack == 'none') & (g.aggregator == a)].rmse_norm.mean():.1f}")
            for att in ["whitebox_strong", "collusion", "collusion_jitter"]:
                cells.append(f"{g[(g.attack == att) & (g.aggregator == a)].excess.mean():.2f}")
            wv = g[(g.attack == "frozen") & (g.aggregator == a)].w_attacker.mean()
            cells.append("--" if not np.isfinite(wv) else f"{wv:.3f}")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if a in ("psnr_v3_unif", "psnr_v3_run5"):
            lines.append(r"\midrule")
    sub = "Rank & nRMSE & SWB & Coll. & Jit. & $w$ fr."
    write("tab_ablation", r"""\begin{table*}[!htbp]
\centering
\caption{Ablation (E8: GMSC and Taiwan; feature, Dirichlet and size-skewed partitions; 20\% attackers).
First block: FedXRA and the designs obtained by undoing its two changes (``--~X'' replaces X by the setting in
parentheses). Second block: components removed from or changed in the shared design with capped $n$-weights
(FedXRA-n). Rank: average rank among the """ + str(len(ABL)) + r""" variants and references; nRMSE: normalized RMSE
$\times 10^3$ without attack; SWB, Coll., Jit.: excess success of the strong white-box coalition, collusion and
jittered collusion; $w$ fr.: weight share of frozen attackers (nominal 0.20).}
\label{tab:ablation}
\small
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
 & \multicolumn{6}{c}{$K = 10$} & \multicolumn{6}{c}{$K = 3$} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-13}
Variant & """ + sub + " & " + sub + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


# ------------------------------------------------------------------ decision-relevant fidelity (no attack)
def table_decision() -> None:
    """Decision-relevant fidelity per condition group (no attack, honest unreliability, misreporting), averaged
    over the SUMMARY experiments: top-5 risk-driver agreement and protected-attribute stability (|rank shift| <= 1)."""
    honest = ("none", "input_noise", "label_flip")
    groups = {"No attack": lambda x: x.attack == "none",
              "Honest unrel.": lambda x: x.attack.isin(honest),
              "Misreporting": lambda x: ~x.attack.isin(honest)}
    top5 = {g: {} for g in groups}
    keep = {g: {} for g in groups}
    for e in SUMMARY:
        d = load(e)
        for g, sel in groups.items():
            x = d[sel(d)]
            top5[g][e] = x.groupby("aggregator").top5_jaccard.mean()
            keep[g][e] = x.assign(k=x.target_rank_shift.abs() <= 1).groupby("aggregator").k.mean()
    cols = {}
    for g in groups:
        cols["T5 " + g] = pd.DataFrame(top5[g]).mean(axis=1)
    for g in groups:
        cols["PK " + g] = pd.DataFrame(keep[g]).mean(axis=1)
    lines = rows_table(cols, 3, lower=False)
    head = " & ".join(["No attack", r"\makecell{Honest\\unreliability}", "Misreporting"] * 2)
    write("tab_decision", r"""\begin{table*}[!htbp]
\centering
\caption{Decision-relevant fidelity, averaged over the eight experiments on honest unreliability and attacks
(""" + ", ".join(SUMMARY) + r"""). Top-5 agreement: Jaccard index between the five most important risk drivers of
the federated and of the centralized explanation (1 = the same five features). Protected attribute kept: share of
cells in which the protected feature stays within one rank of its centralized rank. Honest unreliability: the
attack-free condition and the noisy-input and flipped-label banks; misreporting: all report and coalition attacks.
Higher is better; best per column in bold.}
\label{tab:decision}
\small
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{3}{c}{Top-5 risk-driver agreement} & \multicolumn{3}{c}{Protected attribute kept} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Rule & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def fig_case_study() -> None:
    """Credit case study (tools/case_study.py): the ten most important risk drivers of the centralized model and their
    importance share in the centralized, FedAvg and FedXRA explanations of the selected cell."""
    import matplotlib.pyplot as plt
    import case_study as C
    s = C.select()
    f, sh = s["features"], s["share"]
    top = C.top(sh["ORACLE"], f, 10)
    idx = [f.index(x) for x in top]
    labels = {"ORACLE": "Centralized model", "fedavg": "FedAvg", REF: "FedXRA"}
    colors = {"ORACLE": F.OKABE_ITO["black"], "fedavg": F.OKABE_ITO["orange"], REF: F.OKABE_ITO["blue"]}
    with F.paper_style():
        fig, ax = plt.subplots(figsize=F.figure_size("double", aspect=0.45))
        y = np.arange(len(top))
        for i, a in enumerate(C.SHOWN):
            ax.barh(y + (i - 1) * 0.27, sh[a][idx], 0.27, color=colors[a], label=labels[a])
        ax.set_yticks(y, [t.replace("_", " ") for t in top])
        ax.invert_yaxis()
        ax.set_xlabel("Share of total attribution (normalized mean |SHAP|)")
        ax.legend(loc="lower right", frameon=False)
        F.save_figure(fig, FIG, "fig_case_study")


# ------------------------------------------------------------------ figures
def fig_amplification() -> None:
    """Centrality amplification: weight share the white-box coalition obtains, per rule (E6 and E9)."""
    import matplotlib.pyplot as plt
    rows = []
    for e in ["E6", "E9"]:
        d = load(e)
        g = d[(d.attack == "whitebox") & d.aggregator.isin(MAIN)].groupby("aggregator").w_attacker
        rows.append(g.mean().rename(e))
    t = pd.concat(rows, axis=1).dropna().sort_values("E6")
    with F.paper_style():
        fig, ax = plt.subplots(figsize=F.figure_size("double", aspect=0.42))
        y = np.arange(len(t))
        ax.barh(y - 0.2, t["E6"], 0.4, color=F.OKABE_ITO["blue"], label="Main benchmark (E6)")
        ax.barh(y + 0.2, t["E9"], 0.4, color=F.OKABE_ITO["orange"], label="Heterogeneous quality (E9)")
        ax.axvline(0.2, color="k", ls="--", lw=1)
        ax.text(0.205, len(t) - 0.6, "nominal share 0.20", fontsize=7)
        ax.set_yticks(y, [PLAIN[a] for a in t.index])
        ax.set_xlabel("Weight share obtained by the white-box coalition (20% of the banks)")
        ax.legend(loc="lower right", frameon=False)
        F.save_figure(fig, FIG, "fig_amplification")


def figures() -> None:
    with F.paper_style():
        pass
    for e in ["E6", "E9"]:
        r, nb = avg_ranks(load(e))
        r.index = [PLAIN[a] for a in r.index]
        F.plot_cd_diagram(r, nemenyi_cd(len(MAIN), nb), out_dir=FIG, name=f"fig_cd_{e.lower()}", width="double")
    e3 = with_excess(load("E3"))
    e3 = e3[e3.attack.isin(["whitebox_strong", "collusion", "collusion_jitter", "alie"])].copy()
    e3["attack"] = e3.attack.map({"whitebox_strong": "Strong white-box", "collusion": "Collusion",
                                  "collusion_jitter": "Jittered collusion", "alie": "ALIE"})
    F.plot_metric_grid(e3, "attack_frac", "excess", col="attack", aggregators=SHOWN_FIG, out_dir=FIG,
                       name="fig_e3_fraction", xlabel="Attacker fraction", ylabel="Excess attack success")
    e4 = load("E4")
    e4 = e4[e4.attack.isin(["none", "whitebox_strong"])].copy()
    e4["attack"] = e4.attack.map({"none": "No attack", "whitebox_strong": "Strong white-box coalition"})
    F.plot_metric_grid(e4, "K", "rmse_norm", col="attack", aggregators=SHOWN_FIG, out_dir=FIG, name="fig_e4_K",
                       logx=True, xlabel=r"Number of banks $K$", ylabel="Normalized RMSE")
    e10 = load("E10")
    keep = [a for a in MAIN if a not in ("ivw", "dersimonian_laird", "foolsgold")]
    F.plot_pareto(e10[(e10.attack == "frozen") & e10.aggregator.isin(keep)], "rmse_norm", "rmse_norm_minority",
                  aggregators=keep, out_dir=FIG, name="fig_e10_pareto", annotate=False)
    fig_amplification()
    fig_case_study()
    print("figures done")


if __name__ == "__main__":
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    for f in list(TAB.glob("tab_attacks.tex")):  # replaced by tab_e6_targeted / tab_e6_untargeted
        f.unlink()
    table_datasets()
    table_ranks()
    table_e6()
    table_e6_attacks()
    table_e7()
    table_e9()
    table_ablation()
    table_decision()
    figures()
