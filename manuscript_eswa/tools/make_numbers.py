"""Compute every number quoted in the manuscript from the frozen evidence run (run 7, seeds 5000+; tools/common.py).

Writes generated/numbers.tex (LaTeX lookup table used as \\num{key}) and generated/numbers.csv
(the same values with their provenance). Run `python tools/evidence.py verify` first.

Rule names in keys (common.rule_key): psnr_v3_fit -> fedxra, psnr_v3 -> fedxra_n, psnr_v3_unif -> fedxra_u,
psnr_v3_<x> -> fedxra_n_<x>, psnr_screen -> psnrv2, trimmed_mean_0.3 -> trimmed_mean_0p3. Attack parameters / fractions use p for the decimal point.

Key scheme (all lower-case, dot separated):
    <exp>.rank.<rule>                        average rank of normalized RMSE (1 = best) among the 17 rules
    <exp>.rankpos.<rule>                     position of the rule in that ordering (1 = best)
    e6.rank.<dataset>.<rule>                 per-dataset average rank (main benchmark)
    <exp>.null.<rule>                        rate of "target >= 2 ranks lower" WITHOUT attack (null rate)
    <exp>.nullpromo.<rule>                   rate of "target >= 2 ranks higher" without attack
    <exp>.succ.<attack>[.f<frac>].<rule>     attack success rate
    <exp>.ex.<attack>[.f<frac>].<rule>       success minus the same rule's null rate in the same cell
    <exp>.w.<attack>.<rule>                  attacker weight share (rules with weights)
    <exp>.nrmse.<attack>.<rule>              mean normalized RMSE (x 1000)
    <exp>.rawrmse.none.<rule>, <exp>.tau.none.<rule>
    <exp>.friedman.p, <exp>.cd, <exp>.nblocks  Friedman test and Nemenyi critical difference (0.05)
    <exp>.wil.<rule>                         Holm-adjusted Wilcoxon p, FedXRA vs rule (per-seed means)
    <exp>.wilbetter / .wilworse              number of rules FedXRA is significantly better / worse than
    e8k10.*                                  ablation restricted to K = 10
    e10.<metric>.<attack>.<rule>             minority fidelity and weights
"""
from __future__ import annotations

import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE, MAIN, REF, RUN, SEED0, rule_key

OUT = HERE / "generated"
# experiments summarised by the mean / worst average rank (main.*): every experiment on honest unreliability
# and attacks except E1 (fallacy demonstration), E8 (ablation) and E11 (model-family sweep, reported apart)
SUMMARY = ["E9", "E10", "E2", "E5", "E3", "E4", "E6", "E7"]
# condition groups for the decision metrics: honest unreliability = attack-free, noisy inputs, flipped labels
HONEST_CONDITIONS = ("none", "input_noise", "label_flip")
DEC_GROUPS = {"none": lambda x: x.attack == "none",
              "honest": lambda x: x.attack.isin(HONEST_CONDITIONS),
              "attack": lambda x: ~x.attack.isin(HONEST_CONDITIONS)}
BLOCK = ["dataset", "model", "scheme", "scheme_param", "K", "attack", "attack_frac", "attack_param"]
UNIT = ["dataset", "model", "scheme", "scheme_param", "K", "seed"]
CELL = UNIT + ["aggregator"]

values: list[tuple[str, str, str]] = []


def num_key(v) -> str:
    return f"{v:g}".replace(".", "p").replace("-", "m")


def put(key: str, value: float, prov: str, fmt: str = "{:.2f}") -> None:
    if not re.fullmatch(r"[a-z0-9_.]+", key):
        raise ValueError(f"bad key {key!r}")
    if value is None or not np.isfinite(float(value)):
        return  # undefined (e.g. weight share of a rule without weights): simply no key
    values.append((key, fmt.format(value), prov))


def load(exp: str) -> tuple[pd.DataFrame, str]:
    hits = glob.glob(str(RUN / "*" / "results" / f"*_{exp}" / "results.parquet"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{exp}: expected one results.parquet, found {hits}")
    d = pd.read_parquet(hits[0])
    assert d.seed.min() >= SEED0, "evidence-run seeds only"
    d["attack_param"] = d.attack_param.astype(str)
    # the strong white-box coalition that also claims n = 1000 is reported as its own attack
    d.loc[(d.attack == "whitebox_strong") & (d.attack_param != "nan"), "attack"] = "wbs_claimn"
    return d, str(Path(hits[0]).relative_to(RUN)).replace("\\", "/")


def block_means(d: pd.DataFrame, aggs: list[str]) -> pd.DataFrame:
    return d[d.aggregator.isin(aggs)].groupby(BLOCK + ["aggregator"], dropna=False).rmse_norm.mean().unstack()


def ranks(d, aggs, exp, src, prefix=None):
    t = block_means(d, aggs)
    r = t.rank(axis=1).mean().sort_values()
    p = prefix or exp
    for pos, (a, v) in enumerate(r.items(), 1):
        put(f"{p}.rank.{rule_key(a)}", v, f"{src}: mean over blocks of rank(rmse_norm) among {len(aggs)} rules")
        put(f"{p}.rankpos.{rule_key(a)}", pos, f"{src}: position in the average-rank ordering", "{:d}")
    if len(t) > 1 and t.shape[1] > 2:
        fr = stats.friedmanchisquare(*[t[c].values for c in t.columns])
        k, N = t.shape[1], t.shape[0]
        q = stats.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2)
        put(f"{p}.friedman.p", fr.pvalue, f"{src}: Friedman test over {N} blocks", "{:.1e}" if fr.pvalue < 1e-3 else "{:.3f}")
        put(f"{p}.cd", q * np.sqrt(k * (k + 1) / (6 * N)), f"{src}: Nemenyi CD at 0.05, k={k}, N={N}")
        put(f"{p}.nblocks", N, src, "{:d}")
    return r


def null_and_excess(d, exp, src, attacks=None):
    nn = d[d.attack == "none"].assign(null=lambda x: (x.target_rank_shift >= 2).astype(float),
                                      nullp=lambda x: (x.target_rank_shift <= -2).astype(float))
    for a, v in nn.groupby("aggregator").null.mean().items():
        put(f"{exp}.null.{rule_key(a)}", v, f"{src}: P(target rank shift >= 2) without attack")
    for a, v in nn.groupby("aggregator").nullp.mean().items():
        put(f"{exp}.nullpromo.{rule_key(a)}", v, f"{src}: P(target rank shift <= -2) without attack")
    nmap = nn.set_index(CELL)[["null", "nullp"]]
    x = d[d.attack != "none"].join(nmap, on=CELL)
    x = x[x.attack_success.notna()]
    if attacks is not None:
        x = x[x.attack.isin(attacks)]
    multi = x.groupby("attack").attack_frac.nunique()
    x["exv"] = np.where(x.attack == "feature_promotion", x.attack_success - x.nullp, x.attack_success - x.null)
    for (att, fr, a), g in x.groupby(["attack", "attack_frac", "aggregator"]):
        lab = att if multi[att] == 1 else f"{att}.f{num_key(fr)}"
        put(f"{exp}.succ.{lab}.{rule_key(a)}", g.attack_success.mean(), f"{src}: mean attack_success")
        if att not in ("frozen", "zero", "scaling", "input_noise", "label_flip", "minmax"):
            put(f"{exp}.ex.{lab}.{rule_key(a)}", g.exv.mean(), f"{src}: success minus null rate of the same cell")


def weights_and_errors(d, exp, src):
    for (att, a), g in d[d.attack_frac.isin([0.0, 0.2])].groupby(["attack", "aggregator"]):
        put(f"{exp}.w.{att}.{rule_key(a)}", g.w_attacker.mean(), f"{src}: mean attacker weight share, frac 0.2", "{:.3f}")
        put(f"{exp}.nrmse.{att}.{rule_key(a)}", 1e3 * g.rmse_norm.mean(), f"{src}: mean normalized RMSE x1000", "{:.1f}")
    nn = d[d.attack == "none"]
    for a, g in nn.groupby("aggregator"):
        put(f"{exp}.rawrmse.none.{rule_key(a)}", 1e3 * g.rmse_raw.mean(), f"{src}: raw RMSE x1000, no attack", "{:.2f}")
        put(f"{exp}.tau.none.{rule_key(a)}", g.kendall_tau.mean(), f"{src}: Kendall tau, no attack")


def decision(d, exp, src):
    """Decision-relevant fidelity without attack: top-3 / top-5 risk-driver agreement with the centralized model,
    rank agreement, and the share of cells where the protected feature stays within one rank of its true rank."""
    nn = d[d.attack == "none"]
    for a, g in nn.groupby("aggregator"):
        k = rule_key(a)
        put(f"{exp}.top3.none.{k}", g.top3_jaccard.mean(), f"{src}: top-3 Jaccard with the oracle, no attack")
        put(f"{exp}.top5.none.{k}", g.top5_jaccard.mean(), f"{src}: top-5 Jaccard with the oracle, no attack")
        put(f"{exp}.keep.none.{k}", (g.target_rank_shift.abs() <= 1).mean(),
            f"{src}: share of cells with |protected-feature rank shift| <= 1, no attack")


def wilcoxon(d, exp, src, aggs):
    u = d[d.aggregator.isin(aggs)].groupby(UNIT + ["aggregator"], dropna=False).rmse_norm.mean().unstack()
    raw = {}
    for a in aggs:
        if a == REF or a not in u:
            continue
        diff = (u[REF] - u[a]).dropna()
        if (diff != 0).sum() < 5:
            continue
        raw[a] = (stats.wilcoxon(diff).pvalue, float(np.median(diff)))
    names = sorted(raw, key=lambda k: raw[k][0])
    m, adj, prev = len(names), {}, 0.0
    for i, a in enumerate(names):  # Holm step-down
        prev = max(prev, min(1.0, (m - i) * raw[a][0]))
        adj[a] = prev
    better = sum(adj[a] < 0.05 and raw[a][1] < 0 for a in adj)
    worse = sum(adj[a] < 0.05 and raw[a][1] > 0 for a in adj)
    for a, p in adj.items():
        put(f"{exp}.wil.{rule_key(a)}", p, f"{src}: Holm-adjusted Wilcoxon, FedXRA vs rule, per-seed mean rmse_norm",
            "{:.1e}" if p < 1e-3 else "{:.3f}")
    put(f"{exp}.wilbetter", better, f"{src}: rules FedXRA beats (Holm 0.05)", "{:d}")
    put(f"{exp}.wilworse", worse, f"{src}: rules that beat FedXRA (Holm 0.05)", "{:d}")


def case_study_numbers() -> None:
    """Numbers of the credit case study (tools/case_study.py: the median cell of E9 under honest unreliability)."""
    import case_study as C
    s = C.select()
    cell, f, sh, t = s["cell"], s["features"], s["share"], s["candidates"]
    src = (f"E9 seed {s['seed']}, {cell['dataset']} K={cell['K']} {cell['attack']} {cell['attack_param']} "
           "(tools/case_study.py)")
    put("case.ncells", len(t), "E9 honest-unreliability cells with saved explanations", "{:d}")
    put("case.nfedavgbetter", int((t["diff"] > 0).sum()), "cells where FedAvg has the lower nRMSE than FedXRA", "{:d}")
    put("case.k", int(cell["K"]), src, "{:d}")
    for a in ["fedavg", REF]:
        put(f"case.nrmse.{rule_key(a)}", 1e3 * s["rmse"][a], f"{src}: nRMSE x1000", "{:.1f}")
    o = C.top(sh["ORACLE"], f, 5)
    for a in ["fedavg", REF]:
        top = C.top(sh[a], f, 5)
        put(f"case.top5.{rule_key(a)}", len(set(o) & set(top)) / len(set(o) | set(top)), f"{src}: top-5 Jaccard")
        put(f"case.top5common.{rule_key(a)}", len(set(o) & set(top)), f"{src}: top-5 features in common", "{:d}")
    if "age" in f:
        for a, lab in [("ORACLE", "oracle"), ("fedavg", "fedavg"), (REF, "fedxra")]:
            put(f"case.agerank.{lab}", int((sh[a] > sh[a][f.index("age")]).sum()) + 1,
                f"{src}: rank of age (1 = most important)", "{:d}")
            put(f"case.ageshare.{lab}", 100 * sh[a][f.index("age")], f"{src}: share of total attribution (%)",
                "{:.1f}")
    put("case.top1share.oracle", 100 * sh["ORACLE"].max(), f"{src}: largest share (%)", "{:.1f}")


def main() -> None:
    total, allranks, alldec = 0, {}, {}
    for i in range(1, 12):
        exp = f"E{i}"
        d, src = load(exp)
        e = exp.lower()
        total += len(d)
        put(f"{e}.seeds", d.seed.nunique(), src, "{:d}")
        put(f"{e}.rows", len(d), src, "{:,d}")
        put(f"{e}.cells", d[UNIT].drop_duplicates().shape[0], src, "{:,d}")
        if exp == "E8":
            continue
        ranks(d, MAIN, e, src)
        null_and_excess(d, e, src)
        weights_and_errors(d, e, src)
        wilcoxon(d, e, src, MAIN)
        decision(d, e, src)
        allranks[exp] = block_means(d, MAIN).rank(axis=1).mean()
        if exp in SUMMARY:
            g = d[d.aggregator.isin(MAIN)]
            for grp, sel in DEC_GROUPS.items():
                nn = g[sel(g)]
                alldec[(exp, grp)] = pd.DataFrame({
                    "top3": nn.groupby("aggregator").top3_jaccard.mean(),
                    "top5": nn.groupby("aggregator").top5_jaccard.mean(),
                    "tau": nn.groupby("aggregator").kendall_tau.mean(),
                    "keep": nn.assign(k=nn.target_rank_shift.abs() <= 1).groupby("aggregator").k.mean()})
    put("all.rows", total, "sum of results.parquet rows over E1-E11", "{:,d}")
    r = pd.DataFrame({e: allranks[e] for e in SUMMARY})
    mean, worst = r.mean(axis=1), r.max(axis=1)
    src = "average ranks of " + ", ".join(SUMMARY)
    for pos, (a, v) in enumerate(mean.sort_values().items(), 1):
        put(f"main.rankmean.{rule_key(a)}", v, f"{src}: mean over experiments")
        put(f"main.rankmeanpos.{rule_key(a)}", pos, f"{src}: position by mean", "{:d}")
    for pos, (a, v) in enumerate(worst.sort_values().items(), 1):
        put(f"main.rankworst.{rule_key(a)}", v, f"{src}: worst experiment")
        put(f"main.rankworstpos.{rule_key(a)}", pos, f"{src}: position by worst experiment", "{:d}")
    put("main.nexp", len(SUMMARY), src, "{:d}")
    # decision-relevant fidelity, averaged over the same experiments, per condition group:
    # main.<m>.<rule> (no attack), main.<m>h.<rule> (honest unreliability), main.<m>a.<rule> (misreporting)
    for grp, suffix in [("none", ""), ("honest", "h"), ("attack", "a")]:
        src = f"decision metrics ({grp}) averaged over " + ", ".join(SUMMARY)
        for m in ["top3", "top5", "tau", "keep"]:
            v = pd.DataFrame({e: alldec[(e, grp)][m] for e in SUMMARY})
            mean = v.mean(axis=1)
            for pos, (a, x) in enumerate(mean.sort_values(ascending=False).items(), 1):
                put(f"main.{m}{suffix}.{rule_key(a)}", x, f"{src}: mean of {m}", "{:.3f}")
                put(f"main.{m}{suffix}pos.{rule_key(a)}", pos, f"{src}: position by mean {m} (1 = highest)", "{:d}")
            put(f"main.{m}{suffix}min.fedxra", v.loc[REF].min(), f"{src}: FedXRA's lowest experiment for {m}")
    case_study_numbers()

    d, src = load("E6")
    t = d[d.aggregator.isin(MAIN)].groupby(["dataset"] + BLOCK[1:] + ["aggregator"], dropna=False)\
        .rmse_norm.mean().unstack().rank(axis=1).groupby(level=0).mean()
    for ds, row in t.iterrows():
        for a, r in row.items():
            put(f"e6.rank.{ds}.{rule_key(a)}", r, f"{src}: per-dataset mean rank")
        put(f"e6.rankpos.{ds}.fedxra", int((row < row[REF]).sum()) + 1, f"{src}: per-dataset position", "{:d}")
    for ds, g in d.groupby("dataset"):
        wilcoxon(g, f"e6.{ds}", src, MAIN)

    d, src = load("E8")
    abl = sorted(d.aggregator.unique())
    for K, g in d.groupby("K"):
        p = f"e8k{K}"
        ranks(g, abl, p, src, prefix=p)
        null_and_excess(g, p, src)
        weights_and_errors(g, p, src)

    d, src = load("E10")
    for m in ["rmse_norm_minority", "w_minority"]:
        for (att, a), v in d.groupby(["attack", "aggregator"])[m].mean().items():
            put(f"e10.{m.replace('_', '')}.{att}.{rule_key(a)}", 1e3 * v if "rmse" in m else v, f"{src}: mean {m}",
                "{:.1f}" if "rmse" in m else "{:.3f}")

    OUT.mkdir(exist_ok=True)
    keys = [k for k, _, _ in values]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        sys.exit(f"duplicate keys: {sorted(dup)[:5]}")
    lines = ["% Generated by tools/make_numbers.py from the frozen evidence run -- do not edit by hand.",
             r"\makeatletter",
             r"\newcommand{\num}[1]{\ifcsname num@#1\endcsname\csname num@#1\endcsname"
             r"\else\PackageError{numbers}{Undefined number key #1}{}\fi}"]
    lines += [rf"\expandafter\def\csname num@{k}\endcsname{{{v.replace(',', '{,}')}}}" for k, v, _ in values]
    lines.append(r"\makeatother")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT / "numbers.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("key", "value", "provenance"), *values])
    print(f"{len(values)} numbers -> {OUT / 'numbers.tex'}")


if __name__ == "__main__":
    main()
