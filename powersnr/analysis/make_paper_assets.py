"""Produce every paper figure/table available in a results.parquet.

Usage::

    python -m analysis.make_paper_assets --run results/<run> --out paper_assets/

For each `experiment` value present (E1, E2, ... ; matched case-insensitively on a leading
"E<number>", e.g. "E1", "e1_attacks", "exp6"), writes into <out>/<experiment>/:

* summary_<experiment>.csv   mean/std/n/95% CI of every metric per aggregator x varying keys
* stats_<experiment>.csv     Wilcoxon-Holm paired tests vs psnr_screen and vs psnr_base
* the experiment-specific figures (PDF + PNG) and LaTeX tables (see HANDLERS)

Experiments absent from the results are skipped; experiments without a dedicated handler
get only the summary/stats CSVs. A manifest.json listing produced and skipped assets is
written to <out>/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import KEY_COLS, METRIC_LOWER_IS_BETTER
from . import figures as F
from . import tables as T
from .stats import friedman_nemenyi, paired_tests, summarize

MAIN_METRICS = ["rmse_norm", "kendall_tau", "cosine", "top5_jaccard", "w_attacker", "w_minority",
                "rmse_norm_minority", "attack_success", "target_rank_shift"]
SUMMARY_METRICS = list(METRIC_LOWER_IS_BETTER) + ["fallback"]
REFERENCES = ["psnr_screen", "psnr_base"]
CURVE_METRICS = ["rmse_norm", "kendall_tau", "w_attacker"]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def experiment_number(name) -> int | None:
    m = re.match(r"(?i)\s*e(?:xp(?:eriment)?)?[_\- ]?(\d+)", str(name))
    return int(m.group(1)) if m else None


def varying_keys(d: pd.DataFrame, exclude=("seed", "experiment")) -> list[str]:
    return [c for c in KEY_COLS if c in d.columns and c not in exclude
            and d[c].nunique(dropna=False) > 1]


def has_metric(d: pd.DataFrame, m: str) -> bool:
    return m in d.columns and pd.to_numeric(d[m], errors="coerce").notna().any()


def lib(m: str) -> bool:
    return METRIC_LOWER_IS_BETTER.get(m, True)


def pick_col(d: pd.DataFrame, prefs) -> str:
    """First column in `prefs` that varies in `d`; else the first one present."""
    for c in prefs:
        if c in d.columns and d[c].nunique(dropna=False) > 1:
            return c
    for c in prefs:
        if c in d.columns:
            return c
    raise KeyError(f"none of {prefs} in data")


class Ctx:
    def __init__(self, out: Path, exp: str, n_boot: int):
        self.out = out
        self.exp = exp
        self.n_boot = n_boot
        self.produced: list[str] = []
        self.skipped: list[str] = []

    def add(self, paths):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.produced.extend(str(p) for p in paths)

    def skip(self, what: str, why: str):
        msg = f"{self.exp}: skipped {what} ({why})"
        print(msg, file=sys.stderr)
        self.skipped.append(msg)


# --------------------------------------------------------------------------------------
# generic assets (every experiment)
# --------------------------------------------------------------------------------------
def write_summary(d: pd.DataFrame, ctx: Ctx) -> None:
    by = ["aggregator"] + varying_keys(d)
    parts = []
    for m in SUMMARY_METRICS:
        if has_metric(d, m):
            s = summarize(d.assign(**{m: pd.to_numeric(d[m], errors="coerce")}), by, m,
                          n_boot=ctx.n_boot)
            s.insert(0, "metric", m)
            parts.append(s)
    if not parts:
        ctx.skip("summary CSV", "no metric columns with data")
        return
    p = ctx.out / f"summary_{ctx.exp}.csv"
    pd.concat(parts, ignore_index=True).to_csv(p, index=False)
    ctx.add(p)


def write_stats(d: pd.DataFrame, ctx: Ctx) -> None:
    by = varying_keys(d)
    parts = []
    for ref in REFERENCES:
        if ref not in set(d["aggregator"]):
            continue
        for m in MAIN_METRICS:
            if not has_metric(d, m):
                continue
            pt = paired_tests(d, by, m, reference=ref, lower_is_better=lib(m))
            pt.insert(0, "metric", m)
            parts.append(pt)
    if not parts:
        ctx.skip("stats CSV", "no reference aggregator (psnr_screen/psnr_base) present")
        return
    p = ctx.out / f"stats_{ctx.exp}.csv"
    pd.concat(parts, ignore_index=True).to_csv(p, index=False)
    ctx.add(p)


def metric_table(d: pd.DataFrame, ctx: Ctx, metric: str, col: str, name: str, caption: str,
                 decimals: int = 3) -> None:
    if not has_metric(d, metric):
        ctx.skip(f"table {name}", f"{metric} missing")
        return
    dd = d.assign(**{metric: pd.to_numeric(d[metric], errors="coerce")})
    s = summarize(dd, ["aggregator", col], metric, n_boot=ctx.n_boot)
    pv = paired_tests(dd, [col], metric, reference="psnr_screen", lower_is_better=lib(metric)) \
        if "psnr_screen" in set(d["aggregator"]) else None
    p = ctx.out / f"{name}.tex"
    T.to_latex_table(s, row="aggregator", col=col, decimals=decimals, lower_is_better=lib(metric),
                     pvalues=pv, caption=caption, label=f"tab:{name}", path=p,
                     col_header=col.replace("_", " ").capitalize(),
                     resize=s[col].nunique(dropna=False) > 4)
    ctx.add(p)


def multi_table(d: pd.DataFrame, ctx: Ctx, metrics, col: str, name: str, caption: str) -> None:
    metrics = [m for m in metrics if has_metric(d, m)]
    if not metrics:
        ctx.skip(f"table {name}", "no metrics")
        return
    summ, pvs = {}, {}
    for m in metrics:
        dd = d.assign(**{m: pd.to_numeric(d[m], errors="coerce")})
        summ[m] = summarize(dd, ["aggregator", col], m, n_boot=ctx.n_boot)
        if "psnr_screen" in set(d["aggregator"]):
            pvs[m] = paired_tests(dd, [col], m, reference="psnr_screen", lower_is_better=lib(m))
    labels = {m: T.latex_escape(F.metric_label(m)) if "$" not in F.metric_label(m)
              else F.metric_label(m) for m in metrics}
    p = ctx.out / f"{name}.tex"
    T.to_latex_multi_metric(summ, row="aggregator", col=col,
                            lower_is_better={m: lib(m) for m in metrics}, metric_labels=labels,
                            pvalues=pvs, caption=caption, label=f"tab:{name}", path=p)
    ctx.add(p)


def curves(d: pd.DataFrame, ctx: Ctx, x: str, metrics=CURVE_METRICS, logx: bool = False,
           facet_prefs=("attack",)) -> None:
    if x not in d.columns or d[x].notna().sum() == 0 or d[x].nunique() < 2:
        ctx.skip(f"curves vs {x}", f"{x} does not vary")
        return
    dd = d[d[x].notna()]
    facet = next((c for c in facet_prefs if c in dd.columns and dd[c].nunique() > 1), None)
    for m in metrics:
        if not has_metric(dd, m):
            ctx.skip(f"{m} vs {x}", f"{m} missing")
            continue
        dm = dd.assign(**{m: pd.to_numeric(dd[m], errors="coerce")})
        name = f"{ctx.exp}_{m}_vs_{x}"
        if facet:
            ctx.add(F.plot_metric_grid(dm, x, m, facet, out_dir=ctx.out, name=name, logx=logx,
                                       n_boot=ctx.n_boot))
        else:
            ctx.add(F.plot_metric_vs(dm, x, m, out_dir=ctx.out, name=name, logx=logx,
                                     n_boot=ctx.n_boot))
    shares = [s for s in ("w_honest", "w_minority", "w_attacker") if has_metric(dd, s)]
    if shares:
        ctx.add(F.plot_weight_shares(dd, x, out_dir=ctx.out, name=f"{ctx.exp}_weights_vs_{x}",
                                     logx=logx, n_boot=ctx.n_boot))


# --------------------------------------------------------------------------------------
# experiment handlers
# --------------------------------------------------------------------------------------
def e1(d, ctx):
    col = pick_col(d, ["attack", "dataset", "scheme"])
    for m in ("rmse_norm", "kendall_tau", "w_attacker"):
        metric_table(d, ctx, m, col, f"{ctx.exp}_{m}",
                     f"{F.metric_label(m)} by aggregator and {col.replace('_', ' ')} "
                     r"(mean $\pm$ std over seeds; best in bold; $^{\dagger}$: Wilcoxon--Holm "
                     r"$p<0.05$ vs Power-SNR (screened)).")
    multi_table(d, ctx, ["rmse_norm", "kendall_tau", "w_attacker"], col, f"{ctx.exp}_main",
                "Explanation fidelity and attacker weight share by aggregator and attack.")


def e2(d, ctx):
    curves(d, ctx, "attack_param")


def e3(d, ctx):
    curves(d, ctx, "attack_frac")


def e4(d, ctx):
    curves(d, ctx, "K", logx=bool(d["K"].max() / max(d["K"].min(), 1) >= 8) if "K" in d else False)


def _alpha_column(d: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    # The runner stores the Dirichlet alpha in scheme_param; IID is plotted as alpha = 1000.
    if "scheme_param" in d.columns and "scheme" in d.columns and (d["scheme"] == "dirichlet").any():
        alpha = pd.to_numeric(d["scheme_param"], errors="coerce").where(d["scheme"] == "dirichlet", 1000.0)
        if alpha.nunique() > 1:
            return d.assign(alpha=alpha), "alpha"
    ap =pd.to_numeric(d.get("attack_param"), errors="coerce") if "attack_param" in d else None
    if ap is not None and ap.nunique() > 1:
        return d.assign(alpha=ap), "alpha"
    if "scheme" in d.columns:
        num = d["scheme"].astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
        num = pd.to_numeric(num, errors="coerce")
        if num.notna().all() and num.nunique() > 1:
            return d.assign(alpha=num), "alpha"
        if d["scheme"].nunique() > 1:
            return d, "scheme"
    return d, None


def e5(d, ctx):
    dd, x = _alpha_column(d)
    if x is None:
        ctx.skip("alpha curves", "neither attack_param nor scheme varies")
        return
    curves(dd, ctx, x, metrics=["rmse_norm", "kendall_tau", "rmse_norm_minority", "w_minority"],
           logx=(x == "alpha"))


def e6(d, ctx):
    col = pick_col(d, ["dataset", "model", "attack", "scheme"])
    for m in ("rmse_norm", "kendall_tau"):
        metric_table(d, ctx, m, col, f"{ctx.exp}_main_{m}",
                     f"{F.metric_label(m)} across {col}s (mean $\\pm$ std; best in bold; "
                     r"$^{\dagger}$: Wilcoxon--Holm $p<0.05$ vs Power-SNR (screened)).")
    blocks = varying_keys(d)
    if not blocks:
        blocks = ["seed"]
    rows = []
    for m in ("rmse_norm", "kendall_tau"):
        if not has_metric(d, m):
            continue
        dd = d.assign(**{m: pd.to_numeric(d[m], errors="coerce")})
        if dd["aggregator"].nunique() < 2:
            ctx.skip("CD diagram", "fewer than 2 aggregators")
            return
        stat, p, ranks, cd = friedman_nemenyi(dd, blocks, m, lower_is_better=lib(m))
        rows.append({"metric": m, "block_cols": "+".join(blocks), "friedman_chi2": stat,
                     "p_value": p, "cd": cd, **{f"rank_{a}": r for a, r in ranks.items()}})
        ctx.add(F.plot_cd_diagram(ranks, cd, out_dir=ctx.out, name=f"{ctx.exp}_cd_{m}"))
    if rows:
        p = ctx.out / f"{ctx.exp}_friedman.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        ctx.add(p)


def e7(d, ctx):
    col = pick_col(d, ["attack", "dataset"])
    for m in ("w_attacker", "attack_success", "rmse_norm"):
        metric_table(d, ctx, m, col, f"{ctx.exp}_defense_{m}",
                     f"Defense evaluation: {F.metric_label(m)} by aggregator and attack.")
    multi_table(d, ctx, ["rmse_norm", "w_attacker", "attack_success"], col, f"{ctx.exp}_defense",
                "Defense evaluation: fidelity, attacker weight share and attack success rate.")


def e8(d, ctx):
    col = pick_col(d, ["attack", "scheme", "dataset"])
    multi_table(d, ctx, ["rmse_norm", "kendall_tau", "w_attacker", "rmse_norm_minority"], col,
                f"{ctx.exp}_ablation", "Ablation of the Power-SNR components.")
    metric_table(d, ctx, "rmse_norm", col, f"{ctx.exp}_ablation_rmse_norm",
                 "Ablation: RMSE (L$_1$-normalised), mean $\\pm$ std over seeds.")


def e10(d, ctx):
    if not (has_metric(d, "rmse_norm") and has_metric(d, "rmse_norm_minority")):
        ctx.skip("pareto", "rmse_norm or rmse_norm_minority missing")
        return
    ctx.add(F.plot_pareto(d, "rmse_norm", "rmse_norm_minority", out_dir=ctx.out,
                          name=f"{ctx.exp}_pareto", n_boot=ctx.n_boot))
    col = pick_col(d, ["scheme", "attack", "dataset"])
    multi_table(d, ctx, ["rmse_norm", "rmse_norm_minority", "w_minority"], col,
                f"{ctx.exp}_fairness", "Global vs minority fidelity (fairness trade-off).")


def e9(d, ctx):
    # Quality heterogeneity: one column per (attack, intensity), e.g. "label_flip 0.2".
    ap = pd.to_numeric(d["attack_param"], errors="coerce")
    label = d["attack"].where(ap.isna(), d["attack"] + " " + ap.map(lambda v: f"{v:g}"))
    e1(d.assign(attack=label), ctx)


HANDLERS: dict[int, Callable] = {1: e1, 2: e2, 3: e3, 4: e4, 5: e5, 6: e6, 7: e7, 8: e8, 9: e9, 10: e10}


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def load_results(run: str | Path) -> pd.DataFrame:
    p = Path(run)
    if p.is_dir():
        p = p / "results.parquet"
    if not p.exists():
        raise FileNotFoundError(f"results file not found: {p}")
    return pd.read_parquet(p)


def make_assets(df: pd.DataFrame, out: str | Path, n_boot: int = 2000,
                experiments: list[str] | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for c in ("experiment", "aggregator"):
        if c not in df.columns:
            raise KeyError(f"results lack required column {c!r}")
    manifest: dict = {"experiments": {}, "skipped": []}
    present = list(pd.unique(df["experiment"].astype(str)))
    for exp in sorted(present, key=lambda e: (experiment_number(e) or 10 ** 6, e)):
        if experiments and exp not in experiments:
            continue
        d = df[df["experiment"].astype(str) == exp].copy()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", exp)
        ctx = Ctx(out / safe, safe, n_boot)
        ctx.out.mkdir(parents=True, exist_ok=True)
        write_summary(d, ctx)
        write_stats(d, ctx)
        num = experiment_number(exp)
        handler = HANDLERS.get(num) if num is not None else None
        if handler is None:
            ctx.skip("experiment-specific assets", "no handler for this experiment id")
        else:
            handler(d, ctx)
        manifest["experiments"][exp] = ctx.produced
        manifest["skipped"].extend(ctx.skipped)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="results/<run> directory or a .parquet file")
    ap.add_argument("--out", required=True, help="output directory for paper assets")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for CIs")
    ap.add_argument("--experiments", nargs="*", default=None, help="restrict to these experiments")
    a = ap.parse_args(argv)
    df = load_results(a.run)
    man = make_assets(df, a.out, n_boot=a.n_boot, experiments=a.experiments)
    n = sum(len(v) for v in man["experiments"].values())
    print(f"wrote {n} files for {len(man['experiments'])} experiment(s) into {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
