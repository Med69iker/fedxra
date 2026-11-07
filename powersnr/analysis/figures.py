"""Publication figures (matplotlib only).

Every public plotting function saves a vector PDF and a 300-dpi PNG into `out_dir`
(when given) and returns the list of written paths. Colours follow the Okabe-Ito
colour-blind-safe palette with a fixed aggregator -> (colour, marker, linestyle) map,
so the same aggregator looks identical in every figure of the paper.
All LaTeX-like labels are raw strings (r"...") to avoid escape bugs such as "\t".
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import PRETTY_NAMES  # noqa: E402
from .stats import summarize  # noqa: E402

# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999",
}

# Fixed mapping: name -> dict(color, marker, ls, lw, zorder)
AGG_STYLE: dict[str, dict] = {
    # FedXRA (the proposed rule, fitted-variance weights) is the only thick black line; its n-weight and
    # equal-weight variants are thinner black lines
    "psnr_v3_fit": dict(color=OKABE_ITO["black"], marker="o", ls="-", lw=2.6, zorder=12),
    "psnr_v3": dict(color=OKABE_ITO["black"], marker="s", ls=":", lw=1.6, zorder=11),
    "psnr_v3_unif": dict(color=OKABE_ITO["black"], marker="o", ls="--", lw=1.6, zorder=11),
    # Power-SNR family (warm colours, solid/dash variants)
    "psnr_screen": dict(color=OKABE_ITO["vermillion"], marker="o", ls="-", lw=1.6, zorder=9),
    "screen_only": dict(color=OKABE_ITO["vermillion"], marker="o", ls=":", lw=1.3, zorder=7),
    "screen_geomedian": dict(color=OKABE_ITO["vermillion"], marker="s", ls=":", lw=1.3, zorder=7),
    "psnr_screen_v1": dict(color=OKABE_ITO["orange"], marker="s", ls="-.", lw=1.3, zorder=7),
    "psnr_robust": dict(color=OKABE_ITO["orange"], marker="o", ls="--", lw=1.3, zorder=6),
    "psnr_legacy": dict(color=OKABE_ITO["purple"], marker="d", ls=":", lw=1.1, zorder=5),
    "psnr_base": dict(color=OKABE_ITO["purple"], marker="D", ls="-", lw=1.3, zorder=6),
    "psnr_gate": dict(color=OKABE_ITO["orange"], marker="^", ls="-", lw=1.3, zorder=6),
    "psnr_temp": dict(color=OKABE_ITO["yellow"], marker="v", ls="-", lw=1.3, zorder=6),
    "psnr_energy_only": dict(color=OKABE_ITO["purple"], marker="x", ls="--", lw=1.1, zorder=5),
    # Baselines (cool colours / greys)
    "mean": dict(color=OKABE_ITO["black"], marker="s", ls=":", lw=1.1, zorder=4),
    "fedavg": dict(color=OKABE_ITO["grey"], marker="s", ls="--", lw=1.1, zorder=4),
    "median": dict(color=OKABE_ITO["blue"], marker="P", ls="-", lw=1.1, zorder=4),
    "trimmed_mean": dict(color=OKABE_ITO["skyblue"], marker="X", ls="-", lw=1.1, zorder=4),
    "geomedian": dict(color=OKABE_ITO["green"], marker="h", ls="-", lw=1.1, zorder=4),
    "krum": dict(color=OKABE_ITO["blue"], marker="<", ls="--", lw=1.1, zorder=4),
    "multikrum": dict(color=OKABE_ITO["skyblue"], marker=">", ls="--", lw=1.1, zorder=4),
    "ivw": dict(color=OKABE_ITO["green"], marker="d", ls="--", lw=1.1, zorder=4),
    "dersimonian_laird": dict(color=OKABE_ITO["black"], marker="*", ls="-.", lw=1.1, zorder=4),
    "fltrust_like": dict(color=OKABE_ITO["grey"], marker="p", ls="-.", lw=1.1, zorder=4),
    "trimmed_mean_0.3": dict(color=OKABE_ITO["skyblue"], marker="X", ls=":", lw=1.1, zorder=4),
    "bulyan": dict(color=OKABE_ITO["purple"], marker="^", ls="--", lw=1.1, zorder=4),
    "centered_clipping": dict(color=OKABE_ITO["orange"], marker="v", ls="-", lw=1.1, zorder=4),
    "foolsgold": dict(color=OKABE_ITO["yellow"], marker="D", ls="-.", lw=1.1, zorder=4),
}
_FALLBACK_COLORS = [OKABE_ITO[c] for c in ("blue", "orange", "green", "purple", "skyblue",
                                           "vermillion", "black", "yellow", "grey")]
_FALLBACK_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "h", "*"]

# Order in which aggregators appear in legends.
AGG_ORDER = ["psnr_v3_fit", "psnr_v3", "psnr_v3_unif", "psnr_screen", "psnr_screen_v1", "screen_geomedian", "screen_only", "psnr_robust", "psnr_legacy", "psnr_base", "psnr_gate", "psnr_temp", "psnr_energy_only",
             "ivw", "dersimonian_laird", "mean", "fedavg", "median", "trimmed_mean",
             "trimmed_mean_0.3", "geomedian", "krum", "multikrum", "bulyan", "centered_clipping", "foolsgold",
             "fltrust_like"]


METRIC_LABELS = {
    "rmse_raw": r"RMSE (raw)",
    "rmse_norm": r"RMSE (L$_1$-normalised)",
    "cosine": r"Cosine similarity",
    "kendall_tau": r"Kendall's $\tau$",
    "spearman": r"Spearman's $\rho$",
    "top3_jaccard": r"Top-3 Jaccard",
    "top5_jaccard": r"Top-5 Jaccard",
    "target_rank_shift": r"Target rank shift",
    "w_honest": r"Weight share (honest)",
    "w_minority": r"Weight share (minority)",
    "w_attacker": r"Weight share (attackers)",
    "rmse_raw_minority": r"RMSE minority (raw)",
    "rmse_norm_minority": r"RMSE minority (normalised)",
    "kendall_tau_minority": r"Kendall's $\tau$ (minority)",
    "attack_success": r"Attack success rate",
    "attack_param": r"Attack strength $\lambda$",
    "attack_frac": r"Fraction of attackers $f$",
    "K": r"Number of clients $K$",
    "alpha": r"Dirichlet $\alpha$",
}

WIDTHS = {"single": 3.5, "double": 7.0}

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.markersize": 3.5,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


@contextmanager
def paper_style():
    with plt.rc_context(RC):
        yield


def agg_style(name: str) -> dict:
    """Style dict for an aggregator (deterministic fallback for unknown names)."""
    if name in AGG_STYLE:
        return dict(AGG_STYLE[name])
    h = sum(ord(c) for c in str(name))
    return dict(color=_FALLBACK_COLORS[h % len(_FALLBACK_COLORS)],
                marker=_FALLBACK_MARKERS[h % len(_FALLBACK_MARKERS)], ls="-", lw=1.1, zorder=3)


def pretty(name) -> str:
    return PRETTY_NAMES.get(str(name), str(name))


def metric_label(m: str) -> str:
    return METRIC_LABELS.get(m, m.replace("_", " "))


def order_aggregators(names: Iterable) -> list:
    names = list(dict.fromkeys(names))
    known = [a for a in AGG_ORDER if a in names]
    return known + sorted(a for a in names if a not in AGG_ORDER)


def figure_size(width: str | float = "single", aspect: float = 0.72, nrows: int = 1,
                ncols: int = 1) -> tuple[float, float]:
    w = WIDTHS[width] if isinstance(width, str) else float(width)
    panel_w = w / ncols
    return w, panel_w * aspect * nrows + (0.35 if nrows > 1 else 0.0)


def save_figure(fig, out_dir: str | Path, name: str, dpi: int = 300) -> list[Path]:
    """Save `fig` as <out_dir>/<name>.pdf (vector) and .png (dpi). Closes the figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=dpi)
    plt.close(fig)
    return paths


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _finish(fig, out_dir, name) -> list[Path]:
    if out_dir is None:
        return []
    return save_figure(fig, out_dir, name)


# --------------------------------------------------------------------------------------
# Line plots with CI bands
# --------------------------------------------------------------------------------------
def _draw_lines(ax, df: pd.DataFrame, x: str, metric: str, hue: str, aggregators, logx: bool,
                n_boot: int, seed: int) -> list:
    d = df[df[metric].notna()]
    if aggregators is not None:
        d = d[d[hue].isin(aggregators)]
    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return []
    s = summarize(d, [hue, x], metric, n_boot=n_boot, seed=seed)
    numeric = _is_numeric(d[x])
    if numeric:
        s = s.sort_values(x)
        xmap = None
    else:
        cats = list(pd.unique(d[x].astype(str)))
        xmap = {c: i for i, c in enumerate(cats)}
    handles = []
    hues = order_aggregators(s[hue].unique()) if hue == "aggregator" else list(pd.unique(s[hue]))
    for h in hues:
        sh = s[s[hue] == h]
        xs = sh[x].to_numpy(dtype=float) if numeric else np.array([xmap[str(v)] for v in sh[x]])
        st = agg_style(h) if hue == "aggregator" else agg_style(str(h))
        line, = ax.plot(xs, sh["mean"], color=st["color"], marker=st["marker"], ls=st["ls"],
                        lw=st["lw"], zorder=st["zorder"], label=pretty(h))
        ax.fill_between(xs, sh["ci_low"], sh["ci_high"], color=st["color"], alpha=0.15,
                        lw=0, zorder=st["zorder"] - 1)
        handles.append(line)
    if not numeric:
        ax.set_xticks(range(len(xmap)))
        ax.set_xticklabels(list(xmap), rotation=30, ha="right")
    if logx and numeric:
        ax.set_xscale("log")
    return handles


def plot_metric_vs(df: pd.DataFrame, x: str, metric: str, hue: str = "aggregator", *,
                   out_dir: str | Path | None = None, name: str | None = None,
                   aggregators: Sequence[str] | None = None, width: str | float = "single",
                   logx: bool = False, xlabel: str | None = None, ylabel: str | None = None,
                   title: str | None = None, legend: bool = True, ax=None,
                   n_boot: int = 2000, seed: int = 0) -> list[Path]:
    """Mean of `metric` vs `x` (one line per `hue`), with 95% bootstrap CI band."""
    with paper_style():
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=figure_size(width))
        else:
            fig = ax.figure
        handles = _draw_lines(ax, df, x, metric, hue, aggregators, logx, n_boot, seed)
        ax.set_xlabel(xlabel if xlabel is not None else metric_label(x))
        ax.set_ylabel(ylabel if ylabel is not None else metric_label(metric))
        if title:
            ax.set_title(title)
        if legend and handles and own:
            ncol = 2 if len(handles) > 4 else 1
            ax.legend(handles=handles, ncol=ncol, loc="best")
        if not own:
            return []
        return _finish(fig, out_dir, name or f"{metric}_vs_{x}")


def plot_metric_grid(df: pd.DataFrame, x: str, metric: str, col: str, hue: str = "aggregator", *,
                     out_dir: str | Path | None = None, name: str | None = None,
                     aggregators: Sequence[str] | None = None, ncols: int | None = None,
                     width: str | float = "double", logx: bool = False, sharey: bool = False,
                     xlabel: str | None = None, ylabel: str | None = None,
                     n_boot: int = 2000, seed: int = 0) -> list[Path]:
    """Facet helper: one panel per value of `col`, mean +/- CI of `metric` vs `x`."""
    panels = list(pd.unique(df[col]))
    n = len(panels)
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    with paper_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=figure_size(width, aspect=0.8, nrows=nrows,
                                                                    ncols=ncols),
                                 squeeze=False, sharey=sharey)
        handles_all: dict[str, object] = {}
        for i, p in enumerate(panels):
            ax = axes[i // ncols][i % ncols]
            sub = df[df[col] == p] if not pd.isna(p) else df[df[col].isna()]
            hs = _draw_lines(ax, sub, x, metric, hue, aggregators, logx, n_boot, seed)
            for h in hs:
                handles_all.setdefault(h.get_label(), h)
            ax.set_title(f"{p}" if col != "attack" else str(p).replace("_", " "))
            ax.set_xlabel(xlabel if xlabel is not None else metric_label(x))
            if i % ncols == 0:
                ax.set_ylabel(ylabel if ylabel is not None else metric_label(metric))
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        if handles_all:
            fig.legend(list(handles_all.values()), list(handles_all), loc="lower center",
                       ncol=min(len(handles_all), 5), bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout()
        return _finish(fig, out_dir, name or f"{metric}_vs_{x}_by_{col}")


def plot_weight_shares(df: pd.DataFrame, x: str, *, out_dir: str | Path | None = None,
                       name: str | None = None, aggregators: Sequence[str] | None = None,
                       shares: Sequence[str] = ("w_honest", "w_minority", "w_attacker"),
                       width: str | float = "double", logx: bool = False,
                       n_boot: int = 2000, seed: int = 0) -> list[Path]:
    """One panel per weight-share column (skipping all-NaN ones), lines per aggregator.

    Non weight-based aggregators (NaN weights) drop out automatically.
    """
    shares = [s for s in shares if s in df.columns and df[s].notna().any()]
    if not shares:
        raise ValueError("no non-NaN weight-share columns to plot")
    long = df.melt(id_vars=[c for c in df.columns if c not in shares], value_vars=shares,
                   var_name="share", value_name="weight_share")
    long["share"] = long["share"].map(lambda s: metric_label(s))
    return plot_metric_grid(long, x, "weight_share", "share", out_dir=out_dir,
                            name=name or f"weight_shares_vs_{x}", aggregators=aggregators,
                            ncols=len(shares), width=width, logx=logx, sharey=True,
                            ylabel=r"Weight share", n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------------------
# Critical-difference diagram (Demsar 2006)
# --------------------------------------------------------------------------------------
def _cliques(ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Maximal groups [i, j] (indices into sorted ranks) whose rank span < cd."""
    k = len(ranks)
    spans = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i:
            spans.append((i, j))
    # remove spans contained in another span
    maximal = [s for s in spans if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in spans)]
    return maximal


def plot_cd_diagram(avg_ranks: pd.Series | dict, cd: float, *, out_dir: str | Path | None = None,
                    name: str = "cd_diagram", width: str | float = "single",
                    title: str | None = None) -> list[Path]:
    """Demsar-style critical difference diagram drawn manually.

    Best (lowest) ranks are on the left; aggregators not significantly different
    (rank difference < CD, Nemenyi) are joined by a thick horizontal bar.
    """
    r = pd.Series(avg_ranks, dtype=float).sort_values()
    names = list(r.index)
    ranks = r.to_numpy()
    k = len(ranks)
    lo, hi = 1, max(k, int(np.ceil(ranks.max())))
    n_left = int(np.ceil(k / 2))
    cliques = _cliques(ranks, cd) if np.isfinite(cd) else []
    w = WIDTHS[width] if isinstance(width, str) else float(width)
    row_h = 0.16
    height = 0.75 + row_h * (max(n_left, k - n_left) + 1) + 0.08 * len(cliques)
    with paper_style():
        fig = plt.figure(figsize=(w, height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.grid(False)
        # coordinate system: x in [0, 1], y downward from 0 (top) to `ymax`
        ymax = height / w  # keep units roughly isotropic
        ax.set_xlim(0, 1)
        ax.set_ylim(ymax, 0)
        x0, x1 = 0.22, 0.78
        def xr(v):  # rank -> x
            return x0 + (v - lo) / (hi - lo) * (x1 - x0) if hi > lo else (x0 + x1) / 2
        y_axis = 0.12 * (3.5 / w) + 0.06
        ax.plot([x0, x1], [y_axis, y_axis], color="k", lw=0.8)
        for t in range(lo, hi + 1):
            ax.plot([xr(t), xr(t)], [y_axis, y_axis - 0.015], color="k", lw=0.8)
            ax.text(xr(t), y_axis - 0.022, str(t), ha="center", va="bottom", fontsize=8)
        for t in np.arange(lo + 0.5, hi, 1.0):
            ax.plot([xr(t), xr(t)], [y_axis, y_axis - 0.008], color="k", lw=0.5)
        # CD bar
        if np.isfinite(cd) and cd > 0:
            yb = y_axis - 0.075
            ax.plot([xr(lo), xr(min(lo + cd, hi + 1e-9))], [yb, yb], color="k", lw=1.0)
            for xx in (xr(lo), xr(min(lo + cd, hi))):
                ax.plot([xx, xx], [yb - 0.008, yb + 0.008], color="k", lw=1.0)
            ax.text((xr(lo) + xr(min(lo + cd, hi))) / 2, yb - 0.012, f"CD = {cd:.2f}",
                    ha="center", va="bottom", fontsize=8)
        # clique bars just below axis
        yc0 = y_axis + 0.03
        dy_c = 0.022
        for ci, (i, j) in enumerate(cliques):
            yc = yc0 + ci * dy_c
            ax.plot([xr(ranks[i]) - 0.006, xr(ranks[j]) + 0.006], [yc, yc], color="k", lw=2.2,
                    solid_capstyle="round")
        y_names0 = yc0 + len(cliques) * dy_c + 0.04
        dy = row_h / w * 1.0
        for idx, (nm, rv) in enumerate(zip(names, ranks)):
            st = agg_style(nm)
            emph = nm == "psnr_screen"
            if idx < n_left:
                yy = y_names0 + idx * dy
                xe, ha = x0 - 0.02, "right"
            else:
                yy = y_names0 + (k - 1 - idx) * dy
                xe, ha = x1 + 0.02, "left"
            ax.plot([xr(rv), xr(rv), xe], [y_axis, yy, yy], color=st["color"],
                    lw=1.4 if emph else 0.8)
            label = f"{pretty(nm)} ({rv:.2f})" if ha == "right" else f"({rv:.2f}) {pretty(nm)}"
            ax.text(xe + (-0.005 if ha == "right" else 0.005), yy, label, ha=ha, va="center",
                    fontsize=8, fontweight="bold" if emph else "normal")
        if title:
            ax.text(0.5, 0.005, title, ha="center", va="top", fontsize=9)
        ax.set_ylim(max(ymax, y_names0 + max(n_left, k - n_left) * dy + 0.02), 0)
        return _finish(fig, out_dir, name)


# --------------------------------------------------------------------------------------
# Pareto plot
# --------------------------------------------------------------------------------------
def pareto_front(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (both coordinates lower-is-better)."""
    P = np.asarray(points, dtype=float)
    mask = np.ones(len(P), dtype=bool)
    for i in range(len(P)):
        dominated = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def plot_pareto(df: pd.DataFrame, x_metric: str, y_metric: str, hue: str = "aggregator", *,
                out_dir: str | Path | None = None, name: str | None = None,
                aggregators: Sequence[str] | None = None, width: str | float = "single",
                n_boot: int = 2000, seed: int = 0, annotate: bool = True) -> list[Path]:
    """Mean (x, y) per aggregator with 95% CI error bars; Pareto front (lower-left) dashed."""
    d = df[df[x_metric].notna() & df[y_metric].notna()]
    if aggregators is not None:
        d = d[d[hue].isin(aggregators)]
    if d.empty:
        raise ValueError(f"no rows with both {x_metric} and {y_metric}")
    sx = summarize(d, [hue], x_metric, n_boot=n_boot, seed=seed).set_index(hue)
    sy = summarize(d, [hue], y_metric, n_boot=n_boot, seed=seed).set_index(hue)
    names = order_aggregators(sx.index) if hue == "aggregator" else list(sx.index)
    with paper_style():
        fig, ax = plt.subplots(figsize=figure_size(width, aspect=0.85))
        pts = np.array([[sx.loc[a, "mean"], sy.loc[a, "mean"]] for a in names])
        for a, (px, py) in zip(names, pts):
            st = agg_style(a)
            xerr = [[px - sx.loc[a, "ci_low"]], [sx.loc[a, "ci_high"] - px]]
            yerr = [[py - sy.loc[a, "ci_low"]], [sy.loc[a, "ci_high"] - py]]
            ax.errorbar(px, py, xerr=xerr, yerr=yerr, color=st["color"], marker=st["marker"],
                        ms=6 if a == "psnr_screen" else 4.5, lw=0.8, capsize=1.5,
                        zorder=st["zorder"], label=pretty(a), ls="none",
                        mec="k" if a == "psnr_screen" else st["color"])
        front = pareto_front(pts)
        fp = pts[front]
        fp = fp[np.argsort(fp[:, 0])]
        if len(fp) > 1:
            ax.plot(fp[:, 0], fp[:, 1], color="k", ls="--", lw=0.7, zorder=1)
        ax.set_xlabel(metric_label(x_metric))
        ax.set_ylabel(metric_label(y_metric))
        ax.legend(loc="best", ncol=2 if len(names) > 4 else 1, fontsize=7)
        return _finish(fig, out_dir, name or f"pareto_{x_metric}_vs_{y_metric}")


# --------------------------------------------------------------------------------------
# Feature bars
# --------------------------------------------------------------------------------------
def plot_feature_bars(phi_table: pd.DataFrame, *, out_dir: str | Path | None = None,
                      name: str = "feature_bars", top: int | None = 10, sort_by: str | None = None,
                      normalize: bool = True, highlight: str | None = None,
                      width: str | float = "single", xlabel: str = r"Global importance (share)"
                      ) -> list[Path]:
    """Grouped horizontal bars: rows of `phi_table` are features, columns are sources
    (e.g. "oracle" and aggregator names). Values are L1-normalised per column if `normalize`.
    `highlight` names a feature whose label is set in bold (e.g. the protected feature).
    """
    t = phi_table.astype(float).copy()
    if normalize:
        t = t / t.abs().sum(axis=0).replace(0, np.nan)
    key = sort_by if sort_by is not None else t.columns[0]
    t = t.sort_values(key, ascending=False)
    if top is not None:
        t = t.head(top)
    t = t.iloc[::-1]
    cols = list(t.columns)
    n_f, n_c = t.shape
    with paper_style():
        fig, ax = plt.subplots(figsize=(WIDTHS[width] if isinstance(width, str) else width,
                                        0.6 + 0.12 * n_f * max(1, n_c) ** 0.5))
        bh = 0.8 / n_c
        y = np.arange(n_f)
        for j, c in enumerate(cols):
            st = agg_style(c) if c in AGG_STYLE else dict(color=OKABE_ITO["grey"] if c == "oracle"
                                                          else agg_style(c)["color"])
            ax.barh(y + (j - (n_c - 1) / 2) * bh, t[c].to_numpy(), height=bh, color=st["color"],
                    edgecolor="k" if c == "oracle" else "none", lw=0.4, label=pretty(c))
        ax.set_yticks(y)
        ax.set_yticklabels([str(f) for f in t.index])
        if highlight is not None:
            for lab in ax.get_yticklabels():
                if lab.get_text() == highlight:
                    lab.set_fontweight("bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=7)
        return _finish(fig, out_dir, name)


__all__ = ["AGG_STYLE", "OKABE_ITO", "PRETTY_NAMES", "METRIC_LABELS", "agg_style", "pretty",
           "metric_label", "paper_style", "figure_size", "save_figure", "plot_metric_vs",
           "plot_metric_grid", "plot_weight_shares", "plot_cd_diagram", "plot_pareto",
           "pareto_front", "plot_feature_bars", "order_aggregators"]
