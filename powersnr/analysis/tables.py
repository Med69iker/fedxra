"""Booktabs LaTeX tables (mean +/- std, best in bold, optional significance dagger).

Typical use::

    s = summarize(df, ["aggregator", "attack"], "rmse_norm")
    pt = paired_tests(df, ["attack"], "rmse_norm", reference="psnr_screen")
    tex = to_latex_table(s, row="aggregator", col="attack", pvalues=pt,
                         caption="...", label="tab:e1", path="tab_e1.tex")

Requires \\usepackage{booktabs} (and graphicx when resize=True) in the preamble.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import PRETTY_NAMES

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def latex_escape(s) -> str:
    """Escape LaTeX special characters (underscores etc.); en dash -> '--'."""
    out = "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(s))
    return out.replace("–", "--").replace("—", "---")


def pretty_name(name) -> str:
    """Pretty, LaTeX-escaped display name (aggregators mapped through PRETTY_NAMES)."""
    return latex_escape(PRETTY_NAMES.get(str(name), str(name)))


def _fmt_cell(mean: float, std: float, decimals: int, bold: bool, dagger: bool,
              show_std: bool) -> str:
    if not np.isfinite(mean):
        return "--"
    m = f"{mean:.{decimals}f}"
    body = m
    if show_std and np.isfinite(std):
        body = f"{m} \\pm {std:.{decimals}f}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    if dagger:
        body += "^{\\dagger}"
    return f"${body}$"


def _sig_lookup(pvalues: pd.DataFrame | None, row: str, col: str, reference: str,
                alpha: float) -> set:
    """Set of (row_value, col_value) cells significantly different from `reference`."""
    if pvalues is None or pvalues.empty:
        return set()
    if "aggregator" not in (row, col):
        raise ValueError("pvalues require one of row/col to be 'aggregator'")
    other = col if row == "aggregator" else row
    pv = pvalues
    if "reference" in pv.columns:
        pv = pv[pv["reference"] == reference]
    p = pv["p_holm"] if "p_holm" in pv.columns else pv["p_value"]
    sig = pv[p < alpha]
    cells = set()
    for _, r in sig.iterrows():
        o = _key(r[other]) if other in sig.columns else None
        if row == "aggregator":
            cells.add((r["aggregator"], o))
        else:
            cells.add((o, r["aggregator"]))
    return cells


def _key(v):
    return "__nan__" if (isinstance(v, float) and np.isnan(v)) else v


def to_latex_table(summary_df: pd.DataFrame, row: str = "aggregator", col: str = "attack", *,
                   mean_col: str = "mean", std_col: str = "std", decimals: int = 3,
                   bold_best: bool = True, lower_is_better: bool = True,
                   best_within: str = "col", show_std: bool = True,
                   pvalues: pd.DataFrame | None = None, reference: str = "psnr_screen",
                   alpha: float = 0.05, caption: str | None = None, label: str | None = None,
                   row_order: Sequence | None = None, col_order: Sequence | None = None,
                   row_header: str | None = None, col_header: str | None = None,
                   path: str | Path | None = None, resize: bool = False,
                   table_env: str = "table") -> str:
    """Render a summary (output of `stats.summarize`, by=[row, col]) as a booktabs table.

    best_within: "col" -> the best cell of each column is bold (rows = aggregators);
                 "row" -> the best cell of each row is bold (cols = aggregators).
    Ties at the displayed precision are all bold. If `pvalues` (output of
    `stats.paired_tests` with by=[the non-aggregator dimension]) is given, cells whose
    Holm-corrected p < alpha vs `reference` get a dagger.
    """
    if best_within not in ("row", "col"):
        raise ValueError("best_within must be 'row' or 'col'")
    s = summary_df.copy()
    for c in (row, col, mean_col):
        if c not in s.columns:
            raise KeyError(f"summary_df lacks column {c!r}")
    if std_col not in s.columns:
        s[std_col] = np.nan
    s["_r"] = s[row].map(_key)
    s["_c"] = s[col].map(_key)
    rows = list(row_order) if row_order is not None else list(dict.fromkeys(s["_r"]))
    if row == "aggregator" and row_order is None:
        from .figures import order_aggregators  # local import keeps tables light
        rows = order_aggregators(rows)
    cols = list(col_order) if col_order is not None else list(dict.fromkeys(s["_c"]))
    if col == "aggregator" and col_order is None:
        from .figures import order_aggregators
        cols = order_aggregators(cols)
    means = s.pivot_table(index="_r", columns="_c", values=mean_col, aggfunc="mean")
    stds = s.pivot_table(index="_r", columns="_c", values=std_col, aggfunc="mean")
    means = means.reindex(index=rows, columns=cols)
    stds = stds.reindex(index=rows, columns=cols)
    M = means.to_numpy(dtype=float)
    R = np.round(M, decimals)
    best = np.zeros_like(M, dtype=bool)
    if bold_best:
        axis = 0 if best_within == "col" else 1
        with np.errstate(all="ignore"):
            ref = np.nanmin(R, axis=axis, keepdims=True) if lower_is_better else \
                np.nanmax(R, axis=axis, keepdims=True)
        best = np.isfinite(R) & (R == ref)
    sig = _sig_lookup(pvalues, row, col, reference, alpha)

    def disp(v, dim):
        if v == "__nan__":
            return "--"
        return pretty_name(v) if dim == "aggregator" else latex_escape(v)

    lines = []
    lines.append(f"\\begin{{{table_env}}}[t]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    if resize:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append("\\toprule")
    if col_header:
        lines.append(f" & \\multicolumn{{{len(cols)}}}{{c}}{{{latex_escape(col_header)}}} \\\\")
        lines.append(f"\\cmidrule(lr){{2-{len(cols) + 1}}}")
    head = latex_escape(row_header if row_header is not None else row.replace("_", " ").capitalize())
    lines.append(head + " & " + " & ".join(disp(c, col) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for i, r in enumerate(rows):
        cells = []
        for j, c in enumerate(cols):
            is_ref = (r == reference) if row == "aggregator" else (c == reference)
            dag = (not is_ref) and ((r, c) in sig)
            cells.append(_fmt_cell(M[i, j], float(stds.iat[i, j]), decimals, bool(best[i, j]),
                                   dag, show_std))
        lines.append(disp(r, row) + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if resize:
        lines.append("}")
    lines.append(f"\\end{{{table_env}}}")
    tex = "\n".join(lines) + "\n"
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tex, encoding="utf-8")
    return tex


def to_latex_multi_metric(summaries: Mapping[str, pd.DataFrame], row: str = "aggregator",
                          col: str = "attack", *, lower_is_better: Mapping[str, bool] | None = None,
                          decimals: int | Mapping[str, int] = 3, show_std: bool = False,
                          metric_labels: Mapping[str, str] | None = None,
                          pvalues: Mapping[str, pd.DataFrame] | None = None,
                          reference: str = "psnr_screen", alpha: float = 0.05,
                          caption: str | None = None, label: str | None = None,
                          path: str | Path | None = None, resize: bool = True) -> str:
    """Several metrics side by side: column groups = metrics, sub-columns = values of `col`.

    Best is bolded within each (metric, col) column, direction-aware per metric.
    """
    lower_is_better = dict(lower_is_better or {})
    metric_labels = dict(metric_labels or {})
    metrics = list(summaries)
    from .figures import order_aggregators
    all_rows: list = []
    cols_per: dict[str, list] = {}
    for m in metrics:
        s = summaries[m]
        all_rows.extend(s[row].map(_key).tolist())
        cols_per[m] = list(dict.fromkeys(s[col].map(_key)))
    rows = list(dict.fromkeys(all_rows))
    if row == "aggregator":
        rows = order_aggregators(rows)
    blocks = []
    for m in metrics:
        s = summaries[m].copy()
        s["_r"] = s[row].map(_key); s["_c"] = s[col].map(_key)
        dec = decimals[m] if isinstance(decimals, Mapping) else decimals
        mean = s.pivot_table(index="_r", columns="_c", values="mean", aggfunc="mean").reindex(
            index=rows, columns=cols_per[m])
        std = s.pivot_table(index="_r", columns="_c", values="std", aggfunc="mean").reindex(
            index=rows, columns=cols_per[m]) if "std" in s.columns else mean * np.nan
        M = mean.to_numpy(dtype=float)
        R = np.round(M, dec)
        lib = lower_is_better.get(m, True)
        with np.errstate(all="ignore"):
            ref = np.nanmin(R, axis=0, keepdims=True) if lib else np.nanmax(R, axis=0, keepdims=True)
        best = np.isfinite(R) & (R == ref)
        sig = _sig_lookup((pvalues or {}).get(m), row, col, reference, alpha)
        blocks.append((m, cols_per[m], M, std.to_numpy(dtype=float), best, sig, dec))
    lines = ["\\begin{table*}[t]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    if resize:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l" + "".join("c" * len(b[1]) for b in blocks) + "}")
    lines.append("\\toprule")
    head1, cmid, start = [""], [], 2
    for m, cl, *_ in blocks:
        arrow = "$\\downarrow$" if lower_is_better.get(m, True) else "$\\uparrow$"
        lab = metric_labels.get(m, latex_escape(m))
        head1.append(f"\\multicolumn{{{len(cl)}}}{{c}}{{{lab} {arrow}}}")
        cmid.append(f"\\cmidrule(lr){{{start}-{start + len(cl) - 1}}}")
        start += len(cl)
    lines.append(" & ".join(head1) + " \\\\")
    lines.append(" ".join(cmid))
    head2 = [latex_escape(row.replace("_", " ").capitalize())]
    for _, cl, *_ in blocks:
        head2.extend("--" if c == "__nan__" else latex_escape(c) for c in cl)
    lines.append(" & ".join(head2) + " \\\\")
    lines.append("\\midrule")
    for i, r in enumerate(rows):
        cells = [pretty_name(r) if row == "aggregator" else latex_escape(r)]
        for m, cl, M, S, best, sig, dec in blocks:
            for j, c in enumerate(cl):
                dag = r != reference and (r, c) in sig
                cells.append(_fmt_cell(M[i, j], S[i, j], dec, bool(best[i, j]), dag, show_std))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if resize:
        lines.append("}")
    lines.append("\\end{table*}")
    tex = "\n".join(lines) + "\n"
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tex, encoding="utf-8")
    return tex


__all__ = ["to_latex_table", "to_latex_multi_metric", "latex_escape", "pretty_name", "PRETTY_NAMES"]
