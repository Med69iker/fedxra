"""Generate the protocol tables of Sections 3 and 6 from the evidence-run configurations (run 7) and the code.

Run from the manuscript folder:
    python tools/make_protocol_tables.py

Inputs (never edited):
    ../powersnr/kaggle_runs/run7/{A,B,C}/results/*_E*/config.yaml     (what was actually run)
    ../powersnr/kaggle_runs/run7/{A,B,C}/results/*_E*/results.parquet  (row counts, seed ranges)
    ../powersnr/kaggle_runs/run7/{A,B,C}/results/*_E*/run_meta.json    (versions, runtime)
    ../powersnr/kaggle_runs/run7/{A,B,C}/*.log                         (LightGBM version, printed by the notebooks)
    ../powersnr/kaggle_runs/run{2,3}/**/results/*_E*/config.yaml       (development seed ranges only)
    ../powersnr/powersnr/attacks.py      (threat-model docstrings, incl. _alie, _minmax, whitebox_variants)
    ../powersnr/powersnr/aggregators.py  (rule definitions and default parameters, incl. psnr_v3)
    ../powersnr/powersnr/runner.py       (oracle size, partition floors, white-box selection)

Outputs:
    tables/tab_experiments.tex       (label tab:experiments)
    tables/tab_threat.tex            (label tab:threat: data attacks and single-bank report attacks)
    tables/tab_threat_coalition.tex  (label tab:threatcoalition: stealthy, coalition and energy-inflation attacks)
    tables/tab_threat_whitebox.tex   (label tab:threatwhitebox: published rule-agnostic and white-box attacks)
    tables/tab_baselines.tex         (label tab:baselines: the 17 rules with the parameters as run)
    tables/protocol_numbers.tex      (\\Proto... macros used by sections 3 and 6)

Every cell is derived from the files above. Free-text cells (threat model, rule descriptions) are
condensed by hand from the docstrings; the script checks that each condensed cell still matches the
docstring/parameters it summarises (key phrases and default values), so a change in the code makes
this script fail instead of silently producing a stale table.

Display names (tools/common.py): psnr_v3_fit = FedXRA (the proposed rule), psnr_v3 = FedXRA-n,
psnr_v3_unif = FedXRA-U, psnr_screen = Power-SNR (earlier design),
adaptive_psnr = energy inflation (calibrated on SNR-type weights; NOT adaptive to each rule).
"""
from __future__ import annotations

import glob
import inspect
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

from common import CODE, HERE, MAIN, RUN, RUN_NAME, SEED0

DEV_RUNS = {"run2": CODE / "kaggle_runs" / "run2", "run3": CODE / "kaggle_runs" / "run3"}
TABLES = HERE / "tables"
sys.path.insert(0, str(CODE))

from powersnr import aggregators as agg  # noqa: E402
from powersnr import attacks as atk  # noqa: E402

EXPERIMENTS = [f"E{i}" for i in range(1, 12)]
# explain.py is not imported (it needs shap); its module constant is read from the source.
KERNEL_MAX_EXPLAIN = int(re.search(r"^KERNEL_MAX_EXPLAIN = (\d+)$",
                                   (CODE / "powersnr" / "explain.py").read_text(encoding="utf-8"), re.M).group(1))
# MAIN (common.py): the 17 rules of the benchmark (config keys). Anything else in a config is an ablation variant.
N_CAP_RULES = {"psnr_v3", "psnr_screen"}  # rules whose n_cap must equal n_explain

DATASET_NAMES = {"gmsc": "GMSC", "taiwan": "Taiwan", "german": "German", "homecredit": "Home Credit",
                 "lendingclub": "LendingClub"}
MODEL_NAMES = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "rf": "RF", "logreg": "LR", "mlp": "MLP"}

PURPOSE = {  # short purpose of each experiment (text only; parameters come from the configs)
    "E1": "Zero-variance fallacy",
    "E2": "Noise intensity, three banks",
    "E3": "Attacker fraction",
    "E4": "Number of banks",
    "E5": "Degree of label skew",
    "E6": "Main benchmark",
    "E7": "Attack strength and coalitions",
    "E8": "Ablation of FedXRA",
    "E9": "Bank size and data quality",
    "E10": "Minority banks (fairness)",
    "E11": "Local model families",
}

# Attack display name and the symbol of each parameter in the tables and text.
ATTACK_NAMES = {
    "input_noise": ("Input noise", {"lam": r"\eta"}),
    "label_flip": ("Label flipping", {"p": r"\pi"}),
    "frozen": ("Frozen", {}),
    "zero": ("Zero", {}),
    "scaling": ("Scaling", {"lam": r"\gamma"}),
    "feature_hiding": ("Feature hiding", {}),
    "feature_promotion": ("Feature promotion", {}),
    "mimic": ("Mimicry", {"strength": r"\xi"}),
    "collusion": ("Collusion", {}),
    "collusion_jitter": ("Jittered collusion", {"jitter": r"\zeta"}),
    "adaptive_psnr": ("Energy inflation", {"boost": "b", "cap": "c_a"}),
    "alie": ("ALIE", {}),
    "minmax": ("Min-max", {}),
    "whitebox": ("White-box coalition", {}),
    "whitebox_strong": ("Strong white-box coalition", {"claim_n": r"\tilde n"}),
}


# ----------------------------------------------------------------------------- helpers
def fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def setfmt(vals) -> str:
    vals = [fmt(v) for v in vals]
    return vals[0] if len(vals) == 1 else r"\{" + ",".join(vals) + r"\}"


def big(n: int) -> str:
    return f"{n:,d}".replace(",", "{,}")


def run_dirs() -> dict[str, Path]:
    out = {}
    for exp in EXPERIMENTS:
        hits = glob.glob(str(RUN / "*" / "results" / f"*_{exp}"))
        if len(hits) != 1:
            raise FileNotFoundError(f"{exp}: expected exactly one {RUN_NAME} result folder, found {hits}")
        out[exp] = Path(hits[0])
    return out


def load_cfg(d: Path) -> dict:
    return yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))


def write(name: str, text: str) -> None:
    (TABLES / name).write_text(text, encoding="utf-8")
    print(f"wrote tables/{name}")


# ----------------------------------------------------------------------------- tab:experiments
def partition_cell(p: dict, cfg: dict) -> str:
    s = p["scheme"]
    if s == "iid":
        return "IID"
    if s == "dirichlet":
        return rf"Dirichlet $\omega={fmt(p['alpha'])}$"
    if s == "quantity":
        if p["sigma"] != 1.0:
            raise ValueError("log-normal shape other than 1 is not covered by Section 6")
        return "log-normal sizes"
    if s == "feature":
        # threshold per dataset after the run's dataset_overrides
        parts = []
        for ds in cfg["datasets"]:
            q = {**p, **cfg.get("dataset_overrides", {}).get(ds, {}).get("partition", {})}
            grp = {"age": "age", "grade_num": "grade"}[q["group"]]
            parts.append((DATASET_NAMES[ds], grp, fmt(q["threshold"])))
        # the caption and Section 6 state the minority group: age > 60, > 50 for Taiwan
        expected = {"GMSC": ("age", "60"), "Taiwan": ("age", "50")}
        for n, g, t in parts:
            if expected.get(n, (g, t)) != (g, t) or n not in expected:
                raise ValueError(f"{cfg['experiment']}: minority group {n}: {g}>{t} not covered by the caption")
        nm = int(p.get("n_minority", 1))
        return f"{nm} minority bank{'s' if nm > 1 else ''}"
    raise ValueError(s)


def merged_attacks(cfg: dict) -> list[dict]:
    """Merge consecutive config entries of the same attack with the same fractions (e.g. the two
    energy-inflation entries of E7: boost values and the cap variant)."""
    out: list[dict] = []
    for a in cfg.get("attacks", []):
        a = {"name": a["name"], "fracs": list(a.get("fracs", [0.2])), "params": list(a.get("params") or [])}
        if out and out[-1]["name"] == a["name"] and out[-1]["fracs"] == a["fracs"]:
            out[-1]["params"] += a["params"] or [{}]
        else:
            out.append(a)
    return out


def attack_cell(cfg: dict) -> str:
    """Attacks with their parameters, grouped by attacker-fraction set: "attack, attack ($\\alpha=...$);
    attack ($\\alpha\\in...$)". Groups appear in the order of their first attack in the config."""
    attacks = merged_attacks(cfg)

    def alpha(fr):
        return rf"\alpha={fmt(fr[0])}" if len(fr) == 1 else rf"\alpha\in{setfmt(fr)}"

    groups: dict[tuple, list[str]] = {}
    for a in attacks:
        name, sym = ATTACK_NAMES[a["name"]]
        plist = [p for p in a["params"] if p]
        if a["name"] == "whitebox_strong":  # claim_n is a variant (run with and without), not a parameter value
            groups.setdefault(tuple(a["fracs"]), []).append(
                name + (rf" (also claiming $\tilde n={fmt(plist[0]['claim_n'])}$)" if plist else ""))
            continue
        if not plist and sym:  # parameter not set in the config: the code default was used
            fn = {**atk._DATA_REGISTRY, **atk._REPORT_REGISTRY}[a["name"]]
            sig = inspect.signature(fn).parameters
            plist = [{k: sig[k].default for k in sym if sig[k].default is not None}]
        unknown = {k for p in plist for k in p} - set(sym)
        if unknown:
            raise ValueError(f"{cfg['experiment']}: attack {a['name']} has parameters without a symbol: {unknown}")
        chunks = []
        for k in sorted({k for p in plist for k in p}):
            vals = [p[k] for p in plist if k in p]
            chunks.append(f"{sym[k]}={fmt(vals[0])}" if len(vals) == 1 else rf"{sym[k]}\in{setfmt(vals)}")
        groups.setdefault(tuple(a["fracs"]), []).append(name + (f" (${'; '.join(chunks)}$)" if chunks else ""))
    return "; ".join(", ".join(items) + f" [${alpha(fr)}$]" for fr, items in groups.items())


def rules_cell(cfg: dict) -> str:
    names = list(cfg["aggregators"])
    main = [a for a in names if a in MAIN]
    extra = [a for a in names if a not in MAIN]
    for a in extra:  # every non-benchmark rule must be a FedXRA variant
        if (cfg["aggregators"][a] or {}).get("rule") != "psnr_v3":
            raise ValueError(f"{cfg['experiment']}: {a} is neither a benchmark rule nor a FedXRA variant")
    if not extra and len(main) == len(MAIN):
        return ""
    return f"; {len(main)} rules" + (f" + {len(extra)} FedXRA variants" if extra else "")


def table_experiments(dirs: dict[str, Path]) -> dict:
    rows, totals = [], {"rows": 0, "cells": 0, "runtime_s": 0.0}
    shap_cfgs, cs_cfgs, seeds = {}, {}, {}
    for exp in EXPERIMENTS:
        cfg = load_cfg(dirs[exp])
        meta = json.loads((dirs[exp] / "run_meta.json").read_text(encoding="utf-8"))
        res = pd.read_parquet(dirs[exp] / "results.parquet", columns=["seed", "aggregator", "attack"])
        if cfg["experiment"] != exp:
            raise ValueError(f"{exp}: config says {cfg['experiment']}")
        if len(res) != meta["rows"]:
            raise ValueError(f"{exp}: results.parquet has {len(res)} rows, run_meta says {meta['rows']}")
        s0, s1, ns = int(res.seed.min()), int(res.seed.max()), int(res.seed.nunique())
        if (s0, ns) != (cfg["seed_offset"], cfg["seeds"]) or s1 != s0 + ns - 1:
            raise ValueError(f"{exp}: seeds in results ({s0}-{s1}, {ns}) do not match the config")
        if set(res.aggregator) != set(cfg["aggregators"]):
            raise ValueError(f"{exp}: rules in results differ from the config")
        if set(res.attack) - {"none"} != {a["name"] for a in cfg.get("attacks", [])}:
            raise ValueError(f"{exp}: attacks in results differ from the config")
        totals["rows"] += len(res)
        totals["cells"] += int(meta["cells"])
        totals["runtime_s"] += float(meta["runtime_s"])
        shap_cfgs[exp] = (cfg["shap"]["n_explain"], cfg["shap"]["n_background"], cfg["shap"]["n_repeats"])
        cs_cfgs[exp] = cfg["client_size"]
        seeds[exp] = (s0, s1, ns)
        dirs_ = [p for p in cfg["partitions"] if p["scheme"] == "dirichlet"]
        if len(dirs_) > 1:  # several label-skew levels: one cell "Dirichlet, omega in {...}"
            cells, done = [], False
            for p in cfg["partitions"]:
                if p["scheme"] != "dirichlet":
                    cells.append(partition_cell(p, cfg))
                elif not done:
                    cells.append(rf"Dirichlet $\omega\in{setfmt([q['alpha'] for q in dirs_])}$")
                    done = True
            parts = " or ".join(cells)
        else:
            parts = "; ".join(partition_cell(p, cfg) for p in cfg["partitions"])
        if s0 != cfg["seed_offset"] or s0 != SEED0:
            raise ValueError(f"{exp}: final seeds must start at {SEED0} (caption)")
        setting = "; ".join([
            ", ".join(DATASET_NAMES[d] for d in cfg["datasets"]),
            parts,
            "$K=" + ",".join(str(k) for k in cfg["K"]) + "$",
        ] + ([", ".join(MODEL_NAMES[m] for m in cfg["models"])] if cfg["models"] != ["xgboost"] else []))
        rows.append(" & ".join([
            exp, PURPOSE[exp], setting + rules_cell(cfg), attack_cell(cfg), str(ns), big(len(res)),
        ]) + r" \\")
    # modal settings and the experiments that deviate from them (stated in the caption)
    modal_shap = pd.Series(list(shap_cfgs.values())).mode()[0]
    modal_cs = pd.Series(list(cs_cfgs.values())).mode()[0]
    notes = []
    for exp, v in shap_cfgs.items():
        if v != modal_shap:
            notes.append(rf"{exp} explains {v[0]} rows with {v[1]} background rows (MLP: at most "
                         rf"{KERNEL_MAX_EXPLAIN}) and uses $\bar n={v[0]}$")
    for exp, v in cs_cfgs.items():
        if v != modal_cs:
            notes.append(f"{exp} caps banks at {big(v)} rows")
    caption = (
        r"Experimental design of the final run. Setting: datasets; partition (Section~\ref{sec:protocol_data}; "
        r"minority banks: borrowers older than 60, 50 for Taiwan); $K$; local models if not XGBoost. Attacks: "
        r"Tables~\ref{tab:threat}--\ref{tab:threatwhitebox}, attacker fraction $\alpha$ in brackets; every "
        r"experiment also runs without attack; the strong white-box coalition runs with and without claiming "
        r"$\tilde n$ examples per bank. All " + str(len(MAIN)) + r" rules of Table~\ref{tab:baselines} are run "
        r"unless the setting says otherwise. Seeds are numbered from " + str(SEED0) + ". "
        rf"Each bank explains up to {big(int(modal_shap[0]))} held-out rows ({modal_shap[1]} "
        rf"background rows, $R={modal_shap[2]}$) and holds at most {big(int(modal_cs))} rows"
        + (", except: " + "; ".join(notes) if notes else "") + ". Rows: result rows (seed $\\times$ condition "
        r"$\times$ rule).")
    rag = r">{\raggedright\arraybackslash}"
    tex = "\n".join([
        r"\begin{table*}[p]",
        r"\centering\linespread{1}\selectfont",
        r"\caption{" + caption + "}",
        r"\label{tab:experiments}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{0.9}",
        rf"\begin{{tabular}}{{@{{}}l {rag}p{{0.1\textwidth}} {rag}p{{0.25\textwidth}} {rag}p{{0.445\textwidth}} "
        r"r r@{}}",
        r"\toprule",
        r"ID & Purpose & Setting & Attacks [attacker fraction] & Seeds & Rows \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    write("tab_experiments.tex", tex)
    return {"totals": totals, "modal_shap": modal_shap, "modal_cs": int(modal_cs), "shap_cfgs": shap_cfgs,
            "seeds": seeds}


# ----------------------------------------------------------------------------- tab:threat
# (knowledge, capability, goal) condensed from the docstrings; `must` lists phrases that must occur in
# the docstring (whitespace-normalised) so the condensation stays faithful. `part`: A = data attacks and
# single-bank report attacks (tab:threat), B = stealthy / coalition / energy inflation (tab:threatcoalition),
# C = published rule-agnostic and white-box attacks (tab:threatwhitebox).
THREAT = {
    "input_noise": dict(
        part="A", family="Data",
        knowledge="Own local data only",
        capability=r"Adds $\mathcal{N}(0,(\eta\,\mathrm{sd}_j)^2)$ noise to each feature $j$ of its training "
                   r"data ($\mathrm{sd}_j$: its own standard deviation of feature $j$), then follows the protocol honestly",
        goal="Untargeted: a diluted, less stable report",
        must=["only the attacker's own local data", "N(0, (lam * s_j)^2)", "follows the protocol honestly",
              "untargeted degradation"]),
    "label_flip": dict(
        part="A", family="Data",
        knowledge="Own local data only",
        capability=r"Flips each training label with probability $\pi$, then follows the protocol honestly",
        goal="Untargeted degradation of the local model and its report",
        must=["only the attacker's own local data", "flips each local label independently with probability p",
              "untargeted degradation"]),
    "frozen": dict(
        part="A", family="Report",
        knowledge=r"Average $\ell_1$ mass of the honest reports",
        capability=r"Reports a constant vector with that mass and $\sigma^2_k=0$; does no local work",
        goal=r"Free-riding and a flattened ranking; maximal weight under $1/\sigma^2$ rules",
        must=["average L1", "constant vector", "sigma2 = 0", "free-riding"]),
    "zero": dict(
        part="A", family="Report",
        knowledge="None",
        capability=r"Reports $\mu_k\approx 0$ ($10^{-8}$ per feature) and $\sigma^2_k=0$",
        goal="Untargeted shrinkage; maximal weight under inverse-variance rules",
        must=["Knowledge: none", "default 1e-8", "sigma2 = 0", "untargeted shrinkage"]),
    "scaling": dict(
        part="A", family="Report",
        knowledge=r"Own $\mu_k$",
        capability=r"Reports $\gamma\mu_k$; $\sigma^2_k$ unchanged",
        goal="Dominate magnitude-sensitive rules while keeping a plausible ranking",
        must=["own honest mu only", "multiplies its honest mu by ``lam``", "sigma2 is", "reported unchanged",
              "dominate magnitude-sensitive"]),
    "feature_hiding": dict(
        part="A", family="Report",
        knowledge=r"Own $\mu_k$; the protected feature",
        capability=r"Sets the protected feature to 0 and redistributes its mass over the other features "
                   r"($\ell_1$ mass kept); $\sigma^2_k$ unchanged",
        goal="Push the protected feature down the global ranking",
        must=["the protected attribute", "zeroes the target's importance", "L1 mass preserved",
              "push the target feature down the global ranking"]),
    "feature_promotion": dict(
        part="A", family="Report",
        knowledge=r"Own $\mu_k$; the target feature",
        capability=r"Raises the target to twice the largest entry of its own $\mu_k$; rest unchanged",
        goal="Push the target feature up the global ranking",
        must=["raises the target's reported importance", "(default 2) times the maximum of its own mu",
              "push the target feature up the global ranking"]),
    "mimic": dict(
        part="B", family="Stealthy",
        knowledge="Honest reports; protected feature",
        capability=r"Reports $(1-\xi)\,\bar\mu_{\mathcal H}+\xi\,\mathrm{hide}(\bar\mu_{\mathcal H})$ "
                   r"with the median honest $\sigma^2$",
        goal="Lower the protected feature while evading outlier filters",
        must=["all honest reports of the round", "(1 - strength) * mean_honest + strength * hide(mean_honest)",
              "the median honest sigma2", "evade distance-, cosine- and outlier-based filters"]),
    "collusion": dict(
        part="B", family="Coalition",
        knowledge="Honest reports; coordination",
        capability=r"All attackers send the identical vector $\mathrm{hide}(\bar\mu_{\mathcal H})$, each with "
                   r"its own honest $\sigma^2_k$",
        goal="Lower the protected feature with a tight coalition",
        must=["all honest reports", "coordination with the", "*identical* vector", "own honest sigma2",
              "tight, consistent coalition"]),
    "collusion_jitter": dict(
        part="B", family="Coalition",
        knowledge="As collusion, plus the honest noise level and the duplicate test",
        capability=r"Sends $\mathrm{hide}(\bar\mu_{\mathcal H})$ plus independent Gaussian noise of variance "
                   r"$\zeta\,\tilde\sigma^2/(4d)$ per feature (clipped at 0) and declares the median honest "
                   r"$\sigma^2$, $\tilde\sigma^2$",
        goal="As collusion, while looking statistically independent",
        must=["as `collusion`", "groups reports that are closer than sampling noise allows",
              "jitter * s2 / (4 d)", "Declares sigma2 = s2", "clipped at 0",
              "looking statistically independent"]),
    "alie": dict(
        part="C", family="Published",
        knowledge="Honest reports; $m$; not the rule",
        capability=r"Reports $\bar\mu_{\mathcal H}$ with the protected coordinate lowered by "
                   r"$z_{\max}$ honest standard deviations (clipped at 0), $z_{\max}=\Phi^{-1}((K-s)/K)$, "
                   r"$s=\lfloor K/2+1\rfloor-m$; median honest $\sigma^2$",
        goal="Hide the protected feature inside the honest spread",
        must=["Baruch et al., 2019", "all honest reports (omniscient); the number of attackers",
              "lowered by z_max honest standard deviations (clipped at 0)", "z_max = Phi^-1((K - s) / K)",
              "s = floor(K / 2 + 1) - m", "Declares the median honest sigma2",
              "staying inside the honest spread on every coordinate"]),
    "minmax": dict(
        part="C", family="Published",
        knowledge="Honest reports; not the rule",
        capability=r"Reports $\max(0,\bar\mu_{\mathcal H}-g\,\mathrm{sd}_{\mathcal H})$ (coordinate-wise honest "
                   r"standard deviation) with the largest $g$ whose largest distance to an honest report stays within "
                   r"the largest honest pairwise distance; median honest $\sigma^2$",
        goal="Untargeted: the largest shift that looks honest",
        must=["Shejwalkar and Houmansadr, 2021", "untargeted", "no knowledge of the rule",
              "p = -std (inverse-std direction)", "largest value (bisection)",
              "does not exceed the maximum honest pairwise distance", "clipped at 0",
              "median honest sigma2", "looking no more distant than an honest bank"]),
    "adaptive_psnr": dict(
        part="B", family="Weight inflation",
        knowledge=r"Honest reports; SNR-type weight $\|\mu\|_2^2/\sigma^2$, cap $c_a$; not the deployed rule",
        capability=r"Reports the hiding direction of its own $\mu_k$ with $\sigma^2_k=0$, rescaled to $b$ times "
                   r"the largest honest energy $\|\mu\|_2^2$, or (cap variant) to the energy that puts "
                   r"$\|\mu\|_2^2/\sigma^2$ at $c_a$ times the median weight",
        goal="Large weight under SNR-type weights; hide the protected feature",
        must=["all honest reports of the round", "feature-hiding direction of its own mu",
              "(sigma2 = sigma2_floor if a floor is enforced, else 0)",
              "``boost`` (default 10) x the", "maximum honest energy",
              "clips w_k <= cap * median(w)"]),
    "whitebox": dict(
        part="C", family="White-box",
        knowledge="Honest reports; the deployed rule, which it can evaluate; not the oracle",
        capability=r"For each rule, evaluates 24 coordinated variants (hiding strength $\theta\in\{0.3,0.6,1\}$; "
                   r"no camouflage, jitter of variance $J\tilde\sigma^2/(4d)$ with $J\in\{1.5,3\}$, or hiding applied "
                   r"to distinct honest reports; claimed $\sigma^2$ zero or the honest median) and submits the one "
                   r"that ranks the protected feature lowest in the rule's output",
        goal="Hide the protected feature against this rule",
        must=["all honest reports; the aggregation rule, which it can evaluate (white-box)",
              "a family of 24 variants", "J * s2med / (4 d) for J in {1.5, 3}",
              "hiding applied to a distinct random honest bank's report", "claimed sigma2 0 or the honest median",
              "pushes the target lowest in the rule's output", "the attackers do not know the oracle",
              "against the deployed rule specifically"]),
    "whitebox_strong": dict(
        part="C", family="White-box",
        knowledge="As the white-box coalition, plus the form of trimming, clipping and duplicate tests",
        capability=r"Adds rank-aware reports to the 24 variants: lowers the protected feature and raises the 1--3 "
                   r"features ranked just below it, or places them just inside the reach of clipping and trimming "
                   r"rules; identical copies, jitter beyond the duplicate test, or applied to distinct honest "
                   r"reports; submits the most damaging variant against each rule. Variant with $\tilde n$: every "
                   r"attacker also claims $\tilde n$ examples",
        goal="Hide the protected feature against this rule, at its rank boundary",
        must=["Boundary-rank (BR)", "features ranked just below it", "just inside the reach of clipping rules",
              "beyond the duplicate-grouping threshold", "applied to a distinct random", "honest bank's profile"]),
}

ATTACK_ORDER = {
    "A": ["input_noise", "label_flip", "frozen", "zero", "scaling", "feature_hiding", "feature_promotion"],
    "B": ["mimic", "collusion", "collusion_jitter", "adaptive_psnr"],
    "C": ["alie", "minmax", "whitebox", "whitebox_strong"],
}

RUNNER_SRC = (CODE / "powersnr" / "runner.py").read_text(encoding="utf-8")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def _check_threat_code() -> None:
    """Facts quoted in the threat tables that are not in the docstrings themselves."""
    # defaults quoted in the table must be the code defaults
    assert inspect.signature(atk._zero).parameters["value"].default == 1e-8
    assert inspect.signature(atk._feature_promotion).parameters["factor"].default == 2.0
    # white-box family: 3 hiding strengths x 4 camouflages x 2 claimed sigma2 = 24 variants
    assert atk.WHITEBOX_HIDE == (0.3, 0.6, 1.0)
    assert atk.WHITEBOX_CAMO == ("none", "link1.5", "link3", "honest")
    assert atk.WHITEBOX_SIGMA2 == ("zero", "median")
    assert len(atk.WHITEBOX_HIDE) * len(atk.WHITEBOX_CAMO) * len(atk.WHITEBOX_SIGMA2) == 24
    # ... and the runner picks the variant per rule from the rule's OUTPUT (whitebox_score), never the oracle
    assert "for name, kw in cfg[\"aggregators\"].items():" in RUNNER_SRC
    assert "sc = atk.whitebox_score(r_v.phi, t_idx)" in RUNNER_SRC
    assert "oracle" not in inspect.getsource(atk.whitebox_score)
    # energy inflation: the cap variant solves for the boundary of the SNR-type weight E / sigma2 (not of any
    # deployed rule), and the runner never passes a sigma2 floor, so the attacker always declares sigma2 = 0
    assert "w = E / (s2 + eps)" in inspect.getsource(atk.adaptive_cap_energy)
    assert "sigma2_floor" not in RUNNER_SRC
    # ALIE gets the true number of attackers from the runner
    assert 'a["name"] == "alie"' in RUNNER_SRC and 'params.setdefault("n_attackers", m)' in RUNNER_SRC
    # the strong family is the paper family plus the rank-aware variants; only claim_n changes the attackers' n
    assert 'family="strong" if a["name"] == "whitebox_strong" else "paper"' in RUNNER_SRC
    assert 'n[att_pos] = float(a["params"]["claim_n"])' in RUNNER_SRC
    # n is never changed by a report attack (replace() keeps it) and the white-box variants do not set n
    assert "replace(stats, role=\"attacker\", mu=mu.copy(), sigma2=float(s2)" in inspect.getsource(atk.apply_report_attack)


def _threat_table(part: str, label: str, caption: str) -> str:
    rows, fam_prev = [], None
    for name in ATTACK_ORDER[part]:
        t = THREAT[name]
        if fam_prev is not None and t["family"] != fam_prev:
            rows.append(r"\midrule")
        fam_prev = t["family"]
        rows.append(" & ".join([ATTACK_NAMES[name][0] + rf" \newline {{\itshape {t['family']}}}",
                                t["knowledge"], t["capability"], t["goal"]]) + r" \\")
    return "\n".join([
        r"\begin{table*}[!htbp]",
        r"\centering\linespread{1}\selectfont",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.9}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.13\textwidth} >{\raggedright\arraybackslash}p{0.18\textwidth} "
        r">{\raggedright\arraybackslash}p{0.45\textwidth} >{\raggedright\arraybackslash}p{0.19\textwidth}@{}}",
        r"\toprule",
        r"Attack (type) & Knowledge & Capability & Goal \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])


def table_threat() -> None:
    docs = {**atk.THREAT_MODELS, "whitebox": atk.whitebox_variants.__doc__ or "",
            "whitebox_strong": atk.strong_variants.__doc__ or ""}
    order = atk.DATA_ATTACKS + atk.REPORT_ATTACKS + atk.WHITEBOX_ATTACKS
    if set(order) != set(THREAT) or set(docs) != set(THREAT) or set(ATTACK_NAMES) != set(THREAT):
        raise ValueError("THREAT mapping out of sync with attacks.py")
    if sorted(sum(ATTACK_ORDER.values(), [])) != sorted(THREAT):
        raise ValueError("every attack must appear in exactly one threat table")
    for name, t in THREAT.items():
        doc = _norm(docs[name])
        missing = [p for p in t["must"] if _norm(p) not in doc]
        if missing:
            raise ValueError(f"threat cell for {name} no longer matches its docstring: {missing}")
        if t["part"] not in ATTACK_ORDER or name not in ATTACK_ORDER[t["part"]]:
            raise ValueError(f"{name}: part mismatch")
    _check_threat_code()
    common = (r"$n_k$ is truthful except in the strong white-box variant that claims $\tilde n$. "
              r"$\bar\mu_{\mathcal H}$: mean honest report; "
              r"$\mathrm{hide}(\cdot)$: sets the protected feature (Table~\ref{tab:datasets}) to 0 and "
              r"redistributes its mass proportionally over the other features.")
    write("tab_threat.tex", _threat_table(
        "A", "tab:threat",
        r"Threat model, condensed from the implementation: data attacks (corrupt the training "
        r"data, then follow the protocol) and single-bank report attacks (replace the reported "
        r"$(\mu_k,\sigma^2_k)$). Coalition, published and white-box attacks: Tables~\ref{tab:threatcoalition} "
        r"and~\ref{tab:threatwhitebox} of the supplementary material. " + common))
    write("tab_threat_coalition.tex", _threat_table(
        "B", "tab:threatcoalition",
        r"Threat model, continued from Table~\ref{tab:threat} of the main text: stealthy, coalition and "
        r"energy-inflation attacks. They observe the honest "
        r"reports of the round but not the rule. Energy inflation is calibrated on SNR-type weights and is not "
        r"adapted to each rule. Notation as in Table~\ref{tab:threat}."))
    write("tab_threat_whitebox.tex", _threat_table(
        "C", "tab:threatwhitebox",
        r"Threat model, continued from Table~\ref{tab:threat} of the main text: the published rule-agnostic "
        r"ALIE \citep{baruch2019little} and min-max "
        r"\citep{shejwalkar2021manipulating} attacks, and the white-box coalitions, the only attacks that evaluate "
        r"the deployed rule. Notation as in Table~\ref{tab:threat}."))


# ----------------------------------------------------------------------------- tab:baselines
def run_rule_params(dirs: dict[str, Path]) -> dict[str, dict]:
    """Parameters of the benchmark rules as run; they must be identical in every evidence-run config except
    n_cap, which must equal n_explain. Every configured parameter must be accepted by the rule's function
    (aggregators.aggregate silently drops unknown keyword arguments)."""
    ref: dict[str, dict] = {}
    for exp in EXPERIMENTS:
        cfg = load_cfg(dirs[exp])
        for name, kw in cfg["aggregators"].items():
            kw = dict(kw or {})
            base = kw.pop("rule", name)
            fn = agg.AGGREGATORS[base]
            unknown = set(kw) - set(inspect.signature(fn).parameters)
            if unknown:
                raise ValueError(f"{exp}: {name} sets parameters that {base} ignores: {unknown}")
            if base in N_CAP_RULES and kw.get("n_cap") != cfg["shap"]["n_explain"]:
                raise ValueError(f"{exp}: {name} n_cap {kw.get('n_cap')} != n_explain {cfg['shap']['n_explain']}")
            kw.pop("n_cap", None)
            if name not in MAIN:
                continue
            kw["_rule"] = base
            if name in ref and ref[name] != kw:
                raise ValueError(f"{exp}: parameters of {name} differ across experiments: {kw} vs {ref[name]}")
            ref[name] = kw
    if set(ref) != set(MAIN):
        raise ValueError(f"benchmark rules missing from every config: {set(MAIN) - set(ref)}")
    return ref


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


def table_baselines(dirs: dict[str, Path]) -> dict:
    p = run_rule_params(dirs)
    get = lambda rule, key: p[rule].get(key, _default(agg.AGGREGATORS[p[rule]["_rule"]], key))  # noqa: E731
    beta_a, beta_b = get("trimmed_mean", "beta"), get("trimmed_mean_0.3", "beta")
    assert p["trimmed_mean_0.3"]["_rule"] == "trimmed_mean"
    assert all(p[r]["_rule"] == "psnr_v3" for r in ("psnr_v3_fit", "psnr_v3", "psnr_v3_unif"))
    eps, eps_dl = get("ivw", "eps"), get("dersimonian_laird", "eps")
    # baselines run with their code defaults: the cells quote these defaults, so check them
    for rule in ["fedavg", "median", "geomedian", "krum", "multikrum", "bulyan", "centered_clipping", "foolsgold",
                 "fltrust_like", "dersimonian_laird"]:
        if set(p[rule]) != {"_rule"}:
            raise ValueError(f"{rule} was run with explicit parameters {p[rule]}; update the table cell")
    assert _default(agg.krum, "f") is None and agg._default_f(10, None) == 3
    assert "(K - 3) // 2" in inspect.getsource(agg._default_f)
    assert "m = K - f" in inspect.getsource(agg.multikrum)
    assert _default(agg.bulyan, "f") is None and "f = (K - 3) // 4 if f is None" in inspect.getsource(agg.bulyan)
    assert "theta = K - 2f" in agg.bulyan.__doc__ and "beta = theta - 2f" in agg.bulyan.__doc__
    assert _default(agg.centered_clipping, "tau") is None and _default(agg.centered_clipping, "iters") == 3
    assert "started at the" in agg.centered_clipping.__doc__ and "coordinate-wise median" in agg.centered_clipping.__doc__
    assert "median distance of the reports to the start point" in _norm(agg.centered_clipping.__doc__)
    assert _default(agg.foolsgold, "kappa") == 1.0 and "pardoning" in agg.foolsgold.__doc__
    assert "ReLU(cos(mu_k, root_mu))" in agg.fltrust_like.__doc__
    assert "Weiszfeld" in agg.geomedian.__doc__ and "started at the mean" in agg.geomedian.__doc__
    assert "trimming floor(beta*K) values from each tail" in agg.trimmed_mean.__doc__
    assert "v_k = sigma2_k / d" in agg.dersimonian_laird.__doc__
    # Power-SNR (earlier design)
    ps = p["psnr_screen"]
    ps_mass_z = get("psnr_screen", "mass_z")
    assert "vd = sigma2[idx] / mass[idx] ** 2" in inspect.getsource(agg.psnr_screen)
    assert ps["floor"] == "n" and ps["stage2"] == "geomedian" and ps["dedup"] is True and ps["shrink"] == 0.5
    # FedXRA family: one design, three weightings
    fx, fxn, fxu = p["psnr_v3_fit"], p["psnr_v3"], p["psnr_v3_unif"]
    for q, w in ((fx, "fit_ts"), (fxn, "n"), (fxu, "unif")):
        assert q["weight"] == w and q["combine"] == "trim" and q["dedup"] is True and q["profile_screen"] is False
        assert q["mass_z"] == fx["mass_z"] and q["cap"] == fx["cap"] and q["trim_beta"] == fx["trim_beta"]
    assert get("psnr_v3_fit", "screens") is True
    v3src = inspect.getsource(agg.psnr_v3)
    assert "_cw_trim(Pk, wk, trim_beta)" in v3src and "prof * float(np.median(mass[idx]))" in v3src
    assert "wk = _fit_ts_weights(Pk, nk, trim_beta)" in v3src
    fitsrc = inspect.getsource(agg._fit_ts_weights)
    assert "stats.theilslopes(r2, 1.0 / n)" in fitsrc and "0.05 * float(np.median(r2))" in fitsrc
    assert "len(n) < 4" in fitsrc
    huber_steps = None

    def sci(x):
        m, e = f"{x:.0e}".split("e")
        return rf"10^{{{int(e)}}}" if m == "1" else rf"{m}\times10^{{{int(e)}}}"

    rows = [
        (r"\FedXRA",
         rf"Mass screen ($z_m={fmt(fx['mass_z'])}$); weights $1/(\hat\tau^2+\hat a/\min(n_k,\bar n))$ with "
         r"$(\hat\tau^2,\hat a)$ fitted by Theil--Sen on the squared distances of the profiles to their trimmed "
         rf"mean; duplicate grouping ($\rho={fmt(fx['dup_ratio'])}$); cap $c={fmt(fx['cap'])}$ times the median "
         rf"weight; coordinate-wise weighted trimmed mean of the profiles ($\beta={fmt(fx['trim_beta'])}$), "
         r"rescaled by the median survivor mass", r"This paper (Section~\ref{sec:method})"),
        (r"FedXRA-n",
         rf"As \FedXRA\ with weights $\min(n_k,\bar n)$ and $\rho={fmt(fxn['dup_ratio'])}$",
         r"This paper (Section~\ref{sec:method})"),
        (r"FedXRA-U",
         rf"As \FedXRA\ with equal weights and $\rho={fmt(fxu['dup_ratio'])}$", r"This paper (Section~\ref{sec:method})"),
        (r"\PSNR\ (earlier design)",
         rf"Mass screen ($z_m={fmt(ps_mass_z)}$) and profile screen ($z={fmt(ps['z'])}$); weights "
         rf"$\|p_k\|_2^2/\max(v_k,\kappa/\tilde n_k)$ shrunk to uniform ($\lambda={fmt(ps['shrink'])}$), cap "
         rf"$c={fmt(ps['cap'])}$, duplicate grouping from the reported, floored $v_k$ ($\rho={fmt(ps['dup_ratio'])}$); "
         r"weighted geometric median", "Earlier design of this study"),
        ("FedAvg", r"$\sum_k \frac{n_k}{\sum_l n_l}\,\mu_k$ (sample-size weighted mean)", r"\citet{mcmahan2017fedavg}"),
        ("Median", r"Coordinate-wise median", r"\citet{yin2018byzantine}"),
        (rf"Trimmed mean ($\beta={fmt(beta_a)}$)", r"Coordinate-wise mean after removing the $\lfloor\beta K\rfloor$ "
                                                   r"largest and smallest values", r"\citet{yin2018byzantine}"),
        (rf"Trimmed mean ($\beta={fmt(beta_b)}$)", r"Same, with the larger trimming level", r"\citet{yin2018byzantine}"),
        ("Geometric median (RFA)", r"$\arg\min_z\sum_k\|\mu_k-z\|_2$, Weiszfeld iterations started at the mean",
         r"\citet{pillutla2022rfa}"),
        ("Krum", r"The $\mu_k$ with the smallest sum of squared distances to its $K-f-2$ nearest reports, "
                 r"$f=\max(0,\lfloor (K-3)/2\rfloor)$", r"\citet{blanchard2017krum}"),
        ("Multi-Krum", r"Mean of the $K-f$ reports with the smallest Krum scores (same $f$)",
         r"\citet{blanchard2017krum}"),
        ("Bulyan", r"Selects $K-2f$ reports by repeated Krum, then averages per coordinate the $K-4f$ selected values "
                   r"closest to their median; $f=\max(0,\lfloor (K-3)/4\rfloor)$", r"\citet{elmhamdi2018bulyan}"),
        ("Centered clipping", r"$v\leftarrow v+\frac1K\sum_k \mathrm{clip}_\tau(\mu_k-v)$, 3 steps from the "
                              r"coordinate-wise median; $\tau$: median distance of the reports to the start",
         r"\citet{karimireddy2021history}"),
        ("FoolsGold", r"One-round FoolsGold on cosine similarities with pardoning and the logit with $\kappa=1$: "
                      r"mutually similar reports get low weight", r"\citet{fung2020foolsgold}"),
        ("FLTrust-like", r"Trust $t_k=\max(0,\cos(\mu_k,\mu_{\mathrm{root}}))$; each $\mu_k$ rescaled to "
                         r"$\|\mu_{\mathrm{root}}\|_2$; $t$-weighted mean. $\mu_{\mathrm{root}}$: report of a model "
                         r"trained by the server on its root set", r"\citet{cao2021fltrust}"),
        ("IVW", rf"$w_k\propto 1/(\sigma^2_k+\epsilon)$, $\epsilon={sci(eps)}$", r"\citet{borenstein2010meta}"),
        ("DerSimonian--Laird (DL)", rf"$w_k\propto 1/(\sigma^2_k/d+\hat\tau^2+\epsilon)$, $\hat\tau^2$: "
                                    rf"method-of-moments between-bank variance averaged over features, "
                                    rf"$\epsilon={sci(eps_dl)}$", r"\citet{dersimonian1986meta}"),
    ]
    assert len(rows) == len(MAIN) == 17
    def ref(c: str) -> str:  # reference appended to the definition
        if c.startswith(r"\citet"):
            return " " + c.replace(r"\citet", r"\citep")
        return {"This paper (Section~\\ref{sec:method})": r" (Section~\ref{sec:method})",
                "Earlier design of this study": ""}[c]

    body = [rf"{a} & {b}{ref(c)} \\" for a, b, c in rows]
    body.insert(4, r"\midrule")
    tex = "\n".join([
        r"\begin{table*}[!htbp]",
        r"\centering\linespread{1}\selectfont",
        r"\caption{The " + str(len(MAIN)) + r" aggregation rules, with the parameters used in every experiment (read from the run "
        r"configurations; $\bar n$ equals the number of explained rows). All rules receive the same reports "
        r"$(\mu_k,\sigma^2_k,n_k)$. No baseline parameter was tuned: the baselines use the code defaults shown, "
        r"and Trimmed mean is run at two trimming levels.}",
        r"\label{tab:baselines}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.9}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.2\textwidth} "
        r">{\raggedright\arraybackslash}p{0.78\textwidth}@{}}",
        r"\toprule",
        r"Rule & Definition, parameters and reference \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    write("tab_baselines.tex", tex)
    return {"beta_a": beta_a, "beta_b": beta_b, "fx": fx, "fxn": fxn, "ps": ps}


# ----------------------------------------------------------------------------- macros
# Notebook D re-ran E11 with the corrected FedXRA size cap (Section 6). Its Kaggle image had newer Python and SHAP
# patch versions; only these two may differ (bank reports byte-identical, other rules identical row for row).
RERUN_NOTEBOOK, RERUN_MAY_DIFFER = "D", {"python", "shap"}
VERSION_KEYS = ["python", "numpy", "pandas", "shap", "xgboost", "sklearn"]


def versions(dirs: dict[str, Path]) -> dict[str, str]:
    def meta(d: Path) -> dict:
        return json.loads((d / "run_meta.json").read_text(encoding="utf-8"))

    def unique(metas: list[dict], k: str) -> str:
        vals = {m[k] for m in metas}
        if len(vals) != 1:
            raise ValueError(f"inconsistent {k} versions across runs: {vals}")
        return vals.pop()

    main = [meta(d) for d in dirs.values() if d.parents[1].name != RERUN_NOTEBOOK]
    rerun = [meta(d) for d in dirs.values() if d.parents[1].name == RERUN_NOTEBOOK]
    out = {k: unique(main, k) for k in VERSION_KEYS}
    re_v = {k: unique(rerun, k) for k in VERSION_KEYS} if rerun else dict(out)
    bad = {k for k in VERSION_KEYS if re_v[k] != out[k]} - RERUN_MAY_DIFFER
    if bad:
        raise ValueError(f"re-run notebook {RERUN_NOTEBOOK} differs in {bad}")
    out.update({f"rerun_{k}": re_v[k] for k in VERSION_KEYS})
    jobs = {m["cmd"]["jobs"] for m in main + rerun}
    if len(jobs) != 1:
        raise ValueError(f"inconsistent job counts {jobs}")
    out["jobs"] = str(jobs.pop())
    # LightGBM is not in run_meta.json; the Kaggle notebooks print all versions at start-up.
    logs = [f for f in glob.glob(str(RUN / "*" / "*.log")) if not f.endswith("download.log")]
    notebooks = sorted(Path(f).parent.name for f in logs)
    if notebooks not in (["A", "B", "C"], ["A", "B", "C", RERUN_NOTEBOOK]):
        raise ValueError(f"expected one log per {RUN_NAME} notebook (A, B, C[, {RERUN_NOTEBOOK}]), found {logs}")
    lgb = set()
    for log in logs:
        lgb |= set(re.findall(r"'lightgbm': '([0-9][0-9.]*)'", Path(log).read_text(encoding="utf-8", errors="replace")))
    if len(lgb) != 1:
        raise ValueError(f"could not determine a unique LightGBM version from the logs: {lgb}")
    out["lightgbm"] = lgb.pop()
    for log in logs:  # the logged versions must agree with run_meta
        txt = Path(log).read_text(encoding="utf-8", errors="replace")
        pre = "rerun_" if Path(log).parent.name == RERUN_NOTEBOOK else ""
        for k in ["numpy", "pandas", "sklearn", "shap", "xgboost"]:
            found = set(re.findall(rf"'{k}': '([0-9][0-9.]*)'", txt))
            if found and found != {out[pre + k]}:
                raise ValueError(f"{log}: {k} {found} != run_meta {out[pre + k]}")
    return out


def dev_seed_range(run: str) -> tuple[int, int]:
    """Seed range of the development run (configs of runs 2 and 3; read only for the protocol text)."""
    lo, hi = None, None
    for f in glob.glob(str(DEV_RUNS[run] / "**" / "results" / "*_E*" / "config.yaml"), recursive=True):
        c = yaml.safe_load(Path(f).read_text(encoding="utf-8"))
        s0 = int(c.get("seed_offset", 0))
        s1 = s0 + int(c["seeds"]) - 1
        lo, hi = (s0, s1) if lo is None else (min(lo, s0), max(hi, s1))
    if lo is None:
        raise FileNotFoundError(f"no {run} configs found")
    return lo, hi


def _runner_const(pattern: str) -> str:
    hits = re.findall(pattern, RUNNER_SRC)
    if len(hits) != 1:
        raise ValueError(f"runner.py: expected one match for {pattern!r}, found {hits}")
    return hits[0]


def _oracle_factor() -> int:
    """The oracle is explained on min(factor * n_explain, |eval|) evaluation rows."""
    return int(_runner_const(r'Xe\.sample\(n=min\(shap_cfg\.get\("n_explain", 1000\) \* (\d+), len\(Xe\)\)'))


def write_macros(dirs: dict[str, Path], exp_info: dict, base_info: dict) -> dict[str, str]:
    v = versions(dirs)
    cfgs = {e: load_cfg(dirs[e]) for e in EXPERIMENTS}

    def common(key):
        vals = {json.dumps(c[key]) for c in cfgs.values()}
        if len(vals) != 1:
            raise ValueError(f"{key} differs across experiments: {vals}")
        return json.loads(vals.pop())

    tot = exp_info["totals"]
    seeds_off = common("seed_offset")
    final_max = max(s1 for _, s1, _ in exp_info["seeds"].values())
    d2, d3 = dev_seed_range("run2"), dev_seed_range("run3")
    if not (d2[1] < seeds_off and d3[1] < seeds_off):
        raise ValueError(f"development seeds {d2}, {d3} overlap the final seeds starting at {seeds_off}")
    # runs 4-6 (seeds 2000+, 3000+, 4000+) evaluated earlier designs; the final seeds must not overlap them either
    for old in ("run4", "run5", "run6"):
        cf = [yaml.safe_load(Path(f).read_text(encoding="utf-8"))
              for f in glob.glob(str(CODE / "kaggle_runs" / old / "*" / "results" / "*_E*" / "config.yaml"))]
        hi = max(int(c["seed_offset"]) + int(c["seeds"]) - 1 for c in cf) if cf else -1
        if hi >= seeds_off:
            raise ValueError(f"{old} seeds overlap the final seeds")
    max_seeds = max(c["seeds"] for c in cfgs.values())
    min_seeds = min(c["seeds"] for c in cfgs.values())
    n_attacks = len(set(atk.DATA_ATTACKS + atk.REPORT_ATTACKS + atk.WHITEBOX_ATTACKS))
    used_attacks = {a["name"] for c in cfgs.values() for a in c.get("attacks", [])}
    if len(used_attacks) != n_attacks:
        raise ValueError(f"attacks implemented but not run: {set(THREAT) - used_attacks}")
    fx, fxn, ps = base_info["fx"], base_info["fxn"], base_info["ps"]
    macros = {
        "ProtoRuntimeHours": f"{tot['runtime_s'] / 3600:.1f}",
        "ProtoTotalRows": big(tot["rows"]),
        "ProtoTotalCells": big(tot["cells"]),
        "ProtoJobs": v["jobs"],
        "ProtoPython": v["python"],
        "ProtoNumpy": v["numpy"],
        "ProtoPandas": v["pandas"],
        "ProtoSklearn": v["sklearn"],
        "ProtoXGBoost": v["xgboost"],
        "ProtoLightGBM": v["lightgbm"],
        "ProtoSHAP": v["shap"],
        "ProtoReRunPython": v["rerun_python"],
        "ProtoReRunSHAP": v["rerun_shap"],
        "ProtoSeedOffset": str(seeds_off),
        "ProtoSeedMax": str(final_max),
        "ProtoMinSeeds": str(min_seeds),
        "ProtoMaxSeeds": str(max_seeds),
        "ProtoDevSeedsA": f"{d2[0]}--{d2[1]}",
        "ProtoDevSeedsB": f"{d3[0]}--{d3[1]}",
        "ProtoMaxRows": big(common("max_rows")),
        "ProtoEvalPct": f"{100 * common('eval_frac'):g}",
        "ProtoRootPct": f"{100 * common('root_frac'):g}",
        "ProtoRootPctAll": f"{100 * common('eval_frac') * common('root_frac'):g}",
        "ProtoClientSize": big(exp_info["modal_cs"]),
        "ProtoNExplain": big(exp_info["modal_shap"][0]),
        "ProtoNBackground": str(exp_info["modal_shap"][1]),
        "ProtoRepeats": str(exp_info["modal_shap"][2]),
        "ProtoNExperiments": str(len(EXPERIMENTS)),
        "ProtoNRules": str(len(MAIN)),
        "ProtoNAttacks": str(n_attacks),
        "ProtoWhiteboxVariants": str(len(atk.WHITEBOX_HIDE) * len(atk.WHITEBOX_CAMO) * len(atk.WHITEBOX_SIGMA2)),
        "ProtoTrimBetaA": fmt(base_info["beta_a"]),
        "ProtoTrimBetaB": fmt(base_info["beta_b"]),
        "ProtoFxCap": fmt(fx["cap"]),
        "ProtoFxDupRatio": fmt(fx["dup_ratio"]),
        "ProtoFxnDupRatio": fmt(fxn["dup_ratio"]),
        "ProtoFxMassZ": fmt(fx["mass_z"]),
        "ProtoFxTrim": fmt(fx["trim_beta"]),
        "ProtoPsCap": fmt(ps["cap"]),
        # constants of the code, read from the sources
        "ProtoKernelMax": str(KERNEL_MAX_EXPLAIN),
        "ProtoOracleRows": big(_oracle_factor() * int(exp_info["modal_shap"][0])),
        "ProtoDirFloor": _runner_const(r'pkw\.setdefault\("min_per_class", max\(2, min\((\d+), smallest // \(2 \* K\)\)\)\)'),
        "ProtoMinBank": _runner_const(r'pkw\.setdefault\("min_size", (\d+)\)'),
    }
    lines = [f"%% Generated by tools/make_protocol_tables.py from the {RUN_NAME} configs, run_meta.json and logs.",
             "%% Do not edit by hand."]
    lines += [rf"\providecommand{{\{k}}}{{}}\renewcommand{{\{k}}}{{{val}}}" for k, val in macros.items()]
    write("protocol_numbers.tex", "\n".join(lines) + "\n")
    return macros


def main() -> None:
    dirs = run_dirs()
    info = table_experiments(dirs)
    table_threat()
    base = table_baselines(dirs)
    macros = write_macros(dirs, info, base)
    for k, val in macros.items():
        print(f"  \\{k} = {val}")


if __name__ == "__main__":
    main()
