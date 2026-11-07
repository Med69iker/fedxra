"""Local SHAP explanations and the client report (mu, sigma2).

Explainer per model (one per call, no fallback; any SHAP error propagates):

=========  ===================================================================  =============
model      explainer                                                             output space
=========  ===================================================================  =============
xgboost    TreeExplainer(model, data=bg, feature_perturbation="interventional",   log-odds
           model_output="raw")
lightgbm   same                                                                  log-odds
rf         same (sklearn RF "raw" output = averaged class-1 probability)         probability
logreg     LinearExplainer(LogisticRegression, masker=scaled bg) on the           log-odds
           standardised inputs (linear SHAP is invariant to the affine scaling,
           so values are attributions of the original features)
mlp        KernelExplainer(p1(x), shap.kmeans(bg, 10), link="logit"),            log-odds
           l1_reg=False, n_explain capped at 200
=========  ===================================================================  =============

All explanations are for class 1 (default). RF is the only model explained in probability
space (TreeExplainer cannot produce interventional log-odds for averaged-probability forests);
since aggregation always combines clients of the same model type this does not mix scales.

KernelExplainer draws from NumPy's *global* RNG internally. To stay deterministic given the
explicit ``rng`` we run it inside :func:`_scoped_global_seed`, which saves the global state,
sets it from a seed drawn from ``rng`` and restores it afterwards (the only place the global
RNG is touched).
"""
from __future__ import annotations

import contextlib
import time

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

from powersnr.data import LocalPreprocessor
from powersnr.models import make_model, model_kind
from powersnr.types import ClientStats

KERNEL_MAX_EXPLAIN = 200
KERNEL_BG_CLUSTERS = 10


@contextlib.contextmanager
def _scoped_global_seed(seed: int):
    state = np.random.get_state()
    np.random.set_state(np.random.RandomState(seed).get_state())
    try:
        yield
    finally:
        np.random.set_state(state)


_UBJ_INT = {b"i": (1, ">b"), b"U": (1, ">B"), b"I": (2, ">h"), b"l": (4, ">i"), b"L": (8, ">q")}


def _patch_base_score(raw: bytes) -> tuple[bytes, bool]:
    """Rewrite the UBJSON string value of ``base_score`` from ``'[x]'`` to ``'x'``.
    Returns (bytes, patched?). Raises on multi-output or unexpected layouts."""
    import struct

    key = b"base_scoreS"
    i = raw.find(key)
    if i < 0:
        raise ValueError("base_score not found in xgboost UBJSON dump")
    j = i + len(key)
    marker = raw[j:j + 1]
    if marker not in _UBJ_INT:
        raise ValueError(f"unexpected UBJSON length marker {marker!r}")
    size, fmt = _UBJ_INT[marker]
    length = struct.unpack(fmt, raw[j + 1:j + 1 + size])[0]
    start = j + 1 + size
    val = raw[start:start + length]
    if not val.startswith(b"["):
        return raw, False
    inner = val[1:-1]
    if b"," in inner or not val.endswith(b"]"):
        raise ValueError(f"multi-output base_score {val!r} not supported")
    new = marker + struct.pack(fmt, len(inner)) + inner
    return raw[:j] + new + raw[start + length:], True


def _xgb_for_shap(model):
    """Compatibility shim (shap 0.49.1 x xgboost >= 3.1).

    xgboost >= 3.1 serialises ``base_score`` as a vector string (e.g. ``'[3.7E-1]'``) and
    shap 0.49.1's ``XGBTreeModelLoader`` does ``float(base_score)`` -> ValueError. We give
    TreeExplainer a Booster subclass whose UBJSON dump has the scalar form for a single-output
    model. Tree structure, leaf values and all other fields are untouched, and the result is
    verified by shap's own additivity check. Raises if the dump has an unexpected format.
    """
    import xgboost as xgb

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    raw = bytes(booster.save_raw(raw_format="ubj"))
    if not _patch_base_score(raw)[1]:
        return booster  # already scalar (older xgboost): nothing to do

    class _ScalarBaseScoreBooster(xgb.Booster):
        def save_raw(self, raw_format: str = "ubj"):
            out = bytes(super().save_raw(raw_format=raw_format))
            if raw_format != "ubj":
                return bytearray(out)
            return bytearray(_patch_base_score(out)[0])

    compat = _ScalarBaseScoreBooster(model_file=bytearray(raw))
    compat.feature_names = booster.feature_names
    compat.feature_types = booster.feature_types
    return compat


def _class1(values, n: int, d: int) -> np.ndarray:
    """Normalise SHAP output to an (n, d) matrix for class 1."""
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError(f"expected 2 class outputs, got {len(values)}")
        values = values[1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        if values.shape[2] != 2:
            raise ValueError(f"unexpected SHAP shape {values.shape}")
        values = values[:, :, 1]
    if values.shape != (n, d):
        raise ValueError(f"unexpected SHAP shape {values.shape}, expected {(n, d)}")
    return values


def output_space(model) -> str:
    return "probability" if model_kind(model) == "rf" else "log-odds"


# LightGBM: shap re-reads split thresholds at lower precision than LightGBM uses, so a row whose
# feature value lies exactly on a threshold can be routed differently (seen once in ~600 rows).
# Such rows are dropped after an explicit additivity check; more than this share raises.
LGBM_MAX_INCONSISTENT = 0.01


def _lightgbm_consistent_rows(expl, model, X_explain: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose SHAP values add up to the model's raw output."""
    base = expl.expected_value
    base = float(np.ravel(base)[-1])
    raw = np.asarray(model.predict(X_explain, raw_score=True), dtype=float)
    err = np.abs(values.sum(axis=1) + base - raw)
    ok = err <= 1e-3 * (1.0 + np.abs(raw))
    if (~ok).mean() > LGBM_MAX_INCONSISTENT:
        raise ValueError(f"LightGBM SHAP additivity fails on {(~ok).sum()}/{len(ok)} rows (max error {err.max():.3g})")
    return ok


def shap_matrix(model, X_explain: pd.DataFrame, X_background: pd.DataFrame,
                seed: int = 0, report: dict | None = None) -> np.ndarray:
    """(n_explain, d) class-1 SHAP values. ``seed`` only matters for KernelExplainer (mlp).
    For LightGBM, rows failing the additivity check are dropped (count in report["dropped_rows"])."""
    kind = model_kind(model)
    n, d = X_explain.shape
    if kind == "lightgbm":
        expl = shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional",
                                  model_output="raw")
        vals = _class1(expl.shap_values(X_explain, check_additivity=False), n, d)
        ok = _lightgbm_consistent_rows(expl, model, X_explain, vals)
        if report is not None:
            report["dropped_rows"] = int((~ok).sum())
        return vals[ok]
    if kind in ("xgboost", "rf"):
        tree_model = _xgb_for_shap(model) if kind == "xgboost" else model
        expl = shap.TreeExplainer(tree_model, data=X_background, feature_perturbation="interventional",
                                  model_output="raw")
        vals = expl.shap_values(X_explain)
    elif kind == "logreg":
        scaler, clf = model.steps[0][1], model.steps[-1][1]
        Xb = scaler.transform(X_background)
        Xe = scaler.transform(X_explain)
        expl = shap.LinearExplainer(clf, Xb)
        vals = expl.shap_values(Xe)
    elif kind == "mlp":
        cols = list(X_explain.columns)

        def f(Z):
            p = model.predict_proba(pd.DataFrame(Z, columns=cols))[:, 1]
            return np.clip(p, 1e-7, 1 - 1e-7)

        k = min(KERNEL_BG_CLUSTERS, len(X_background))
        bg = shap.kmeans(X_background.to_numpy(dtype=float), k)
        expl = shap.KernelExplainer(f, bg, link="logit")
        with _scoped_global_seed(seed):
            vals = expl.shap_values(X_explain.to_numpy(dtype=float), l1_reg=False, silent=True)
    else:  # pragma: no cover - model_kind already raises
        raise TypeError(kind)
    return _class1(vals, n, d)


def global_importance(model, X_explain: pd.DataFrame, X_background: pd.DataFrame,
                      signed: bool = False) -> np.ndarray:
    """(d,) mean |SHAP| (or mean signed SHAP) of class 1 over X_explain."""
    phi = shap_matrix(model, X_explain, X_background, seed=0)
    return phi.mean(axis=0) if signed else np.abs(phi).mean(axis=0)


def split_half_sigma2(phi: np.ndarray, n_repeats: int, rng: np.random.Generator) -> float:
    """mean_r ||mean|phi_A| - mean|phi_B|||^2 over random half splits of the rows of phi."""
    m = phi.shape[0]
    if m < 2:
        raise ValueError("need >= 2 explained rows for a split-half estimate")
    a = np.abs(phi)
    out = []
    for _ in range(n_repeats):
        perm = rng.permutation(m)
        h = m // 2
        mu_a = a[perm[:h]].mean(axis=0)
        mu_b = a[perm[h:2 * h]].mean(axis=0)
        out.append(float(np.sum((mu_a - mu_b) ** 2)))
    return float(np.mean(out))


def client_stats(model, X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, client_id: int,
                 role: str, n_repeats: int = 5, n_explain: int = 1000, n_background: int = 100,
                 test_frac: float = 0.2) -> ClientStats:
    """Client report computed on its held-out rows (X, y), already preprocessed.

    * background: ``n_background`` rows drawn (rng) from X;
    * explained rows: up to ``n_explain`` rows drawn (rng) from X (<= 200 for mlp);
    * the SHAP matrix is computed ONCE; sigma2 re-uses it over ``n_repeats`` random half-splits;
    * ``n`` = number of held-out rows (proportional to the client's size because test_frac is
      shared by all clients); ``test_frac`` is accepted for interface compatibility only.
    """
    t0 = time.perf_counter()
    m_total = len(X)
    if m_total < 2:
        raise ValueError(f"client {client_id}: need >= 2 held-out rows, got {m_total}")
    kind = model_kind(model)
    cap = min(n_explain, KERNEL_MAX_EXPLAIN) if kind == "mlp" else n_explain
    bg_idx = rng.choice(m_total, size=min(n_background, m_total), replace=False)
    ex_idx = rng.choice(m_total, size=min(cap, m_total), replace=False)
    X_bg = X.iloc[bg_idx]
    X_ex = X.iloc[ex_idx]
    seed = int(rng.integers(0, 2**31 - 1))
    rep: dict = {}
    phi = shap_matrix(model, X_ex, X_bg, seed=seed, report=rep)
    mu = np.abs(phi).mean(axis=0)
    mu_signed = phi.mean(axis=0)
    sigma2 = split_half_sigma2(phi, n_repeats, rng)
    y = np.asarray(y)
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    else:
        auc = float("nan")
    return ClientStats(client_id=client_id, role=role, mu=mu, sigma2=sigma2, n=int(m_total),
                       mu_signed=mu_signed, local_auc=auc, runtime_s=time.perf_counter() - t0,
                       extra={"n_explain": len(ex_idx), "n_background": len(bg_idx),
                              "explainer": kind, "output_space": output_space(model),
                              "dropped_rows": rep.get("dropped_rows", 0)})


def _stratified_holdout(y: np.ndarray, test_frac: float, rng: np.random.Generator) -> np.ndarray:
    test = []
    for c in np.unique(y):
        idx_c = rng.permutation(np.flatnonzero(y == c))
        k = int(round(test_frac * len(idx_c)))
        if len(idx_c) >= 2:
            k = min(max(k, 1), len(idx_c) - 1)  # keep >= 1 of each class on both sides
        test.append(idx_c[:k])
    return np.sort(np.concatenate(test)).astype(int)


def fit_client(model_name: str, X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, seed: int,
               test_frac: float = 0.2):
    """Stratified train/test split (rng), median imputation fitted on train only, model fit.

    Returns (model, preprocessor, X_test_preprocessed, y_test). With ``test_frac == 0`` all
    rows are used for training and X_test / y_test are empty (same columns / dtype).
    """
    y = np.asarray(y).astype(int)
    if not 0.0 <= test_frac < 1.0:
        raise ValueError("test_frac must be in [0, 1)")
    if test_frac > 0:
        te = _stratified_holdout(y, test_frac, rng)
        tr = np.setdiff1d(np.arange(len(y)), te)
    else:
        te = np.array([], dtype=int)
        tr = np.arange(len(y))
    if len(np.unique(y[tr])) < 2:
        raise ValueError(f"training split has a single class (n={len(tr)}); cannot fit {model_name}")
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    prep = LocalPreprocessor().fit(X_tr)
    model = make_model(model_name, seed)
    model.fit(prep.transform(X_tr), y[tr])
    return model, prep, prep.transform(X_te).reset_index(drop=True), y[te]
