"""Local model factory. All estimators are sklearn-compatible, single-threaded and seeded.

logreg and mlp are ``sklearn.pipeline.Pipeline(StandardScaler, clf)``: scaling is part of the
model (fitted on the client's training rows only). ``explain.py`` unwraps the logreg pipeline
for ``shap.LinearExplainer`` (linear SHAP values are invariant to the per-feature affine
standardisation, so the attributions refer to the original features); the mlp pipeline is
explained as a black box by KernelExplainer.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ["xgboost", "lightgbm", "rf", "logreg", "mlp"]
TREE_MODELS = {"xgboost", "lightgbm", "rf"}


def make_model(name: str, seed: int, **params):
    if name == "xgboost":
        from xgboost import XGBClassifier

        cfg = dict(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss",
                   n_jobs=1, random_state=seed, tree_method="hist")
        cfg.update(params)
        return XGBClassifier(**cfg)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        cfg = dict(n_estimators=100, learning_rate=0.1, num_leaves=31, verbose=-1, n_jobs=1,
                   random_state=seed)
        cfg.update(params)
        return LGBMClassifier(**cfg)
    if name == "rf":
        cfg = dict(n_estimators=200, max_depth=8, n_jobs=1, random_state=seed)
        cfg.update(params)
        return RandomForestClassifier(**cfg)
    if name == "logreg":
        cfg = dict(max_iter=1000, random_state=seed)
        cfg.update(params)
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**cfg))])
    if name == "mlp":
        cfg = dict(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=seed)
        cfg.update(params)
        return Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(**cfg))])
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def model_kind(model) -> str:
    """Return one of MODELS for a fitted/unfitted estimator built by make_model."""
    if isinstance(model, Pipeline):
        last = model.steps[-1][1]
        if isinstance(last, LogisticRegression):
            return "logreg"
        if isinstance(last, MLPClassifier):
            return "mlp"
        raise TypeError(f"unsupported pipeline {model}")
    cls = type(model).__name__
    if cls == "XGBClassifier":
        return "xgboost"
    if cls == "LGBMClassifier":
        return "lightgbm"
    if isinstance(model, RandomForestClassifier):
        return "rf"
    raise TypeError(f"unsupported model type {cls}")
