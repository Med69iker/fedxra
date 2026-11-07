"""Shared data structures. Every module imports from here; do not redefine them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame                 # numeric features only (one-hot done), NaNs allowed (imputed per client)
    y: np.ndarray                   # int {0,1}, shape (n,)
    feature_names: list[str]
    # Raw attributes used for minority splits / fairness attacks, aligned with X.index.
    # e.g. {"age": Series}. Values may also be present as columns of X.
    groups: dict[str, pd.Series] = field(default_factory=dict)
    # Name of the feature (column of X) that is "protected" for fairness attacks, e.g. "age".
    protected_feature: str | None = None


@dataclass
class ClientStats:
    """What a client reports to the server (plus bookkeeping not sent)."""
    client_id: int
    role: str                       # "honest" | "minority" | "attacker"
    mu: np.ndarray                  # (d,) mean |SHAP| importance vector (reported)
    sigma2: float                   # split-half instability: mean_r ||mu_A - mu_B||^2 (reported)
    n: int                          # local sample size (reported)
    mu_signed: np.ndarray | None = None   # (d,) mean signed SHAP (secondary variant)
    local_auc: float = float("nan")       # bookkeeping only
    runtime_s: float = float("nan")       # bookkeeping only
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggResult:
    phi: np.ndarray                 # (d,) global importance vector
    weights: np.ndarray             # (K,) normalized client weights, sum to 1; NaN where undefined (e.g. median)
    info: dict[str, Any] = field(default_factory=dict)
