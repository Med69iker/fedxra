"""Federated partitioning of a Dataset into K disjoint clients.

Every scheme returns ``(parts, tags)``: K disjoint positional index arrays into ``ds`` and K
tags ("standard" / "minority"). ``client_size`` caps the number of rows per client.

Schemes
-------
iid        uniform random split.
dirichlet  label skew: for each class c, every client first receives ``min_per_class`` rows
           (default 2, so every client can train and compute an AUC); the remaining class rows
           are split with proportions p_c ~ Dir(alpha * 1_K) (Hsu et al. 2019). Clients larger
           than ``client_size`` are randomly truncated (preserves their label mix in expectation).
feature    covariate / demographic skew: the LAST ``n_minority`` clients receive rows with
           ``group > threshold`` (group looked up in ``ds.groups`` then ``ds.X``), split evenly;
           the other clients receive the remaining rows IID (rows with missing group are
           treated as non-minority). Tags: "minority" for those clients.
quantity   size skew: s_k ~ LogNormal(0, sigma). Without ``client_size`` the n rows are split
           proportionally to s_k; with ``client_size`` the largest client gets ``client_size``
           rows and client k gets round(client_size * s_k / max s) (>= ``min_size``, default 20),
           so no client exceeds ``client_size``.
"""
from __future__ import annotations

import numpy as np

from powersnr.types import Dataset

SCHEMES = ["iid", "dirichlet", "feature", "quantity"]


def _stratified_cap(p: np.ndarray, y: np.ndarray, size: int, floor: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Truncate a client to ``size`` rows keeping its label mix and >= ``floor`` rows per class."""
    if len(p) <= size:
        return p
    classes = np.unique(y[p])
    n_c = np.array([np.sum(y[p] == c) for c in classes])
    keep = np.maximum(np.floor(n_c / len(p) * size).astype(int), np.minimum(floor, n_c))
    while keep.sum() > size:            # remove surplus from the largest class
        keep[np.argmax(keep)] -= 1
    out = [rng.choice(p[y[p] == c], size=k, replace=False) for c, k in zip(classes, keep)]
    return np.concatenate(out)


def _iid(idx: np.ndarray, K: int, rng: np.random.Generator, client_size: int | None) -> list[np.ndarray]:
    idx = rng.permutation(idx)
    if client_size is not None and client_size * K <= len(idx):
        return [idx[k * client_size:(k + 1) * client_size] for k in range(K)]
    return [np.asarray(p) for p in np.array_split(idx, K)]


def _dirichlet(ds: Dataset, K: int, rng: np.random.Generator, alpha: float,
               min_per_class: int) -> list[np.ndarray]:
    parts: list[list[np.ndarray]] = [[] for _ in range(K)]
    for c in np.unique(ds.y):
        idx_c = rng.permutation(np.flatnonzero(ds.y == c))
        reserve = K * min_per_class
        if len(idx_c) < reserve:
            raise ValueError(f"dirichlet: class {c} has {len(idx_c)} rows < K*min_per_class={reserve}")
        for k in range(K):  # guaranteed floor so every client sees every class
            parts[k].append(idx_c[k * min_per_class:(k + 1) * min_per_class])
        rest = idx_c[reserve:]
        p = rng.dirichlet(alpha * np.ones(K))
        cuts = (np.cumsum(p) * len(rest)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(rest, cuts)):
            parts[k].append(chunk)
    return [np.concatenate(pk) for pk in parts]


def _group_values(ds: Dataset, group: str) -> np.ndarray:
    if group in ds.groups:
        return np.asarray(ds.groups[group], dtype=float)
    if group in ds.X.columns:
        return ds.X[group].to_numpy(dtype=float)
    raise KeyError(f"group {group!r} neither in ds.groups {list(ds.groups)} nor in ds.X")


def partition(ds: Dataset, scheme: str, K: int, rng: np.random.Generator,
              client_size: int | None = None, **kw) -> tuple[list[np.ndarray], list[str]]:
    if K < 1:
        raise ValueError("K must be >= 1")
    n = len(ds.y)
    all_idx = np.arange(n)
    tags = ["standard"] * K

    if scheme == "iid":
        parts = _iid(all_idx, K, rng, client_size)

    elif scheme == "dirichlet":
        alpha = float(kw.get("alpha", 0.5))
        mpc = int(kw.get("min_per_class", 2))
        parts = _dirichlet(ds, K, rng, alpha, mpc)
        if client_size is not None:
            parts = [_stratified_cap(p, ds.y, client_size, mpc, rng) for p in parts]

    elif scheme == "feature":
        group = kw.get("group", "age")
        threshold = float(kw.get("threshold", 60))
        n_min = int(kw.get("n_minority", 1))
        if not 0 <= n_min < K:
            raise ValueError("need 0 <= n_minority < K")
        g = _group_values(ds, group)
        is_min = np.nan_to_num(g, nan=-np.inf) > threshold
        min_idx = rng.permutation(np.flatnonzero(is_min))
        rest_idx = np.flatnonzero(~is_min)
        if n_min > 0 and len(min_idx) < n_min:
            raise ValueError(f"only {len(min_idx)} rows with {group} > {threshold}")
        std_parts = _iid(rest_idx, K - n_min, rng, client_size)
        if client_size is not None and client_size * n_min <= len(min_idx):
            min_parts = [min_idx[k * client_size:(k + 1) * client_size] for k in range(n_min)]
        else:
            min_parts = [np.asarray(p) for p in np.array_split(min_idx, n_min)] if n_min else []
        parts = std_parts + min_parts
        tags = ["standard"] * (K - n_min) + ["minority"] * n_min

    elif scheme == "quantity":
        sigma = float(kw.get("sigma", 1.0))
        min_size = int(kw.get("min_size", 20))
        s = rng.lognormal(mean=0.0, sigma=sigma, size=K)
        if client_size is None:
            sizes = np.maximum(np.floor(s / s.sum() * n).astype(int), min_size)
        else:
            sizes = np.maximum(np.round(client_size * s / s.max()).astype(int), min_size)
            sizes = np.minimum(sizes, client_size)
        if sizes.sum() > n:
            raise ValueError(f"quantity: requested {sizes.sum()} rows > n={n}")
        perm = rng.permutation(all_idx)
        cuts = np.cumsum(sizes)[:-1]
        parts = list(np.split(perm[: sizes.sum()], cuts))

    else:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")

    parts = [np.sort(np.asarray(p, dtype=int)) for p in parts]
    if len(np.unique(np.concatenate(parts))) != sum(len(p) for p in parts):
        raise AssertionError("partition produced overlapping clients")  # invariant check
    return parts, tags
