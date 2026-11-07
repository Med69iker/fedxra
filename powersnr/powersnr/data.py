"""Dataset loading, evaluation split, subsetting and per-client preprocessing.

Conventions
-----------
* Every loader returns a :class:`powersnr.types.Dataset` with a RangeIndex, numeric float
  features (one-hot done with a *fixed* schema so all clients share the same columns),
  NaNs left in place (imputed per client by :class:`LocalPreprocessor`), and ``y`` in {0,1}
  where 1 = bad / default.
* Column names are sanitised to ``[A-Za-z0-9_]`` (LightGBM / XGBoost reject some characters).
* The age column is always called ``"age"`` (in years) when the dataset has age.
* ``rng`` is explicit. When a loader must subsample and ``rng`` is None, a fixed generator
  ``np.random.default_rng(0)`` is used (deterministic, documented; no global state).

Minority groups for the ``"feature"`` partition scheme (see :data:`DEFAULT_GROUP`):
age > 60 for every dataset with age; for LendingClub (no age) the group is ``grade_num``
(A=1 ... G=7) with threshold 5, i.e. sub-prime grades F-G (~2-3 % of resolved loans).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from powersnr.types import Dataset

DATASETS = ["gmsc", "taiwan", "german", "homecredit", "lendingclub"]

# dataset -> (group key in Dataset.groups, threshold); minority = group > threshold
DEFAULT_GROUP: dict[str, tuple[str, float]] = {
    "gmsc": ("age", 60),
    "taiwan": ("age", 60),
    "german": ("age", 60),
    "homecredit": ("age", 60),
    "lendingclub": ("grade_num", 5),
    "synthetic": ("age", 60),
}

LENDINGCLUB_MAX_ROWS = 200_000

# Raw file read by each loader (searched in raw_dir/<name>/, raw_dir/, then /kaggle/input/**/).
DATA_FILES: dict[str, str] = {
    "gmsc": "cs-training.csv",
    "taiwan": "taiwan.csv",            # converted from UCI "default of credit card clients.xls"
    "german": "german.data",           # UCI Statlog german.data (raw codes)
    "homecredit": "application_train.csv",
    "lendingclub": "accepted_2007_to_2018Q4.csv.gz",
}
# Accepted alternatives (e.g. the uncompressed LendingClub CSV shipped in the Kaggle dataset).
ALT_DATA_FILES: dict[str, list[str]] = {
    "lendingclub": ["accepted_2007_to_2018Q4.csv"],
}
KAGGLE_INPUT = "/kaggle/input"


# --------------------------------------------------------------------------- helpers
def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return s or "col"


def _one_hot_fixed(s: pd.Series, prefix: str, categories: list[str]) -> pd.DataFrame:
    """One-hot with a fixed category list; values outside the list go to ``<prefix>_rest``
    (column always present so the schema never depends on the data)."""
    s = s.astype("object")
    out = {}
    known = s.isin(categories)
    for c in categories:
        out[f"{prefix}_{c}"] = (s == c).astype(float)
    out[f"{prefix}_rest"] = (~known & s.notna()).astype(float)
    df = pd.DataFrame(out, index=s.index)
    df[s.isna().to_numpy()] = np.nan  # missing category -> NaN in every dummy (imputed later)
    return df


def _finalize(name: str, X: pd.DataFrame, y, groups: dict[str, pd.Series],
              protected: str | None) -> Dataset:
    X = X.copy()
    X.columns = [_sanitize(c) for c in X.columns]
    if X.columns.duplicated().any():
        raise ValueError(f"{name}: duplicated column names after sanitising: "
                         f"{X.columns[X.columns.duplicated()].tolist()}")
    X = X.apply(pd.to_numeric, errors="raise").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y = np.asarray(y).astype(int)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError(f"{name}: target must be binary 0/1, got {np.unique(y)}")
    if len(y) != len(X):
        raise ValueError(f"{name}: X/y length mismatch")
    groups = {k: pd.Series(np.asarray(v, dtype=float), index=X.index, name=k) for k, v in groups.items()}
    if protected is not None and protected not in X.columns:
        raise ValueError(f"{name}: protected feature {protected!r} not a column")
    return Dataset(name=name, X=X, y=y, feature_names=list(X.columns), groups=groups,
                   protected_feature=protected)


def _find_file(raw_dir: str | Path, name: str) -> Path:
    """Locate the raw file of ``name``: first ``raw_dir/<name>/<file>``, then
    ``raw_dir/<file>``, then any ``/kaggle/input/**/<file>`` (Kaggle notebooks).
    Accepted file names: ``DATA_FILES[name]`` then ``ALT_DATA_FILES[name]``."""
    fnames = [DATA_FILES[name], *ALT_DATA_FILES.get(name, [])]
    searched: list[str] = []
    raw_dir = Path(raw_dir)
    for base in (raw_dir / name, raw_dir):
        for f in fnames:
            cand = base / f
            searched.append(str(cand))
            if cand.is_file():
                return cand
    kaggle = Path(KAGGLE_INPUT)
    for f in fnames:
        searched.append(f"{kaggle}/**/{f}")
        if kaggle.is_dir():
            hits = sorted(kaggle.glob(f"**/{f}"))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"raw file for dataset {name!r} not found. Searched:\n  "
                            + "\n  ".join(searched)
                            + "\n(run scripts/download_data.py or attach the Kaggle input)")


# --------------------------------------------------------------------------- loaders
def _load_gmsc(path: Path) -> Dataset:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    y = df.pop("SeriousDlqin2yrs").to_numpy()
    # age == 0 is a known data error (1 row) -> missing
    df.loc[df["age"] <= 0, "age"] = np.nan
    return _finalize("gmsc", df, y, {"age": df["age"]}, "age")


def _load_taiwan(path: Path) -> Dataset:
    df = pd.read_csv(path)
    y = df.pop("default payment next month").to_numpy()
    df = df.drop(columns=["ID"])
    sex = df.pop("SEX")                      # 1 = male, 2 = female
    edu = df.pop("EDUCATION").replace({0: 4, 5: 4, 6: 4})   # undocumented codes -> other(4)
    mar = df.pop("MARRIAGE").replace({0: 3})                # undocumented 0 -> other(3)
    df = df.rename(columns={"AGE": "age"})
    df["sex_female"] = (sex == 2).astype(float)
    df = pd.concat([df,
                    _one_hot_fixed(edu.map({1: "graduate", 2: "university", 3: "highschool", 4: "other"}),
                                   "education", ["graduate", "university", "highschool"]),
                    _one_hot_fixed(mar.map({1: "married", 2: "single", 3: "other"}),
                                   "marriage", ["married", "single"])], axis=1)
    return _finalize("taiwan", df, y, {"age": df["age"], "sex": (sex == 2).astype(float)}, "age")


_GERMAN_COLS = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
                "savings", "employment", "installment_rate", "personal_status_sex", "other_debtors",
                "residence_since", "property", "age", "other_installment_plans", "housing",
                "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "class"]
_GERMAN_CATS = {  # fixed schema from the UCI documentation
    "checking_status": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
    "savings": ["A61", "A62", "A63", "A64", "A65"],
    "employment": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status_sex": ["A91", "A92", "A93", "A94", "A95"],
    "other_debtors": ["A101", "A102", "A103"],
    "property": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}


def _load_german(path: Path) -> Dataset:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_GERMAN_COLS)
    y = (df.pop("class") == 2).astype(int).to_numpy()   # 2 = bad -> 1
    parts = [df[[c for c in df.columns if c not in _GERMAN_CATS]]]
    for c, cats in _GERMAN_CATS.items():
        # drop the "_rest" column for German: the UCI code list is exhaustive
        oh = _one_hot_fixed(df[c], c, cats)
        if oh[f"{c}_rest"].sum() > 0:
            raise ValueError(f"german: unexpected category in {c}")
        parts.append(oh.drop(columns=[f"{c}_rest"]))
    X = pd.concat(parts, axis=1)
    return _finalize("german", X, y, {"age": df["age"]}, "age")


_HC_NUM = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_CHILDREN",
           "CNT_FAM_MEMBERS", "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "OWN_CAR_AGE",
           "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
           "DAYS_LAST_PHONE_CHANGE", "AMT_REQ_CREDIT_BUREAU_YEAR", "OBS_30_CNT_SOCIAL_CIRCLE",
           "DEF_30_CNT_SOCIAL_CIRCLE", "FLAG_DOCUMENT_3", "REG_CITY_NOT_WORK_CITY",
           "HOUR_APPR_PROCESS_START"]
_HC_CATS = {
    "CODE_GENDER": ["F", "M"],
    "FLAG_OWN_CAR": ["Y"],
    "FLAG_OWN_REALTY": ["Y"],
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "NAME_INCOME_TYPE": ["Working", "Commercial associate", "Pensioner", "State servant"],
    "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education",
                            "Incomplete higher", "Lower secondary", "Academic degree"],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents", "Municipal apartment",
                          "Rented apartment", "Office apartment", "Co-op apartment"],
}


def _load_homecredit(path: Path) -> Dataset:
    cols = ["TARGET", "DAYS_BIRTH", "DAYS_EMPLOYED"] + _HC_NUM + list(_HC_CATS)
    df = pd.read_csv(path, usecols=cols)
    y = df.pop("TARGET").to_numpy()
    X = df[_HC_NUM].copy()
    X["age"] = -df["DAYS_BIRTH"] / 365.25
    emp = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != 365243)  # sentinel = not employed
    X["years_employed"] = -emp / 365.25
    parts = [X]
    for c, cats in _HC_CATS.items():
        oh = _one_hot_fixed(df[c], c.lower(), cats)
        if len(cats) == 1:                       # binary Y/N flag -> single indicator
            oh = oh[[f"{c.lower()}_{cats[0]}"]]
        parts.append(oh)
    X = pd.concat(parts, axis=1)
    gender = (df["CODE_GENDER"] == "F").astype(float).where(df["CODE_GENDER"].isin(["F", "M"]))
    return _finalize("homecredit", X, y, {"age": X["age"], "gender_female": gender}, "age")


# LendingClub: origination-time features only (no total_pymnt, recoveries, last_pymnt_*,
# out_prncp*, last_fico_*, settlement/hardship fields, etc.).
_LC_NUM = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
           "fico_range_low", "fico_range_high", "inq_last_6mths", "mths_since_last_delinq",
           "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
           "pub_rec_bankruptcies"]
_LC_CATS = {
    "home_ownership": ["MORTGAGE", "RENT", "OWN"],
    "verification_status": ["Not Verified", "Source Verified", "Verified"],
    "purpose": ["debt_consolidation", "credit_card", "home_improvement", "other",
                "major_purchase", "small_business", "car", "medical"],
    "initial_list_status": ["w"],
    "application_type": ["Joint App"],
}
_LC_BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
_LC_GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
_LC_USECOLS = (_LC_NUM + list(_LC_CATS) + ["term", "grade", "sub_grade", "emp_length",
                                          "earliest_cr_line", "issue_d", "loan_status"])


def _lc_read_sample(path: Path, n: int, rng: np.random.Generator, chunksize: int = 250_000) -> pd.DataFrame:
    """Uniform sample without replacement of n resolved loans, streaming over chunks:
    each kept row gets a U(0,1) key from ``rng``; we keep the n smallest keys."""
    kept: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, usecols=_LC_USECOLS, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["loan_status"].isin(_LC_BAD | _LC_GOOD)]
        chunk = chunk.assign(_key=rng.random(len(chunk)))
        kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        if len(kept) > n:
            kept = kept.nsmallest(n, "_key")
    if kept is None or kept.empty:
        raise ValueError("lendingclub: no resolved loans found")
    return kept.sort_values("_key").drop(columns="_key").reset_index(drop=True)


def _load_lendingclub(path: Path, max_rows: int | None, rng: np.random.Generator | None) -> Dataset:
    n = min(max_rows, LENDINGCLUB_MAX_ROWS) if max_rows is not None else LENDINGCLUB_MAX_ROWS
    rng = np.random.default_rng(0) if rng is None else rng
    df = _lc_read_sample(path, n, rng)
    y = df["loan_status"].isin(_LC_BAD).astype(int).to_numpy()
    X = df[_LC_NUM].astype(float).copy()
    X["term_months"] = df["term"].str.extract(r"(\d+)")[0].astype(float)
    grade_num = df["grade"].map({g: i + 1 for i, g in enumerate("ABCDEFG")}).astype(float)
    X["grade_num"] = grade_num
    sub = df["sub_grade"].astype(str)
    X["sub_grade_num"] = (sub.str[0].map({g: i for i, g in enumerate("ABCDEFG")}) * 5
                          + pd.to_numeric(sub.str[1:], errors="coerce")).astype(float)
    emp = df["emp_length"].astype("object")
    X["emp_length"] = (emp.str.replace("< 1 year", "0", regex=False)
                       .str.extract(r"(\d+)")[0].astype(float))
    issue = pd.to_datetime(df["issue_d"], format="%b-%Y")
    ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    X["credit_history_years"] = (issue - ecl).dt.days / 365.25
    parts = [X]
    for c, cats in _LC_CATS.items():
        oh = _one_hot_fixed(df[c], c, cats)
        if len(cats) == 1:
            oh = oh[[f"{c}_{cats[0]}"]]
        parts.append(oh)
    X = pd.concat(parts, axis=1)
    return _finalize("lendingclub", X, y,
                     {"grade_num": grade_num, "annual_inc": X["annual_inc"]}, "annual_inc")


def load_dataset(name: str, raw_dir: str | Path, max_rows: int | None = None,
                 rng: np.random.Generator | None = None) -> Dataset:
    """Load a dataset (file located by :func:`_find_file`). If ``max_rows`` is set and the dataset is
    larger, a uniform random subsample (without replacement, ``rng``) is returned."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    path = _find_file(raw_dir, name)
    if name == "lendingclub":
        return _load_lendingclub(path, max_rows, rng)
    ds = {"gmsc": _load_gmsc, "taiwan": _load_taiwan, "german": _load_german,
          "homecredit": _load_homecredit}[name](path)
    if max_rows is not None and len(ds.y) > max_rows:
        rng = np.random.default_rng(0) if rng is None else rng
        idx = np.sort(rng.choice(len(ds.y), size=max_rows, replace=False))
        ds = subset(ds, idx)
    return ds


# --------------------------------------------------------------------------- synthetic
def make_synthetic(n: int, d: int, rng: np.random.Generator, n_informative: int = 4) -> Dataset:
    """Learnable binary task. Column 0 is ``age`` (uniform 20-80, informative: older = lower
    risk), columns 1..n_informative-1 are informative Gaussians with decreasing coefficients,
    the remaining columns are pure noise. y ~ Bernoulli(sigmoid(logit))."""
    if d < n_informative or n_informative < 1:
        raise ValueError("need 1 <= n_informative <= d")
    Z = rng.normal(size=(n, d))
    age = rng.uniform(20, 80, size=n)
    coefs = np.zeros(d)
    coefs[:n_informative] = np.linspace(2.0, 0.75, n_informative)
    age_std = (age - 50.0) / 17.3
    Z[:, 0] = age_std
    logit = Z @ coefs - 1.0
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    names = ["age"] + [f"x{j}" for j in range(1, d)]
    X = pd.DataFrame(Z, columns=names)
    X["age"] = age
    return _finalize("synthetic", X, y, {"age": X["age"]}, "age")


# --------------------------------------------------------------------------- splits
def subset(ds: Dataset, idx: np.ndarray) -> Dataset:
    """Positional subset; the result has a fresh RangeIndex (groups re-aligned)."""
    idx = np.asarray(idx, dtype=int)
    X = ds.X.iloc[idx].reset_index(drop=True)
    groups = {k: v.iloc[idx].reset_index(drop=True) for k, v in ds.groups.items()}
    return Dataset(name=ds.name, X=X, y=ds.y[idx].copy(), feature_names=list(ds.feature_names),
                   groups=groups, protected_feature=ds.protected_feature)


def split_eval(ds: Dataset, frac: float, rng: np.random.Generator) -> tuple[Dataset, Dataset]:
    """Stratified (on y) split -> (pool, eval) with ``frac`` of each class in eval."""
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1)")
    ev = []
    for c in np.unique(ds.y):
        idx_c = np.flatnonzero(ds.y == c)
        idx_c = rng.permutation(idx_c)
        ev.append(idx_c[: int(round(frac * len(idx_c)))])
    ev_idx = np.sort(np.concatenate(ev))
    pool_idx = np.setdiff1d(np.arange(len(ds.y)), ev_idx)
    return subset(ds, pool_idx), subset(ds, ev_idx)


# --------------------------------------------------------------------------- preprocessing
class LocalPreprocessor:
    """Per-client median imputation, fitted on that client's training rows only.

    Columns that are entirely missing in the client's training data have no median; they are
    imputed with 0.0 and listed in ``all_nan_cols_`` (explicit, not silent).
    """

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None
        self.columns_: list[str] | None = None
        self.all_nan_cols_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "LocalPreprocessor":
        med = X.median(axis=0, skipna=True, numeric_only=False)
        self.all_nan_cols_ = med.index[med.isna()].tolist()
        self.medians_ = med.fillna(0.0)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("LocalPreprocessor.transform called before fit")
        if list(X.columns) != self.columns_:
            raise ValueError("column mismatch between fit and transform")
        return X.fillna(self.medians_).astype(float)
