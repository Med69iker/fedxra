import numpy as np
import pandas as pd
import pytest

from powersnr.data import (DATA_FILES, DATASETS, LocalPreprocessor, _find_file, load_dataset,
                           make_synthetic, split_eval, subset)


def test_synthetic_shape_and_age():
    ds = make_synthetic(1000, 6, np.random.default_rng(0))
    assert ds.X.shape == (1000, 6) and ds.y.shape == (1000,)
    assert "age" in ds.X.columns and ds.protected_feature == "age"
    assert set(np.unique(ds.y)) == {0, 1}
    assert ds.X["age"].between(20, 80).all()
    # age is informative
    assert abs(ds.y[ds.X["age"] > 60].mean() - ds.y[ds.X["age"] < 40].mean()) > 0.2


def test_synthetic_deterministic():
    a = make_synthetic(200, 5, np.random.default_rng(3))
    b = make_synthetic(200, 5, np.random.default_rng(3))
    pd.testing.assert_frame_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)


def test_split_eval_stratified_disjoint():
    ds = make_synthetic(3000, 5, np.random.default_rng(1))
    pool, ev = split_eval(ds, 0.2, np.random.default_rng(2))
    assert len(pool.y) + len(ev.y) == len(ds.y)
    assert abs(len(ev.y) - 600) <= 2
    assert abs(ev.y.mean() - ds.y.mean()) < 0.005
    assert abs(pool.y.mean() - ds.y.mean()) < 0.005


def test_subset_realigns_groups():
    ds = make_synthetic(100, 4, np.random.default_rng(0))
    idx = np.array([5, 7, 50])
    s = subset(ds, idx)
    assert list(s.X.index) == [0, 1, 2]
    assert np.allclose(s.groups["age"].to_numpy(), ds.X["age"].to_numpy()[idx])
    assert np.array_equal(s.y, ds.y[idx])


def test_local_preprocessor_no_leakage():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [np.nan] * 4})
    test = pd.DataFrame({"a": [np.nan, 1000.0], "b": [np.nan, 5.0]})
    prep = LocalPreprocessor().fit(train)
    out = prep.transform(test)
    assert out.loc[0, "a"] == 2.0          # train median, not influenced by the test value 1000
    assert out.loc[0, "b"] == 0.0 and prep.all_nan_cols_ == ["b"]
    assert out.loc[1, "a"] == 1000.0
    assert not out.isna().any().any()
    assert LocalPreprocessor().fit(test).medians_["a"] == 1000.0


def test_local_preprocessor_requires_fit_and_columns():
    with pytest.raises(RuntimeError):
        LocalPreprocessor().transform(pd.DataFrame({"a": [1.0]}))
    prep = LocalPreprocessor().fit(pd.DataFrame({"a": [1.0]}))
    with pytest.raises(ValueError):
        prep.transform(pd.DataFrame({"z": [1.0]}))


def test_missing_raw_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Searched"):
        _find_file(tmp_path, "gmsc")
    assert set(DATA_FILES) == set(DATASETS)


def test_find_file_in_raw_dir(tmp_path):
    (tmp_path / "german").mkdir()
    f = tmp_path / "german" / DATA_FILES["german"]
    f.write_text("x")
    assert _find_file(tmp_path, "german") == f


def test_unknown_dataset():
    with pytest.raises(ValueError):
        load_dataset("nope", ".")
