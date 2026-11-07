import numpy as np
import pandas as pd
import pytest

from powersnr.data import make_synthetic
from powersnr.explain import client_stats, fit_client, global_importance, split_half_sigma2
from powersnr.models import MODELS, make_model


@pytest.fixture(scope="module")
def ds():
    return make_synthetic(1500, 6, np.random.default_rng(0))


@pytest.mark.parametrize("model_name", MODELS)
def test_client_stats_shapes_and_determinism(ds, model_name):
    model, prep, Xt, yt = fit_client(model_name, ds.X, ds.y, np.random.default_rng(1), seed=0)
    kw = dict(n_explain=60 if model_name in ("rf", "mlp") else 150, n_background=20)
    s1 = client_stats(model, Xt, yt, np.random.default_rng(5), 3, "honest", **kw)
    s2 = client_stats(model, Xt, yt, np.random.default_rng(5), 3, "honest", **kw)
    d = ds.X.shape[1]
    assert s1.mu.shape == (d,) and s1.mu_signed.shape == (d,)
    assert (s1.mu >= 0).all() and s1.sigma2 >= 0
    assert s1.client_id == 3 and s1.role == "honest" and s1.n == len(yt)
    assert 0.5 < s1.local_auc <= 1.0
    assert np.array_equal(s1.mu, s2.mu) and s1.sigma2 == s2.sigma2
    # the strongest synthetic column (age, coefficient 2.0) dominates
    assert int(np.argmax(s1.mu)) == 0


def test_mlp_explain_cap(ds):
    model, _, Xt, yt = fit_client("mlp", ds.X, ds.y, np.random.default_rng(1), seed=0)
    s = client_stats(model, Xt, yt, np.random.default_rng(0), 0, "honest", n_explain=1000, n_background=10)
    assert s.extra["n_explain"] <= 200


def test_global_importance(ds):
    model, _, Xt, _ = fit_client("xgboost", ds.X, ds.y, np.random.default_rng(1), seed=0)
    g = global_importance(model, Xt.iloc[:100], Xt.iloc[100:150])
    gs = global_importance(model, Xt.iloc[:100], Xt.iloc[100:150], signed=True)
    assert g.shape == (6,) and (g >= 0).all() and (np.abs(gs) <= g + 1e-12).all()


def test_split_half_sigma2():
    assert split_half_sigma2(np.ones((50, 4)), 5, np.random.default_rng(0)) == 0.0
    phi = np.random.default_rng(1).normal(size=(200, 4))
    assert split_half_sigma2(phi, 10, np.random.default_rng(0)) > 0


def test_fit_client_no_test_split(ds):
    model, prep, Xt, yt = fit_client("logreg", ds.X, ds.y, np.random.default_rng(1), seed=0, test_frac=0.0)
    assert len(Xt) == 0 and len(yt) == 0 and list(Xt.columns) == list(ds.X.columns)
    assert model.predict_proba(prep.transform(ds.X.iloc[:5])).shape == (5, 2)


def test_fit_client_stratified_and_imputed(ds):
    X = ds.X.copy()
    X.iloc[::7, 1] = np.nan
    model, prep, Xt, yt = fit_client("lightgbm", X, ds.y, np.random.default_rng(2), seed=0, test_frac=0.2)
    assert abs(len(yt) - 300) <= 2 and not Xt.isna().any().any()
    assert abs(yt.mean() - ds.y.mean()) < 0.01


def test_noise_model_has_smaller_energy():
    """Power-SNR premise: a model trained on pure-noise features yields a smaller ||mu||."""
    rng = np.random.default_rng(0)
    ds = make_synthetic(2000, 6, rng, n_informative=4)
    X_noise = pd.DataFrame(rng.normal(size=ds.X.shape), columns=ds.X.columns)
    m_inf, _, Xt_inf, yt_inf = fit_client("xgboost", ds.X, ds.y, np.random.default_rng(1), seed=0)
    m_noi, _, Xt_noi, yt_noi = fit_client("xgboost", X_noise, ds.y, np.random.default_rng(1), seed=0)
    kw = dict(n_explain=300, n_background=50)
    s_inf = client_stats(m_inf, Xt_inf, yt_inf, np.random.default_rng(2), 0, "honest", **kw)
    s_noi = client_stats(m_noi, Xt_noi, yt_noi, np.random.default_rng(2), 1, "honest", **kw)
    e_inf, e_noi = float(np.sum(s_inf.mu ** 2)), float(np.sum(s_noi.mu ** 2))
    print(f"\n||mu||^2 informative={e_inf:.4f} noise={e_noi:.4f}; "
          f"sigma2 informative={s_inf.sigma2:.2e} noise={s_noi.sigma2:.2e}; "
          f"SNR informative={e_inf / s_inf.sigma2:.1f} noise={e_noi / s_noi.sigma2:.1f}; "
          f"AUC informative={s_inf.local_auc:.3f} noise={s_noi.local_auc:.3f}")
    assert e_noi < e_inf


def test_unknown_model():
    with pytest.raises(ValueError):
        make_model("svm", 0)
