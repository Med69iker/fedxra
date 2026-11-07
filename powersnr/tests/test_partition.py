import numpy as np
import pytest

from powersnr.data import make_synthetic
from powersnr.partition import partition


@pytest.fixture(scope="module")
def ds():
    return make_synthetic(5000, 6, np.random.default_rng(0))


def _disjoint(parts):
    allidx = np.concatenate(parts)
    return len(np.unique(allidx)) == len(allidx)


@pytest.mark.parametrize("scheme,kw", [("iid", {}), ("dirichlet", {"alpha": 0.5}),
                                       ("feature", {"threshold": 60}), ("quantity", {"sigma": 1.0})])
def test_disjoint_and_sizes(ds, scheme, kw):
    parts, tags = partition(ds, scheme, 10, np.random.default_rng(1), client_size=300, **kw)
    assert len(parts) == 10 and len(tags) == 10
    assert _disjoint(parts)
    assert all(0 < len(p) <= 300 for p in parts)
    assert all(p.min() >= 0 and p.max() < len(ds.y) for p in parts)


def test_iid_exact_sizes(ds):
    parts, tags = partition(ds, "iid", 8, np.random.default_rng(1), client_size=250)
    assert [len(p) for p in parts] == [250] * 8 and set(tags) == {"standard"}
    parts, _ = partition(ds, "iid", 8, np.random.default_rng(1))
    assert sum(len(p) for p in parts) == len(ds.y)


def test_deterministic(ds):
    a, _ = partition(ds, "dirichlet", 5, np.random.default_rng(7), alpha=0.3)
    b, _ = partition(ds, "dirichlet", 5, np.random.default_rng(7), alpha=0.3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_dirichlet_label_skew(ds):
    def spread(scheme, **kw):
        parts, _ = partition(ds, scheme, 10, np.random.default_rng(2), **kw)
        return np.std([ds.y[p].mean() for p in parts])

    assert spread("dirichlet", alpha=0.1) > 5 * spread("iid")
    parts, _ = partition(ds, "dirichlet", 10, np.random.default_rng(2), alpha=0.1)
    for p in parts:  # every client keeps both classes
        assert len(np.unique(ds.y[p])) == 2
    parts, _ = partition(ds, "dirichlet", 10, np.random.default_rng(2), client_size=150, alpha=0.1)
    for p in parts:  # still true after truncation to client_size
        assert len(p) <= 150 and np.bincount(ds.y[p], minlength=2).min() >= 2


def test_feature_scheme_minority(ds):
    parts, tags = partition(ds, "feature", 6, np.random.default_rng(3), client_size=200,
                            group="age", threshold=60, n_minority=2)
    age = ds.groups["age"].to_numpy()
    assert tags == ["standard"] * 4 + ["minority"] * 2
    for p, t in zip(parts, tags):
        if t == "minority":
            assert (age[p] > 60).all() and len(p) == 200
        else:
            assert (age[p] <= 60).all() and len(p) == 200
    assert _disjoint(parts)


def test_quantity_skew(ds):
    parts, _ = partition(ds, "quantity", 10, np.random.default_rng(4), client_size=400, sigma=1.5)
    sizes = np.array([len(p) for p in parts])
    assert sizes.max() == 400 and sizes.min() < 200


def test_unknown_scheme(ds):
    with pytest.raises(ValueError):
        partition(ds, "bogus", 3, np.random.default_rng(0))
