import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from fake_results import ALL_COLUMNS, GRIDS, make_fake_results  # noqa: E402

from analysis import figures as F  # noqa: E402
from analysis import make_paper_assets as MPA  # noqa: E402
from analysis import tables as T  # noqa: E402
from analysis.stats import (cliffs_delta, friedman_nemenyi, holm_correction,  # noqa: E402
                            nemenyi_cd, paired_tests, summarize, wilcoxon_test)


@pytest.fixture
def fast_fonts(monkeypatch):
    # Windows' Times New Roman makes Type-42 PDF subsetting slow (~0.7 s/figure);
    # the pipeline is what is tested here, not the font.
    monkeypatch.setitem(F.RC, "font.serif", ["DejaVu Serif"])


@pytest.fixture(scope="module")
def df1():
    return make_fake_results(("E1",), n_seeds=10)


def test_fixture_has_all_columns(df1):
    assert list(df1.columns) == ALL_COLUMNS
    assert df1["aggregator"].nunique() == 3
    assert df1["seed"].nunique() == 10


def test_summarize(df1):
    s = summarize(df1, ["aggregator"], "rmse_norm", n_boot=500)
    assert set(s.columns) >= {"aggregator", "mean", "std", "n", "ci_low", "ci_high"}
    s = s.set_index("aggregator")
    assert (s["n"] == 30).all()
    assert (s["ci_low"] <= s["mean"]).all() and (s["mean"] <= s["ci_high"]).all()
    assert s.loc["psnr_screen", "mean"] < s.loc["mean", "mean"]
    # seeded -> reproducible
    s2 = summarize(df1, ["aggregator"], "rmse_norm", n_boot=500).set_index("aggregator")
    assert np.allclose(s["ci_low"], s2["ci_low"])
    np.testing.assert_allclose(s.loc["mean", "mean"],
                               df1.loc[df1.aggregator == "mean", "rmse_norm"].mean())


def test_summarize_handles_nan_keys(df1):
    s = summarize(df1, ["aggregator", "attack_param"], "rmse_norm", n_boot=100)
    assert len(s) == 3 and s["attack_param"].isna().all()


def test_paired_tests_detects_best(df1):
    pt = paired_tests(df1, ["attack"], "rmse_norm", reference="psnr_screen", lower_is_better=True)
    assert len(pt) == 3 * 2
    assert (pt["n_pairs"] == 10).all()
    assert (pt["p_holm"] < 0.05).all()
    assert (pt["p_holm"] >= pt["p_value"]).all()
    assert (pt["wins"] >= 8).all() and (pt["wins"] > pt["losses"]).all()
    assert (pt["cliffs_delta"] > 0.3).all()
    assert pt["ref_better"].all()
    # higher-is-better metric
    pt2 = paired_tests(df1, ["attack"], "kendall_tau", reference="psnr_screen",
                       lower_is_better=False)
    assert (pt2["cliffs_delta"] > 0).all() and pt2["ref_better"].all()
    # reversed reference: the worst aggregator loses
    pt3 = paired_tests(df1, ["attack"], "rmse_norm", reference="mean")
    assert (pt3["losses"] >= 8).all() and not pt3["ref_better"].any()


def test_wilcoxon_all_ties_and_helpers():
    x = np.arange(10.0)
    assert wilcoxon_test(x, x) == (0.0, 1.0)
    p = holm_correction([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(p[:3], [0.03, 0.06, 0.06])
    assert np.isnan(p[3])
    assert cliffs_delta(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0


def test_paired_tests_identical_aggregators():
    df = make_fake_results(("E1",), aggregators={"psnr_screen": 1.0, "clone": 1.0}, n_seeds=5)
    df.loc[df.aggregator == "clone", "rmse_norm"] = df.loc[df.aggregator == "psnr_screen",
                                                           "rmse_norm"].to_numpy()
    pt = paired_tests(df, ["attack"], "rmse_norm")
    assert (pt["p_value"] == 1.0).all() and (pt["ties"] == 5).all()


def test_friedman_nemenyi_order():
    df = make_fake_results(("E6",), n_seeds=10)
    stat, p, ranks, cd = friedman_nemenyi(df, ["dataset", "model"], "rmse_norm")
    assert list(ranks.index) == ["psnr_screen", "psnr_base", "mean"]
    assert ranks.iloc[0] == pytest.approx(1.0)
    assert cd > 0 and p < 0.05 and stat > 0
    assert cd == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 6)))
    _, _, ranks_hi, _ = friedman_nemenyi(df, ["dataset", "model"], "kendall_tau",
                                         lower_is_better=False)
    assert ranks_hi.index[0] == "psnr_screen"
    assert nemenyi_cd(10, 20) == pytest.approx(3.164 * np.sqrt(110 / 120))


def test_latex_table(df1, tmp_path):
    s = summarize(df1, ["aggregator", "attack"], "rmse_norm", n_boot=100)
    pv = paired_tests(df1, ["attack"], "rmse_norm")
    tex = T.to_latex_table(s, row="aggregator", col="attack", pvalues=pv, decimals=3,
                           caption="Test caption", label="tab:test", path=tmp_path / "t.tex")
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert "\\caption{Test caption}" in tex and "\\label{tab:test}" in tex
    robust_line = next(l for l in tex.splitlines() if l.startswith("Power-SNR &"))
    assert robust_line.count("\\mathbf") == 3
    mean_line = next(l for l in tex.splitlines() if l.startswith("Mean &"))
    assert "\\mathbf" not in mean_line and "\\dagger" in mean_line
    assert "\\dagger" not in robust_line
    assert "feature\\_hiding" in tex
    assert (tmp_path / "t.tex").read_text(encoding="utf-8") == tex
    # higher is better flips the bold
    s2 = summarize(df1, ["aggregator", "attack"], "kendall_tau", n_boot=100)
    tex2 = T.to_latex_table(s2, lower_is_better=False)
    rl = next(l for l in tex2.splitlines() if l.startswith("Power-SNR &"))
    assert rl.count("\\mathbf") == 3


def test_pretty_names():
    assert T.pretty_name("psnr_screen") == "Power-SNR"
    assert T.pretty_name("dersimonian_laird") == "DerSimonian--Laird"
    assert T.pretty_name("ivw") == "Inverse-variance"
    assert T.pretty_name("some_new_agg") == "some\\_new\\_agg"


def test_multi_metric_table(df1):
    summ = {m: summarize(df1, ["aggregator", "attack"], m, n_boot=50)
            for m in ("rmse_norm", "kendall_tau")}
    tex = T.to_latex_multi_metric(summ, lower_is_better={"rmse_norm": True, "kendall_tau": False})
    assert "\\toprule" in tex and "\\cmidrule" in tex
    rl = next(l for l in tex.splitlines() if l.startswith("Power-SNR &"))
    assert rl.count("\\mathbf") == 6


def test_figures_write_files(tmp_path, fast_fonts):
    df = make_fake_results(("E2", "E6", "E10"), n_seeds=5)
    e2 = df[df.experiment == "E2"]
    paths = F.plot_metric_vs(e2, "attack_param", "rmse_norm", out_dir=tmp_path, name="a",
                             n_boot=100)
    paths += F.plot_metric_grid(e2, "attack_param", "kendall_tau", "attack", out_dir=tmp_path,
                                name="b", n_boot=100)
    paths += F.plot_weight_shares(e2, "attack_param", out_dir=tmp_path, name="c", n_boot=100)
    _, _, ranks, cd = friedman_nemenyi(df[df.experiment == "E6"], ["dataset", "model"], "rmse_norm")
    paths += F.plot_cd_diagram(ranks, cd, out_dir=tmp_path, name="d")
    paths += F.plot_pareto(df[df.experiment == "E10"], "rmse_norm", "rmse_norm_minority",
                           out_dir=tmp_path, name="e", n_boot=100)
    phi = pd.DataFrame({"oracle": [0.4, 0.3, 0.2, 0.1], "psnr_screen": [0.38, 0.32, 0.2, 0.1],
                        "mean": [0.1, 0.3, 0.2, 0.4]}, index=["age", "income", "debt", "x"])
    paths += F.plot_feature_bars(phi, out_dir=tmp_path, name="f", highlight="age")
    assert len(paths) == 12
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
    assert {p.suffix for p in paths} == {".pdf", ".png"}


def test_cd_many_aggregators(tmp_path, fast_fonts):
    ranks = pd.Series({a: i + 1.0 + 0.1 * i for i, a in enumerate(F.AGG_ORDER)})
    paths = F.plot_cd_diagram(ranks, nemenyi_cd(len(ranks), 30), out_dir=tmp_path, name="cd15",
                              width="double")
    assert all(p.exists() for p in paths)


def test_style_mapping_distinct():
    psnr = [a for a in F.AGG_STYLE if a.startswith("psnr")]
    combos = {(F.AGG_STYLE[a]["color"], F.AGG_STYLE[a]["marker"], F.AGG_STYLE[a]["ls"]) for a in psnr}
    assert len(combos) == len(psnr)
    assert F.AGG_STYLE["psnr_v3"]["lw"] > max(F.AGG_STYLE[a]["lw"] for a in F.AGG_STYLE
                                               if a != "psnr_v3")
    assert "\t" not in "".join(F.METRIC_LABELS.values())


def test_experiment_number():
    assert MPA.experiment_number("E1") == 1
    assert MPA.experiment_number("e10_fairness") == 10
    assert MPA.experiment_number("exp6") == 6
    assert MPA.experiment_number("pilot") is None


def test_make_paper_assets_end_to_end(tmp_path, fast_fonts, monkeypatch):
    # E2/E4 share the `curves` code path with E3 (faceting and log-x are covered by
    # test_figures_write_files); left out to keep this file under 30 s.
    monkeypatch.setattr(MPA, "MAIN_METRICS", ["rmse_norm", "kendall_tau", "w_attacker"])
    exps = ("E1", "E3", "E5", "E6", "E7", "E8", "E9", "E10")
    df = make_fake_results(exps, n_seeds=4)
    run = tmp_path / "run"
    run.mkdir()
    df.to_parquet(run / "results.parquet")
    out = tmp_path / "assets"
    assert MPA.main(["--run", str(run), "--out", str(out), "--n-boot", "100"]) == 0
    import json
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(man["experiments"]) == set(exps)
    for e in exps:
        assert (out / e / f"summary_{e}.csv").exists()
        assert (out / e / f"stats_{e}.csv").exists()
    st = pd.read_csv(out / "E1" / "stats_E1.csv")
    assert set(st["reference"]) == {"psnr_screen", "psnr_base"}
    assert (out / "E1" / "E1_rmse_norm.tex").exists()
    assert (out / "E1" / "E1_main.tex").exists()
    assert (out / "E3" / "E3_kendall_tau_vs_attack_frac.png").exists()
    assert (out / "E3" / "E3_weights_vs_attack_frac.pdf").exists()
    assert (out / "E5" / "E5_rmse_norm_vs_alpha.pdf").exists()
    assert (out / "E6" / "E6_cd_rmse_norm.pdf").exists()
    assert (out / "E6" / "E6_main_rmse_norm.tex").exists()
    assert (out / "E6" / "E6_friedman.csv").exists()
    assert (out / "E7" / "E7_defense.tex").exists()
    assert (out / "E8" / "E8_ablation.tex").exists()
    assert (out / "E10" / "E10_pareto.pdf").exists()
    # E9 (quality heterogeneity) reuses the E1 tables with one column per attack intensity
    assert (out / "E9" / "E9_main.tex").exists()
    # only a subset of experiments present -> others are simply absent
    out2 = tmp_path / "assets2"
    MPA.make_assets(df[df.experiment == "E1"], out2, n_boot=50)
    assert (out2 / "E1").exists() and not (out2 / "E3").exists()
