"""Rewrite the aggregator blocks of configs/E*.yaml for the final (confirmatory) runs:
old Power-SNR variants removed, Power-SNR v2 (screened) as the main method, 20 split-half
repeats and fresh seeds (seed_offset)."""
from __future__ import annotations

import glob
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEED_OFFSET = 2000


def v2(n_cap: int, **over) -> dict:
    kw = {"z": 2.0, "cap": 2.0, "floor": "n", "n_cap": n_cap, "shrink": 0.5, "stage2": "geomedian", "n_repeats": 20,
          "dedup": True, "dup_ratio": 1.0}
    kw.update(over)
    return kw


def main_set(n_cap: int) -> dict:
    return {
        "fedavg": {}, "median": {}, "trimmed_mean": {"beta": 0.2}, "geomedian": {}, "krum": {}, "multikrum": {},
        "ivw": {"eps": 1.0e-12}, "dersimonian_laird": {}, "fltrust_like": {},
        "psnr_screen": v2(n_cap),
    }


def ablation_set(n_cap: int) -> dict:
    r = {"rule": "psnr_screen"}
    return {
        "fedavg": {}, "geomedian": {}, "ivw": {"eps": 1.0e-12},
        "psnr_screen": v2(n_cap),
        "psnr_screen_mean_stage": {**r, **v2(n_cap, stage2="mean")},
        "psnr_screen_quantile_floor": {**r, **v2(n_cap, floor="quantile", floor_q=0.25)},
        "psnr_screen_no_shrink": {**r, **v2(n_cap, shrink=None)},
        "psnr_screen_auto_shrink": {**r, **v2(n_cap, shrink="auto")},
        "psnr_screen_no_mass_screen": {**r, **v2(n_cap, mass_z=1.0e9)},
        "psnr_screen_no_profile_screen": {**r, **v2(n_cap, z=1.0e9)},
        "psnr_screen_no_cap": {**r, **v2(n_cap, cap=None)},
        "psnr_screen_no_dedup": {**r, **v2(n_cap, dedup=False)},
        "screen_geomedian": {**r, "z": 2.0, "snr": False, "stage2": "geomedian"},
        "psnr_screen_v1": {**r, "z": 2.0, "cap": 2.0, "floor_q": 0.25},
    }


def fairness_set(n_cap: int) -> dict:
    r = {"rule": "psnr_screen"}
    return {
        **main_set(n_cap),
        "psnr_screen_soft": {**r, **v2(n_cap, soft=True)},
        "psnr_screen_no_shrink": {**r, **v2(n_cap, shrink=None)},
        "psnr_screen_cap1": {**r, **v2(n_cap, cap=1.0)},
        "psnr_screen_z3": {**r, **v2(n_cap, z=3.0)},
    }


def main():
    for f in sorted(glob.glob(str(ROOT / "configs" / "E*.yaml"))):
        text = Path(f).read_text(encoding="utf-8")
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        cfg = yaml.safe_load(text)
        cfg["shap"] = {**cfg.get("shap", {}), "n_repeats": 20}
        n_cap = int(cfg["shap"].get("n_explain", 1000))
        exp = cfg["experiment"]
        cfg["aggregators"] = ablation_set(n_cap) if exp == "E8" else fairness_set(n_cap) if exp == "E10" else main_set(n_cap)
        cfg["seed_offset"] = SEED_OFFSET
        Path(f).write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(Path(f).name, len(cfg["aggregators"]))


if __name__ == "__main__":
    main()
