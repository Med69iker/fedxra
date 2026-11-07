"""Rewrite configs/E*.yaml for run 5 (confirmation of Power-SNR v3 on fresh seeds 3000+).

Changes with respect to run 4 (whose configs are archived next to its results):
  * aggregators: the run-4 set + Power-SNR v3 + FoolsGold, centered clipping, Bulyan and a Trimmed mean
    tuned to beta = 0.3; E8 becomes the v3 ablation;
  * attacks: the white-box coalition (chosen per rule), targeted ALIE and min-max are added;
  * seed_offset = 3000 (no seed of runs 1-4 is reused).
"""
from __future__ import annotations

import glob
from pathlib import Path

import yaml

from set_final_aggregators import v2

ROOT = Path(__file__).resolve().parents[1]
SEED_OFFSET = 3000

V3 = {"rule": "psnr_v3", "z": 2.0, "mass_z": 3.0, "weight": "n", "cap": 3.0, "dedup": True, "dup_ratio": 4.0,
      "combine": "huber", "huber_c": 0.5, "scale_q": 0.8}


def main_set(n_cap: int) -> dict:
    return {
        "fedavg": {}, "median": {}, "trimmed_mean": {"beta": 0.2},
        "trimmed_mean_0.3": {"rule": "trimmed_mean", "beta": 0.3},
        "geomedian": {}, "krum": {}, "multikrum": {}, "bulyan": {}, "centered_clipping": {}, "foolsgold": {},
        "ivw": {"eps": 1.0e-12}, "dersimonian_laird": {}, "fltrust_like": {},
        "psnr_screen": v2(n_cap), "psnr_v3": dict(V3, n_cap=n_cap),
    }


def ablation_set(n_cap: int) -> dict:
    v3 = dict(V3, n_cap=n_cap)
    return {
        "fedavg": {}, "trimmed_mean": {"beta": 0.2}, "geomedian": {}, "psnr_screen": v2(n_cap),
        "psnr_v3": v3,
        "psnr_v3_mean_stage": {**v3, "combine": "mean"},
        "psnr_v3_geomedian_stage": {**v3, "combine": "geomedian"},
        "psnr_v3_mad_scale": {**v3, "scale_q": 0.5},
        "psnr_v3_uniform_weights": {**v3, "weight": "unif"},
        "psnr_v3_snr_weights": {**v3, "weight": "snr"},
        "psnr_v3_no_screens": {**v3, "screens": False},
        "psnr_v3_no_dedup": {**v3, "dedup": False},
        "psnr_v3_no_cap": {**v3, "cap": None},
    }


NEW_ATTACKS = {  # experiment -> attacker fractions for the white-box and ALIE coalitions (min-max at 0.2)
    "E3": [0.1, 0.2, 0.3, 0.4], "E4": [0.2], "E5": [0.2], "E6": [0.2], "E7": [0.1, 0.2, 0.3],
    "E8": [0.2], "E9": [0.2], "E10": [0.2], "E11": [0.2],
}


def main():
    for f in sorted(glob.glob(str(ROOT / "configs" / "E*.yaml"))):
        text = Path(f).read_text(encoding="utf-8")
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        cfg = yaml.safe_load(text)
        n_cap = int(cfg["shap"].get("n_explain", 1000))
        exp = cfg["experiment"]
        cfg["aggregators"] = ablation_set(n_cap) if exp == "E8" else main_set(n_cap)
        attacks = [a for a in cfg.get("attacks", []) if a["name"] not in ("whitebox", "alie", "minmax")]
        if exp in NEW_ATTACKS:
            fr = NEW_ATTACKS[exp]
            attacks += [{"name": "whitebox", "fracs": fr}, {"name": "alie", "fracs": fr},
                        {"name": "minmax", "fracs": [0.2]}]
        cfg["attacks"] = attacks
        cfg["seed_offset"] = SEED_OFFSET
        Path(f).write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(Path(f).name, len(cfg["aggregators"]), [a["name"] for a in attacks])


if __name__ == "__main__":
    main()
