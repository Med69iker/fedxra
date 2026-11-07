"""Rewrite configs/E*.yaml for run 6 (confirmation of the revised FedXRA on fresh seeds 4000+).

Changes with respect to run 5 (whose configs are archived next to its results):
  * psnr_v3 (FedXRA) is the round-8 design: mass screen only, n weights capped at 1.5 x median, duplicate
    grouping, coordinate-wise weighted trimmed mean (beta = 0.3); psnr_v3_unif is its uniform-weight mode
    (for consortia that cannot audit n); the run-5 design stays in the E8 ablation;
  * attacks: the stronger white-box coalition (whitebox_strong: paper family + boundary-rank + boundary
    families, chosen per rule) is added next to the paper family; in E6 and E9 it is also run with the
    attackers claiming n = 1000 (outside the threat model);
  * seed_offset = 4000 (no seed of runs 1-5 is reused).
"""
from __future__ import annotations

import glob
from pathlib import Path

import yaml

from set_final_aggregators import v2

ROOT = Path(__file__).resolve().parents[1]
SEED_OFFSET = 4000

V3 = {"rule": "psnr_v3", "mass_z": 3.0, "weight": "n", "cap": 1.5, "dedup": True, "dup_ratio": 4.0,
      "combine": "trim", "trim_beta": 0.3, "profile_screen": False}
V3_RUN5 = {"rule": "psnr_v3", "z": 2.0, "mass_z": 3.0, "weight": "n", "cap": 3.0, "dedup": True, "dup_ratio": 4.0,
           "combine": "huber", "huber_c": 0.5, "scale_q": 0.8, "profile_screen": True}


def main_set(n_cap: int) -> dict:
    return {
        "fedavg": {}, "median": {}, "trimmed_mean": {"beta": 0.2},
        "trimmed_mean_0.3": {"rule": "trimmed_mean", "beta": 0.3},
        "geomedian": {}, "krum": {}, "multikrum": {}, "bulyan": {}, "centered_clipping": {}, "foolsgold": {},
        "ivw": {"eps": 1.0e-12}, "dersimonian_laird": {}, "fltrust_like": {},
        "psnr_screen": v2(n_cap), "psnr_v3": dict(V3, n_cap=n_cap),
        "psnr_v3_unif": dict(V3, n_cap=n_cap, weight="unif"),
    }


def ablation_set(n_cap: int) -> dict:
    v3 = dict(V3, n_cap=n_cap)
    return {
        "fedavg": {}, "trimmed_mean": {"beta": 0.2}, "trimmed_mean_0.3": {"rule": "trimmed_mean", "beta": 0.3},
        "geomedian": {}, "psnr_screen": v2(n_cap),
        "psnr_v3": v3,
        "psnr_v3_unif": {**v3, "weight": "unif"},
        "psnr_v3_run5": dict(V3_RUN5, n_cap=n_cap),
        "psnr_v3_huber_stage": {**v3, "combine": "huber", "huber_c": 0.5, "scale_q": 0.8},
        "psnr_v3_median_stage": {**v3, "combine": "median"},
        "psnr_v3_mean_stage": {**v3, "combine": "mean"},
        "psnr_v3_profile_screen": {**v3, "profile_screen": True, "z": 2.0},
        "psnr_v3_no_screens": {**v3, "screens": False},
        "psnr_v3_no_dedup": {**v3, "dedup": False},
        "psnr_v3_cap3": {**v3, "cap": 3.0},
        "psnr_v3_trim0.2": {**v3, "trim_beta": 0.2},
    }


WB_FRACS = {  # experiment -> attacker fractions for the white-box and ALIE coalitions (min-max at 0.2)
    "E3": [0.1, 0.2, 0.3, 0.4], "E4": [0.2], "E5": [0.2], "E6": [0.2], "E7": [0.1, 0.2, 0.3],
    "E8": [0.2], "E9": [0.2], "E10": [0.2], "E11": [0.2],
}
CLAIM_N = {"E6", "E9"}


def main():
    for f in sorted(glob.glob(str(ROOT / "configs" / "E*.yaml"))):
        text = Path(f).read_text(encoding="utf-8")
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        cfg = yaml.safe_load(text)
        n_cap = int(cfg["shap"].get("n_explain", 1000))
        exp = cfg["experiment"]
        cfg["aggregators"] = ablation_set(n_cap) if exp == "E8" else main_set(n_cap)
        attacks = [a for a in cfg.get("attacks", []) if a["name"] not in ("whitebox", "whitebox_strong", "alie", "minmax")]
        if exp in WB_FRACS:
            fr = WB_FRACS[exp]
            strong = {"name": "whitebox_strong", "fracs": fr}
            if exp in CLAIM_N:
                strong["params"] = [{}, {"claim_n": n_cap}]
            attacks += [{"name": "whitebox", "fracs": fr}, strong, {"name": "alie", "fracs": fr},
                        {"name": "minmax", "fracs": [0.2]}]
        cfg["attacks"] = attacks
        cfg["seed_offset"] = SEED_OFFSET
        Path(f).write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(Path(f).name, len(cfg["aggregators"]), [a["name"] for a in attacks])


if __name__ == "__main__":
    main()
