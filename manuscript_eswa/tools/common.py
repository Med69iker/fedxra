"""Settings shared by the manuscript generators: the evidence run, the benchmark rules and their names.

Evidence: Kaggle run 7 (seeds 5000+, kaggle_runs/run7/FINAL_RESULTS.md). Config keys -> paper names:
    psnr_v3_fit  -> FedXRA     (proposed rule: screened, fitted-variance weights, trimmed combination)
    psnr_v3      -> FedXRA-n   (same rule with capped sample-size weights; the run-6 design)
    psnr_v3_unif -> FedXRA-U   (same rule with equal weights; for consortia that cannot audit n)
    psnr_screen  -> Power-SNR  (earlier design)
Number keys use rule_key(): fedxra, fedxra_n, fedxra_u, psnrv2, trimmed_mean_0p3, ...
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CODE = HERE.parent / "powersnr"
RUN_NAME = "run7"
RUN = CODE / "kaggle_runs" / RUN_NAME
SEED0 = 5000
REF = "psnr_v3_fit"

MAIN = ["psnr_v3_fit", "psnr_v3", "psnr_v3_unif", "psnr_screen", "fedavg", "median", "trimmed_mean",
        "trimmed_mean_0.3", "geomedian", "krum", "multikrum", "bulyan", "centered_clipping", "foolsgold",
        "fltrust_like", "ivw", "dersimonian_laird"]
FEDXRA_FAMILY = ("psnr_v3_fit", "psnr_v3", "psnr_v3_unif")

NAMES = {"psnr_v3_fit": r"\textbf{FedXRA}", "psnr_v3": "FedXRA-n", "psnr_v3_unif": "FedXRA-U",
         "psnr_screen": "Power-SNR", "fedavg": "FedAvg", "median": "Median",
         "trimmed_mean": r"Trimmed mean ($\beta{=}0.2$)", "trimmed_mean_0.3": r"Trimmed mean ($\beta{=}0.3$)",
         "geomedian": "Geometric median", "krum": "Krum", "multikrum": "Multi-Krum", "bulyan": "Bulyan",
         "centered_clipping": "Centered clipping", "foolsgold": "FoolsGold", "fltrust_like": "FLTrust-like",
         "ivw": "Inverse-variance", "dersimonian_laird": "DerSimonian--Laird"}
PLAIN = {k: v.replace(r"\textbf{", "").replace("}", "").replace(r"($\beta{=", "(").replace(r"$)", ")")
         for k, v in NAMES.items()}
PLAIN.update({"trimmed_mean": "Trimmed mean", "trimmed_mean_0.3": "Trimmed mean (0.3)"})


def rule_key(a: str) -> str:
    """Config key -> number-key fragment (psnr_v3_fit* -> fedxra*, psnr_v3* -> fedxra_n*, '.' -> 'p')."""
    if a.startswith("psnr_v3_fit"):
        a = "fedxra" + a[len("psnr_v3_fit"):]
    elif a == "psnr_v3_unif":
        a = "fedxra_u"
    elif a.startswith("psnr_v3"):
        a = "fedxra_n" + a[len("psnr_v3"):]
    a = a.replace("psnr_screen", "psnrv2")
    return a.replace(".", "p")
