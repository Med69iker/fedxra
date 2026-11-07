"""Analysis utilities for the Power-SNR experiments: statistics, figures, LaTeX tables.

Modules
-------
stats    : summarize, paired_tests, friedman_nemenyi
figures  : matplotlib figures (PDF + 300 dpi PNG), Okabe-Ito palette
tables   : booktabs LaTeX tables
make_paper_assets : CLI producing every figure/table available in a results.parquet
"""

KEY_COLS = ["experiment", "dataset", "model", "scheme", "K", "attack",
            "attack_frac", "attack_param", "seed"]

# Metric direction: True -> lower is better, False -> higher is better.
# Metrics absent from this map have no natural direction (e.g. raw ranks).
METRIC_LOWER_IS_BETTER = {
    "rmse_raw": True,
    "rmse_norm": True,
    "cosine": False,
    "kendall_tau": False,
    "spearman": False,
    "top3_jaccard": False,
    "top5_jaccard": False,
    "target_rank_shift": True,
    "w_honest": False,
    "w_minority": False,
    "w_attacker": True,
    "rmse_raw_minority": True,
    "rmse_norm_minority": True,
    "kendall_tau_minority": False,
    "attack_success": True,
}

# Display names of aggregators (unicode; tables.py converts to LaTeX).
PRETTY_NAMES = {
    "psnr_screen": "Power-SNR",
    "psnr_v3_fit": "FedXRA",
    "psnr_v3": "FedXRA-n",
    "psnr_v3_unif": "FedXRA-U",
    "trimmed_mean_0.3": "Trimmed mean (0.3)",
    "foolsgold": "FoolsGold",
    "centered_clipping": "Centered clipping",
    "bulyan": "Bulyan",
    "screen_only": "Screening only",
    "screen_geomedian": "Screening + geometric median",
    "psnr_screen_v1": "Power-SNR (screened v1)",
    "psnr_robust": "Power-SNR (robust v1)",
    "psnr_legacy": "Power-SNR (original)",
    "psnr_base": "Power-SNR (base)",
    "psnr_gate": "Power-SNR (gate)",
    "psnr_temp": "Power-SNR (temp.)",
    "psnr_energy_only": "Energy only",
    "mean": "Mean",
    "fedavg": "FedAvg",
    "median": "Median",
    "trimmed_mean": "Trimmed mean",
    "geomedian": "Geometric median",
    "krum": "Krum",
    "multikrum": "Multi-Krum",
    "ivw": "Inverse-variance",
    "dersimonian_laird": "DerSimonian–Laird",
    "fltrust_like": "FLTrust-like",
}

__all__ = ["KEY_COLS", "METRIC_LOWER_IS_BETTER", "PRETTY_NAMES"]
