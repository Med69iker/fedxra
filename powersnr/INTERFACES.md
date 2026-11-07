# Interface Contract (all agents must follow this exactly)

Package root: `powersnr/` (this folder). Python 3.10. Import as `from powersnr.types import ...`.
Shared dataclasses live in `powersnr/types.py`: `Dataset`, `ClientStats`, `AggResult`. **Do not modify types.py**; if a field is missing, use `extra` dicts / `info` dicts.

Randomness: every function that is random takes `rng: np.random.Generator` explicitly. Never call `np.random.seed` or module-level `np.random.*`.
Errors: never swallow exceptions silently (no bare `except:`). Fail loudly.
Tests: `tests/test_<module>.py`, runnable with `python -m pytest -q tests/test_<module>.py` from the `powersnr/` folder. Tests must be fast (< 30 s per file) and use synthetic data.

---

## powersnr/data.py  (Agent A)
```python
DATASETS = ["gmsc", "taiwan", "german", "homecredit", "lendingclub"]
def load_dataset(name: str, raw_dir: str | Path, max_rows: int | None = None, rng: np.random.Generator | None = None) -> Dataset
def make_synthetic(n: int, d: int, rng: np.random.Generator, n_informative: int = 4) -> Dataset   # groups={"age": ...}, protected_feature="age" (an informative column named "age")
def split_eval(ds: Dataset, frac: float, rng: np.random.Generator) -> tuple[Dataset, Dataset]       # stratified on y -> (pool, eval)
def subset(ds: Dataset, idx: np.ndarray) -> Dataset                                                  # positional indices
class LocalPreprocessor:   # fit on one client's training data only (median imputation)
    def fit(self, X: pd.DataFrame) -> "LocalPreprocessor"
    def transform(self, X: pd.DataFrame) -> pd.DataFrame
```
Raw files are expected in `data/raw/<name>/`. `scripts/download_data.py` downloads them.

## powersnr/partition.py  (Agent A)
```python
def partition(ds: Dataset, scheme: str, K: int, rng: np.random.Generator,
              client_size: int | None = None, **kw) -> tuple[list[np.ndarray], list[str]]
```
Returns (list of K disjoint positional index arrays into ds, list of K tags "standard"/"minority").
Schemes: `"iid"`; `"dirichlet"` (kw `alpha`); `"feature"` (kw `group`="age", `threshold`=60, `n_minority`=1: minority clients get rows with group > threshold, others get the rest IID); `"quantity"` (kw `sigma` lognormal size skew).
If `client_size` given, each client gets at most that many rows.

## powersnr/models.py  (Agent A)
```python
MODELS = ["xgboost", "lightgbm", "rf", "logreg", "mlp"]
def make_model(name: str, seed: int, **params)   # sklearn-compatible estimator with predict_proba
```

## powersnr/explain.py  (Agent A)
```python
def global_importance(model, X_explain: pd.DataFrame, X_background: pd.DataFrame, signed: bool = False) -> np.ndarray   # (d,) mean |SHAP| (or mean signed), class-1 logit/prob output
def client_stats(model, X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, client_id: int, role: str,
                 n_repeats: int = 5, n_explain: int = 1000, n_background: int = 100, test_frac: float = 0.2) -> ClientStats
```
`client_stats`: model is already fitted on the client's train split; X,y here are the client's held-out rows. mu = mean|SHAP| on up to n_explain rows; sigma2 = mean over n_repeats of ||mu_A - mu_B||_2^2 for random half-splits; also fills mu_signed, local_auc, runtime_s.
Explainer choice by model type: Tree models -> `shap.TreeExplainer(model, data=background, feature_perturbation="interventional")`; logreg -> `shap.LinearExplainer`; mlp -> `shap.KernelExplainer` (cap n_explain at 200). One explainer per call, no silent fallback.

Also:
```python
def fit_client(model_name: str, X: pd.DataFrame, y: np.ndarray, rng, seed: int, test_frac: float = 0.2)
    -> tuple[model, LocalPreprocessor, pd.DataFrame, np.ndarray]   # fitted model, preprocessor, X_test(preprocessed), y_test
```

## powersnr/aggregators.py  (Agent B)
```python
AGGREGATORS: dict[str, Callable]  # name -> function
def aggregate(name: str, mus: np.ndarray, sigma2: np.ndarray, n: np.ndarray, **kw) -> AggResult
```
mus (K,d) non-negative, sigma2 (K,), n (K,). weights returned normalized (sum 1) or NaN if rule is not weight-based.
Names: `mean, fedavg, median, trimmed_mean (beta=0.2), geomedian (Weiszfeld), krum (f), multikrum (f, m), ivw (eps), dersimonian_laird, fltrust_like (root_mu required), psnr_energy_only, psnr_base (eps), psnr_gate (eps, gate=0.1), psnr_robust (eps, cap=3.0, sigma2_floor: array|float|None, consistency=True), psnr_temp (eps, tau)`.
Definitions: signal_k = ||mu_k||_2^2; psnr_base w ∝ signal/(sigma2+eps); ivw w ∝ 1/(sigma2+eps).
psnr_robust: (1) sigma2_eff = max(sigma2, sigma2_floor) (if floor None: floor = median(sigma2 of all clients) * 0.1); (2) w = signal/(sigma2_eff+eps); (3) cap w_k <= cap * median(w); (4) if consistency: zero the weight of clients whose cosine to the capped-weighted consensus < 0 or is a >3 MAD low outlier; renormalize.
If all weights are zero -> fall back to `median` and set info["fallback"]=True.
Default eps = 1e-12 (scale-free relative to data), configurable.

## powersnr/attacks.py  (Agent C)
```python
DATA_ATTACKS = ["input_noise", "label_flip"]
REPORT_ATTACKS = ["frozen", "zero", "scaling", "feature_hiding", "feature_promotion", "mimic", "collusion", "adaptive_psnr"]
def apply_data_attack(name: str, X: pd.DataFrame, y: np.ndarray, rng, **params) -> tuple[pd.DataFrame, np.ndarray]
def apply_report_attack(name: str, stats: ClientStats, honest_reports: list[ClientStats], rng, *,
                        feature_names: list[str], target_feature: str | None = None, **params) -> ClientStats
```
Data attacks run before local training (input_noise: lambda * column std Gaussian noise; label_flip: prob p).
Report attacks replace the reported mu/sigma2 (return a NEW ClientStats with role="attacker"; keep n).
- frozen: mu = constant vector (uniform with the honest mean L1 mass), sigma2 = 0
- zero: mu ≈ 0 (1e-8), sigma2 = 0
- scaling: mu = lambda * own honest mu (lambda default 10), sigma2 unchanged
- feature_hiding: own mu with target_feature set to 0, mass redistributed proportionally to others
- feature_promotion: target_feature (least important if None) raised to 2x the max of own mu
- mimic: mean of honest mus + targeted shift toward hiding target_feature (strength param)
- collusion: identical to `feature_hiding` vector; the runner gives the same vector to all attackers (deterministic given honest_reports)
- adaptive_psnr: knows Power-SNR: sigma2 = 0 (or the declared floor param `sigma2_floor`), mu = the feature_hiding vector scaled so its energy is `boost` (default 10) x the max honest energy. Also accepts `cap` param: if given, scale so its weight equals exactly the cap boundary (worst case under psnr_robust).
Attacks with a target_feature: default to `Dataset.protected_feature` passed by the runner.

## powersnr/metrics.py  (Agent C)
```python
def fidelity_metrics(phi: np.ndarray, oracle: np.ndarray, feature_names: list[str], target_feature: str | None = None, ks=(3, 5)) -> dict[str, float]
```
Keys: `rmse_raw, rmse_norm (L1-normalized vectors), cosine, kendall_tau, spearman, top3_jaccard, top5_jaccard, target_rank_true, target_rank_est, target_rank_shift` (rank 1 = most important; NaN if no target).
```python
def weight_shares(weights: np.ndarray, roles: list[str]) -> dict[str, float]   # keys w_honest, w_minority, w_attacker (sum over role); NaN if weights NaN
def attack_success(metrics: dict, attack: str, threshold_rank_shift: int = 2) -> float  # 1.0/0.0; hiding: target dropped >= threshold ranks; promotion: rose >= threshold; others: rmse_norm > 2x reference passed via metrics["rmse_norm_ref"] if present else NaN
```

## analysis/  (Agent D)
Input: tidy results DataFrame read from `results/<run>/results.parquet` with columns:
`experiment, dataset, model, scheme, K, attack, attack_frac, attack_param, seed, aggregator,` + every fidelity_metrics key + `w_honest, w_minority, w_attacker, rmse_raw_minority, rmse_norm_minority, kendall_tau_minority, attack_success, fallback`.
- `analysis/stats.py`: `summarize(df, by, metric) -> DataFrame` (mean, std, n, 95% bootstrap CI); `paired_tests(df, by, metric, reference="psnr_robust", lower_is_better=True) -> DataFrame` (Wilcoxon signed-rank over seeds vs each other aggregator, Holm-corrected p, Cliff's delta, win count); `friedman_nemenyi(df, block_cols, metric) -> (stat, p, avg_ranks, cd)`.
- `analysis/figures.py`: functions each saving vector PDF + PNG(300dpi) into an output dir, colorblind-safe palette (Okabe–Ito), mean line + 95% CI band: `plot_metric_vs(df, x, metric, hue="aggregator", ...)`, `plot_weight_shares(df, x, ...)`, `plot_cd_diagram(avg_ranks, cd, ...)`, `plot_pareto(df, x_metric, y_metric, ...)`, `plot_feature_bars(phi_table, ...)`.
- `analysis/tables.py`: `to_latex_table(summary_df, ..., bold_best=True, lower_is_better=True) -> str` producing booktabs LaTeX with mean ± std; write to file.
- `analysis/make_paper_assets.py`: CLI `python -m analysis.make_paper_assets --run results/<run> --out paper_assets/` that produces all figures/tables available in the results.
