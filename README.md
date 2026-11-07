# FedXRA: robust aggregation of SHAP explanations from unreliable banks

Code, configurations and results for the article *Trustworthy federated learning explanations for credit scoring:
aggregating SHAP explanations from unreliable banks* (submitted to Expert Systems with Applications).

## Contents

| Path | What it holds |
|---|---|
| `powersnr/powersnr/` | The package: data loading and partitions, local models and SHAP reports, attacks, the 17 aggregation rules (FedXRA in `psnr_v3_fit`), metrics, experiment runner |
| `powersnr/configs/` | One YAML file per experiment (E1–E11) and the development configurations |
| `powersnr/tests/` | Unit tests (`pytest`) |
| `powersnr/scripts/` | Data download and helper scripts |
| `powersnr/kaggle/` | Notebook builders and kernel folders used to run the experiments on Kaggle CPU containers |
| `powersnr/kaggle_runs/run7/` | The final run (seeds 5000+): per-experiment results, saved bank reports, run metadata and logs; `D/` is the re-run of the model-family experiment (E11) |
| `manuscript_eswa/evidence/MANIFEST.json` | SHA-256 hash of every result file used in the article |
| `manuscript_eswa/tools/` | Scripts that verify the manifest and generate every number, table and figure, and the numerical checks of the propositions (`check_theory.py`) |

The raw datasets are not redistributed (Kaggle and UCI terms); `powersnr/scripts/download_data.py` fetches them.

## Quick start

```bash
pip install -r powersnr/requirements.txt
cd powersnr && python -m pytest -q && cd ..
cd manuscript_eswa
python tools/evidence.py verify      # checks the hashes of the run-7 result files
python tools/check_theory.py         # numerical checks of the propositions
python tools/make_numbers.py         # every number quoted in the article
```
