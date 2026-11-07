# Run 7 (confirmation of FedXRA with fit_ts weights, seeds 5000+, run on Kaggle 2026-10-31)

Rule under test: `psnr_v3_fit` = run-6 FedXRA with weights 1 / (tau2 + a / n_k) fitted by Theil-Sen on the
screened banks (`weight: fit_ts`) and duplicate threshold ratio 1. Designed on development seeds (research
rounds 9-10, research/NOTES.md); not tuned on these seeds. Full tables: research/out/run7_summary.md
(research/analyze_run7.py). Kaggle tests: 195 passed, 1 failed (test_missing_raw_file_message, as in runs 5-6).

## Verdict: confirmed
- Accuracy rank over E9 E10 E2 E5 E3 E4 E6 E7 (all 17 rules): FedXRA-fit mean 5.93, worst 8.04 (E4) -
  best mean and best worst case of all rules. Run-6 FedXRA: mean 7.84, worst 12.00 (E7).
- Cost: E9 4.50 vs 3.10 and E10 4.43 vs 3.50 (still 2nd-4th of 17). Gains: E3 6.89 vs 11.24, E4 8.04 vs
  11.11, E7 6.44 vs 12.00, E5 6.43 vs 9.24, E6 6.89 vs 8.71. E1 4.78 vs 6.50, E11 7.05 vs 7.44.
- Decision metrics (no attack): top-5 Jaccard and Kendall tau equal or higher than run-6 FedXRA in 6 of 8 experiments.
- Attacks: inflated-n coalition clearly weaker (E9 0.40 vs 0.475; E6 0.42 vs 0.56); strong white-box equal
  within 0.03 except E3 at 20-30% (0.29/0.48 vs 0.19/0.33); collusion, mimicry, ALIE unchanged.
- E8 ablation: the fit_ts weights and the dd1 threshold each contribute little alone (dd1 with n weights 8.98,
  fit_ts with dd4 9.40, both 9.12, run-6 9.58 among the E8 variants).

## SHA-256 of the result files

- `033a380b5acd9870439b157ee9917b75f8606890c334fe7340a3d4cd72dc0d4a`  kaggle_runs/run7/A/results/20261031-215226_E1/client_reports.parquet
- `3c2a161f0e2f32db16b0f161735ec0f37d2b243202f2d83acf1ece230d8871e7`  kaggle_runs/run7/A/results/20261031-215226_E1/phi_vectors.parquet
- `092c3901d10a94ba17e8143fbc524ff34fbba048463c8be2c5b1dcc7191156bd`  kaggle_runs/run7/A/results/20261031-215226_E1/results.parquet
- `269085d068fc177631b6ec558cc9367a4a021babe08c50bd12fc8feec2cbe50a`  kaggle_runs/run7/A/results/20261031-220452_E2/client_reports.parquet
- `6de412cacca3fa7640f40eebf303218c2ef9de8c76f6245c501a2da1338aecb5`  kaggle_runs/run7/A/results/20261031-220452_E2/phi_vectors.parquet
- `126a07caf87c69c38f4f85720711132a96e411e27fd9818a25d601233a0ab26b`  kaggle_runs/run7/A/results/20261031-220452_E2/results.parquet
- `3f5be0dd259754e0e34bff9670592c8e197dbe4b1ccf4963c52e5509861e8667`  kaggle_runs/run7/A/results/20261031-220858_E7/client_reports.parquet
- `d4ef46d77a702f9a157dc2154028a3bd9d7927a681ddc9dfe858bdaeb7a636ef`  kaggle_runs/run7/A/results/20261031-220858_E7/phi_vectors.parquet
- `11bb92623b865738460ce62a5956ed586f3e3169d3c45c6d6b7caa8a058dbca0`  kaggle_runs/run7/A/results/20261031-220858_E7/results.parquet
- `19857a475660735e7e76a7e5845b25e3dc5099eccddd633d8cfa2afb8f19c3d7`  kaggle_runs/run7/A/results/20261031-222135_E9/client_reports.parquet
- `cddbfd70cdf94503aa1c15f1909d6397f5d3e69809f948ced8dbd609f4830217`  kaggle_runs/run7/A/results/20261031-222135_E9/phi_vectors.parquet
- `bcdbb8600e5479a4eb0e9d407e844f510499f98fa4915774b55586b149c18a1f`  kaggle_runs/run7/A/results/20261031-222135_E9/results.parquet
- `beb721e971ecf90a4dec41ef2f9a1d49c18b2a06aa95b1062c7e2df5db63a2b8`  kaggle_runs/run7/B/results/20261031-215235_E3/client_reports.parquet
- `216ac0bde7796b09a174bfcf9e5c6a9d0d6b349da8e953676e10694ac157ae2b`  kaggle_runs/run7/B/results/20261031-215235_E3/phi_vectors.parquet
- `adb138f1d1de70ab1338cd117999fc3eac81ea25be2e939fadf8bbb05ddfc1aa`  kaggle_runs/run7/B/results/20261031-215235_E3/results.parquet
- `82709e3a441e701aa60da64751607333ef0ae605fd73d5cc381b74eccb4d1cc9`  kaggle_runs/run7/B/results/20261031-221306_E4/client_reports.parquet
- `d240ac742777cecdde759245bdcf2e052f277adebfd0961f889c11fd714893f8`  kaggle_runs/run7/B/results/20261031-221306_E4/phi_vectors.parquet
- `690ca4ba3308f64b9e06826aefd69d6d5869e23b9e5422794f9c4cee83422b58`  kaggle_runs/run7/B/results/20261031-221306_E4/results.parquet
- `fa57c159bc31b94237442d84ab9d2ff76bf7a2d0e151241281cc062ac0f62800`  kaggle_runs/run7/B/results/20261031-222354_E5/client_reports.parquet
- `4d9b388d2d37c298eda2fa81680222c8edb15a0452523ec4b75847fd1f946113`  kaggle_runs/run7/B/results/20261031-222354_E5/phi_vectors.parquet
- `f1a981879c37d3c9dcab91a93ec1e5b36f76c27540c036189b62ee3cc5807d16`  kaggle_runs/run7/B/results/20261031-222354_E5/results.parquet
- `4a2922715fc0a56ce2667230a82ca36ea57de5f4801ab922cbdb3a93c89033a1`  kaggle_runs/run7/B/results/20261031-224855_E8/client_reports.parquet
- `fb348c8b17bbfab58c8c217deebcb2f477086c0b7d7618737ceb52b5ab6191d3`  kaggle_runs/run7/B/results/20261031-224855_E8/phi_vectors.parquet
- `ce655d7df12d9fe4eca3b651564f03838611037ca57a5189aabdf9387e72f136`  kaggle_runs/run7/B/results/20261031-224855_E8/results.parquet
- `c151a3961f0fce5555869f67bdd90238f4992223ae2bb3cc1aa360948f2a4bf9`  kaggle_runs/run7/B/results/20261031-234338_E10/client_reports.parquet
- `bb8ab0847780b5099d3a82ac11f32b30cf6da248f550dc4f7d2164b9cf0cf818`  kaggle_runs/run7/B/results/20261031-234338_E10/phi_vectors.parquet
- `abc386986807974a9d72c237f531bccf148010feee2a72f768679cddb3b7d8ae`  kaggle_runs/run7/B/results/20261031-234338_E10/results.parquet
- `1d3143652d9f13d5e5cd7b59093a70534590e9405c901e0df68ec346eb50ed10`  kaggle_runs/run7/C/results/20261031-215247_E6/client_reports.parquet
- `5a6c0a1e9bab0c8ea0beee9c672ffbaff846c608dd038811c3c402534f39fa33`  kaggle_runs/run7/C/results/20261031-215247_E6/phi_vectors.parquet
- `22b763ca41313fc0f09f6ee5fba87745f3985e6725324156a0ca3536edeafa07`  kaggle_runs/run7/C/results/20261031-215247_E6/results.parquet
- `64303037f7526c719d309e427abd0d42ff09faf09259da5ace89da0a6fbfb7d9`  kaggle_runs/run7/C/results/20261031-221345_E11/client_reports.parquet
- `f3ff808b2214687a3cee98bf1171ba593719e4c7ba9d91fc45002f6f400da9e0`  kaggle_runs/run7/C/results/20261031-221345_E11/phi_vectors.parquet
- `f0ed3524213519ad467a139e6a248749f43c85c8c9aa153bc8192c73683076a2`  kaggle_runs/run7/C/results/20261031-221345_E11/results.parquet

## E11 re-run (2026-11-05, downloaded and swapped 2026-11-06)
Configuration error found after the run: in E11 (400 explained rows), FedXRA (`psnr_v3_fit`) had n_cap 1000 instead of
400. E11 was re-run on Kaggle (private notebook `powersnr-exp-e11`, same seeds 5000–5009, guard printed
`psnr_v3_fit n_cap = 400`). Checks: config differs only in that n_cap; 13 600 rows, same keys; every other rule
identical row for row (16 rules); only psnr_v3_fit rows differ (654 of 800; mean nRMSE x1e3 41.94 -> 41.89);
client_reports.parquet byte-identical. The old folder is kept, unmodified, in `kaggle_runs/run7_superseded/C/results/`;
the new one is `run7/D/results/` (raw download in `kaggle_runs/run7_e11/`). Tests in the notebook: 194 passed, 2 failed
(test_missing_raw_file_message, as in run 7; test_style_mapping_distinct, a plot line-width check, also fails locally;
neither touches results).
- `64303037f7526c719d309e427abd0d42ff09faf09259da5ace89da0a6fbfb7d9`  kaggle_runs/run7/D/results/20261105-221945_E11/client_reports.parquet
- `2f1ae73a8cf18d3174b150736b8087e4f7a3b7a82ebd7170905f398449e34cc9`  kaggle_runs/run7/D/results/20261105-221945_E11/config.yaml
- `04bda7e15c4556f81b34582da841b9e81b2ec4f0eb3ba346d85a156b8858ac2a`  kaggle_runs/run7/D/results/20261105-221945_E11/phi_vectors.parquet
- `1efa69229064e25dae407ac914f66a889d5ee112fab0e0d090f8c96fd62a46b5`  kaggle_runs/run7/D/results/20261105-221945_E11/results.parquet
- `44f296e06313604a45b2c1d4daa4af675ad01ca2dc9b9675da860fdcf8123db1`  kaggle_runs/run7/D/results/20261105-221945_E11/run_meta.json
