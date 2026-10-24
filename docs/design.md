# Cloud Workload Forecasting — Design (Option B)

Original design, written before implementation on 2026-10-23. Where the implementation diverged from
it, `devlog.txt` records the decision.

## 1. Intent

The solution has four aims:

1. rigour — every trap in the data found and proven with numbers;
2. modern methods — global ML and time-series foundation models (TSFMs), benchmarked honestly;
3. an operational policy an infra team could adopt;
4. production-shaped, reproducible engineering.

Sophistication must earn its place against simple baselines with statistical evidence, a forecast is
treated as an input to a provisioning decision rather than the decision itself, and the policy pairs a
predictive baseline with a reactive floor, a rollback trigger and human sign-off. All inference runs
locally; no external forecasting API is used.

## 2. Facts established from `data-v1` (probe run 2026-10-23)

| Fact | Value | Consequence |
|---|---|---|
| Grid | 134 timestamps, 0..39,900 s, step 300 s, no dups/NaN | clean, but the contract is still enforced in code |
| VMs per timestamp | exactly 10,000 at every timestamp | `active_vm_count` is constant; persistence is exact |
| Lifetimes | all `timestamp_vm_created = 0`, none deleted in window | no churn; constancy is a selection artefact |
| Aggregate CPU | mean 9.39 %, sd 0.195 pp, range 8.98–9.91 | tiny signal; differences need significance tests |
| Autocorrelation | lag-1 = 0.68 | short-memory, mean-reverting |
| Regime | bump to ≈9.8 % around steps 53–72 | the peak/anomaly case for error analysis |
| Train vs holdout mean | 9.41 vs 9.27 | mild downward drift in holdout |
| Metadata `avg_cpu`,`max_cpu`,`p95_max_cpu` | 30-day lifetime stats | **leak** — excluded from features |
| `timestamp_vm_deleted` | min 40,500 s (just past window) | **leak** — excluded from features |

## 3. Goals, non-goals, anti-goals

Goals
- Required `predictions.csv` (24 rows, contract in SUBMISSION.md) from one command.
- Three model families compared under one leakage-safe rolling-origin protocol:
  baselines/classical, global LightGBM, zero-shot TSFM.
- Calibrated prediction intervals and a capacity policy built on them.
- Notebook and a 4–6 page report.

Non-goals
- A data platform, streaming, a model registry, a service. Scaling is *argued* in the report, not built.
- Deep-learning models trained from scratch (110 aggregate points; not justified).

Anti-goals (failure even if it "works")
- Any use of the final 24 targets for fitting, feature selection, tuning or model selection.
- Claiming a model is better on a single metric or without a significance test.
- Hard-coding 10,000 / 134 / 110 anywhere — the reviewer smoke dataset must run.
- Committing data, caches or large artefacts.

## 4. Constraints (from ASSIGNMENT/DATASET/SUBMISSION)

Holdout = last 24 timestamps of whatever data is supplied; train = everything before. Features
computable at the forecast origin only. No random splits. Baseline on same scheme. Metrics: MAE,
RMSE, normalised/relative, peak-period error, failure analysis. Output finite, count ≥ 0,
CPU in [0,100], no index column. Python ≥ 3.11. State every assumption.

## 5. Targets

Required (built from the full series, then split):
- `active_vm_count(t)` = distinct `vm_id` with a valid reading at t. "Valid" = `avg_cpu` finite and in [0,100].
- `aggregate_cpu_utilization(t)` = mean of valid `avg_cpu` at t. Invalid readings are dropped, not
  imputed; a timestamp with zero valid readings is a gap and is reported, not invented.
  Missing grid timestamps are detected and reported; models see the gap as missing.

Extensions (analysis only, not in `predictions.csv`):
- `core_weighted_cpu(t)` = Σ avg_cpu·cores / Σ cores — what capacity actually tracks.
  Bucket `>24` needs an assumed core count (32); a sensitivity check (32 vs 48) is reported.
- `peak_core_demand(t)` = Σ max_cpu·cores / 100 — busy-core upper envelope for headroom.

## 6. Validation protocol

- Holdout: last H = 24 steps. Never touched until the single final evaluation.
- Model selection on rolling origins inside the training window: horizon 24, first origin at 48,
  step 2 → origins 48, 50, …, 86 → 20 folds on the 110-step window (parametrised; derived from
  data length, not hard-coded).
- For panel models, a training row for horizon h at origin o' is allowed in fold with origin o only if
  o' + h ≤ o (its label is observed by the origin). This is the subtle leak most pipelines miss.
- Existing `validation.py` is fixed: integer-second timestamps (`unit="s"`), index-based splits,
  raise instead of silently returning `[]`.
- Significance: Diebold–Mariano with Harvey small-sample correction on per-fold losses, plus a
  moving-block bootstrap CI on the MAE difference. Folds overlap, so this is reported as indicative.

## 7. Models

Baselines / classical (industry-standard floor): persistence, rolling mean (window chosen on folds),
drift, AutoETS and AutoARIMA (`statsforecast`). Daily seasonal-naive is not applicable (11 h of data);
short seasonality is tested via ACF and stated.

Global LightGBM (bottom-up)
- One model over the VM panel; direct multi-horizon with horizon as a feature.
- Features at origin: per-VM lags (1–12) of avg/min/max, rolling mean/sd (6/12/24), EWM,
  fleet-level lags (cross-sectional context), static metadata known at creation
  (category, core bucket, memory bucket). Excluded: lifetime CPU stats, deletion time.
- Row budget: 10k VMs × every origin × 24 horizons ≈ 20M rows per fold, too many. Training origins
  are strided (every 6 steps) → ≈2–3M rows per fold; stride is a parameter, reported.
- L2 objective on per-VM level; aggregate = mean of VM predictions (MSE-consistent for a mean).
  VM-level quantiles do not add up to fleet quantiles, so intervals are computed at fleet level (§8).
- Explainability: gain importance + SHAP summary in the report.

Zero-shot TSFM (local, GPU with CPU fallback)
- Primary: Chronos-Bolt (Amazon). Secondary if time allows: TimesFM (Google).
- Run on the aggregate series and bottom-up over VM series; same folds.
- Before trusting any TSFM, check its published pretraining corpus for Azure/cloud VM traces;
  contamination is reported. Latency and memory per 1,000 series are measured (GPU and CPU).

Selection rule: lowest mean fold MAE for each target; if not significantly better (DM, 5 %) than a
simpler candidate, the simpler one wins (parsimony). The rule is coded, so the CLI applies it
deterministically.

`active_vm_count`: same protocol is run; persistence is expected to win exactly. The report explains
why the target is constant and why that is a property of the sample.

## 8. Uncertainty and policy

- Split-conformal intervals per horizon from fold residuals of the selected model (80/90/95 %),
  empirical coverage reported on folds and on the holdout. Only 20 residuals exist per horizon, too few
  for a stable 95 % quantile, so residuals are pooled across neighbouring horizons (h±2) and the
  finite-sample correction ⌈(n+1)(1−α)⌉/n is applied. Adaptive conformal (ACI) is discussed as the
  production variant for drift.
- Policy: provision to the upper conformal quantile of core-weighted demand, plus a reactive floor
  (scale out if observed demand exceeds the forecast band). The quantile is set per segment:
  Interactive (latency-sensitive) higher than Delay-insensitive.
- Trade-off curve on folds: over-provisioned core-hours vs breach rate per quantile level, with a
  stated cost ratio assumption. Rollback trigger: rolling coverage below target for N steps → fall
  back to reactive scaling. Human sign-off on policy changes.

## 9. Metrics and error analysis

MAE, RMSE, MASE (scale = in-sample one-step naive MAE), relative MAE vs persistence, pinball loss and
coverage for intervals, error by horizon step, error on top-quartile-load timestamps, and the 53–72
regime as a case study. sMAPE is reported with an explicit epsilon note (no near-zero denominators here).

## 10. Engineering

Layout (extends the existing `cwfc` package; fewest files):
```
src/cwfc/data.py        load + data contract checks
src/cwfc/targets.py     target construction
src/cwfc/features.py    origin-safe panel features
src/cwfc/models.py      baselines/classical, LightGBM, TSFM adapters
src/cwfc/backtest.py    rolling-origin engine + selection rule (replaces validation.py logic)
src/cwfc/metrics.py     + MASE, pinball, coverage, DM test
src/cwfc/policy.py      conformal intervals + capacity policy
src/cwfc/predict.py     CLI: python -m cwfc.predict --data challenge_data --output predictions.csv
notebooks/analysis.ipynb   narrative, imports cwfc only
reports/final_report.md    4–6 pages + appendices; figures in reports/figures/
```
- Dependencies: core = numpy, pandas, lightgbm, statsforecast, scipy, matplotlib; optional extra
  `[fm]` = torch + chronos-forecasting (+ timesfm). If the selected model needs `[fm]` and it is not
  installed, the CLI exits with a clear message (no silent change of model).
- Tests: data contract; target construction; split correctness; **holdout-perturbation test**
  (changing the final 24 targets leaves `predictions.csv` byte-identical — proves no leakage);
  **smoke test** on a tiny synthetic dataset with churn and gaps; output contract test.
- `.gitignore` extended to exclude data files in `challenge_data/` (keep its README), caches,
  notebook checkpoints, model artefacts.

## 11. Design assumptions

- Keep 5-minute cadence (15-min would leave ≈37 points).
- Extra targets core-weighted CPU and peak core demand; `>24` bucket assumed 32 cores.
- 20 folds, origin 48, step 2.
- Parsimony tie-break via DM test at 5 %.
