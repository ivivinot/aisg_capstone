# 1. Architecture

## The shape of it

Two entry points over one library. `main.py` owns the data and refuses to hand it on quietly if a
check fails; `train.py` reuses every one of `main.py`'s stages and then does the modelling that is
specific to this project.

```
                         main.py                            train.py
data/synthetic_data_126.csv
        │
        ▼
  1  load + audit ─────────────► outputs/data_audit.json            ▲
  2  row filtering  (opt-in; nothing removed by default)            │
  3  split 80/20 stratified on value deciles ── test split sealed   │
  4  CLVPreprocessor.fit(train)                                     │
        │   encode → rules → impute → clip │ derive → log → poly → scale → select
        ▼                                                           │
  5  FEATURE ENGINEERING     created columns, selection             │
  6  OUTPUT VALIDATION        8 checks                              │
  7  TRANSFORMER TESTS       12 checks                              │   the gate:
  8  FEATURE TESTS            7 checks                              │   train.py stops
  9  STATISTICAL VALIDATION   5 checks                              │   if any failed
 10  BASELINE LADDER          6 checks  ─► baseline_leaderboard.csv │
 11  ADVANCED MODELS          8 checks  ─► model_comparison.csv     │
 12  MODEL OPTIMIZATION       9 checks  ─► hyperparameter_search.csv│
        │                                                           │
        ▼                                                           │
 13  artifacts ───────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                                            14  model selection (8 candidates)
                                            15  randomised search over 3 finalists
                                            16  refit winner, score test split ONCE
                                            17  importance, value tiers
                                            18  artifacts ─► model.joblib → predict.py
```

## Modules

| Module | Owns | Lines |
|---|---|---|
| `src/preprocessing.py` | The data contract, audit, row filtering, the split, the cleaning transformers, the `CLVPreprocessor` facade — **and the 25 checks that validate its output**. | ~1540 |
| `src/feature_engineering.py` | The catalogue of engineered columns, `DerivedFeatures`, the log/expand/scale steps, `FeatureSelector`, and the measurements that say whether any of it helped. | ~900 |
| `src/models.py` | The **baseline ladder** for all three tasks: naive floors, simple model baselines, cross-validated evaluation, the metric guide and the cost benchmark. | ~1130 |
| `src/advanced_models.py` | **Two further architectures per task**, the paired fold-by-fold comparison against the ladder, the corrected significance test, and the generated Markdown report. | ~930 |
| `src/model_optimization.py` | **Cross-validation strategy, hyperparameter search and the over/underfit diagnosis** for every model above, plus the one-standard-error selection and the single held-out validation. | ~1050 |
| `src/metrics.py` | The scorecard: Spearman, normalized Gini, top-decile capture, decile MAPE, MAE, R², and the value tiers. | ~240 |
| `src/evaluation.py` | Held-out scoring of a chosen model, both importance measures, tiers, figures, and the deployable bundle. | ~450 |

## Five decisions that explain the rest

**1. The preprocessor is a fitted scikit-learn transformer, not a script.** Imputation fills, clip
bounds, scaler statistics and the polynomial expansion are all learned, so `CLVPreprocessor` drops
into a `Pipeline` and is **refitted inside every cross-validation fold**. Leakage would have to
come from the split itself. The alternative — transform once, then cross-validate — reports scores
that cannot be reproduced in production.

**2. A model is a `(representation, estimator)` pair.** EDA §7 found the target multiplicative;
EDA §9–10 found that representation beat model choice decisively. So each estimator is paired with
the feature space it deserves — logged and expanded for the linear family, raw for trees — and the
search ranges over the representation too.

**3. Checks live beside the code they check, and run on every run.** 55 of them, in four modules,
collected rather than raised so one run reports every failure. `main.py` exits non-zero on any
failure; `train.py` refuses to model on data that failed.

**4. One optimization protocol for every model.** Baselines and challengers are tuned, diagnosed
and selected by the same code in `src/model_optimization.py`. A protocol applied to half the field
cannot be used to choose between the halves.

**5. Every module runs standalone.** `python src/models.py` executes a self-contained smoke test on
synthetic data and exits 0 or 1. This is why each file carries an import shim:

```python
if __package__ in (None, ""):        # run as a script: no parent package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.metrics import scorecard
else:
    from .metrics import scorecard
```

## Dependency direction

```
preprocessing  ←── feature_engineering
      ↑                    ↑
      │                    │ (lazy, inside CLVPreprocessor._build)
      ├──────────── models ─────────── metrics
      │               ↑  ↑
      │               │  └── advanced_models ──┐
      │               │                        │
      └──────── model_optimization ←───────────┘
                      ↑
             main.py ─┴─ train.py ── evaluation ── predict.py
```

`preprocessing` is the lower layer and never imports upward at module level. `CLVPreprocessor._build`
reaches for `feature_engineering.build_feature_steps` lazily, which is what keeps the cycle from
closing at import time.

## Task-agnostic by construction

`infer_task(y)` reads the target — none means clustering, few distinct values means classification,
otherwise regression — and every stage dispatches on it:

| | regression | classification | clustering |
|---|---|---|---|
| CV scheme | decile-stratified K-fold | stratified K-fold | none; subsample stability |
| Headline metric | Spearman | F1-macro | silhouette |
| Naive floors | mean, median, heuristic | most-frequent, stratified, uniform | one cluster, random labels |
| Model baselines | linear, ridge, tree, k-NN | logistic, tree, k-NN | k-means, agglomerative |
| Architectures 2 & 3 | gradient boosting, MLP | gradient boosting, MLP | Gaussian mixture, DBSCAN |

The capstone exercises the regression column; the other two are tested against synthetic fixtures
and are why the project is described as non-regression-capable rather than regression-only.
