# 5. Operations

Running it, watching it, and fixing it.

## Monitoring

Three families, in the order they tend to fail.

**1. Input drift — watch first, it moves first.**

| Signal | Source | Act when |
|---|---|---|
| Share of scored rows hitting a clip bound | `QuantileClipper` bounds in `preprocess_config.json` | > 1% — the population has moved past the training range |
| Feature medians and skew vs the audit | `outputs/data_audit.json` | a median moves > 20%, or skew changes sign |
| Missing-value rate per column | your scoring job | any column newly missing at all — the training data had none |
| Unseen categories | logged by `CategoricalEncoder` | any occurrence |

**2. Prediction drift — cheap, and needs no labels.**

Compare each batch's predicted distribution with the training one: median, top-decile share, and
the tier mix. The tier mix is the most legible: if "VIP" stops being ~5% of customers, either the
population changed or the model did.

**3. Accuracy — only once labels exist.**

Rank quality (Spearman, normalized Gini) and calibration (decile MAPE) fail *independently*. A
model can keep ranking correctly while its levels drift, which matters if the numbers feed a
forecast rather than a targeting list. Track both.

## Retraining

Retrain when any of these is true, not on a calendar:

- clipped-value share above 1% for a week
- decile MAPE on a labelled sample doubles from its held-out value (0.0019)
- Spearman on a labelled sample drops below the heuristic baseline (0.877) — at that point the
  model is no longer earning its complexity
- the schema changes in any way

```bash
python main.py && python train.py     # the gate, then the model
```

`main.py` exiting non-zero blocks the retrain, which is the point. Compare the new bundle's
`test_scorecard` with the old before swapping, and keep the previous `model.joblib` until the new
one has scored a real batch.

On real transactional data, retrain **per cohort** and use `--cv-strategy timeseries`: a temporal
split is the only valid design once timestamps exist (readme §7.1).

## Interpreting predictions for a stakeholder

Use the **elasticities**, not the coefficients of the degree-3 model:

| Feature | Elasticity | Reading |
|---|---|---|
| `average_order_value` | **0.86** | a 1% larger basket is worth ~0.86% more value — the dominant lever |
| `days_since_last_purchase` | −0.32 | recency decay; lapse prevention is the second lever |
| `total_purchase_count` | 0.29 | strongly diminishing: doubling purchases does *not* double value |
| `days_since_first_purchase` | −0.015 | near zero alone, real through interactions |
| `loyalty_program_membership` | ~0 | contributes nothing |

Drop-column importance agrees and is the one to trust. **Permutation importance is misleading on
this model** — shuffling a column of a degree-3 polynomial invents combinations the surface never
saw — and it ranks the date features above `average_order_value`, the reverse of the truth. Both
are computed so the disagreement is visible rather than accidental.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: attempted relative import with no known parent package` | A library module was run as a script from an older checkout | Each `src/*.py` now has an import shim; run `python src/<module>.py` or `python -m src.<module>` from `capstone/` |
| `ValueError: ... search parameters not present on the estimator` | A search space names a parameter the estimator lacks | Known cross-library names are translated by `PARAM_SYNONYMS`; a genuine typo is meant to raise |
| `gradient boosting` scores differently on another machine | xgboost present vs absent — the fallback is `HistGradientBoosting` | Pin xgboost, or accept it: the tuning table records the renamed and dropped parameters |
| `Import "xgboost" could not be resolved` in the editor | The editor's interpreter differs from the run interpreter | *Python: Select Interpreter* → the one with the packages |
| `main.py` exits 1 | A check failed — this is the gate working | Read the `FAILED:` line; every check names what it measured |
| A check fails only on new data | The data moved | Compare `data_audit.json` between runs before touching thresholds |
| Predictions all near the mean | The heuristic or a naive baseline was deployed by mistake | Check `bundle["name"]` |
| Scores much better in CV than on held-out data | Selection overfitted the folds | `validate_final_model` reports this as *optimism*; a large positive value is the diagnosis |

## Cost and runtime

| Command | Time | Notes |
|---|---|---|
| `python main.py` | ~24 s | includes 55 checks and three model stages |
| `python main.py --no-optimize --no-advanced` | ~3 s | data and checks only |
| `python train.py` | ~41 s | adds the candidate zoo and the search |
| `pytest tests/ -q` | ~62 s | 139 tests |
| `python predict.py` | < 1 s | dominated by loading the bundle |

Prediction is ~9 µs per row; the fitted bundle is ~87 KB.

## Runbook: a failed nightly scoring job

1. `python predict.py --input <the batch> --top 5` — reproduce it. A `ValueError` naming a column
   is a schema change upstream; fix the source, do not patch the model.
2. If it scores but the numbers look wrong, compare the tier mix with the last good run.
3. If the tier mix moved, check clipped-value share and the feature medians against
   `data_audit.json`.
4. If the inputs are unchanged, re-run `python main.py` against the training data — a difference
   there means the environment changed (library versions), not the data.
5. Only then consider a retrain, and gate it on `main.py` exiting 0.
