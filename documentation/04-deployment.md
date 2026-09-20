# 4. Deployment guide

What ships, how to serve it, and what has to be true before it does.

## Before you deploy: read this

This model predicts the dataset's `estimated_lifetime_value` column, which is itself an estimate
and — per EDA §7 — a near-deterministic function of the six input columns. **It is not a 12-month
forecast**, and its near-perfect scores are formula recovery. Deploying it is reasonable for
*tiering customers by an existing value estimate*; it is not reasonable to present its output as a
prediction of future spend. See [06-review.md](06-review.md#what-the-numbers-do-and-do-not-mean).

## 1. What gets deployed

One file: `outputs/model.joblib`, written by `train.py`.

```python
{
  "model":   LogTargetRegressor(Pipeline([("prep", CLVPreprocessor), ("model", Ridge)])),
  "name":    "Polynomial on logs + Ridge",
  "config":  PreprocessConfig(...),          # every preprocessing knob used
  "input_features": [...6 raw column names...],
  "target":  "estimated_lifetime_value",
  "test_scorecard": {...},                   # the scores it earned, for the record
  "tier_edges": [0, 0.5, 0.8, 0.95, 1.0],
  "tier_labels": ["Standard (bottom 50%)", ..., "VIP (top 5%)"],
  "train_prediction_quantiles": {"0.5": ..., "0.8": ..., "0.95": ...},
}
```

The bundle is self-contained: preprocessing (imputation fills, clip bounds, scaler, polynomial
expansion), the estimator, the Duan smearing factor, the input contract and the tier thresholds all
travel together. A model without its smearing factor and thresholds is not deployable, and a bundle
that can drift apart from its metadata will.

## 2. Build it

```bash
pip install -r requirements.txt
python main.py           # must exit 0: 55 checks, this is the gate
python train.py          # writes outputs/model.joblib
pytest tests/ -q         # 139 tests
```

`main.py` returning non-zero means the data failed its own checks — do not build a model from it.
`train.py` enforces the same gate and stops before modelling.

## 3. Serve it

### Batch (the supported path)

```bash
python predict.py --input new_customers.csv --output scored.csv
```

Input needs the six raw columns in any order; extra columns are carried through untouched. Output
adds `predicted_clv` and `tier`.

### Behind an API

```python
from fastapi import FastAPI
import pandas as pd
from src.evaluation import load_bundle
from predict import score            # the same scoring path as the CLI

app = FastAPI()
BUNDLE = load_bundle("outputs/model.joblib")     # load once, at startup

@app.post("/score")
def score_customers(rows: list[dict]):
    frame = pd.DataFrame(rows)
    scored = score(frame, BUNDLE)                # raises ValueError on a missing column
    return scored[["predicted_clv", "tier"]].to_dict(orient="records")
```

Three things this gets right and a hand-rolled version usually does not:

- **Load the bundle once**, not per request. It carries a fitted preprocessor.
- **Use `predict.score`**, not `bundle["model"].predict` — the tier assignment and the column
  contract live in that function.
- **Let the `ValueError` propagate** as a 400. A missing feature must be rejected, not imputed into
  a plausible-looking answer.

Single-row latency is ~9 µs of prediction on top of preprocessing; the pipeline is verified to
produce identical output for one row and for a batch (a check that runs on every pipeline run).

## 4. Tier thresholds are fixed at training time

`train_prediction_quantiles` holds the dollar cut-offs learned on the training split, and
`predict.py` applies those. It does **not** re-derive quantiles per batch — otherwise a customer's
tier would depend on who else happened to be scored that day, and "VIP" would mean something
different every run. A retention budget can only be attached to a stable definition.

## 5. Unseen-data safety

| Situation | What happens |
|---|---|
| A feature is missing | `ValueError` naming the column — rejected, not guessed |
| A value is `NaN` | Imputed with the **training** median |
| An unseen category | Recorded and handled; the row still scores |
| A value of 0 or negative in a logged column | Floored, so `log()` stays finite |
| An absurd value (1e9) | Clipped to the training range widened by 50% — a bounded prediction instead of an extrapolated one |
| Extra columns | Ignored |

These are not aspirations: each is a check in `check_transformer_behaviour`, run on every pipeline
run.

## 6. Configuration

Everything is in `PreprocessConfig`, serialised beside the model as
`outputs/preprocess_config.json`. To deploy a different representation, change the config and
rebuild — never edit a fitted bundle.

Useful production-shaped flags:

```bash
python main.py --recency-policy clip          # repair impossible rows instead of flagging
python main.py --outlier-quantiles .001 .999  # winsorise (costs accuracy — see readme §16.2)
python train.py --no-tune                     # faster rebuild, takes the best default candidate
python main.py --cv-strategy timeseries       # when you move to transactional data
```

## 7. Environment

| Requirement | Note |
|---|---|
| Python ≥ 3.10 | tested on 3.12.10 |
| scikit-learn ≥ 1.3 | hard floor: the preprocessor uses `set_output(transform="pandas")` |
| pandas ≥ 2.0, numpy ≥ 1.24, scipy ≥ 1.10, joblib ≥ 1.3 | |
| xgboost ≥ 2.0 | **optional** — gradient boosting falls back to `HistGradientBoosting` |
| matplotlib ≥ 3.7 | figures only; Agg backend, no display needed |

Pin the versions you build with. A `joblib` bundle is not guaranteed to load across scikit-learn
minor versions — rebuild rather than unpickle across an upgrade.

### Containerising

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY predict.py main.py train.py ./
COPY outputs/model.joblib outputs/
CMD ["python", "predict.py", "--input", "/data/in.csv", "--output", "/data/out.csv"]
```

Copy the *bundle*, not `data/`. If the image must be able to retrain, copy `data/` and run
`main.py && train.py` at build time so a failing check fails the build.

## 8. Deployment checklist

- [ ] `python main.py` exits 0 — all 55 checks pass
- [ ] `pytest tests/ -q` — 139 passed
- [ ] `outputs/model.joblib` rebuilt from the current code, not carried over
- [ ] `test_scorecard` in the bundle matches the run you intend to ship
- [ ] Interpreter has the pinned versions; xgboost present *or* its absence accepted
- [ ] Scoring smoke test on a held-back file: `python predict.py --input sample.csv --top 5`
- [ ] Tier thresholds reviewed with whoever owns the retention budget
- [ ] Monitoring in place ([05-operations.md](05-operations.md))
- [ ] The caveat in §"Before you deploy" is written into whatever the business reads

## 9. What is deliberately not here

No model registry, no feature store, no online serving, no A/B harness. This is a capstone
pipeline with a clean batch interface; those belong to whatever platform adopts it. The bundle
format and `predict.score` are the seams to build them against.
