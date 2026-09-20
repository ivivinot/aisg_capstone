# 6. Documentation review

What is verified, what is claimed, and where the two are not the same thing.

## Verification run

Everything below was executed against the current working tree, not recalled.

| Check | Result |
|---|---|
| `pytest tests/` | **139 passed** in 80 s |
| `python main.py` | exit 0 — **55 checks**: 52 passed, 0 failed, 3 skipped |
| `python train.py` | exit 0 |
| `python predict.py` | exit 0 |
| Every `src/*.py` standalone | 8/8 exit 0 |
| `python -m src.<module>` | 7/7 exit 0 |
| Docstring coverage | **104/104** exported symbols |

The three skips are correct behaviour, not gaps: the derived-feature checks skip when no derived
columns were requested (the default).

## Docstring coverage

Measured by introspecting `__all__` in each module and reading `inspect.getdoc`:

| Module | Exports | Documented |
|---|---|---|
| `preprocessing` | 24 | 24 |
| `feature_engineering` | 13 | 13 |
| `models` | 17 | 17 |
| `advanced_models` | 13 | 13 |
| `model_optimization` | 17 | 17 |
| `metrics` | 9 | 9 |
| `evaluation` | 8 | 8 |
| **Total** | **104** | **104** |

Docstrings here carry *reasoning*, not restatement: why `median` rather than `mean`, why the clip
bounds are the training min/max widened 50%, why VIF is the wrong tool on a polynomial basis. Each
module header maps EDA findings to the code that answers them, and several carry the measured cost
of the alternative.

## Codebase at a glance

| | Lines |
|---|---|
| `src/` (7 modules) | 6,622 |
| `tests/` (4 modules) | 1,718 |
| `documentation/` (6 pages) | 866 |
| Entry points (`main`, `train`, `predict`) | ~1,700 |

## What the numbers do and do not mean

The single most important thing in this documentation set.

`estimated_lifetime_value` is **already an estimate**, not observed spend, and EDA §7 shows it is a
near-deterministic smooth function of the six feature columns — cross-validated R² rises to 0.99994
as polynomial degree increases, with residual spread falling to 0.4%. That cannot happen with real
customer behaviour.

Therefore:

| Claim | Status |
|---|---|
| "The pipeline recovers the generating function almost exactly" | **Supported** — test Spearman 0.9999, residual spread 0.7% in log space, held-out optimism +0.0006 |
| "Representation beats architecture on this data" | **Supported** — established three independent ways: the candidate zoo (§15.5), the paired test (§15.3, p = 0.0070 against boosting), and the 1-SE selection (§15.4) |
| "This predicts 12-month customer value" | **Not supported.** No timestamps, no cut-off date, no future window. A deployed CLV model reports Spearman ≈ 0.56 |
| "These scores would transfer to real customers" | **Not supported.** The data is synthetic, with fractional purchase counts and negative recencies |
| "The loyalty programme has no effect" | **Not supported as a business claim** — it describes the generator, not a real programme |

Every generated report repeats the caveat, and `main.py` prints it at the end of every run. That is
deliberate: a number this good is more likely to be misquoted than a mediocre one.

## Where the documentation could mislead, and does not

- **§15.4's optimization numbers come from a run with xgboost installed.** Without it, gradient
  boosting falls back to `HistGradientBoosting` — different parameter names, different tuned score.
  The conclusion (the baseline wins) holds in both; the specific row does not. The tuning table
  records the renamed and dropped parameters on every run.
- **The "one-standard-error rule" chose linear-on-log over ridge by 0.0002.** That is the rule
  working as designed, not evidence that linear is better. The honest statement is "they are
  indistinguishable and one is simpler".
- **Clustering numbers on the blob fixtures are near-perfect** because the fixtures are three
  well-separated Gaussians. They demonstrate the plumbing, not the difficulty of real segmentation.
- **`fit_time_ratio` is wall-clock on one machine.** Comparable within a run, not to your hardware.

## Known limitations

1. **Not a forecast.** The temporal design that defines CLV prediction cannot be built from this
   file.
2. **Random split, not temporal.** Correct for a cross-section, wrong for a forecasting claim.
3. **No uncertainty quantification.** One point prediction per customer; intervals would need a
   distributional model (readme §5.1).
4. **Metrics are at their ceiling**, which makes them useless for separating a good model from a
   great one on this data.
5. **The search is not nested.** Tuning and selection share folds with evaluation, so the CV score
   is optimistic. This is the standard shortcut, it is documented, and `validate_final_model`
   measures the resulting optimism against a held-out split rather than assuming it away.
6. **Clustering has no held-out validation** — there is no such thing. Stability across subsamples
   is the substitute.

## Documentation map

| Page | Covers | Audience |
|---|---|---|
| [README.md](README.md) | the index of this folder | anyone arriving |
| [01-architecture.md](01-architecture.md) | module map, data flow, the five design decisions | a developer changing the code |
| [02-api-reference.md](02-api-reference.md) | 104 public symbols, signatures, a worked non-CLV example | someone calling the library |
| [03-models.md](03-models.md) | every model, its assumptions, the measured results, how to choose | a reviewer or a data scientist |
| [04-deployment.md](04-deployment.md) | the bundle, serving, safety, config, checklist | whoever ships it |
| [05-operations.md](05-operations.md) | monitoring, retraining, interpretation, troubleshooting, runbook | whoever is on call |
| [06-review.md](06-review.md) | what is verified, what is claimed, the limits | an assessor |

The root [`../readme.md`](../readme.md) is the report and the single entry point: literature
review (§1–12), findings (§13–17), the objective, the quick start, the project layout and this
index. These pages do not duplicate it — they point at it for *why* and cover *how*.

## Still open

Two housekeeping items, neither affecting behaviour:

- **Nothing is committed.** The capstone repo also tracks `__pycache__/`; a `.gitignore` covering
  `__pycache__/`, `outputs/` and `*.joblib` belongs in the first commit.
- **`readme.md` §15.4** would benefit from a one-line note that the boosting row depends on whether
  xgboost is installed — the same caveat made above.
