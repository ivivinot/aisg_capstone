# 2. API reference

Every public symbol, by module — 104 in total, all with docstrings (verified, see
[06-review.md](06-review.md#docstring-coverage)). Signatures are abbreviated; the authoritative
version is `help(obj)` or the source, where each entry carries the reasoning behind the default.

Conventions: `X` is a `DataFrame` of raw customer rows, `y` a `Series` target (or `None` for
clustering), `task` one of `"regression" | "classification" | "clustering"`, and every function
that fits something accepts a `cv` splitter so the caller controls the protocol.

---

## `src.preprocessing` — data contract, cleaning, validation

### Constants
| Symbol | Meaning |
|---|---|
| `TARGET` | `"estimated_lifetime_value"` |
| `SKEWED_NUMERIC` | the five continuous features, all logged before modelling |
| `CATEGORICAL_MAP` | `{"loyalty_program_membership": {"Enrolled": 1, "Not Enrolled": 0}}` |
| `EXPECTED_FEATURES_BY_DEGREE` | `{1: 7, 2: 35, 3: 119}` — the representation's fingerprint |
| `EXPECTED_CV_R2`, `CV_R2_TOLERANCE` | `0.99936 ± 5e-4`, asserted on every run |
| `SHUFFLE_R2_CEILING` | `0.05` — a label-shuffled pipeline must score below this |

### Configuration and the pipeline

```python
PreprocessConfig(target=TARGET, numeric_features=SKEWED_NUMERIC, ..., random_state=42)
```
Every knob in one serialisable dataclass: imputation, the recency policy, outlier bounds, the log
transform, `poly_degree`, scaling, derived features, feature selection, split sizes. `.to_json(path)`
persists it. Defaults reproduce the representation that won model selection.

```python
CLVPreprocessor(config=None)                 # scikit-learn transformer
    .fit(X, y=None) .transform(X) .get_feature_names_out()
    .quality_report() -> dict                # fills, bounds, flags, selection
    .save(path) / CLVPreprocessor.load(path)
```
Nine steps: encode → domain rules → impute → clip → **derive → log → expand → scale → select**.
The second half is built by `feature_engineering.build_feature_steps`. Pass `y` to `fit` when a
supervised selection strategy is configured.

```python
LogTargetTransformer(smearing=True)
    .fit(y) .transform(y) .fit_smearing(residuals) .inverse_transform(y_log)
```
`log(y)` for fitting, Duan's smearing estimator for the way back — `exp(E[log y])` under-states
`E[y]` for a skewed target.

**Component transformers** (usually reached through `CLVPreprocessor`):
`CategoricalEncoder`, `DomainRuleTransformer` (the 52 impossible-recency rows: flag / clip / none),
`FrameImputer` (fills learned on train only), `QuantileClipper` (an extrapolation guard, not tail
removal).

### Loading, auditing, splitting
```python
load_raw(path, target=TARGET) -> DataFrame           # validates the columns exist
audit_dataset(frame, target=TARGET) -> dict          # the EDA §2 checks, machine-readable
filter_rows(frame, config=None, drop_invalid_recency=False, ...) -> (DataFrame, dict)
stratified_split(frame, config=None) -> (train, test)   # 80/20 on value deciles
get_logger(name="capstone", level=INFO)  /  quiet(level=WARNING)   # context manager
```

### Validation
```python
Check(name, status, detail="", group="")             # status: PASS | FAIL | SKIP
check_output_contract(X_train, X_test, y_train, y_test, prep) -> List[Check]   # 8
check_transformer_behaviour(prep, X_train_raw, X_test_raw, y_train_log=None) -> List[Check]  # 12
check_statistics(train_raw, config=None, cv_folds=5, seed=42) -> List[Check]   # 5
summarise(checks) -> dict                            # counts + the names that failed
```

---

## `src.feature_engineering` — creation and selection

```python
FeatureSpec(name, inputs, formula, kind, new_in_log_space, rationale, build)
FEATURE_SPECS: Dict[str, FeatureSpec]                # the 8-feature catalogue
DEFAULT_DERIVED / INDEPENDENT_DERIVED / ALL_DERIVED   # named sets
resolve_derived("all" | "a,b" | [...]) -> Tuple[str, ...]
```
`new_in_log_space` is the field that matters: `False` means the feature is a product or ratio, so
after the log step it is a linear combination of columns the model already has.

```python
DerivedFeatures(features=DEFAULT_DERIVED, floor=1e-3)     # appended before the log
SafeLogTransformer(columns=SKEWED_NUMERIC, floor=1e-3)    # the core normalisation
FeatureSelector(strategy="variance", k=None, correlation_threshold=0.999, vif_threshold=6.0)
SELECTION_STRATEGIES = ("none", "variance", "correlation", "mutual_info", "model", "vif")
build_feature_steps(cfg) -> List[Tuple[str, object]]      # used by CLVPreprocessor._build
```

```python
feature_report(X, y=None, top=None) -> DataFrame          # std, zero share, Spearman, VIF
compare_feature_sets(train_raw, config, cv_folds=5) -> DataFrame   # Ridge vs Random Forest
check_feature_engineering(prep, X_train_raw, X_test_raw, y_train_log=None) -> List[Check]  # 7
```

---

## `src.models` — the baseline ladder (all tasks)

```python
TASKS = ("regression", "classification", "clustering")
HEADLINE_METRIC = {"regression": ("spearman", True), "classification": ("f1_macro", True),
                   "clustering": ("silhouette", True)}
infer_task(y=None, max_classes=20) -> str
has_xgboost() -> bool                    # find_spec, no import
```

```python
BaselineSpec(name, task, kind, rationale, build, log_target=False, raw_input=False)
BASELINE_SPECS: Tuple[BaselineSpec, ...]         # 18 baselines across 3 tasks
build_baselines(task="regression", preprocessor=None, include=None, **kw) -> Dict[str, estimator]
```
`kind` is `"naive"` (learns one number; makes the metrics readable) or `"model"` (simple but real;
the bar a complex model must clear). `raw_input=True` takes a baseline *out* of the preprocessing
pipeline — a constant predictor ignores `X`, and a domain heuristic needs the raw columns.

```python
evaluate_baselines(X, y=None, task=None, preprocessor=None, cv=None,
                   benchmark=True, benchmark_repeats=3, **kw) -> BaselineResults
```
The whole ladder in one call: build → cross-validate → analyse → benchmark → check.

```python
BaselineResults(task, board, predictions, benchmark, analysis, checks)
    .best -> str          .to_dict() -> dict
```

```python
cross_validate_model(name, model, X, y=None, cv=None, task="regression") -> (row, oof)
score_predictions(name, task, y_true=None, y_pred=None, **extra) -> dict
benchmark_baselines(models, X, y=None, task="regression", repeats=3) -> DataFrame
analyse_baselines(board, task) -> dict            # winner per metric, lift, the bar
check_baselines(results, X, y=None) -> List[Check]                             # 6
METRIC_GUIDE: Dict[str, Dict[str, str]]           # question / direction / how it misleads
```

**Estimators this module provides:** `LogTargetRegressor` (fit on `log(y)`, predict in units with
smearing; `.predict_log()`, `.smearing_factor_`) and `ColumnProductRegressor` (the analyst's
`purchases × AOV`, as a proper estimator so it can be cross-validated like everything else).

---

## `src.advanced_models` — architectures 2 and 3, and the comparison

```python
ModelSpec(name, task, architecture, bias, cost, build, needs_scaling=False)
ADVANCED_SPECS: Tuple[ModelSpec, ...]              # 2 per task, distinct inductive biases
architecture_table(task=None) -> DataFrame
build_advanced_models(task="regression", preprocessor=None, include=None, **kw) -> Dict
```

```python
run_comparison(X, y=None, task=None, preprocessor=None, cv=None, baselines=None,
               tune=False, n_iter=20, report_path=None, **kw) -> AdvancedResults
```
Baselines (reused or run) → advanced models → paired comparison → analysis → checks → report.
The single call `main.py` makes.

```python
evaluate_advanced(...) -> AdvancedResults          # fit, score, time; nothing compared
fold_scores(name, model, X, y=None, cv=None, task=..., n_subsamples=5) -> np.ndarray
compare_models(baselines, advanced, X, y=None, cv=None, reference=None) -> DataFrame
paired_test(differences, n_folds=None) -> dict     # corrected resampled t-test + Wilcoxon
analyse_comparison(baselines, advanced, comparison) -> dict     # verdict, error reduction
document_results(results, path=None) -> str        # the Markdown report
check_advanced_models(results, baselines) -> List[Check]                       # 8
ALPHA = 0.05
```

`compare_models` returns, per model: `score`, `fold_mean`, `delta`, `delta_sd`, `folds_won`,
`reliable`, `p_value`, `significant`, `fit_time_ratio`. `score` pools every out-of-fold prediction
(the deployment question); `fold_mean` ranks within folds (the paired-test question). They can
disagree, and the analysis says so rather than picking the flattering one.

---

## `src.model_optimization` — CV, tuning, over/underfitting, selection

```python
CV_STRATEGIES = ("auto", "kfold", "stratified", "repeated", "shuffle", "timeseries")
make_cv(task="regression", y=None, strategy="auto", n_splits=5, n_repeats=3, random_state=42)
cv_report(cv, X, y=None, task="regression") -> DataFrame   # what each fold received
```
`auto` gives regression **decile-stratified** folds, classification stratified folds, clustering
`None`.

```python
SEARCH_SPACES: Dict[str, Callable[[], Dict[str, dict]]]     # per task, every tunable model
SEARCH_METHODS = ("random", "grid", "halving")
search_space_for(name, task="regression") -> Optional[dict]
scoring_for(task)                       # the headline metric itself, not a proxy
tune_models(models, X, y=None, task=..., cv=None, method="random", n_iter=20,
            spaces=None, random_state=42, n_jobs=-1) -> (Dict[str, estimator], DataFrame)
PARAM_SYNONYMS: Dict[str, Tuple[str, ...]]   # XGBoost ↔ HistGradientBoosting names
```
`tune_models` adapts parameter names to whatever shape the estimator has (bare, pipeline, or
wrapped), renames across implementations, drops parameters with no counterpart **and reports the
drop** — while a genuine typo still raises.

```python
diagnose_fit(model, X, y=None, task=..., cv=None, name="model") -> dict
    # train vs CV score → verdict: overfitting | underfitting | balanced, plus the action
learning_curve_report(model, X, y=None, fractions=(0.2,...,1.0)) -> DataFrame
validation_curve_report(model, X, y=None, param_name="model__alpha", param_range=...) -> DataFrame
select_within_one_se(scores, standard_errors, complexity=None, higher_is_better=True) -> (str, dict)
COMPLEXITY_RANK: Dict[str, int]         # what "simpler" means, since arithmetic cannot know
OVERFIT_GAP = 0.05  /  UNDERFIT_SCORE = 0.60
```

```python
optimise_models(models, X, y=None, task=None, cv=None, strategy="auto", method="random",
                n_iter=20, diagnose=True, learning_curves=True) -> OptimizationResults
select_final_model(results, X, y=None, complexity=None) -> (name, model)
validate_final_model(results, X_train, y_train, X_test, y_test) -> dict   # reports optimism
check_optimization(results) -> List[Check]                                # 9
```

---

## `src.metrics` — the scorecard

```python
scorecard(name, y_true_raw, y_pred_raw, y_true_log=None, y_pred_log=None) -> dict
leaderboard(rows, sort_by="spearman") -> DataFrame
normalized_gini(y_true, y_pred) -> float
decile_mape(y_true, y_pred, n_deciles=10) -> float     # calibration, not point accuracy
top_decile_capture(y_true, y_pred) -> float            # ceiling is the data's concentration
value_tiers(actual, predicted, edges=TIER_EDGES, labels=TIER_LABELS) -> DataFrame
SCORE_COLUMNS, TIER_EDGES, TIER_LABELS
```

---

## `src.evaluation` — held-out scoring and the deployable bundle

```python
fit_final_model(model, X_train, y_train) -> LogTargetRegressor
evaluate_on_test(final, X_test, y_test, name) -> (row, pred_raw, pred_log)
drop_column_importance(model, X, y_log, cv=None, columns=None) -> DataFrame
permutation_scores(model, X_test, y_test_log, n_repeats=30, seed=42) -> DataFrame
tier_report(y_test, pred_test) -> DataFrame
save_bundle(path, final, metadata) -> Path   /   load_bundle(path) -> dict
write_figures(outdir, leaderboard, y_test, pred_test, pred_test_log, importance, tiers) -> List[Path]
```

Prefer **drop-column** importance on a high-degree polynomial: shuffling one column invents feature
combinations the fitted surface never saw, and permutation importance then ranks the wrong features
first. Both are computed so the disagreement stays visible.

---

## Worked example: this library on someone else's problem

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models import evaluate_baselines, infer_task
from src.advanced_models import run_comparison
from src.model_optimization import make_cv, optimise_models, validate_final_model

X = pd.read_csv("churn.csv"); y = X.pop("churned")     # a classification problem
print(infer_task(y))                                    # -> 'classification'

cv = make_cv("classification", y)                       # stratified folds
baselines = evaluate_baselines(X, y, preprocessor=StandardScaler(), cv=cv)
print(baselines.board, baselines.analysis["bar_for_a_real_model"])

comparison = run_comparison(X, y, preprocessor=StandardScaler(), cv=cv,
                            baselines=baselines, report_path="comparison.md")
print(comparison.analysis["verdict"])

models = {**baselines.models, **comparison.models} if hasattr(baselines, "models") else {}
optimised = optimise_models(models or {}, X, y, task="classification", cv=cv)
print(validate_final_model(optimised, X_train, y_train, X_test, y_test)["verdict"])
```

Nothing in that snippet is CLV-specific. The capstone's own columns appear only in
`src.preprocessing`, which you would replace with your own transformer.
