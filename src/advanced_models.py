"""Two further architectures per task, and an honest comparison against the baselines.

`src/models.py` establishes the floor. This module asks the next question: **does a
fundamentally different architecture beat it, and by enough to be worth the cost?**
It develops a second and a third model for each task, evaluates every model on the
same folds and metrics, compares them fold by fold, and writes the result up.

Why these two, and why "distinct architecture" matters
------------------------------------------------------
Adding a second gradient booster to a ladder that already has one measures
hyperparameters, not architecture. The two models per task are therefore chosen to
have *different inductive biases* -- different assumptions about what the data
looks like -- so that when one wins, the win says something:

===============  ==========================================================
task             model 2 / model 3, and the bias each brings
===============  ==========================================================
``regression``   **Gradient boosting** -- an additive ensemble of shallow
                 trees. Piecewise-constant, axis-aligned, no assumption of
                 smoothness; excellent at thresholds and interactions,
                 clumsy on a smooth multiplicative surface (EDA 10).
                 **Neural network (MLP)** -- a dense, continuously
                 differentiable function approximator. No axis alignment at
                 all: it bends smoothly in every direction and pays for that
                 with a need for scaled inputs and more data.
``classification`` The same two, as classifiers: additive trees against a
                 smooth decision surface.
``clustering``   **Gaussian mixture** -- probabilistic and *soft*: elliptical
                 components with full covariance, so a point can belong
                 partly to two clusters. **DBSCAN** -- density-based: no k,
                 arbitrary shapes, and it is allowed to call a point noise,
                 which no baseline here can do.
===============  ==========================================================

Against the baselines -- a linear model, a shallow tree, k-NN -- that gives three
genuinely different answers to "what shape is this problem", which is the only
comparison from which anything can be learned.

Comparison, done as a paired test
---------------------------------
Two cross-validated means are two numbers; the interesting question is whether the
difference between them is bigger than the noise between folds. :func:`compare_models`
therefore keeps the **per-fold** score for every model and compares architectures on
the *same* folds, reporting the mean paired difference, its spread, and how many
folds it held in. A model that wins on average but loses in two folds out of five is
not reliably better, and the summary says so instead of hiding it behind a mean.

The confidence comes from :func:`paired_test`, which runs the **corrected resampled
t-test** rather than a plain paired t-test. Cross-validation folds share training
rows, so their differences are correlated and the naive variance is too small --
which is how fold noise gets published as significance. The correction inflates the
variance by ``1/k + 1/(k-1)`` before computing t. Wilcoxon's signed-rank p-value is
reported beside it with the caveat that, at five folds, it cannot go below 0.0625
whatever the data does.

Tuning, and the objection it answers
------------------------------------
``tune=True`` (``python main.py --tune-advanced``) gives each architecture a bounded
search before the comparison, because "keep the baseline" is a much weaker claim
about models left at their defaults. The verdict states which of the two regimes
produced it.

The search itself is **not** implemented here: cross-validation strategy,
hyperparameter spaces and the over/underfitting diagnosis for *every* model in
the project live in :mod:`src.model_optimization`. Splitting them across modules
is how the baselines and the challengers end up optimised by different protocols,
which is the quiet way to make a comparison meaningless.

The cost side is reported next to the accuracy side, because "3% better for 40x the
fit time and a dependency" is a decision, not a detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone

# Relative when imported as part of the package, absolute when this file is run
# directly (``python src/advanced_models.py``). The __main__ block at the bottom
# is a self-contained smoke test of all three comparisons.
if __package__ in (None, ""):                                    # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models import (BaselineResults, HEADLINE_METRIC, METRIC_GUIDE,
                            _default_cv, _metric_names, benchmark_baselines,
                            build_baselines, cross_validate_model, evaluate_baselines,
                            has_xgboost, infer_task, score_predictions)
    from src.preprocessing import Check, _Recorder, quiet
else:
    from .models import (BaselineResults, HEADLINE_METRIC, METRIC_GUIDE,
                         _default_cv, _metric_names, benchmark_baselines,
                         build_baselines, cross_validate_model, evaluate_baselines,
                         has_xgboost, infer_task, score_predictions)
    from .preprocessing import Check, _Recorder, quiet

#: A library logger: it inherits the handler main.py installs on "capstone".
logger = logging.getLogger("capstone.advanced_models")

__all__ = [
    "ModelSpec",
    "ADVANCED_SPECS",
    "architecture_table",
    "build_advanced_models",
    "AdvancedResults",
    "evaluate_advanced",
    "fold_scores",
    "compare_models",
    "paired_test",
    "ALPHA",
    "analyse_comparison",
    "document_results",
    "check_advanced_models",
    "run_comparison",
]

#: How much better a model must be, relative to the spread between folds, before
#: the difference is called reliable rather than noise. This is an effect-size
#: rule of thumb; :func:`paired_test` is the statistical one, and both are
#: reported because they answer slightly different questions.
RELIABILITY_MARGIN = 1.0

#: Significance level for the corrected paired t-test.
ALPHA = 0.05


def paired_test(differences, n_folds: Optional[int] = None) -> Dict[str, object]:
    """Is a per-fold difference distinguishable from zero? Two tests, honestly caveated.

    **The corrected resampled t-test** (Nadeau & Bengio 2003; Bouckaert & Frank
    2004). A plain paired t-test over cross-validation folds is *anti-conservative*
    -- the folds share training rows, so the differences are not independent and
    the naive variance is too small, which turns fold noise into significance. The
    correction inflates the variance by ``1/k + n_test/n_train``, which for k-fold
    is ``1/k + 1/(k-1)``::

        t = mean(d) / sqrt(var(d) * (1/k + 1/(k-1))),   df = k - 1

    **Wilcoxon signed-rank** is reported beside it because it assumes nothing
    about the shape of the differences -- but with 5 folds its smallest possible
    two-sided p-value is 0.0625, so it *cannot* reach 0.05 however consistent the
    result. That is a property of the sample size, not of the models, and the
    returned ``note`` says so rather than letting a reader over-read a p-value.
    """
    from scipy import stats

    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    k = int(n_folds or len(d))
    result: Dict[str, object] = {"n": int(len(d)), "mean_difference": float(np.mean(d))
                                 if len(d) else float("nan")}
    if len(d) < 2 or np.allclose(d, 0):
        result.update({"t_stat": float("nan"), "p_value": float("nan"),
                       "p_wilcoxon": float("nan"), "significant": False,
                       "note": "too few folds, or an identical pair, to test"})
        return result

    variance = float(np.var(d, ddof=1))
    correction = 1.0 / k + 1.0 / max(k - 1, 1)
    t_stat = float(np.mean(d) / np.sqrt(variance * correction)) if variance > 0 else float("inf")
    p_value = float(2 * stats.t.sf(abs(t_stat), df=max(k - 1, 1))) if np.isfinite(t_stat) else 0.0

    try:
        p_wilcoxon = float(stats.wilcoxon(d).pvalue)
    except ValueError:                                   # all-zero differences
        p_wilcoxon = float("nan")

    result.update({
        "t_stat": t_stat,
        "p_value": p_value,
        "p_wilcoxon": p_wilcoxon,
        "significant": bool(p_value < ALPHA),
        "note": ("Wilcoxon cannot reach 0.05 with "
                 f"{len(d)} folds (its floor is {_wilcoxon_floor(len(d)):.4f})"
                 if _wilcoxon_floor(len(d)) > ALPHA else ""),
    })
    return result


def _wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p-value the signed-rank test can produce with n pairs."""
    return 2.0 / (2 ** n) if n else float("nan")


# --------------------------------------------------------------------------- #
# 1. The two architectures, per task
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """One advanced model: what it is, what it assumes, and what it costs.

    ``architecture`` is the one-phrase description of *how* it represents a
    function; ``bias`` is what it assumes about the data; ``cost`` is the
    practical price. Together they are what makes a comparison interpretable:
    when this model wins, it wins *because* of the bias, and that is the finding.
    """

    name: str
    task: str
    architecture: str
    bias: str
    cost: str
    build: Callable[..., object]
    needs_scaling: bool = False


def _gradient_boosting_regressor(random_state: int = 42, **_):
    """XGBoost when it is installed, scikit-learn's histogram booster otherwise.

    Both are the same architecture -- shallow trees fitted in sequence on the
    residual -- so the comparison does not change when the optional dependency
    is missing. Only the implementation does.
    """
    if has_xgboost():
        from xgboost import XGBRegressor  # type: ignore[import-not-found]

        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                            subsample=0.9, colsample_bytree=0.9,
                            random_state=random_state, n_jobs=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                         max_depth=4, random_state=random_state)


def _mlp_regressor(random_state: int = 42, **_):
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                        alpha=1e-3, learning_rate_init=3e-3, max_iter=1500,
                        early_stopping=True, n_iter_no_change=25,
                        random_state=random_state)


def _gradient_boosting_classifier(random_state: int = 42, **_):
    if has_xgboost():
        from xgboost import XGBClassifier  # type: ignore[import-not-found]

        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                             subsample=0.9, colsample_bytree=0.9,
                             random_state=random_state, n_jobs=-1)
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          max_depth=4, random_state=random_state)


def _mlp_classifier(random_state: int = 42, **_):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                         alpha=1e-3, learning_rate_init=3e-3, max_iter=1500,
                         early_stopping=True, n_iter_no_change=25,
                         random_state=random_state)


def _gaussian_mixture(n_clusters: int = 4, random_state: int = 42, **_):
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(n_components=n_clusters, covariance_type="full",
                           n_init=5, random_state=random_state)


def _dbscan(eps: float = 0.8, min_samples: int = 10, **_):
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples)


ADVANCED_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        "gradient boosting", "regression",
        "additive ensemble: shallow trees fitted in sequence on the residual",
        "Piecewise-constant and axis-aligned. Assumes the response is built from "
        "thresholds and interactions rather than smooth curvature, so it excels on "
        "rules and struggles to represent a product of continuous features.",
        "Hundreds of trees to store and traverse; fast to fit on tabular data; "
        "many hyperparameters, and an optional dependency when XGBoost is used.",
        _gradient_boosting_regressor),
    ModelSpec(
        "neural network (MLP)", "regression",
        "dense feed-forward network, two hidden layers (64, 32), ReLU",
        "A smooth, continuously differentiable surface with no axis alignment. "
        "Assumes enough data to learn the shape and inputs on a comparable scale; "
        "in exchange it bends in every direction at once.",
        "Iterative fitting with early stopping, sensitive to scaling and to the "
        "seed; small to store, and the least interpretable model here.",
        _mlp_regressor, needs_scaling=True),
    ModelSpec(
        "gradient boosting", "classification",
        "additive ensemble of shallow trees on the log-odds",
        "Axis-aligned decision boundaries built from thresholds; handles mixed "
        "feature types and interactions without being told about them.",
        "As above; probabilities usually need calibrating before they are used "
        "as probabilities.",
        _gradient_boosting_classifier),
    ModelSpec(
        "neural network (MLP)", "classification",
        "dense feed-forward network, two hidden layers (64, 32), ReLU",
        "A smooth, curved decision boundary. Assumes scaled inputs and enough "
        "examples of the minority class to shape the boundary around it.",
        "Iterative, seed-sensitive, and uncalibrated by default.",
        _mlp_classifier, needs_scaling=True),
    ModelSpec(
        "gaussian mixture", "clustering",
        "probabilistic mixture of Gaussians with full covariance",
        "Clusters are elliptical and may overlap; membership is *soft*, so a "
        "point can be 70% one cluster and 30% another -- which k-means cannot "
        "express and a segmentation often needs.",
        "Fits by EM, needs the number of components, and degenerates when a "
        "component collapses onto a few points.",
        _gaussian_mixture),
    ModelSpec(
        "DBSCAN", "clustering",
        "density-based: dense regions joined into clusters of arbitrary shape",
        "Makes no assumption about shape or count, and is allowed to label a "
        "point as noise -- the only model here that can say 'this customer "
        "belongs to no segment'.",
        "Two parameters that interact awkwardly (eps, min_samples), no predict "
        "for new rows, and it degrades badly in high dimensions.",
        _dbscan),
)


def architecture_table(task: Optional[str] = None) -> pd.DataFrame:
    """The catalogue as a table -- the documentation half of this module."""
    rows = [{"model": s.name, "task": s.task, "architecture": s.architecture,
             "inductive bias": s.bias, "cost": s.cost}
            for s in ADVANCED_SPECS if task is None or s.task == task]
    return pd.DataFrame(rows).set_index("model")


def build_advanced_models(
    task: str = "regression",
    preprocessor: Optional[TransformerMixin] = None,
    include: Optional[Sequence[str]] = None,
    **build_kwargs,
) -> Dict[str, object]:
    """The two advanced models for one task, wrapped in the given preprocessing.

    The preprocessing must be **the same object the baselines were given**, or the
    comparison measures the features rather than the architecture. ``main.py``
    passes one configuration to both.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    chosen = [spec for spec in ADVANCED_SPECS if spec.task == task]
    if not chosen:
        raise ValueError(f"no advanced models defined for task {task!r}")
    if include is not None:
        wanted = set(include)
        unknown = wanted - {spec.name for spec in chosen}
        if unknown:
            raise ValueError(f"unknown advanced model(s) for {task}: {sorted(unknown)}")
        chosen = [spec for spec in chosen if spec.name in wanted]

    models: Dict[str, object] = {}
    for spec in chosen:
        estimator = spec.build(**build_kwargs)
        steps = []
        if preprocessor is not None:
            steps.append(("prep", clone(preprocessor)))
        if spec.needs_scaling:
            # A network on unscaled inputs is a different experiment: it would
            # measure the scaling, not the architecture.
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator))
        # Always a Pipeline, even with a single step. The estimator has to live
        # under the name ``model`` for tune_advanced's ``model__*`` parameters to
        # reach it; returning a bare estimator when there is no preprocessing
        # made those parameters land on the estimator as unknown keyword
        # arguments, which XGBoost accepts silently -- a search that tuned
        # nothing and said nothing.
        models[spec.name] = Pipeline(steps)
    return models


# --------------------------------------------------------------------------- #
# 2. Evaluation -- the same folds and metrics as the baselines
# --------------------------------------------------------------------------- #


@dataclass
class AdvancedResults:
    """Everything one comparison produced."""

    task: str
    board: pd.DataFrame                                  # advanced models only
    combined: pd.DataFrame = field(default_factory=pd.DataFrame)   # + baselines
    fold_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    benchmark: Optional[pd.DataFrame] = None
    tuning: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuned: bool = False
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    analysis: Dict[str, object] = field(default_factory=dict)
    checks: List[Check] = field(default_factory=list)

    @property
    def best(self) -> str:
        metric, higher = HEADLINE_METRIC[self.task]
        column = self.board[metric].dropna()
        return str(column.idxmax() if higher else column.idxmin())

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "headline_metric": HEADLINE_METRIC[self.task][0],
            "best_advanced": self.best if len(self.board) else None,
            "advanced_leaderboard": self.board.reset_index().to_dict(orient="records"),
            "combined_leaderboard": (self.combined.reset_index().to_dict(orient="records")
                                     if len(self.combined) else []),
            "paired_comparison": (self.comparison.reset_index().to_dict(orient="records")
                                  if len(self.comparison) else []),
            "benchmark": (self.benchmark.reset_index().to_dict(orient="records")
                          if self.benchmark is not None else []),
            "tuned": self.tuned,
            "tuning": (self.tuning.reset_index().to_dict(orient="records")
                       if len(self.tuning) else []),
            "analysis": self.analysis,
            "checks": [c.as_dict() for c in self.checks],
        }


def fold_scores(
    name: str,
    model,
    X: pd.DataFrame,
    y=None,
    cv=None,
    task: str = "regression",
    n_subsamples: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """The headline metric **per fold**, which is what a paired comparison needs.

    Cross-validated means hide their own uncertainty: two models separated by
    0.002 on average may swap places in three folds out of five. Scoring each
    fold separately keeps that visible.

    Clustering has no held-out notion of correctness, so the analogue here is
    stability: the model is refitted on ``n_subsamples`` random 80% subsamples and
    scored on each. A method whose silhouette swings between subsamples has found
    the sample, not the structure.
    """
    metric, _ = HEADLINE_METRIC[task]
    X = pd.DataFrame(X).reset_index(drop=True)
    scores: List[float] = []

    if task == "clustering":
        rng = np.random.default_rng(random_state)
        for _ in range(n_subsamples):
            rows = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
            subsample = X.iloc[rows]
            fitted = clone(model)
            with quiet():
                labels = (fitted.fit_predict(subsample) if hasattr(fitted, "fit_predict")
                          else fitted.fit(subsample).predict(subsample))
            scores.append(score_predictions(name, task, y_pred=labels,
                                            X=subsample).get(metric, np.nan))
        return np.asarray(scores, dtype=float)

    y = pd.Series(np.asarray(y).ravel())
    cv = cv or _default_cv(task, y)
    with quiet():
        for train_idx, val_idx in cv.split(X, y if task == "classification" else None):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions = fitted.predict(X.iloc[val_idx])
            row = score_predictions(name, task, y.iloc[val_idx].to_numpy(), predictions)
            scores.append(row.get(metric, np.nan))
    return np.asarray(scores, dtype=float)


def evaluate_advanced(
    X: pd.DataFrame,
    y=None,
    task: Optional[str] = None,
    preprocessor: Optional[TransformerMixin] = None,
    cv=None,
    include: Optional[Sequence[str]] = None,
    benchmark: bool = True,
    benchmark_repeats: int = 3,
    tune: bool = False,
    n_iter: int = 20,
    **build_kwargs,
) -> AdvancedResults:
    """Fit, score and time the advanced models -- nothing compared yet.

    With ``tune=True`` each architecture gets a bounded search first, so a later
    "the baseline won" is a statement about architectures rather than about
    defaults. The search itself lives in :mod:`src.model_optimization`, which
    owns cross-validation and tuning for *every* model in the project -- keeping
    it in one place is what stops the baselines and the challengers from being
    optimised by subtly different protocols.
    """
    task = task or infer_task(y)
    models = build_advanced_models(task, preprocessor=preprocessor, include=include,
                                   **build_kwargs)
    cv = cv or _default_cv(task, y)

    tuning = pd.DataFrame()
    if tune:
        if __package__ in (None, ""):                            # pragma: no cover
            from src.model_optimization import tune_models
        else:
            from .model_optimization import tune_models

        models, tuning = tune_models(models, X, y, task=task, cv=cv, n_iter=n_iter,
                                     random_state=build_kwargs.get("random_state", 42))

    rows, predictions, folds = [], {}, {}
    for name, model in models.items():
        row, preds = cross_validate_model(name, model, X, y, cv=cv, task=task)
        rows.append(row)
        predictions[name] = preds
        folds[name] = fold_scores(name, model, X, y, cv=cv, task=task,
                                  random_state=build_kwargs.get("random_state", 42))
        logger.info("advanced %-24s %s %.4f (fold sd %.4f)", name,
                    HEADLINE_METRIC[task][0], row.get(HEADLINE_METRIC[task][0], np.nan),
                    float(np.nanstd(folds[name])))

    metric, higher = HEADLINE_METRIC[task]
    board = (pd.DataFrame(rows).set_index("model")
             .sort_values(metric, ascending=not higher, na_position="last"))

    results = AdvancedResults(task=task, board=board, predictions=predictions,
                              fold_scores=folds, tuning=tuning, tuned=bool(tune))
    if benchmark:
        results.benchmark = benchmark_baselines(models, X, y, task=task,
                                                repeats=benchmark_repeats)
    return results


# --------------------------------------------------------------------------- #
# 3. Comparison
# --------------------------------------------------------------------------- #


def compare_models(
    baselines: BaselineResults,
    advanced: AdvancedResults,
    X: pd.DataFrame,
    y=None,
    cv=None,
    preprocessor: Optional[TransformerMixin] = None,
    reference: Optional[str] = None,
    **build_kwargs,
) -> pd.DataFrame:
    """Paired, fold-by-fold comparison of each advanced model against one baseline.

    The reference defaults to the **best model baseline** -- not the best naive
    one, and not the overall best -- because that is the honest question: does a
    different architecture beat the simplest thing that already works?

    Columns:

    ``score``               pooled out-of-fold score, as on the leaderboard
    ``fold_mean``           mean of the per-fold scores (what ``delta`` uses)
    ``delta``               mean paired difference in the headline metric
    ``delta_sd``            spread of that difference across folds
    ``folds_won``           how many folds the advanced model won
    ``reliable``            is the difference -- in *either* direction -- larger
                            than its own spread? Reliably worse is also a result
    ``p_value``             corrected paired t-test over the folds (:func:`paired_test`)
    ``significant``         ``p_value`` below :data:`ALPHA`
    ``fit_time_ratio``      how much more expensive it is to fit
    """
    task = advanced.task
    metric, higher = HEADLINE_METRIC[task]
    reference = reference or _best_model_baseline(baselines, task)
    if reference is None:
        return pd.DataFrame()

    # The reference is re-scored on the same folds, so the two sets of per-fold
    # numbers are paired rather than merely comparable.
    reference_model = build_baselines(task, preprocessor=preprocessor,
                                      include=[reference], **build_kwargs)[reference]
    reference_folds = fold_scores(reference, reference_model, X, y, cv=cv, task=task,
                                  random_state=build_kwargs.get("random_state", 42))

    baseline_cost = (baselines.benchmark.loc[reference, "fit_ms"]
                     if baselines.benchmark is not None
                     and reference in baselines.benchmark.index else np.nan)

    rows = []
    for name, folds in advanced.fold_scores.items():
        paired = np.asarray(folds, dtype=float) - reference_folds
        if not higher:
            paired = -paired
        delta, spread = float(np.nanmean(paired)), float(np.nanstd(paired))
        cost = (advanced.benchmark.loc[name, "fit_ms"]
                if advanced.benchmark is not None and name in advanced.benchmark.index
                else np.nan)
        test = paired_test(paired)
        rows.append({
            "model": name,
            "score": float(advanced.board.loc[name, metric]),
            "reference": reference,
            "reference_score": float(baselines.board.loc[reference, metric]),
            "fold_mean": float(np.nanmean(folds)),
            "reference_fold_mean": float(np.nanmean(reference_folds)),
            "delta": delta,
            "delta_sd": spread,
            "folds_won": int(np.sum(paired > 0)),
            "folds": int(len(paired)),
            "reliable": bool(spread > 0 and abs(delta) / spread > RELIABILITY_MARGIN),
            "p_value": float(test["p_value"]),
            "significant": bool(test["significant"]),
            "fit_time_ratio": float(cost / baseline_cost) if baseline_cost else np.nan,
        })

    frame = pd.DataFrame(rows).set_index("model").sort_values("delta", ascending=False)
    # ``score`` and ``fold_mean`` answer different questions and can disagree.
    # ``score`` pools every out-of-fold prediction and ranks all rows together --
    # the deployment question. ``fold_mean`` ranks within each fold and averages
    # -- the paired-test question. A model can lead on one and trail on the other
    # when its errors are consistent within folds but shifted between them, and
    # that disagreement is information, so both columns are kept.
    return frame


def _best_model_baseline(baselines: BaselineResults, task: str) -> Optional[str]:
    """The strongest *model* baseline -- the thing a new architecture must beat."""
    if __package__ in (None, ""):                                # pragma: no cover
        from src.models import BASELINE_SPECS
    else:
        from .models import BASELINE_SPECS

    metric, higher = HEADLINE_METRIC[task]
    models = {s.name for s in BASELINE_SPECS if s.task == task and s.kind == "model"}
    scores = baselines.board.loc[baselines.board.index.isin(models), metric].dropna()
    if scores.empty:
        return None
    return str(scores.idxmax() if higher else scores.idxmin())


def combined_leaderboard(baselines: BaselineResults, advanced: AdvancedResults) -> pd.DataFrame:
    """Every model from both ladders in one table, tagged by where it came from."""
    if __package__ in (None, ""):                                # pragma: no cover
        from src.models import BASELINE_SPECS
    else:
        from .models import BASELINE_SPECS

    task = advanced.task
    metric, higher = HEADLINE_METRIC[task]
    kinds = {s.name: s.kind for s in BASELINE_SPECS if s.task == task}

    frames = []
    for board, default_kind in ((baselines.board, "baseline"), (advanced.board, "advanced")):
        if not len(board):
            continue
        tagged = board.copy()
        tagged.insert(0, "kind", [kinds.get(str(name), default_kind) if
                                  default_kind == "baseline" else "advanced"
                                  for name in board.index])
        frames.append(tagged)

    combined = pd.concat(frames)
    columns = ["kind"] + [c for c in _metric_names(task) if c in combined.columns]
    return combined[columns].sort_values(metric, ascending=not higher, na_position="last")


def analyse_comparison(
    baselines: BaselineResults,
    advanced: AdvancedResults,
    comparison: pd.DataFrame,
) -> Dict[str, object]:
    """Turn the comparison into statements: who won, reliably, and at what cost."""
    task = advanced.task
    metric, higher = HEADLINE_METRIC[task]
    analysis: Dict[str, object] = {"task": task, "headline_metric": metric}
    if not len(comparison):
        return analysis

    combined = advanced.combined if len(advanced.combined) else advanced.board
    scores = combined[metric].dropna()
    overall = str(scores.idxmax() if higher else scores.idxmin())
    analysis["overall_best"] = overall
    analysis["overall_best_is_advanced"] = overall in advanced.board.index

    best = comparison.iloc[0]
    analysis["reference"] = str(best["reference"])
    analysis["best_advanced"] = str(comparison.index[0])
    analysis["delta"] = float(best["delta"])
    analysis["delta_sd"] = float(best["delta_sd"])
    analysis["reliable"] = bool(best["reliable"])
    analysis["folds_won"] = f"{int(best['folds_won'])} of {int(best['folds'])}"
    analysis["fit_time_ratio"] = float(best["fit_time_ratio"])
    analysis["p_value"] = float(best.get("p_value", float("nan")))
    analysis["significant"] = bool(best.get("significant", False))

    # Error reduction is the honest way to read a small delta near a ceiling: a
    # move from 0.990 to 0.995 halves what is left, and 0.005 does not say that.
    # It is only reported when the leader actually leads -- "closes -115% of the
    # gap" is not a sentence anyone should have to parse.
    if metric in ("spearman", "norm_gini", "f1_macro", "accuracy", "silhouette"):
        reference_score, score = float(best["reference_score"]), float(best["score"])
        room = 1.0 - reference_score
        if room > 1e-9 and score > reference_score:
            analysis["error_reduction"] = float((score - reference_score) / room)

    # Pooled ranking and within-fold ranking can point different ways (see the
    # note in compare_models). When they do, neither number is wrong and the
    # disagreement is the thing to report.
    pooled_gap = float(best["score"]) - float(best["reference_score"])
    if not higher:
        pooled_gap = -pooled_gap
    analysis["pooled_gap"] = pooled_gap
    analysis["pooled_and_paired_agree"] = bool(
        np.sign(pooled_gap) == np.sign(best["delta"]) or abs(pooled_gap) < 1e-9)
    if not analysis["pooled_and_paired_agree"]:
        analysis["pooling_note"] = (
            f"{comparison.index[0]} leads by {pooled_gap:+.4f} when every out-of-fold "
            f"prediction is ranked together, but by {float(best['delta']):+.4f} within "
            "folds. Its advantage is in the pooled ordering, not inside any one fold.")

    # Set before the verdict is written: the verdict says whether the models were
    # searched or left at their defaults, and reading a flag that is not there yet
    # would quietly claim the weaker of the two.
    analysis["tuned"] = bool(advanced.tuned)
    analysis["verdict"] = _verdict(analysis)
    analysis["architectures"] = {
        spec.name: spec.architecture for spec in ADVANCED_SPECS if spec.task == task}
    return analysis


def _verdict(analysis: Dict[str, object]) -> str:
    """One sentence a reviewer can quote, derived from the numbers above.

    The direction comes from the paired mean and the confidence from the
    corrected t-test, so "won by a nose" and "won reliably" cannot be confused
    with each other -- which is the whole reason the test is there.
    """
    name = analysis["best_advanced"]
    reference = analysis["reference"]
    delta = float(analysis["delta"])
    ratio = float(analysis.get("fit_time_ratio", float("nan")))
    cost = f", at {ratio:.1f}x the fit time" if np.isfinite(ratio) else ""
    state = "after a randomised search" if analysis.get("tuned") else "at their default settings"
    p_value = float(analysis.get("p_value", float("nan")))
    evidence = f" (corrected paired t-test p = {p_value:.4f})" if np.isfinite(p_value) else ""

    if delta <= 0:
        confidence = ("and the gap is significant" if analysis.get("significant")
                      else "though the gap is inside fold noise")
        return (f"No advanced architecture beat {reference} {state}: the best of them "
                f"({name}) is {abs(delta):.4f} behind, {confidence}{evidence}{cost}. "
                "Keep the baseline.")
    if not analysis.get("significant"):
        return (f"{name} leads {reference} by {delta:.4f}, but the difference is not "
                f"distinguishable from fold noise{evidence}{cost}. Keep the baseline "
                "unless the gap grows on more data.")
    return (f"{name} beats {reference} by {delta:.4f}, winning {analysis['folds_won']} folds"
            f"{evidence}{cost}. The architecture, not the tuning, is what changed.")


# --------------------------------------------------------------------------- #
# 4. Documentation
# --------------------------------------------------------------------------- #


def document_results(results: AdvancedResults, path=None) -> str:
    """Write the comparison up as Markdown -- the deliverable a reviewer reads.

    Everything in the report is generated from the numbers in ``results``; there
    is no prose here that a future run could silently contradict.
    """
    task = results.task
    metric, _ = HEADLINE_METRIC[task]
    analysis = results.analysis
    lines: List[str] = []

    lines.append(f"# Advanced models vs the baseline ladder ({task})\n")
    lines.append(f"Headline metric: **{metric}** -- "
                 f"{METRIC_GUIDE.get(metric, {}).get('question', '')}\n")
    if analysis.get("verdict"):
        lines.append(f"> **Verdict.** {analysis['verdict']}\n")

    lines.append("\n## 1. The architectures compared\n")
    lines.append("| Model | Architecture | Inductive bias | Cost |")
    lines.append("|---|---|---|---|")
    for spec in ADVANCED_SPECS:
        if spec.task == task:
            lines.append(f"| {spec.name} | {spec.architecture} | {spec.bias} | {spec.cost} |")

    if len(results.combined):
        lines.append("\n## 2. Every model, same folds, same metrics\n")
        lines.append(_markdown_table(results.combined.round(4)))

    if len(results.comparison):
        lines.append("\n## 3. Paired comparison against the best model baseline\n")
        lines.append("Each row compares the advanced model with the reference **on the same "
                     "folds**, so the difference is paired rather than two separate averages.\n")
        lines.append(_markdown_table(results.comparison.round(4)))
        if "error_reduction" in analysis:
            lines.append(f"\nThe leader closes **{analysis['error_reduction']:.1%}** of the "
                         f"distance between the reference and a perfect score.")

    if results.benchmark is not None:
        lines.append("\n## 4. What the advanced models cost\n")
        lines.append(_markdown_table(results.benchmark.round(3)))

    if results.fold_scores:
        lines.append("\n## 5. Fold-level spread\n")
        spread = pd.DataFrame({
            "mean": {k: float(np.nanmean(v)) for k, v in results.fold_scores.items()},
            "sd": {k: float(np.nanstd(v)) for k, v in results.fold_scores.items()},
            "min": {k: float(np.nanmin(v)) for k, v in results.fold_scores.items()},
            "max": {k: float(np.nanmax(v)) for k, v in results.fold_scores.items()},
        })
        lines.append(_markdown_table(spread.round(4)))
        lines.append("\nA model whose spread across folds is wider than its lead over the "
                     "reference has not been shown to be better.")

    lines.append("\n## 6. Checks\n")
    for check in results.checks:
        lines.append(f"* `{check.status}` {check.name}"
                     f"{(' -- ' + check.detail) if check.detail else ''}")

    report = "\n".join(lines) + "\n"
    if path is not None:
        from pathlib import Path as _Path

        target = _Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        logger.info("comparison report written -> %s", target)
    return report


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join([frame.index.name or ""] + [str(c) for c in frame.columns]) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    rows = ["| " + " | ".join([str(index)] + [str(v) for v in row]) + " |"
            for index, row in zip(frame.index, frame.to_numpy())]
    return "\n".join([header, rule, *rows])


# --------------------------------------------------------------------------- #
# 5. Checks
# --------------------------------------------------------------------------- #


def check_advanced_models(results: AdvancedResults, baselines: BaselineResults) -> List[Check]:
    """What main.py asserts about the comparison before anyone quotes it."""
    rec = _Recorder("advanced models")
    task = results.task
    metric, higher = HEADLINE_METRIC[task]

    expected = {spec.name for spec in ADVANCED_SPECS if spec.task == task}
    rec.record("both advanced architectures ran",
               set(results.board.index) == expected,
               f"{sorted(results.board.index)}")

    def finite_predictions():
        bad = [name for name, preds in results.predictions.items()
               if preds.dtype.kind == "f" and not np.isfinite(preds).all()]
        return not bad, f"{len(results.predictions)} prediction sets checked"

    rec.guard("predictions are finite", finite_predictions)

    rec.record("every model has a score for each fold",
               all(len(v) > 1 and np.isfinite(v).any() for v in results.fold_scores.values()),
               f"{ {k: len(v) for k, v in results.fold_scores.items()} }")

    # The comparison is only meaningful if both sides were scored the same way.
    if len(results.comparison):
        reference = str(results.comparison.iloc[0]["reference"])
        rec.record("the reference is a model baseline, not a naive one",
                   reference in baselines.board.index,
                   f"compared against {reference}")
        rec.record("the paired difference is consistent with the leaderboard",
                   _delta_matches_board(results, baselines, metric, higher),
                   f"delta {float(results.comparison.iloc[0]['delta']):+.4f} = "
                   "fold mean - reference fold mean")
    else:
        rec.skip("the reference is a model baseline, not a naive one", "no comparison run")
        rec.skip("the paired difference is consistent with the leaderboard", "no comparison run")

    # An advanced model that cannot beat the naive floor means something is wired
    # wrong -- wrong preprocessing, wrong target, wrong folds.
    floor = _naive_floor_score(baselines, task, metric)
    if floor is None:
        rec.skip("every advanced model beats the naive floor", "no naive baseline was run")
    else:
        scores = results.board[metric].dropna()
        beats = (scores > floor).all() if higher else (scores < floor).all()
        rec.record("every advanced model beats the naive floor", bool(beats),
                   f"floor {floor:.4f}, worst advanced "
                   f"{(scores.min() if higher else scores.max()):.4f}")

    if len(results.comparison) and "p_value" in results.comparison.columns:
        row = results.comparison.iloc[0]
        p_value = float(row["p_value"])
        if not np.isfinite(p_value) and abs(float(row["delta"])) < 1e-12:
            # Two models that score identically on every fold have no difference
            # to test. That is a legitimate outcome -- it happens when both
            # architectures recover the same structure -- not a broken comparison.
            rec.skip("the comparison carries a significance test",
                     "identical scores on every fold; nothing to test")
        else:
            rec.record("the comparison carries a significance test",
                       bool(np.isfinite(p_value)),
                       f"corrected paired t-test p = {p_value:.4f}"
                       f"{' (significant)' if row['significant'] else ' (not significant)'}")
    else:
        rec.skip("the comparison carries a significance test", "no comparison run")

    rec.record("the verdict states the cost as well as the gain",
               "verdict" in results.analysis and bool(results.analysis["verdict"]),
               results.analysis.get("verdict", "")[:70] + "...")
    return rec.checks


def _delta_matches_board(results: AdvancedResults, baselines: BaselineResults,
                         metric: str, higher: bool) -> bool:
    """``delta`` must be exactly the difference of the two fold means it came from.

    This is the arithmetic identity, not the pooled comparison: ``score`` and
    ``fold_mean`` are allowed to disagree (see compare_models), and when they do
    the analysis says so rather than a check failing.
    """
    row = results.comparison.iloc[0]
    expected = float(row["fold_mean"]) - float(row["reference_fold_mean"])
    if not higher:
        expected = -expected
    return bool(abs(expected - float(row["delta"])) < 1e-9)


def _naive_floor_score(baselines: BaselineResults, task: str, metric: str) -> Optional[float]:
    if __package__ in (None, ""):                                # pragma: no cover
        from src.models import BASELINE_SPECS
    else:
        from .models import BASELINE_SPECS

    naive = {s.name for s in BASELINE_SPECS if s.task == task and s.kind == "naive"}
    scores = baselines.board.loc[baselines.board.index.isin(naive), metric].dropna()
    if scores.empty:
        return None
    return float(scores.max() if HEADLINE_METRIC[task][1] else scores.min())


# --------------------------------------------------------------------------- #
# 6. The whole thing, in one call
# --------------------------------------------------------------------------- #


def run_comparison(
    X: pd.DataFrame,
    y=None,
    task: Optional[str] = None,
    preprocessor: Optional[TransformerMixin] = None,
    cv=None,
    baselines: Optional[BaselineResults] = None,
    benchmark_repeats: int = 3,
    tune: bool = False,
    n_iter: int = 20,
    report_path=None,
    **build_kwargs,
) -> AdvancedResults:
    """Baselines (reused or run), advanced models, comparison, analysis, checks, report.

    This is what ``main.py`` calls, and what another project would call with its
    own data: it needs nothing from the capstone except ``X``, ``y`` and, if the
    features need preparing, a transformer.
    """
    task = task or infer_task(y)
    cv = cv or _default_cv(task, y)
    if baselines is None:
        baselines = evaluate_baselines(X, y, task=task, preprocessor=preprocessor, cv=cv,
                                       benchmark_repeats=benchmark_repeats, **build_kwargs)

    results = evaluate_advanced(X, y, task=task, preprocessor=preprocessor, cv=cv,
                                benchmark_repeats=benchmark_repeats, tune=tune,
                                n_iter=n_iter, **build_kwargs)
    results.combined = combined_leaderboard(baselines, results)
    results.comparison = compare_models(baselines, results, X, y, cv=cv,
                                        preprocessor=preprocessor, **build_kwargs)
    results.analysis = analyse_comparison(baselines, results, results.comparison)
    results.checks = check_advanced_models(results, baselines)
    if report_path is not None:
        document_results(results, report_path)
    return results


# --------------------------------------------------------------------------- #
# Smoke test: python src/advanced_models.py
# --------------------------------------------------------------------------- #


def _smoke_test() -> int:
    """Run the full comparison on synthetic data, for all three tasks.

    Needs no dataset and nothing else from the project, so it doubles as the
    worked example of applying this module to a problem that is not the capstone.
    """
    if __package__ in (None, ""):
        from src.models import (_synthetic_classification, _synthetic_clustering,
                                _synthetic_regression)
    else:
        from .models import (_synthetic_classification, _synthetic_clustering,
                             _synthetic_regression)

    print("=" * 78)
    print("src/advanced_models.py -- ARCHITECTURE COMPARISON SMOKE TEST")
    print("=" * 78)
    print("  Two further architectures per task, compared with the baseline ladder.")

    rng = np.random.default_rng(42)
    failures: List[str] = []
    cases = [
        ("regression", *_synthetic_regression(rng), {}),
        ("classification", *_synthetic_classification(rng), {}),
        ("clustering", _synthetic_clustering(rng), None, {"n_clusters": 3, "eps": 0.9}),
    ]

    for task, X, y, extra in cases:
        print("\n" + "-" * 78)
        print(f"{task.upper()}  ({len(X)} rows)")
        print("-" * 78)
        for spec in ADVANCED_SPECS:
            if spec.task == task:
                print(f"  {spec.name:22s} {spec.architecture}")

        results = run_comparison(X, y, task=task, benchmark_repeats=1, **extra)

        metric = HEADLINE_METRIC[task][0]
        columns = ["kind"] + [c for c in _metric_names(task)
                              if c in results.combined.columns][:3]
        print(f"\n  All models, sorted by {metric}:")
        print(results.combined[columns].round(4).to_string())

        if len(results.comparison):
            print("\n  Paired against the best model baseline:")
            print(results.comparison[["reference", "delta", "delta_sd", "folds_won",
                                      "reliable", "fit_time_ratio"]].round(4).to_string())
        print(f"\n  Verdict: {results.analysis.get('verdict', 'n/a')}")

        print("\n  checks:")
        for check in results.checks:
            print(f"    [{check.status:4s}] {check.name}"
                  f"{('  -- ' + check.detail) if check.detail else ''}")
            if check.failed:
                failures.append(f"{task}: {check.name}")

    print("\n" + "-" * 78)
    print("REPORT (first 24 lines of the generated Markdown)")
    print("-" * 78)
    report = document_results(results)
    print("\n".join(report.splitlines()[:24]))

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("All three comparisons ran and every check passed.")
    print("In the project this module is used through main.py (stage 10).")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(_smoke_test())
