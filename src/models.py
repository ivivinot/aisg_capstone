"""Baseline models: the bar a real model has to clear, for any tabular task.

Why this module exists
----------------------
`readme.md` 7.3 is blunt about it: **reporting a metric without a baseline next to
it is uninterpretable.** A CLV model with Spearman 0.99 sounds excellent until the
``purchases x order value`` heuristic on the same split reaches 0.88 -- and then
the question is what the other 0.11 cost, not how good 0.99 looks. Most published
CLV models fail to beat a persistence baseline (`readme.md` 7.3), and nobody
notices because the baseline is never fitted.

So baselines are treated here as a first-class deliverable, not a footnote: they
are implemented, trained, scored on the same folds and metrics as everything
else, benchmarked for cost, and checked.

Task coverage
-------------
The module is task-agnostic. :func:`infer_task` reads the target and dispatches:

===============  =========================================  ====================
task             naive floor(s)                              model baselines
===============  =========================================  ====================
``regression``   mean, median, a domain heuristic            linear, log-target
                 (a product of two columns, when named)      linear, ridge,
                                                             shallow tree, k-NN
``classification`` most-frequent class, stratified random,   logistic regression,
                 uniform random                              shallow tree, k-NN
``clustering``   one cluster for everyone, random labels     k-means at a few k,
                                                             agglomerative
===============  =========================================  ====================

The capstone exercises the regression half (``main.py`` stage 5); the other two
are here so the same ladder can be run on a classification or clustering
assignment without rewriting anything. Every task follows the same four steps --
**select, train and evaluate, analyse the metrics, benchmark the cost** -- and
returns the same :class:`BaselineResults` object.

What each step means
--------------------
**Selection** (:data:`BASELINE_SPECS`). Two kinds of baseline, and the difference
matters: a *naive floor* learns one number and exists to make metrics readable; a
*model baseline* is a real but deliberately simple estimator, and it is the one a
complicated model actually has to beat. A tuned gradient booster that ties with
ridge is a negative result, however good its absolute score looks.

**Training and initial evaluation** (:func:`evaluate_baselines`). K-fold
out-of-fold predictions, one scorecard row per baseline, on the training split
only. Any preprocessing is passed in as a transformer and refitted inside every
fold, so the scores carry no leakage.

**Metrics analysis** (:data:`METRIC_GUIDE`, :func:`analyse_baselines`). Every
metric is documented with the question it answers, its direction, and how it
misleads. The analysis then reports whether the metrics *agree* on a winner --
when ranking and calibration disagree, that disagreement is the finding.

**Performance benchmarking** (:func:`benchmark_baselines`). Fit time, prediction
throughput and serialised model size. A baseline that is 400x cheaper and 3%
worse is often the right answer in production, and that trade-off cannot be seen
from a score column alone.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold

# Imported as part of the package (``from src.models import ...``) the relative
# imports below are correct. Run as a script (``python src/models.py``, or the
# editor's Run button) there is no parent package for the leading dot to resolve
# against, so the module puts the project root on sys.path and imports absolutely.
# The __main__ block at the bottom is a self-contained smoke test of all three
# ladders, which is what makes running this file directly worth doing.
if __package__ in (None, ""):                                    # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.metrics import scorecard
    from src.preprocessing import (AOV_COL, COUNT_COL, Check, LogTargetTransformer,
                                   _Recorder, quiet)
else:
    from .metrics import scorecard
    from .preprocessing import (AOV_COL, COUNT_COL, Check, LogTargetTransformer,
                                _Recorder, quiet)

#: A library logger: it inherits the handler main.py installs on "capstone".
logger = logging.getLogger("capstone.models")

__all__ = [
    "TASKS",
    "infer_task",
    "LogTargetRegressor",
    "ColumnProductRegressor",
    "BaselineSpec",
    "BASELINE_SPECS",
    "build_baselines",
    "BaselineResults",
    "evaluate_baselines",
    "benchmark_baselines",
    "analyse_baselines",
    "check_baselines",
    "cross_validate_model",
    "score_predictions",
    "METRIC_GUIDE",
    "HEADLINE_METRIC",
    "has_xgboost",
]

TASKS = ("regression", "classification", "clustering")

#: The metric each task is ranked by, and whether more is better.
HEADLINE_METRIC = {
    "regression": ("spearman", True),
    "classification": ("f1_macro", True),
    "clustering": ("silhouette", True),
}


def has_xgboost() -> bool:
    """Is XGBoost installed? It is optional -- nothing in this module needs it.

    ``find_spec`` asks the import system whether the package is available without
    importing it. That avoids paying for a heavy import just to answer a yes/no
    question, and it keeps this file free of an import that a type checker cannot
    resolve when the editor is pointed at an environment where XGBoost is absent.
    """
    import importlib.util

    return importlib.util.find_spec("xgboost") is not None


def infer_task(y=None, max_classes: int = 20) -> str:
    """Decide the task from the target: no target means clustering.

    A float target is a regression; anything with few distinct values (or a
    non-numeric dtype) is a classification. The threshold is a convention, not a
    law, which is why every entry point takes an explicit ``task`` override.
    """
    if y is None:
        return "clustering"
    y = pd.Series(np.asarray(y).ravel())
    if y.dtype.kind in "OSUb" or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    distinct = y.nunique(dropna=True)
    if y.dtype.kind in "iu" and distinct <= max_classes:
        return "classification"
    if distinct <= 2:
        return "classification"
    return "regression"


# --------------------------------------------------------------------------- #
# Estimators the baselines need and scikit-learn does not provide
# --------------------------------------------------------------------------- #


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Fit any regressor on ``log(y)``; predict back on the original scale.

    The back-transform is Duan's smearing estimator -- ``exp(prediction)`` times
    ``mean(exp(residual))`` on the training data -- because ``exp(E[log y])``
    systematically under-states ``E[y]`` for a right-skewed target (EDA 8). The
    factor rescales levels and leaves the ordering untouched, so it moves the
    calibration metrics and not Spearman.

    :meth:`predict_log` gives the untransformed prediction, which is what the
    log-scale metrics are computed on.
    """

    def __init__(self, model, smearing: bool = True):
        self.model = model
        self.smearing = smearing

    def fit(self, X, y) -> "LogTargetRegressor":
        self.model_ = clone(self.model)
        self.target_ = LogTargetTransformer(smearing=self.smearing).fit(y)
        y_log = self.target_.transform(y)
        self.model_.fit(X, y_log)
        with quiet():  # the in-sample factor is a detail; fold-level ones are logged
            self.target_.fit_smearing(y_log - self.model_.predict(X))
        return self

    def predict_log(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict(X), dtype=float)

    def predict(self, X) -> np.ndarray:
        return self.target_.inverse_transform(self.predict_log(X))

    @property
    def smearing_factor_(self) -> float:
        return self.target_.smearing_factor_


class ColumnProductRegressor(BaseEstimator, RegressorMixin):
    """The analyst's answer: multiply two columns and rescale to the training mean.

    This is the domain heuristic of `readme.md` 7.3 -- ``purchases x order
    value`` -- as a proper estimator, so it can be cross-validated on the same
    folds as everything else instead of being computed once on the side. It
    learns exactly one number (the scale factor), which is the point: any model
    that cannot beat it is not earning its complexity.

    Give it any two column names; if either is missing it falls back to the
    training mean and says so, so a generic run never crashes on a dataset that
    has no such pair.
    """

    def __init__(self, left: str = COUNT_COL, right: str = AOV_COL):
        self.left = left
        self.right = right

    def fit(self, X, y) -> "ColumnProductRegressor":
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        self.usable_ = self.left in X.columns and self.right in X.columns
        self.fallback_ = float(np.mean(y))
        if not self.usable_:
            logger.info("%s x %s unavailable; the heuristic falls back to the mean",
                        self.left, self.right)
            self.scale_ = 1.0
            return self
        raw = self._raw(X)
        self.scale_ = float(np.mean(y) / np.mean(raw)) if np.mean(raw) else 1.0
        return self

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.left].to_numpy(dtype=float) * X[self.right].to_numpy(dtype=float))

    def predict(self, X) -> np.ndarray:
        X = pd.DataFrame(X)
        if not self.usable_:
            return np.full(len(X), self.fallback_)
        return self._raw(X) * self.scale_


class ConstantClusterer(BaseEstimator):
    """Everyone in one cluster -- the clustering equivalent of the mean baseline.

    Useless as a segmentation and essential as a reference: it is what "no
    structure found" looks like, and several internal indices are undefined for
    it, which is itself worth seeing once.
    """

    def __init__(self, n_clusters: int = 1):
        self.n_clusters = n_clusters

    def fit(self, X, y=None) -> "ConstantClusterer":
        self.labels_ = np.zeros(len(pd.DataFrame(X)), dtype=int)
        return self

    def predict(self, X) -> np.ndarray:
        return np.zeros(len(pd.DataFrame(X)), dtype=int)

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_


class RandomClusterer(BaseEstimator):
    """Random cluster assignment: the floor every silhouette must clear.

    A k-means silhouette of 0.25 means nothing until you know that random labels
    on the same data score about 0.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None) -> "RandomClusterer":
        rng = np.random.default_rng(self.random_state)
        self.labels_ = rng.integers(0, self.n_clusters, size=len(pd.DataFrame(X)))
        return self

    def predict(self, X) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.integers(0, self.n_clusters, size=len(pd.DataFrame(X)))

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_


# --------------------------------------------------------------------------- #
# 1. Selection -- the catalogue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BaselineSpec:
    """One baseline: how to build it, what it is for, and how to read it.

    ``kind`` is either ``"naive"`` -- learns a constant or a single scale factor,
    and exists so the metrics can be read -- or ``"model"``, a real but
    deliberately simple estimator, which is the bar a complicated model has to
    clear before it is worth its complexity.

    ``raw_input`` takes the baseline *out* of the preprocessing pipeline, and
    every naive baseline sets it. Two reasons, both of which showed up as wrong
    numbers before the flag existed: a constant predictor ignores ``X``, so
    preprocessing it is pure cost and makes the benchmark measure the
    preprocessor instead of the baseline; and a domain heuristic is defined on
    the *raw* columns -- fed a logged, expanded matrix it cannot find them at all
    and silently degrades to the mean.
    """

    name: str
    task: str
    kind: str                       # naive | model
    rationale: str
    build: Callable[..., object]
    log_target: bool = False        # wrap in LogTargetRegressor (regression only)
    raw_input: bool = False         # skip the preprocessor: see below


def _mean_regressor(**_):
    from sklearn.dummy import DummyRegressor

    return DummyRegressor(strategy="mean")


def _median_regressor(**_):
    from sklearn.dummy import DummyRegressor

    return DummyRegressor(strategy="median")


def _heuristic_regressor(left: str = COUNT_COL, right: str = AOV_COL, **_):
    return ColumnProductRegressor(left=left, right=right)


def _linear_regressor(**_):
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def _ridge_regressor(alpha: float = 1.0, **_):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha)


def _tree_regressor(random_state: int = 42, **_):
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(max_depth=3, random_state=random_state)


def _knn_regressor(**_):
    from sklearn.neighbors import KNeighborsRegressor

    return KNeighborsRegressor(n_neighbors=10)


def _majority_classifier(random_state: int = 42, **_):
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def _stratified_classifier(random_state: int = 42, **_):
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="stratified", random_state=random_state)


def _uniform_classifier(random_state: int = 42, **_):
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="uniform", random_state=random_state)


def _logistic_classifier(random_state: int = 42, **_):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, random_state=random_state)


def _tree_classifier(random_state: int = 42, **_):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(max_depth=3, random_state=random_state)


def _knn_classifier(**_):
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(n_neighbors=10)


def _one_cluster(**_):
    return ConstantClusterer()


def _random_clusters(n_clusters: int = 4, random_state: int = 42, **_):
    return RandomClusterer(n_clusters=n_clusters, random_state=random_state)


def _kmeans(n_clusters: int = 4, random_state: int = 42, **_):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)


def _agglomerative(n_clusters: int = 4, **_):
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=n_clusters)


BASELINE_SPECS: Tuple[BaselineSpec, ...] = (
    # -- regression --------------------------------------------------------- #
    BaselineSpec("mean", "regression", "naive",
                 "Predicts one number for everyone. R2 is 0 here by construction and "
                 "rank correlation is noise around zero: out of fold each fold predicts "
                 "its own training mean, so the predictions are not quite constant. That "
                 "is what an uninformative model scores.",
                 _mean_regressor, raw_input=True),
    BaselineSpec("median", "regression", "naive",
                 "The mean's robust twin. On a target with skew 6.4 (EDA 3) it beats the "
                 "mean on MAE and loses on R2, which is the first sign that a metric "
                 "choice is also a business choice.",
                 _median_regressor, raw_input=True),
    BaselineSpec("heuristic (product of two columns)", "regression", "naive",
                 "The domain shortcut of readme.md 7.3: purchases x order value. It "
                 "reaches Spearman 0.88 on this data, so it -- not the mean -- is the "
                 "bar a model has to clear. Defined on the raw columns, so it never "
                 "sees the preprocessor.",
                 _heuristic_regressor, raw_input=True),
    BaselineSpec("linear regression", "regression", "model",
                 "The simplest thing that learns from every feature. If a complex model "
                 "cannot beat it, the extra capacity is buying nothing.",
                 _linear_regressor),
    BaselineSpec("linear regression on log(y)", "regression", "model",
                 "The same model fitted on the log target with Duan smearing (EDA 8). "
                 "On a multiplicative, heavy-tailed target this one line of preprocessing "
                 "is usually worth more than any estimator swap.",
                 _linear_regressor, log_target=True),
    BaselineSpec("ridge on log(y)", "regression", "model",
                 "Regularised, so it survives collinear or expanded feature spaces where "
                 "plain least squares becomes unstable.",
                 _ridge_regressor, log_target=True),
    BaselineSpec("decision tree (depth 3)", "regression", "model",
                 "A deliberately shallow non-linear reference: it shows how much of the "
                 "signal is step-shaped, and it is readable end to end.",
                 _tree_regressor),
    BaselineSpec("k-NN (k=10)", "regression", "model",
                 "Non-parametric contrast: are customers with similar features worth "
                 "similar amounts? A strong k-NN means the geometry already carries the "
                 "answer.",
                 _knn_regressor),
    # -- classification ----------------------------------------------------- #
    BaselineSpec("most frequent class", "classification", "naive",
                 "Predicts the majority class. Its accuracy is the majority share, which "
                 "is exactly why accuracy is the wrong headline metric on imbalanced "
                 "data -- this row makes that impossible to miss.",
                 _majority_classifier, raw_input=True),
    BaselineSpec("stratified random", "classification", "naive",
                 "Draws from the training class distribution. The floor for F1 and AUC: "
                 "AUC ~ 0.5 whatever the imbalance.",
                 _stratified_classifier, raw_input=True),
    BaselineSpec("uniform random", "classification", "naive",
                 "Ignores the class balance entirely; separates 'the model learned the "
                 "prior' from 'the model learned nothing'.",
                 _uniform_classifier, raw_input=True),
    BaselineSpec("logistic regression", "classification", "model",
                 "The linear reference, and a calibrated one: its probabilities can be "
                 "used directly for thresholding.",
                 _logistic_classifier),
    BaselineSpec("decision tree (depth 3)", "classification", "model",
                 "A readable non-linear reference; the rules can be shown to a "
                 "stakeholder.",
                 _tree_classifier),
    BaselineSpec("k-NN (k=10)", "classification", "model",
                 "Local structure without a decision boundary of any particular shape.",
                 _knn_classifier),
    # -- clustering --------------------------------------------------------- #
    BaselineSpec("one cluster", "clustering", "naive",
                 "No structure at all. Several internal indices are undefined for it, "
                 "which is a useful reminder that they are relative measures.",
                 _one_cluster, raw_input=True),
    BaselineSpec("random labels", "clustering", "naive",
                 "Random assignment at the same k. A silhouette is only interesting "
                 "against this number.",
                 _random_clusters, raw_input=True),
    BaselineSpec("k-means", "clustering", "model",
                 "The default partitioning method; spherical clusters of similar size.",
                 _kmeans),
    BaselineSpec("agglomerative (ward)", "clustering", "model",
                 "Hierarchical alternative: finds elongated or nested groups that "
                 "k-means splits by construction.",
                 _agglomerative),
)


def build_baselines(
    task: str = "regression",
    preprocessor: Optional[TransformerMixin] = None,
    include: Optional[Sequence[str]] = None,
    **build_kwargs,
) -> Dict[str, object]:
    """The baselines for one task, each wrapped in the given preprocessing.

    ``preprocessor`` is any scikit-learn transformer -- for this project the
    :class:`~src.preprocessing.CLVPreprocessor`, for another project whatever is
    appropriate. It is cloned into every baseline's pipeline so that
    cross-validation refits it inside each fold, which is what keeps the scores
    honest.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    from sklearn.pipeline import Pipeline

    chosen = [spec for spec in BASELINE_SPECS if spec.task == task]
    if include is not None:
        wanted = set(include)
        unknown = wanted - {spec.name for spec in chosen}
        if unknown:
            raise ValueError(f"unknown baseline(s) for {task}: {sorted(unknown)}")
        chosen = [spec for spec in chosen if spec.name in wanted]

    models: Dict[str, object] = {}
    for spec in chosen:
        estimator = spec.build(**build_kwargs)
        if spec.log_target:
            estimator = LogTargetRegressor(estimator)
        if preprocessor is not None and not spec.raw_input:
            # The log-target wrapper has to sit outside the pipeline: it transforms
            # y, and a Pipeline step never sees the target it is meant to change.
            inner = Pipeline([("prep", clone(preprocessor)),
                              ("model", estimator.model if spec.log_target else estimator)])
            estimator = LogTargetRegressor(inner) if spec.log_target else inner
        models[spec.name] = estimator
    return models


def spec_table(task: Optional[str] = None) -> pd.DataFrame:
    """The catalogue as a table -- the documentation half of this module."""
    rows = [{"baseline": s.name, "task": s.task, "kind": s.kind, "why it is here": s.rationale}
            for s in BASELINE_SPECS if task is None or s.task == task]
    return pd.DataFrame(rows).set_index("baseline")


# --------------------------------------------------------------------------- #
# 2. Metrics -- and what each one is actually telling you
# --------------------------------------------------------------------------- #

#: Every metric this module reports: the question it answers, whether more is
#: better, and the way it misleads. The third column is the one people skip.
METRIC_GUIDE: Dict[str, Dict[str, str]] = {
    # regression
    "spearman": {"question": "Are the rows ranked in the right order?", "direction": "higher",
                 "misleads": "Blind to level: a model can rank perfectly and be out by 10x."},
    "norm_gini": {"question": "How much of the perfect ordering is captured?",
                  "direction": "higher",
                  "misleads": "Concentration-dependent; compare only within one dataset."},
    "top_decile_capture": {"question": "What share of value sits in the predicted top 10%?",
                           "direction": "higher",
                           "misleads": "Its ceiling is the data's own concentration, not 1.0."},
    "decile_mape": {"question": "Are predicted levels right, decile by decile?",
                    "direction": "lower",
                    "misleads": "Averages within deciles, so it forgives individual errors."},
    "mae_raw": {"question": "How far off is a typical prediction, in units?",
                "direction": "lower",
                "misleads": "Dominated by the bulk; a heavy tail barely moves it."},
    "r2_raw": {"question": "What share of variance is explained on the raw scale?",
               "direction": "higher",
               "misleads": "On a skewed target it is a report on the few largest rows."},
    "r2_log": {"question": "Variance explained where the model was actually fitted.",
               "direction": "higher", "misleads": "Not comparable across target transforms."},
    "rmse_log": {"question": "Spread of the log-scale residuals.", "direction": "lower",
                 "misleads": "Reads as a percentage error, which is easy to over-read."},
    # classification
    "accuracy": {"question": "What share of rows is classified correctly?",
                 "direction": "higher",
                 "misleads": "The majority-class baseline already scores the majority share."},
    "balanced_accuracy": {"question": "Mean recall across classes.", "direction": "higher",
                          "misleads": "Ignores precision, so it rewards over-predicting rare "
                                      "classes."},
    "f1_macro": {"question": "Precision and recall, averaged over classes equally.",
                 "direction": "higher",
                 "misleads": "A tiny class counts as much as a huge one -- usually the point, "
                             "occasionally a distortion."},
    "roc_auc": {"question": "Can the model rank a positive above a negative?",
                "direction": "higher",
                "misleads": "Optimistic under heavy imbalance; prefer AUPRC there."},
    "average_precision": {"question": "Area under the precision-recall curve (AUPRC).",
                          "direction": "higher",
                          "misleads": "Its floor is the positive rate, not 0.5."},
    "log_loss": {"question": "Are the predicted probabilities calibrated?",
                 "direction": "lower",
                 "misleads": "One confident mistake can dominate the average."},
    # clustering
    "silhouette": {"question": "Are points closer to their own cluster than the next one?",
                   "direction": "higher",
                   "misleads": "Favours convex, equally sized clusters -- it likes k-means."},
    "calinski_harabasz": {"question": "Between-cluster over within-cluster dispersion.",
                          "direction": "higher", "misleads": "Grows with k almost mechanically."},
    "davies_bouldin": {"question": "Average similarity between each cluster and its closest.",
                       "direction": "lower", "misleads": "Also assumes convex clusters."},
    "n_clusters": {"question": "How many groups were produced?", "direction": "n/a",
                   "misleads": "Not a quality measure; context for the three above."},
}


def metric_guide(task: Optional[str] = None) -> pd.DataFrame:
    """:data:`METRIC_GUIDE` as a table, optionally for the metrics of one task."""
    names = list(METRIC_GUIDE) if task is None else _metric_names(task)
    return pd.DataFrame(
        [{"metric": n, **METRIC_GUIDE[n]} for n in names if n in METRIC_GUIDE]
    ).set_index("metric")


def _metric_names(task: str) -> List[str]:
    return {
        "regression": ["spearman", "norm_gini", "top_decile_capture", "decile_mape",
                       "mae_raw", "r2_raw", "r2_log", "rmse_log"],
        "classification": ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc",
                           "average_precision", "log_loss"],
        "clustering": ["silhouette", "calinski_harabasz", "davies_bouldin", "n_clusters"],
    }[task]


def score_predictions(name: str, task: str, y_true=None, y_pred=None, **extra) -> dict:
    """One scorecard row for any task, so the three ladders read the same way."""
    if task == "regression":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        positive = bool((y_true > 0).all() and (y_pred > 0).all())
        return scorecard(name, y_true, y_pred,
                         np.log(y_true) if positive else None,
                         np.log(y_pred) if positive else None)
    if task == "classification":
        return _score_classification(name, y_true, y_pred, extra.get("proba"))
    return _score_clustering(name, extra["X"], y_pred)


def _score_classification(name: str, y_true, y_pred, proba=None) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, f1_score, log_loss, roc_auc_score)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    row = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    classes = np.unique(y_true)
    if proba is not None and len(classes) == 2:
        positive = np.asarray(proba)
        positive = positive[:, 1] if positive.ndim == 2 else positive
        row["roc_auc"] = float(roc_auc_score(y_true, positive))
        row["average_precision"] = float(average_precision_score(
            y_true, positive, pos_label=classes[-1]))
        row["log_loss"] = float(log_loss(y_true, np.clip(positive, 1e-9, 1 - 1e-9)))
    return row


def _score_clustering(name: str, X, labels) -> dict:
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)

    X = np.asarray(pd.DataFrame(X), dtype=float)
    labels = np.asarray(labels).ravel()
    row = {"model": name, "n_clusters": int(len(np.unique(labels)))}
    if row["n_clusters"] < 2 or row["n_clusters"] >= len(labels):
        # Undefined, not zero: every internal index compares clusters to each other.
        row.update({"silhouette": float("nan"), "calinski_harabasz": float("nan"),
                    "davies_bouldin": float("nan")})
        return row
    row["silhouette"] = float(silhouette_score(X, labels))
    row["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    row["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    return row


# --------------------------------------------------------------------------- #
# 3. Training and initial evaluation
# --------------------------------------------------------------------------- #


@dataclass
class BaselineResults:
    """Everything one run of the ladder produced."""

    task: str
    board: pd.DataFrame                       # one scorecard row per baseline
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    benchmark: Optional[pd.DataFrame] = None
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
            "best": self.best if len(self.board) else None,
            "leaderboard": self.board.reset_index().to_dict(orient="records"),
            "benchmark": (self.benchmark.reset_index().to_dict(orient="records")
                          if self.benchmark is not None else []),
            "analysis": self.analysis,
            "checks": [c.as_dict() for c in self.checks],
        }


def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y=None,
    cv=None,
    task: str = "regression",
) -> Tuple[dict, np.ndarray]:
    """Out-of-fold predictions for one estimator -> one scorecard row.

    The whole estimator -- preprocessing included -- is refitted inside every
    fold, so nothing it saw was computed with the validation rows in it.
    Clustering has no held-out notion of correctness, so it is fitted once on all
    rows and scored by internal indices instead.
    """
    X = pd.DataFrame(X).reset_index(drop=True)

    if task == "clustering":
        fitted = clone(model)
        with quiet():
            labels = (fitted.fit_predict(X) if hasattr(fitted, "fit_predict")
                      else fitted.fit(X).predict(X))
        return score_predictions(name, task, y_pred=labels, X=X), np.asarray(labels)

    y = pd.Series(np.asarray(y).ravel())
    cv = cv or _default_cv(task, y)
    oof = np.empty(len(y), dtype=float if task == "regression" else object)
    proba = np.full(len(y), np.nan)

    with quiet():  # a full refit per fold would otherwise flood the log
        for train_idx, val_idx in cv.split(X, y if task == "classification" else None):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[val_idx] = fitted.predict(X.iloc[val_idx])
            if task == "classification" and hasattr(fitted, "predict_proba"):
                scores = fitted.predict_proba(X.iloc[val_idx])
                if scores.shape[1] == 2:
                    proba[val_idx] = scores[:, 1]

    if task == "regression":
        return score_predictions(name, task, y.to_numpy(), oof.astype(float)), oof.astype(float)

    # The out-of-fold buffer is an object array so it can hold string labels; left
    # that way, scikit-learn's metrics cannot tell what kind of target it is
    # ("a mix of binary and unknown"). Re-inferring the dtype fixes that.
    labels = np.asarray(oof.tolist())
    return (score_predictions(name, task, y.to_numpy(), labels,
                              proba=None if np.isnan(proba).all() else proba),
            labels)


def _default_cv(task: str, y) -> object:
    if task == "classification":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return KFold(n_splits=5, shuffle=True, random_state=42)


def evaluate_baselines(
    X: pd.DataFrame,
    y=None,
    task: Optional[str] = None,
    preprocessor: Optional[TransformerMixin] = None,
    cv=None,
    include: Optional[Sequence[str]] = None,
    benchmark: bool = True,
    benchmark_repeats: int = 3,
    **build_kwargs,
) -> BaselineResults:
    """Run the whole ladder: build, train, score, analyse, benchmark, check.

    This is the one function ``main.py`` calls, and the one a different project
    would call with its own ``X``, ``y`` and preprocessor.
    """
    task = task or infer_task(y)
    models = build_baselines(task, preprocessor=preprocessor, include=include, **build_kwargs)
    if not models:
        raise ValueError(f"no baselines selected for task {task!r}")
    cv = cv or _default_cv(task, y)

    rows, predictions = [], {}
    for name, model in models.items():
        row, preds = cross_validate_model(name, model, X, y, cv=cv, task=task)
        rows.append(row)
        predictions[name] = preds
        logger.info("baseline %-34s %s", name, _headline(row, task))

    metric, higher = HEADLINE_METRIC[task]
    board = (pd.DataFrame(rows).set_index("model")
             .sort_values(metric, ascending=not higher, na_position="last"))

    results = BaselineResults(task=task, board=board, predictions=predictions)
    results.analysis = analyse_baselines(board, task)
    if benchmark:
        results.benchmark = benchmark_baselines(models, X, y, task=task,
                                                repeats=benchmark_repeats)
    results.checks = check_baselines(results, X, y)
    return results


def _headline(row: dict, task: str) -> str:
    metric, _ = HEADLINE_METRIC[task]
    value = row.get(metric)
    return f"{metric} {value:.4f}" if isinstance(value, float) and np.isfinite(value) \
        else f"{metric} n/a"


# --------------------------------------------------------------------------- #
# 4. Metrics analysis
# --------------------------------------------------------------------------- #


def analyse_baselines(board: pd.DataFrame, task: str) -> Dict[str, object]:
    """Read the leaderboard: who won, by how much, and do the metrics agree?

    The last question is the one worth asking. A model that leads on ranking and
    trails on calibration is not "the best model" -- it is the right model for one
    decision and the wrong one for another (`readme.md` 7.2), and a single sorted
    column hides that completely.
    """
    metric, higher = HEADLINE_METRIC[task]
    available = [m for m in _metric_names(task) if m in board.columns]
    analysis: Dict[str, object] = {"headline_metric": metric, "task": task}

    scores = board[metric].dropna()
    if scores.empty:
        return analysis
    best = str(scores.idxmax() if higher else scores.idxmin())
    analysis["best"] = best
    analysis["best_score"] = float(scores.loc[best])

    # Who wins under each metric? Agreement is not guaranteed and is worth naming.
    winners = {}
    for name in available:
        column = board[name].dropna()
        if column.empty or name == "n_clusters":
            continue
        direction = METRIC_GUIDE.get(name, {}).get("direction", "higher")
        winners[name] = str(column.idxmin() if direction == "lower" else column.idxmax())
    analysis["winner_by_metric"] = winners
    analysis["metrics_agree"] = len(set(winners.values())) == 1
    if not analysis["metrics_agree"]:
        analysis["disagreement"] = (
            f"{len(set(winners.values()))} different winners across "
            f"{len(winners)} metrics: {sorted(set(winners.values()))}")

    # How much of the distance from "knows nothing" to the best does each row cover?
    floor_name = _naive_floor(board, task)
    if floor_name is not None and floor_name in scores.index:
        floor = float(scores.loc[floor_name])
        span = analysis["best_score"] - floor
        analysis["naive_floor"] = {"baseline": floor_name, "score": floor}
        if span:
            analysis["lift_over_floor"] = {
                str(name): float((value - floor) / span) for name, value in scores.items()}

    strongest_naive = _strongest_naive(board, task, metric, higher)
    if strongest_naive:
        analysis["bar_for_a_real_model"] = strongest_naive
    return analysis


def _naive_floor(board: pd.DataFrame, task: str) -> Optional[str]:
    """The 'knows nothing' row for this task, if it was run."""
    floors = {"regression": "mean", "classification": "stratified random",
              "clustering": "random labels"}
    name = floors[task]
    return name if name in board.index else None


def _strongest_naive(board: pd.DataFrame, task: str, metric: str, higher: bool) -> Optional[dict]:
    """The best naive baseline -- the number a real model actually has to beat."""
    naive = {s.name for s in BASELINE_SPECS if s.task == task and s.kind == "naive"}
    rows = board.loc[board.index.isin(naive), metric].dropna()
    if rows.empty:
        return None
    name = str(rows.idxmax() if higher else rows.idxmin())
    models = board.loc[~board.index.isin(naive), metric].dropna()
    beaten = (int((models > rows.loc[name]).sum()) if higher
              else int((models < rows.loc[name]).sum()))
    return {"baseline": name, "score": float(rows.loc[name]),
            "model_baselines_beating_it": beaten, "of": int(len(models))}


# --------------------------------------------------------------------------- #
# 5. Performance benchmarking
# --------------------------------------------------------------------------- #


def benchmark_baselines(
    models: Dict[str, object],
    X: pd.DataFrame,
    y=None,
    task: str = "regression",
    repeats: int = 3,
) -> pd.DataFrame:
    """Fit time, prediction throughput and serialised size, per baseline.

    Scores answer "is it good"; this answers "what does it cost". The two
    together are what a deployment decision needs: on this data the mean baseline
    fits in microseconds and ridge in milliseconds, and if the gap in accuracy
    were small, the cheap one would win.

    Timings are wall-clock medians over ``repeats`` fits on the full input, so
    they are comparable to each other, not to a production machine.
    """
    X = pd.DataFrame(X)
    rows = []
    for name, model in models.items():
        fit_times, predict_times = [], []
        fitted = None
        for _ in range(max(1, repeats)):
            candidate = clone(model)
            start = time.perf_counter()
            with quiet():
                if task == "clustering":
                    (candidate.fit_predict(X) if hasattr(candidate, "fit_predict")
                     else candidate.fit(X))
                else:
                    candidate.fit(X, np.asarray(y).ravel())
            fit_times.append(time.perf_counter() - start)
            fitted = candidate

            # Some clusterers (agglomerative, DBSCAN) have no predict at all:
            # they assign labels during the fit and cannot score a new row. That
            # is a deployment fact worth seeing, so it is reported as "n/a"
            # rather than worked around.
            if hasattr(fitted, "predict"):
                start = time.perf_counter()
                with quiet():
                    fitted.predict(X)
                predict_times.append(time.perf_counter() - start)

        rows.append({
            "model": name,
            "fit_ms": 1000 * float(np.median(fit_times)),
            "predict_us_per_row": (1e6 * float(np.median(predict_times)) / max(len(X), 1)
                                   if predict_times else float("nan")),
            "model_kb": len(pickle.dumps(fitted)) / 1024,
        })
    return pd.DataFrame(rows).set_index("model").sort_values("fit_ms")


# --------------------------------------------------------------------------- #
# 6. Checks -- what main.py asserts about the ladder
# --------------------------------------------------------------------------- #


def check_baselines(results: BaselineResults, X: pd.DataFrame, y=None) -> List[Check]:
    """Assertions that catch a broken ladder, a leak, or a silently useless model."""
    rec = _Recorder("baseline models")
    task, board = results.task, results.board
    metric = HEADLINE_METRIC[task][0]

    rec.record("every baseline produced a score",
               bool(len(board)) and not board[metric].isna().all(),
               f"{len(board)} baselines, headline metric {metric}")

    def finite_predictions():
        bad = [name for name, preds in results.predictions.items()
               if preds.dtype.kind == "f" and not np.isfinite(preds).all()]
        return not bad, f"{len(results.predictions)} prediction sets checked"

    rec.guard("predictions are finite", finite_predictions)

    def right_length():
        expected = len(pd.DataFrame(X))
        bad = {name: len(preds) for name, preds in results.predictions.items()
               if len(preds) != expected}
        return not bad, f"all {expected} rows predicted"

    rec.guard("one prediction per row", right_length)

    # The naive floor has to behave like a naive floor, or the metric is misread.
    if task == "regression" and "mean" in board.index:
        r2 = board.loc["mean", "r2_raw"]
        # Out of fold this is slightly negative, never exactly zero: each fold
        # predicts its own training mean against a validation set that has a
        # different one. A large value either way means the ladder is broken.
        rec.record("the mean baseline scores R2 ~ 0",
                   bool(abs(float(r2)) < 0.05),
                   f"got {float(r2):+.4f} out-of-fold")
    elif task == "classification" and "most frequent class" in board.index and y is not None:
        share = float(pd.Series(np.asarray(y).ravel()).value_counts(normalize=True).max())
        accuracy = float(board.loc["most frequent class", "accuracy"])
        rec.record("the majority baseline scores the majority share",
                   abs(accuracy - share) < 0.02,
                   f"accuracy {accuracy:.3f} vs majority share {share:.3f}")
    elif task == "clustering" and "one cluster" in board.index:
        rec.record("internal indices are undefined for a single cluster",
                   bool(np.isnan(board.loc["one cluster", "silhouette"])),
                   "silhouette is NaN, not 0")
    else:
        rec.skip("the naive floor behaves as expected", f"no floor baseline run for {task}")

    # A model baseline that cannot beat the naive floor means the ladder is broken
    # or the features carry nothing -- either way it must be visible.
    bar = results.analysis.get("bar_for_a_real_model")
    if bar:
        rec.record("at least one model baseline beats the strongest naive one",
                   bar["model_baselines_beating_it"] > 0,
                   f"{bar['model_baselines_beating_it']} of {bar['of']} beat "
                   f"{bar['baseline']} ({metric} {bar['score']:.4f})")
    else:
        rec.skip("at least one model baseline beats the strongest naive one",
                 "no naive baseline in this run")

    if results.benchmark is not None:
        rec.record("benchmark covers every baseline",
                   set(results.benchmark.index) == set(board.index),
                   f"{len(results.benchmark)} timed")
    else:
        rec.skip("benchmark covers every baseline", "benchmarking disabled")

    return rec.checks


# --------------------------------------------------------------------------- #
# Smoke test: python src/models.py
# --------------------------------------------------------------------------- #


def _synthetic_regression(rng) -> Tuple[pd.DataFrame, pd.Series]:
    """A multiplicative target, like the capstone's: value = count^a * aov^b * noise."""
    n = 400
    X = pd.DataFrame({
        COUNT_COL: rng.lognormal(1.5, 0.9, n),
        AOV_COL: rng.lognormal(4.2, 0.5, n),
        "days_since_first_purchase": rng.uniform(30, 1200, n),
        "days_since_last_purchase": rng.uniform(1, 400, n),
    })
    y = (np.exp(3.4) * X[COUNT_COL] ** 0.3 * X[AOV_COL] ** 0.86
         * X["days_since_last_purchase"] ** -0.32 * rng.lognormal(0, 0.1, n))
    return X, pd.Series(y, name="value")


def _synthetic_classification(rng) -> Tuple[pd.DataFrame, pd.Series]:
    """An 80/20 split, so the majority-class baseline says something worth hearing."""
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + 0.4 * rng.normal(size=n) > 0.85).astype(int), name="churned")
    return X, y


def _synthetic_clustering(rng) -> pd.DataFrame:
    """Three well-separated blobs: k-means should beat random labels comfortably."""
    def blob(cx: float, cy: float, n: int) -> pd.DataFrame:
        return pd.DataFrame({"x": rng.normal(cx, 0.3, n), "y": rng.normal(cy, 0.3, n)})

    return pd.concat([blob(0, 0, 100), blob(4, 4, 100), blob(0, 5, 100)],
                     ignore_index=True)


def _smoke_test() -> int:
    """Run all three ladders on synthetic data and report; 0 if every check passed.

    This is what ``python src/models.py`` does. It needs no dataset, no
    preprocessor and nothing else from the project, so it also serves as the
    worked example of using this module on a problem that is not the capstone.
    """
    print("=" * 78)
    print("src/models.py -- BASELINE LADDER SMOKE TEST")
    print("=" * 78)
    print("  Synthetic data, all three tasks, no project dependencies.")

    rng = np.random.default_rng(42)
    failures: List[str] = []

    cases = [
        ("regression", *_synthetic_regression(rng), {}),
        ("classification", *_synthetic_classification(rng), {}),
        ("clustering", _synthetic_clustering(rng), None, {"n_clusters": 3}),
    ]

    for task, X, y, extra in cases:
        print("\n" + "-" * 78)
        print(f"{task.upper()}  ({len(X)} rows, inferred as {infer_task(y)!r})")
        print("-" * 78)
        results = evaluate_baselines(X, y, task=task, benchmark_repeats=1, **extra)

        metric = HEADLINE_METRIC[task][0]
        columns = [c for c in _metric_names(task) if c in results.board.columns][:4]
        print(results.board[columns].round(4).to_string())
        print(f"\n  best: {results.best}  ({metric})")

        bar = results.analysis.get("bar_for_a_real_model")
        if bar:
            print(f"  bar to clear: {bar['baseline']} at {bar['score']:.4f} -- "
                  f"{bar['model_baselines_beating_it']} of {bar['of']} model baselines beat it")
        if not results.analysis.get("metrics_agree", True):
            print(f"  {results.analysis['disagreement']}")

        print("\n  cost (median of 1 fit):")
        print(results.benchmark.round(3).to_string())

        print("\n  checks:")
        for check in results.checks:
            print(f"    [{check.status:4s}] {check.name}"
                  f"{('  -- ' + check.detail) if check.detail else ''}")
            if check.failed:
                failures.append(f"{task}: {check.name}")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("All three ladders ran and every check passed.")
    print("In the project this module is used through main.py (stage 9) and train.py.")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(_smoke_test())
