"""Cross-validation, hyperparameter tuning, and the fight against over- and underfitting.

Every model in this project -- the baselines of ``src/models.py`` and the two
architectures of ``src/advanced_models.py`` -- is optimised here, so that the
protocol is one thing rather than three slightly different things. The module is
task-agnostic: regression, classification and clustering each get the
cross-validation scheme, the search space and the diagnosis that suit them.

The four steps
--------------
**1. Cross-validation setup** (:func:`make_cv`, :func:`cv_report`). Which splitter,
and why. The default for regression here is *not* plain K-fold: with skew 6.4 and
38% of value in the top decile (EDA 3), a random split can hand one fold most of
the whales and the fold-to-fold spread then measures the split rather than the
model. :func:`make_cv` therefore stratifies regression folds on value deciles, the
same trick the train/test split uses, and ``cv_report`` shows what each fold got.

**2. Hyperparameter tuning** (:data:`SEARCH_SPACES`, :func:`tune_models`). One
registry covering every developed model, three search methods (random, grid,
successive halving), and scoring that is *the headline metric itself* rather than
a proxy -- tuning a regression on R2 while ranking it by Spearman optimises the
wrong thing on a heavy-tailed target.

**3. Model optimization** (:func:`diagnose_fit`, :func:`learning_curve_report`,
:func:`select_within_one_se`). Tuning finds the best score; this step asks whether
that score is *trustworthy*. Three outcomes and three different actions:

    overfitting     train score >> CV score. The model has memorised the
                    training rows. Act by *reducing effective capacity*: the
                    one-standard-error rule below picks the simplest model whose
                    score is statistically indistinguishable from the best.
    underfitting    both scores low, and close together. The model cannot
                    represent the problem. Act by *adding capacity* -- a richer
                    representation usually beats a bigger estimator (readme.md 16).
    balanced        a small gap, both scores high. Nothing to do.

The **one-standard-error rule** is the principled answer to overfitting in model
*selection*, as opposed to in fitting: among models within one standard error of
the best cross-validated score, prefer the simplest. It is standard practice in
the CART and glmnet traditions and it costs a little accuracy for a lot of
robustness -- which is the right trade when the differences are inside the noise
anyway (see :func:`~src.advanced_models.paired_test`).

**4. Final selection and validation** (:func:`select_final_model`,
:func:`validate_final_model`). One model is chosen, refitted on the whole training
split, and scored **once** on the held-out test set. The gap between the
cross-validated estimate and that test score is reported as *optimism*: a large
positive gap means the selection itself overfitted the folds, which is the failure
mode that survives every other precaution.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone

# Relative when imported as part of the package, absolute when run as a script.
if __package__ in (None, ""):                                    # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models import (HEADLINE_METRIC, build_baselines, cross_validate_model,
                            infer_task, score_predictions)
    from src.preprocessing import Check, _Recorder, quiet
else:
    from .models import (HEADLINE_METRIC, build_baselines, cross_validate_model,
                         infer_task, score_predictions)
    from .preprocessing import Check, _Recorder, quiet

#: A library logger: it inherits the handler main.py installs on "capstone".
logger = logging.getLogger("capstone.model_optimization")

__all__ = [
    "CV_STRATEGIES",
    "make_cv",
    "cv_report",
    "SEARCH_SPACES",
    "search_space_for",
    "SEARCH_METHODS",
    "tune_models",
    "scoring_for",
    "diagnose_fit",
    "learning_curve_report",
    "validation_curve_report",
    "select_within_one_se",
    "OptimizationResults",
    "optimise_models",
    "select_final_model",
    "validate_final_model",
    "check_optimization",
]

#: Train-minus-CV gap above which a model is called overfitted, as a share of the
#: train score. Not a law -- a threshold, stated so it can be argued with.
OVERFIT_GAP = 0.05

#: CV score below which a model is called underfitted, on metrics bounded at 1.
UNDERFIT_SCORE = 0.60


# --------------------------------------------------------------------------- #
# 1. Cross-validation setup
# --------------------------------------------------------------------------- #

CV_STRATEGIES = ("auto", "kfold", "stratified", "repeated", "shuffle", "timeseries")


def make_cv(
    task: str = "regression",
    y=None,
    strategy: str = "auto",
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
):
    """The splitter for a task, and the reason it is that one.

    ``auto`` resolves to:

    * **regression** -> K-fold stratified on target *deciles*. Plain K-fold is the
      usual default and it is the wrong one here: a heavy-tailed target (EDA 3)
      makes fold composition random, so the spread between folds measures the
      split. Stratifying on deciles gives every fold the same share of whales,
      which is also what the train/test split does -- one protocol, not two.
    * **classification** -> stratified K-fold, so every fold carries the class
      balance. On an imbalanced problem plain K-fold can hand a fold no positives
      at all, and the metric for that fold is then undefined or absurd.
    * **clustering** -> ``None``. There is no held-out notion of correctness;
      stability across subsamples is the analogue and lives in
      ``src.advanced_models.fold_scores``.

    ``repeated`` runs the whole thing several times with different shuffles, which
    is the honest way to shrink fold noise when a decision rests on a small
    difference -- at a linear cost in time.
    """
    from sklearn.model_selection import (KFold, RepeatedKFold, ShuffleSplit,
                                         StratifiedKFold, TimeSeriesSplit)

    if strategy not in CV_STRATEGIES:
        raise ValueError(f"strategy must be one of {CV_STRATEGIES}, got {strategy!r}")
    if task == "clustering":
        return None

    if strategy == "auto":
        strategy = "stratified"
    if strategy == "timeseries":
        # Here for completeness and for the day this project moves to
        # transactional data, where a temporal split is the only valid one
        # (readme.md 7.1). It ignores y by construction.
        return TimeSeriesSplit(n_splits=n_splits)
    if strategy == "shuffle":
        return ShuffleSplit(n_splits=n_splits, test_size=1 / n_splits,
                            random_state=random_state)
    if strategy == "repeated":
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                             random_state=random_state)
    if strategy == "kfold" or y is None:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # "stratified": on classes for a classifier, on deciles for a regression.
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return _DecileStratifiedKFold(n_splits=n_splits, random_state=random_state)


class _DecileStratifiedKFold:
    """K-fold that balances a continuous target across folds by binning it.

    scikit-learn has no stratified splitter for regression, so this wraps
    ``StratifiedKFold`` over quantile bins of ``y``. It exposes the splitter API
    (``split``, ``get_n_splits``) and nothing else, which is all a CV consumer
    needs.
    """

    def __init__(self, n_splits: int = 5, n_bins: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        from sklearn.model_selection import KFold, StratifiedKFold

        if y is None:
            yield from KFold(n_splits=self.n_splits, shuffle=True,
                             random_state=self.random_state).split(X)
            return
        values = pd.Series(np.asarray(y, dtype=float).ravel())
        bins = min(self.n_bins, max(2, values.nunique() // self.n_splits))
        strata = pd.qcut(values.rank(method="first"), bins, labels=False, duplicates="drop")
        inner = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                random_state=self.random_state)
        yield from inner.split(X, strata)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def __repr__(self) -> str:
        return (f"DecileStratifiedKFold(n_splits={self.n_splits}, n_bins={self.n_bins}, "
                f"random_state={self.random_state})")


def cv_report(cv, X: pd.DataFrame, y=None, task: str = "regression") -> pd.DataFrame:
    """What each fold actually received -- the check that a splitter did its job.

    For a regression it reports each validation fold's median and top-decile
    share; for a classification, the positive rate. Folds that differ wildly on
    these are the reason a model's score "moves" between runs.
    """
    if cv is None:
        return pd.DataFrame()
    y = None if y is None else pd.Series(np.asarray(y).ravel())
    rows = []
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        row = {"fold": i, "train_rows": len(train_idx), "val_rows": len(val_idx)}
        if y is not None:
            values = y.iloc[val_idx]
            if task == "classification":
                row["positive_rate"] = float(pd.Series(values).value_counts(
                    normalize=True).max())
            else:
                row["median"] = float(values.median())
                top = np.sort(values.to_numpy())[::-1][: max(1, len(values) // 10)]
                row["top_decile_share"] = float(top.sum() / values.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


# --------------------------------------------------------------------------- #
# 2. Hyperparameter tuning
# --------------------------------------------------------------------------- #

SEARCH_METHODS = ("random", "grid", "halving")


def _spaces_regression() -> Dict[str, dict]:
    from scipy.stats import loguniform, randint, uniform

    return {
        # -- baselines (src/models.py) -------------------------------------- #
        "ridge on log(y)": {"model__alpha": loguniform(1e-3, 1e3)},
        "decision tree (depth 3)": {"model__max_depth": [2, 3, 4, 6, 8, None],
                                    "model__min_samples_leaf": randint(1, 20)},
        "k-NN (k=10)": {"model__n_neighbors": randint(2, 40),
                        "model__weights": ["uniform", "distance"]},
        # -- advanced architectures (src/advanced_models.py) ---------------- #
        "gradient boosting": {"model__n_estimators": randint(200, 900),
                              "model__learning_rate": loguniform(0.01, 0.3),
                              "model__max_depth": randint(2, 8),
                              "model__subsample": uniform(0.6, 0.4)},
        "neural network (MLP)": {
            "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
            "model__alpha": loguniform(1e-5, 1e-1),
            "model__learning_rate_init": loguniform(1e-4, 1e-2)},
    }


def _spaces_classification() -> Dict[str, dict]:
    from scipy.stats import loguniform, randint, uniform

    return {
        "logistic regression": {"model__C": loguniform(1e-3, 1e3),
                                "model__class_weight": [None, "balanced"]},
        "decision tree (depth 3)": {"model__max_depth": [2, 3, 4, 6, 8, None],
                                    "model__min_samples_leaf": randint(1, 20),
                                    "model__class_weight": [None, "balanced"]},
        "k-NN (k=10)": {"model__n_neighbors": randint(2, 40),
                        "model__weights": ["uniform", "distance"]},
        "gradient boosting": {"model__n_estimators": randint(200, 900),
                              "model__learning_rate": loguniform(0.01, 0.3),
                              "model__max_depth": randint(2, 8),
                              "model__subsample": uniform(0.6, 0.4)},
        "neural network (MLP)": {
            "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "model__alpha": loguniform(1e-5, 1e-1),
            "model__learning_rate_init": loguniform(1e-4, 1e-2)},
    }


def _spaces_clustering() -> Dict[str, dict]:
    from scipy.stats import randint, uniform

    return {
        "k-means": {"model__n_clusters": [2, 3, 4, 5, 6, 8]},
        "agglomerative (ward)": {"model__n_clusters": [2, 3, 4, 5, 6, 8]},
        "gaussian mixture": {"model__n_components": [2, 3, 4, 5, 6],
                             "model__covariance_type": ["full", "tied", "diag"]},
        "DBSCAN": {"model__eps": uniform(0.3, 1.5), "model__min_samples": randint(3, 20)},
    }


#: Every developed model that has anything worth tuning, per task. Naive
#: baselines are absent on purpose: a mean has no hyperparameters, and tuning a
#: baseline would stop it being a baseline (src/models.py).
SEARCH_SPACES: Dict[str, Callable[[], Dict[str, dict]]] = {
    "regression": _spaces_regression,
    "classification": _spaces_classification,
    "clustering": _spaces_clustering,
}


def search_space_for(name: str, task: str = "regression") -> Optional[dict]:
    """The space for one model, or ``None`` when it has nothing to tune."""
    return SEARCH_SPACES[task]().get(name)


def scoring_for(task: str):
    """Search on the metric the leaderboard uses, not a proxy for it.

    This looks pedantic and is not: scoring a regression search by R2 while the
    comparison ranks by Spearman optimises the wrong thing on a heavy-tailed
    target -- R2 chases the largest customers, Spearman cares only about order --
    and a model tuned on the wrong metric can come back *worse* on the right one.
    """
    if task == "classification":
        return "f1_macro"

    from scipy import stats
    from sklearn.metrics import make_scorer

    def spearman(y_true, y_pred) -> float:
        y_pred = np.asarray(y_pred, dtype=float)
        if np.allclose(y_pred, y_pred[0]):
            return 0.0
        return float(stats.spearmanr(y_pred, y_true)[0])

    return make_scorer(spearman)


def tune_models(
    models: Dict[str, object],
    X: pd.DataFrame,
    y=None,
    task: str = "regression",
    cv=None,
    method: str = "random",
    n_iter: int = 20,
    spaces: Optional[Dict[str, dict]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Search each model's hyperparameters; return the tuned models and the results.

    Three methods, and the trade between them is time against coverage:

    ``random``   N draws from the space. The default, because it covers many
                 dimensions cheaply and the budget is explicit.
    ``grid``     every combination. Only sane for small discrete spaces; used for
                 clustering, where the interesting choices are discrete anyway.
    ``halving``  successive halving: many candidates on little data, survivors on
                 more. Much faster when fits are expensive, at the cost of
                 possibly discarding a candidate that needed the full data.

    The search runs **inside the same folds** the evaluation later uses, which is
    not nested cross-validation: the tuned model has seen every fold's training
    rows through the search, so its cross-validated score is optimistic. The
    returned table records the method and budget, and
    :func:`validate_final_model` measures that optimism against a held-out split
    rather than assuming it away.
    """
    if method not in SEARCH_METHODS:
        raise ValueError(f"method must be one of {SEARCH_METHODS}, got {method!r}")
    spaces = spaces if spaces is not None else SEARCH_SPACES[task]()
    cv = cv if cv is not None else make_cv(task, y, random_state=random_state)

    tuned: Dict[str, object] = {}
    rows: List[dict] = []

    for name, model in models.items():
        space = spaces.get(name)
        if not space:
            tuned[name] = model
            rows.append({"model": name, "method": "none", "n_candidates": 0,
                         "cv_score": np.nan, "best_params": {},
                         "note": "nothing to tune"})
            continue

        space, adaptation = _adapt_space(model, space)
        _assert_tunable(name, model, space)
        if not space:
            tuned[name] = model
            rows.append({"model": name, "method": "none", "n_candidates": 0,
                         "cv_score": np.nan, "best_params": {},
                         "note": "no searchable parameter survived adaptation: "
                                 f"{adaptation['dropped']}"})
            continue

        if task == "clustering":
            tuned[name], row = _search_clustering(model, X, space, n_iter=n_iter,
                                                  random_state=random_state)
        else:
            tuned[name], row = _search_supervised(model, X, y, task, space, cv,
                                                  method=method, n_iter=n_iter,
                                                  random_state=random_state, n_jobs=n_jobs)
        row["note"] = "; ".join(filter(None, [
            row.get("note", ""),
            f"renamed for this implementation: {adaptation['renamed']}"
            if adaptation["renamed"] else "",
            f"no counterpart here, dropped: {adaptation['dropped']}"
            if adaptation["dropped"] else "",
        ]))
        if adaptation["renamed"] or adaptation["dropped"]:
            logger.info("%s: %s", name, row["note"])
        rows.append({"model": name, **row})
        logger.info("tuned %-28s %s search, best %.4f", name, row["method"],
                    row["cv_score"])

    return tuned, pd.DataFrame(rows).set_index("model")


def _search_supervised(model, X, y, task, space, cv, method, n_iter, random_state, n_jobs):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    scoring = scoring_for(task)
    if method == "grid":
        search = GridSearchCV(model, space, cv=cv, scoring=scoring, n_jobs=n_jobs)
    elif method == "halving":
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV

        search = HalvingRandomSearchCV(model, space, cv=cv, scoring=scoring,
                                       random_state=random_state, n_jobs=n_jobs)
    else:
        search = RandomizedSearchCV(model, space, n_iter=n_iter, cv=cv, scoring=scoring,
                                    random_state=random_state, n_jobs=n_jobs)

    with quiet(), warnings.catch_warnings():
        # A search fits hundreds of models; their convergence warnings are noise.
        warnings.simplefilter("ignore")
        search.fit(X, np.asarray(y).ravel())

    results = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, {
        "method": method,
        "n_candidates": int(len(results)),
        "cv_score": float(search.best_score_),
        "cv_score_sd": float(results.loc[search.best_index_, "std_test_score"]),
        "best_params": readable_params(search.best_params_),
        "note": "",
    }


def _search_clustering(model, X: pd.DataFrame, space: dict, n_iter: int = 12,
                       random_state: int = 42):
    """Grid-search a clusterer by silhouette -- there is no held-out score to use."""
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import ParameterGrid, ParameterSampler

    frame = pd.DataFrame(X)
    try:
        candidates = list(ParameterGrid(space))
        method = "grid"
    except TypeError:                              # distributions, not lists
        candidates = list(ParameterSampler(space, n_iter=n_iter, random_state=random_state))
        method = "random"

    best_model, best_score, best_params = model, -np.inf, {}
    for params in candidates:
        candidate = clone(model).set_params(**params)
        try:
            with quiet(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = (candidate.fit_predict(frame) if hasattr(candidate, "fit_predict")
                          else candidate.fit(frame).predict(frame))
            if len(np.unique(labels)) < 2:
                continue                           # one cluster: silhouette undefined
            score = float(silhouette_score(frame, labels))
        except Exception:                          # noqa: BLE001 - a bad corner of the grid
            continue
        if score > best_score:
            best_model, best_score, best_params = candidate, score, params

    return best_model, {"method": method, "n_candidates": len(candidates),
                        "cv_score": best_score, "cv_score_sd": np.nan,
                        "best_params": readable_params(best_params),
                        "note": "scored by silhouette on all rows; no held-out split exists"}


#: The same hyperparameter under two libraries' names, plus the ones that simply
#: do not exist in the other implementation.
#:
#: This exists because the boosting estimator is *not one estimator*: XGBoost
#: when it is installed, scikit-learn's ``HistGradientBoosting`` when it is not
#: (``src/advanced_models.py``). Same architecture, different spelling -- and
#: ``subsample`` has no counterpart at all, because the histogram booster does
#: not subsample rows. A search space written for one and handed to the other
#: would otherwise fail on a machine that merely lacks an optional dependency.
#:
#: An empty tuple means "no equivalent": the parameter is dropped and the drop is
#: reported, never silently swallowed.
PARAM_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "n_estimators": ("max_iter",),
    "max_iter": ("n_estimators",),
    "colsample_bytree": ("max_features",),
    "reg_lambda": ("l2_regularization", "alpha"),
    "min_child_weight": ("min_samples_leaf",),
    "subsample": (),
}


def _adapt_space(model, space: dict) -> Tuple[dict, dict]:
    """Match the space's parameter names to the estimator that will actually be tuned.

    Four ways a name can be right, tried in order:

    1. **exactly as written** -- an estimator at the end of a pipeline, ``model__alpha``;
    2. **bare** -- a model handed in without preprocessing has plain ``alpha``;
    3. **nested deeper** -- a ``LogTargetRegressor`` around a pipeline puts Ridge's
       alpha at ``model__model__alpha``, so the shortest path ending in the
       parameter's own name wins;
    4. **a synonym** -- the same knob under another library's name
       (:data:`PARAM_SYNONYMS`).

    Anything still unmatched is either a parameter with no counterpart in this
    implementation, which is dropped and reported, or a typo, which is left in
    place for :func:`_assert_tunable` to reject loudly. The difference matters:
    silently tuning nothing is the failure this whole layer exists to prevent.

    Returns the adapted space and a note of what was renamed or dropped.
    """
    available = set(model.get_params(deep=True))
    adapted: dict = {}
    renamed: Dict[str, str] = {}
    dropped: List[str] = []

    def resolve(name: str) -> Optional[str]:
        """The path on this estimator that exposes ``name``, if there is one."""
        if name in available:
            return name
        matches = sorted((p for p in available if p.split("__")[-1] == name), key=len)
        return matches[0] if matches else None

    for key, values in space.items():
        if key in available:
            adapted[key] = values
            continue

        bare = key.split("__")[-1]
        target = resolve(bare)
        if target is not None:
            adapted[target] = values
            continue

        synonyms = PARAM_SYNONYMS.get(bare)
        if synonyms is None:
            # Not a known knob at all: keep it so the assertion below names it.
            adapted[key] = values
            continue

        for synonym in synonyms:
            target = resolve(synonym)
            if target is not None:
                adapted[target] = values
                renamed[bare] = target.split("__")[-1]
                break
        else:
            dropped.append(bare)

    return adapted, {"renamed": renamed, "dropped": dropped}


def _assert_tunable(name: str, model, space: dict) -> None:
    """Every searched parameter must actually exist on the estimator.

    scikit-learn raises for an unknown ``Pipeline`` parameter, but an estimator
    that accepts arbitrary keywords -- XGBoost does -- swallows a mis-prefixed one
    and carries on at its defaults. A search that silently tunes nothing is worse
    than no search, so the names are checked before any fitting happens.
    """
    available = set(model.get_params(deep=True))
    unknown = [key for key in space if key not in available]
    if unknown:
        raise ValueError(f"{name}: search parameters not present on the estimator: "
                         f"{unknown}. Expected names like 'model__<param>'.")


def readable_params(params: dict) -> dict:
    """Search results, printable: numpy scalars and tuples become plain values."""
    out = {}
    for key, value in params.items():
        name = key.replace("model__", "")
        if isinstance(value, (bool, str)) or value is None:
            out[name] = value
        elif isinstance(value, (int, np.integer)):
            out[name] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[name] = round(float(value), 5)
        else:
            out[name] = str(value)
    return out


# --------------------------------------------------------------------------- #
# 3. Model optimization: over- and underfitting
# --------------------------------------------------------------------------- #


def diagnose_fit(
    model,
    X: pd.DataFrame,
    y=None,
    task: str = "regression",
    cv=None,
    name: str = "model",
) -> dict:
    """Compare the training score with the cross-validated one, and name the problem.

    The gap between what a model scores on rows it fitted and rows it did not is
    the whole of the bias-variance story in one number:

    * a **large gap** means variance -- it memorised, and the fix is less
      effective capacity (regularisation, fewer features, a simpler model, or
      :func:`select_within_one_se`);
    * **both low and close** means bias -- it cannot represent the problem, and
      the fix is more capacity, which on this project means a better
      *representation* rather than a bigger estimator (readme.md 16.2);
    * a **small gap at a high score** is what success looks like.

    Clustering has no such split, so the diagnosis there is stability across
    subsamples instead, reported as the same fields with ``train`` omitted.
    """
    metric, higher = HEADLINE_METRIC[task]
    cv = cv if cv is not None else make_cv(task, y)

    if task == "clustering":
        return {"model": name, "train_score": float("nan"), "cv_score": float("nan"),
                "gap": float("nan"), "verdict": "not applicable",
                "action": "clustering has no held-out score; see fold stability"}

    fitted = clone(model)
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(X, np.asarray(y).ravel())
        train_pred = fitted.predict(X)
    train_row = score_predictions(name, task, np.asarray(y).ravel(), train_pred)
    train_score = float(train_row.get(metric, np.nan))

    cv_row, _ = cross_validate_model(name, model, X, y, cv=cv, task=task)
    cv_score = float(cv_row.get(metric, np.nan))

    gap = train_score - cv_score if higher else cv_score - train_score
    relative = gap / abs(train_score) if train_score else np.nan

    if np.isfinite(relative) and relative > OVERFIT_GAP:
        verdict = "overfitting"
        action = ("reduce effective capacity: stronger regularisation, fewer features, "
                  "or take the simplest model within one standard error")
    elif np.isfinite(cv_score) and cv_score < UNDERFIT_SCORE:
        verdict = "underfitting"
        action = ("add capacity -- on this project a better representation "
                  "(log, expansion) beats a bigger estimator")
    else:
        verdict = "balanced"
        action = "nothing to fix; the gap is small and the score is high"

    return {"model": name, "train_score": train_score, "cv_score": cv_score,
            "gap": float(gap), "relative_gap": float(relative), "verdict": verdict,
            "action": action}


def learning_curve_report(
    model,
    X: pd.DataFrame,
    y=None,
    task: str = "regression",
    cv=None,
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    random_state: int = 42,
) -> pd.DataFrame:
    """Train and validation score against training-set size.

    The shape answers a question no single score can: *would more data help?*
    Curves that have converged and sit low mean bias -- more rows will not save
    it. Curves still separating at the right edge mean variance, and more data
    will close some of the gap.
    """
    from sklearn.model_selection import learning_curve

    cv = cv if cv is not None else make_cv(task, y, random_state=random_state)
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, train, validation = learning_curve(
            model, X, np.asarray(y).ravel(), cv=cv, scoring=scoring_for(task),
            train_sizes=list(fractions), n_jobs=-1, random_state=random_state)

    return pd.DataFrame({
        "train_rows": sizes,
        "train_score": train.mean(axis=1),
        "cv_score": validation.mean(axis=1),
        "cv_score_sd": validation.std(axis=1),
        "gap": train.mean(axis=1) - validation.mean(axis=1),
    }).set_index("train_rows")


def validation_curve_report(
    model,
    X: pd.DataFrame,
    y=None,
    param_name: str = "model__alpha",
    param_range: Sequence = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    task: str = "regression",
    cv=None,
) -> pd.DataFrame:
    """One hyperparameter against train and validation score -- the complexity curve.

    Where the two lines diverge is where extra capacity starts being spent on
    memorising. It is the picture behind every regularisation choice in the
    project, and it is cheap enough to look at before arguing about one.
    """
    from sklearn.model_selection import validation_curve

    cv = cv if cv is not None else make_cv(task, y)
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train, validation = validation_curve(
            model, X, np.asarray(y).ravel(), param_name=param_name,
            param_range=list(param_range), cv=cv, scoring=scoring_for(task), n_jobs=-1)

    return pd.DataFrame({
        param_name.replace("model__", ""): list(param_range),
        "train_score": train.mean(axis=1),
        "cv_score": validation.mean(axis=1),
        "gap": train.mean(axis=1) - validation.mean(axis=1),
    }).set_index(param_name.replace("model__", ""))


def select_within_one_se(
    scores: Dict[str, float],
    standard_errors: Dict[str, float],
    complexity: Optional[Dict[str, float]] = None,
    higher_is_better: bool = True,
) -> Tuple[str, dict]:
    """The one-standard-error rule: the simplest model that is as good as the best.

    Take the best cross-validated score, add (or subtract) one standard error to
    get a threshold, and among every model that clears it choose the *least
    complex*. The reasoning is that differences inside one standard error are not
    evidence, so spending complexity on them buys variance rather than accuracy --
    the same argument the paired test makes in ``src/advanced_models.py``, applied
    to selection instead of comparison.

    ``complexity`` is any ordering where smaller means simpler; the caller
    supplies it because "simpler" is domain knowledge, not arithmetic. With none
    given, the ranking falls back to the scores themselves, which reduces the rule
    to "take the best".
    """
    if not scores:
        raise ValueError("no scores to select from")
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid:
        raise ValueError("every score is NaN")

    best = max(valid, key=valid.get) if higher_is_better else min(valid, key=valid.get)
    se = float(standard_errors.get(best, 0.0) or 0.0)
    threshold = valid[best] - se if higher_is_better else valid[best] + se

    within = [k for k, v in valid.items()
              if (v >= threshold if higher_is_better else v <= threshold)]
    ranking = complexity or {k: (-v if higher_is_better else v) for k, v in valid.items()}
    choice = min(within, key=lambda k: ranking.get(k, np.inf))

    return choice, {
        "best_by_score": best,
        "best_score": valid[best],
        "chosen_score": valid[choice],
        "standard_error": se,
        "threshold": float(threshold),
        "within_one_se": sorted(within),
        "chosen": choice,
        "traded_accuracy": float(valid[best] - valid[choice]) if higher_is_better
        else float(valid[choice] - valid[best]),
    }


# --------------------------------------------------------------------------- #
# 4. Final selection and validation
# --------------------------------------------------------------------------- #


@dataclass
class OptimizationResults:
    """Everything one optimization pass produced."""

    task: str
    cv: object = None
    cv_folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuning: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnosis: pd.DataFrame = field(default_factory=pd.DataFrame)
    learning_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    models: Dict[str, object] = field(default_factory=dict)
    selection: Dict[str, object] = field(default_factory=dict)
    final_name: str = ""
    final_model: object = None
    validation: Dict[str, object] = field(default_factory=dict)
    checks: List[Check] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "cv": repr(self.cv),
            "cv_folds": self.cv_folds.reset_index().to_dict(orient="records"),
            "tuning": self.tuning.reset_index().to_dict(orient="records")
            if len(self.tuning) else [],
            "diagnosis": self.diagnosis.reset_index(drop=True).to_dict(orient="records")
            if len(self.diagnosis) else [],
            "learning_curve": self.learning_curve.reset_index().to_dict(orient="records")
            if len(self.learning_curve) else [],
            "selection": self.selection,
            "final_model": self.final_name,
            "validation": self.validation,
            "checks": [c.as_dict() for c in self.checks],
        }


#: Rough complexity ordering, smallest = simplest. Used by the one-standard-error
#: rule, which needs to know what "simpler" means and cannot work it out alone.
COMPLEXITY_RANK = {
    "mean": 0, "median": 0, "most frequent class": 0, "stratified random": 0,
    "uniform random": 0, "one cluster": 0, "random labels": 0,
    "heuristic (product of two columns)": 1,
    "linear regression": 2, "linear regression on log(y)": 2, "logistic regression": 2,
    "ridge on log(y)": 3,
    "decision tree (depth 3)": 4,
    "k-NN (k=10)": 5, "k-means": 5, "agglomerative (ward)": 5,
    "gaussian mixture": 6, "DBSCAN": 6,
    "gradient boosting": 8,
    "neural network (MLP)": 9,
}


def optimise_models(
    models: Dict[str, object],
    X: pd.DataFrame,
    y=None,
    task: Optional[str] = None,
    cv=None,
    strategy: str = "auto",
    method: str = "random",
    n_iter: int = 20,
    diagnose: bool = True,
    learning_curves: bool = True,
    random_state: int = 42,
) -> OptimizationResults:
    """Steps 1-3 for a set of models: split, tune, diagnose, and rank by 1-SE.

    Returns everything needed for step 4, which needs a held-out split and so
    stays in :func:`validate_final_model`.
    """
    task = task or infer_task(y)
    cv = cv if cv is not None else make_cv(task, y, strategy=strategy,
                                           random_state=random_state)
    results = OptimizationResults(task=task, cv=cv)
    results.cv_folds = cv_report(cv, X, y, task=task)

    tuned, tuning = tune_models(models, X, y, task=task, cv=cv, method=method,
                                n_iter=n_iter, random_state=random_state)
    results.models, results.tuning = tuned, tuning

    if diagnose and task != "clustering":
        results.diagnosis = pd.DataFrame(
            [diagnose_fit(model, X, y, task=task, cv=cv, name=name)
             for name, model in tuned.items()])

    select_final_model(results, X, y)

    if learning_curves and task != "clustering":
        results.learning_curve = learning_curve_report(results.final_model, X, y,
                                                       task=task, cv=cv,
                                                       random_state=random_state)
    return results


def select_final_model(
    results: OptimizationResults,
    X: pd.DataFrame,
    y=None,
    complexity: Optional[Dict[str, float]] = None,
) -> Tuple[str, object]:
    """Step 4a: pick one model from the tuned set, by the one-standard-error rule.

    Every model is scored on the same folds; the search score is used where a
    search ran, and a model with nothing to tune gets a plain cross-validation so
    that it can still win -- which on this project it repeatedly does.

    The rule then prefers the *simplest* model whose score is within one standard
    error of the best (:func:`select_within_one_se`). ``complexity`` defaults to
    :data:`COMPLEXITY_RANK`, which encodes "a linear model is simpler than a
    booster" because arithmetic cannot know that.
    """
    task = results.task
    metric, higher = HEADLINE_METRIC[task]
    tuned, tuning = results.models, results.tuning

    from sklearn.model_selection import cross_val_score

    n_splits = (results.cv.get_n_splits(X) if results.cv is not None else 1) or 1
    scores = {name: float(tuning.loc[name, "cv_score"]) for name in tuned
              if name in tuning.index}
    # The rule needs the standard error of the cross-validated *mean*, which is
    # the fold-to-fold standard deviation over sqrt(k) -- not the deviation
    # itself. Using the deviation would widen the margin by more than a factor of
    # two and simplify away models that are genuinely better.
    errors = {name: ((float(tuning.loc[name, "cv_score_sd"]) / np.sqrt(n_splits))
                     if "cv_score_sd" in tuning.columns
                     and np.isfinite(tuning.loc[name, "cv_score_sd"]) else 0.0)
              for name in scores}

    for name, model in tuned.items():
        if np.isfinite(scores.get(name, np.nan)):
            continue
        # A model with nothing to tune never went through a search, so it has
        # neither a score nor a spread yet. Scoring it on the same folds gives
        # both -- and without the spread it would enter the one-standard-error
        # rule with a margin of zero, which is how the untuned models (the ones
        # that keep winning here) would be denied the rule's protection.
        if task == "clustering" or y is None:
            row, _ = cross_validate_model(name, model, X, y, cv=results.cv, task=task)
            scores[name] = float(row.get(metric, np.nan))
            errors[name] = 0.0
            continue
        with quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            folds = cross_val_score(model, X, np.asarray(y).ravel(), cv=results.cv,
                                    scoring=scoring_for(task))
        scores[name] = float(np.mean(folds))
        errors[name] = float(np.std(folds, ddof=1) / np.sqrt(len(folds))) \
            if len(folds) > 1 else 0.0

    choice, selection = select_within_one_se(
        scores,
        {k: (0.0 if not np.isfinite(v) else v) for k, v in errors.items()},
        complexity=complexity or {k: COMPLEXITY_RANK.get(k, 7) for k in scores},
        higher_is_better=higher)

    results.final_name, results.final_model = choice, tuned[choice]
    results.selection = selection
    logger.info("selected %s (best was %s at %.4f)", choice,
                selection["best_by_score"], selection["best_score"])
    return choice, tuned[choice]


def validate_final_model(
    results: OptimizationResults,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
) -> Dict[str, object]:
    """Step 4: refit on all of training, score **once** on the held-out split.

    The number that matters here is not the test score itself but the *optimism*:
    cross-validated score minus test score. Selection and tuning both used the
    folds, so a large positive optimism means the choice was fitted to them --
    the failure that survives clean preprocessing, honest folds and a paired test.
    """
    task = results.task
    metric, higher = HEADLINE_METRIC[task]
    model = clone(results.final_model)

    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "clustering":
            labels = (model.fit_predict(X_test) if hasattr(model, "fit_predict")
                      else model.fit(X_train).predict(X_test))
            row = score_predictions(results.final_name, task, y_pred=labels, X=X_test)
        else:
            model.fit(X_train, np.asarray(y_train).ravel())
            row = score_predictions(results.final_name, task,
                                    np.asarray(y_test).ravel(), model.predict(X_test))

    cv_score = float(results.selection.get("chosen_score", np.nan))
    if not np.isfinite(cv_score) and results.final_name in results.tuning.index:
        cv_score = float(results.tuning.loc[results.final_name, "cv_score"])
    test_score = float(row.get(metric, np.nan))
    optimism = (cv_score - test_score) if higher else (test_score - cv_score)

    validation = {
        "model": results.final_name,
        "metric": metric,
        "cv_score": cv_score,
        "test_score": test_score,
        "optimism": float(optimism),
        "scorecard": row,
        "verdict": _optimism_verdict(results.final_name, optimism, cv_score, test_score),
    }
    results.validation = validation
    results.final_model = model
    return validation


def _optimism_verdict(name: str, optimism: float, cv_score: float, test_score: float) -> str:
    if not np.isfinite(optimism):
        return f"{name}: no held-out score available for this task."
    if optimism > 0.05:
        return (f"{name} scored {cv_score:.4f} in cross-validation and {test_score:.4f} on the "
                f"held-out split: an optimism of {optimism:+.4f}. Tuning and selection fitted "
                "the folds; trust the test number.")
    if optimism < -0.05:
        return (f"{name} scored {test_score:.4f} on the held-out split against {cv_score:.4f} "
                "in cross-validation. An easy test split, or folds made harder by their own "
                "stratification -- either way, do not quote the higher one.")
    return (f"{name}: cross-validated {cv_score:.4f}, held-out {test_score:.4f} "
            f"(optimism {optimism:+.4f}). The selection did not overfit the folds.")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_optimization(results: OptimizationResults) -> List[Check]:
    """What main.py asserts about an optimization pass before anyone acts on it."""
    rec = _Recorder("model optimization")
    task = results.task

    rec.record("a cross-validation scheme was chosen",
               results.cv is not None or task == "clustering",
               repr(results.cv) if results.cv is not None else "clustering: no folds")

    if len(results.cv_folds):
        sizes = results.cv_folds["val_rows"]
        rec.record("folds are the same size to within one row",
                   int(sizes.max() - sizes.min()) <= 1,
                   f"{len(sizes)} folds of {int(sizes.min())}-{int(sizes.max())} rows")
        if "top_decile_share" in results.cv_folds.columns:
            spread = float(results.cv_folds["top_decile_share"].max()
                           - results.cv_folds["top_decile_share"].min())
            rec.record("stratification balanced the tail across folds", spread < 0.15,
                       f"top-decile share varies by {spread:.3f} between folds")
    else:
        rec.skip("folds are the same size to within one row", "no folds for this task")

    if len(results.tuning):
        searched = results.tuning[results.tuning["method"] != "none"]
        rec.record("every tunable model was searched", len(searched) > 0,
                   f"{len(searched)} of {len(results.tuning)} models had a space")
        rec.record("search scores are finite where a search ran",
                   bool(searched.empty or np.isfinite(searched["cv_score"]).all()),
                   f"{len(searched)} searches")
    else:
        rec.skip("every tunable model was searched", "no tuning run")
        rec.skip("search scores are finite where a search ran", "no tuning run")

    if len(results.diagnosis):
        overfit = results.diagnosis[results.diagnosis["verdict"] == "overfitting"]
        rec.record("the final model is not diagnosed as overfitting",
                   results.final_name not in set(overfit["model"]),
                   f"{len(overfit)} of {len(results.diagnosis)} models overfit; "
                   f"chosen model is "
                   f"{results.diagnosis.set_index('model').loc[results.final_name, 'verdict']}")
    else:
        rec.skip("the final model is not diagnosed as overfitting", "no diagnosis run")

    if results.selection:
        rec.record("selection used the one-standard-error rule",
                   "within_one_se" in results.selection,
                   f"{len(results.selection.get('within_one_se', []))} model(s) within 1 SE; "
                   f"chose {results.selection.get('chosen')}")
        rec.record("the simpler choice cost little accuracy",
                   abs(float(results.selection.get("traded_accuracy", 0.0))) <=
                   max(float(results.selection.get("standard_error", 0.0)), 1e-9) + 1e-9,
                   f"traded {float(results.selection.get('traded_accuracy', 0.0)):+.4f} "
                   "for a simpler model")
    else:
        rec.skip("selection used the one-standard-error rule", "no selection run")
        rec.skip("the simpler choice cost little accuracy", "no selection run")

    if results.validation:
        optimism = float(results.validation.get("optimism", np.nan))
        # The tolerance has to know about noise. A held-out split of a few hundred
        # rows has a standard error of its own, so a fixed 0.05 would flag honest
        # runs on small data and miss real optimism on large ones. Three standard
        # errors of the fold spread, or 0.05, whichever is the more forgiving.
        fold_sd = 0.0
        if results.final_name in results.tuning.index and \
                "cv_score_sd" in results.tuning.columns:
            value = results.tuning.loc[results.final_name, "cv_score_sd"]
            fold_sd = float(value) if np.isfinite(value) else 0.0
        tolerance = max(0.05, 3 * fold_sd)
        rec.record("the held-out score is close to the cross-validated one",
                   bool(np.isfinite(optimism) and abs(optimism) <= tolerance),
                   f"optimism {optimism:+.4f} ({results.validation['metric']}), "
                   f"tolerance {tolerance:.4f}")
    else:
        rec.skip("the held-out score is close to the cross-validated one",
                 "final validation not run")
    return rec.checks


# --------------------------------------------------------------------------- #
# Smoke test: python src/model_optimization.py
# --------------------------------------------------------------------------- #


def _smoke_test() -> int:
    """All four steps on synthetic data, for every task, with no project dependencies."""
    if __package__ in (None, ""):
        from src.models import (_synthetic_classification, _synthetic_clustering,
                                _synthetic_regression, build_baselines)
        from src.advanced_models import build_advanced_models
    else:
        from .models import (_synthetic_classification, _synthetic_clustering,
                             _synthetic_regression, build_baselines)
        from .advanced_models import build_advanced_models

    print("=" * 78)
    print("src/model_optimization.py -- OPTIMIZATION SMOKE TEST")
    print("=" * 78)
    print("  CV setup -> hyperparameter tuning -> over/underfit diagnosis -> selection")

    rng = np.random.default_rng(42)
    failures: List[str] = []
    cases = [
        ("regression", *_synthetic_regression(rng), {}),
        ("classification", *_synthetic_classification(rng), {}),
        ("clustering", _synthetic_clustering(rng), None, {"n_clusters": 3}),
    ]

    for task, X, y, extra in cases:
        print("\n" + "-" * 78)
        print(f"{task.upper()}  ({len(X)} rows)")
        print("-" * 78)

        models = {**build_baselines(task, **extra), **build_advanced_models(task, **extra)}
        tunable = [n for n in models if search_space_for(n, task)]
        print(f"  {len(models)} developed models, {len(tunable)} of them tunable")

        print("\n1. CROSS-VALIDATION SETUP")
        cv = make_cv(task, y, strategy="auto")
        print(f"   {cv!r}")
        folds = cv_report(cv, X, y, task=task)
        if len(folds):
            print(folds.round(4).to_string())

        print("\n2-3. TUNING, DIAGNOSIS AND SELECTION")
        results = optimise_models(models, X, y, task=task, cv=cv, n_iter=6,
                                  learning_curves=task == "regression")
        searched = results.tuning[results.tuning["method"] != "none"]
        print(searched[["method", "n_candidates", "cv_score"]].round(4).to_string())

        if len(results.diagnosis):
            print("\n   over/underfitting:")
            print(results.diagnosis[["model", "train_score", "cv_score", "gap",
                                     "verdict"]].round(4).to_string(index=False))

        selection = results.selection
        print(f"\n   one-SE rule: best {selection['best_by_score']} "
              f"({selection['best_score']:.4f} +/- {selection['standard_error']:.4f}); "
              f"{len(selection['within_one_se'])} within 1 SE")
        print(f"   chosen: {results.final_name} "
              f"(traded {selection['traded_accuracy']:+.4f} for simplicity)")

        if len(results.learning_curve):
            print("\n   learning curve (does more data help?):")
            print(results.learning_curve.round(4).to_string())

        print("\n4. FINAL VALIDATION ON A HELD-OUT SPLIT")
        # Shuffled, not positional. The clustering fixture is three blobs written
        # one after another, so a positional split hands the "test" half a single
        # blob -- and a silhouette on one cluster is meaningless. Real callers
        # split randomly (or by time); this mirrors that.
        order = np.random.default_rng(0).permutation(len(X))
        cut = int(0.8 * len(X))
        train_idx, test_idx = order[:cut], order[cut:]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        if y is None:
            y_train = y_test = None
        else:
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        validation = validate_final_model(results, X_train, y_train, X_test, y_test)
        print(f"   {validation['verdict']}")

        results.checks = check_optimization(results)
        print("\n   checks:")
        for check in results.checks:
            print(f"     [{check.status:4s}] {check.name}"
                  f"{('  -- ' + check.detail) if check.detail else ''}")
            if check.failed:
                failures.append(f"{task}: {check.name}")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("All four steps ran for every task and every check passed.")
    print("In the project this module is used through main.py (stage 11).")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(_smoke_test())
