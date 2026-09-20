"""Tests for src/model_optimization.py -- CV setup, tuning, over/underfitting, selection.

Two things are worth testing hardest here, because both fail silently:

* a **search that tunes nothing** (the parameter names never reach the estimator),
  which reports a best score and leaves the model at its defaults;
* a **diagnosis that always says "balanced"**, which would make the whole
  over/underfitting step decorative. The fixtures below are built so that the
  right answer is known: a model that must memorise, and one that cannot fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.model_optimization import (
    COMPLEXITY_RANK,
    CV_STRATEGIES,
    OptimizationResults,
    check_optimization,
    cv_report,
    diagnose_fit,
    learning_curve_report,
    make_cv,
    optimise_models,
    scoring_for,
    search_space_for,
    select_final_model,
    select_within_one_se,
    tune_models,
    validate_final_model,
    validation_curve_report,
)
from src.models import TASKS, build_baselines


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def heavy_tailed(rng):
    """A skewed target, like the capstone's: fold composition matters here."""
    n = 400
    X = pd.DataFrame({"a": rng.lognormal(1.5, 0.9, n), "b": rng.lognormal(4.2, 0.5, n)})
    y = pd.Series(X["a"] ** 0.3 * X["b"] ** 0.86 * rng.lognormal(0, 0.1, n))
    return X, y


@pytest.fixture
def imbalanced(rng):
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + 0.3 * rng.normal(size=n) > 1.4).astype(int))
    return X, y


def _pipe(estimator) -> Pipeline:
    return Pipeline([("model", estimator)])


# --------------------------------------------------------------------------- #
# 1. Cross-validation setup
# --------------------------------------------------------------------------- #


def test_regression_folds_are_stratified_on_the_tail(heavy_tailed):
    """The point of the decile-stratified splitter: whales spread across folds."""
    X, y = heavy_tailed
    stratified = cv_report(make_cv("regression", y, "auto"), X, y, task="regression")
    plain = cv_report(make_cv("regression", y, "kfold"), X, y, task="regression")

    spread = stratified["top_decile_share"].max() - stratified["top_decile_share"].min()
    plain_spread = plain["top_decile_share"].max() - plain["top_decile_share"].min()
    assert spread < plain_spread, "stratifying should even out the tail, not worsen it"
    assert stratified["median"].std() < plain["median"].std()


def test_classification_folds_keep_the_class_balance(imbalanced):
    X, y = imbalanced
    folds = cv_report(make_cv("classification", y), X, y, task="classification")
    assert folds["positive_rate"].max() - folds["positive_rate"].min() < 0.05


def test_clustering_has_no_folds():
    assert make_cv("clustering", None) is None
    assert cv_report(None, pd.DataFrame({"a": [1.0]})).empty


@pytest.mark.parametrize("strategy", [s for s in CV_STRATEGIES if s != "auto"])
def test_every_strategy_produces_usable_folds(strategy, heavy_tailed):
    X, y = heavy_tailed
    cv = make_cv("regression", y, strategy=strategy)
    splits = list(cv.split(X, y))
    assert len(splits) >= 5
    for train_idx, val_idx in splits:
        assert len(train_idx) and len(val_idx)
        assert not set(train_idx) & set(val_idx), "a row is in both halves"


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy must be one of"):
        make_cv("regression", None, strategy="magic")


# --------------------------------------------------------------------------- #
# 2. Hyperparameter tuning
# --------------------------------------------------------------------------- #


def test_search_parameters_reach_a_pipeline_estimator(heavy_tailed):
    X, y = heavy_tailed
    models = {"ridge on log(y)": _pipe(Ridge(alpha=1.0))}
    tuned, table = tune_models(models, X, y, n_iter=8)

    assert table.loc["ridge on log(y)", "n_candidates"] == 8
    assert tuned["ridge on log(y)"].get_params()["model__alpha"] != 1.0


def test_search_parameters_reach_a_bare_estimator(heavy_tailed):
    """A model handed in without preprocessing has no ``model__`` prefix."""
    X, y = heavy_tailed
    tuned, table = tune_models({"ridge on log(y)": Ridge(alpha=1.0)}, X, y, n_iter=6)

    assert np.isfinite(table.loc["ridge on log(y)", "cv_score"])
    assert tuned["ridge on log(y)"].get_params()["alpha"] != 1.0


def test_a_parameter_that_exists_nowhere_is_rejected(heavy_tailed):
    """A silent no-op search is worse than no search."""
    X, y = heavy_tailed
    with pytest.raises(ValueError, match="not present on the estimator"):
        tune_models({"ridge on log(y)": _pipe(Ridge())}, X, y,
                    spaces={"ridge on log(y)": {"model__nonexistent": [1, 2]}})


@pytest.fixture
def without_xgboost(monkeypatch):
    """Pretend the optional dependency is absent, as it is on many machines.

    This is the blind spot that shipped a broken default run: every test passed
    on a machine *with* XGBoost, while `python main.py` failed on one without it,
    because the boosting estimator silently falls back to scikit-learn's
    histogram booster -- same architecture, different parameter names.
    """
    import src.advanced_models as advanced
    import src.models as models

    monkeypatch.setattr(models, "has_xgboost", lambda: False)
    monkeypatch.setattr(advanced, "has_xgboost", lambda: False)
    return advanced


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_boosting_tunes_without_xgboost(task, without_xgboost, heavy_tailed, imbalanced):
    """The reported bug: n_estimators/subsample do not exist on HistGradientBoosting."""
    X, y = heavy_tailed if task == "regression" else imbalanced
    models = without_xgboost.build_advanced_models(task, include=["gradient boosting"])
    estimator = models["gradient boosting"].named_steps["model"]
    assert type(estimator).__name__.startswith("HistGradientBoosting")

    tuned, table = tune_models(models, X, y, task=task, n_iter=4)

    row = table.loc["gradient boosting"]
    assert row["method"] == "random"
    assert np.isfinite(row["cv_score"])
    # The rename must happen, and the drop must be reported rather than silent.
    assert "n_estimators" in row["note"] and "max_iter" in row["note"]
    assert "subsample" in row["note"]
    assert tuned["gradient boosting"].get_params()["model__max_iter"] != 400


def test_synonyms_do_not_hide_a_real_typo(heavy_tailed):
    """Dropping a known-absent parameter must not become dropping anything unknown."""
    X, y = heavy_tailed
    with pytest.raises(ValueError, match="not present on the estimator"):
        tune_models({"r": _pipe(Ridge())}, X, y,
                    spaces={"r": {"model__alpha_typo": [1.0, 2.0]}})


def test_xgboost_parameters_are_kept_when_xgboost_is_present(heavy_tailed):
    """With the real dependency, nothing is renamed and nothing is dropped."""
    from src.advanced_models import build_advanced_models
    from src.models import has_xgboost

    if not has_xgboost():
        pytest.skip("xgboost is not installed in this environment")

    X, y = heavy_tailed
    models = build_advanced_models("regression", include=["gradient boosting"])
    _, table = tune_models(models, X, y, n_iter=4)
    assert "dropped" not in table.loc["gradient boosting", "note"]
    assert "renamed" not in table.loc["gradient boosting", "note"]


def test_models_with_nothing_to_tune_are_passed_through(heavy_tailed):
    X, y = heavy_tailed
    models = {"mean": _pipe(DummyRegressor(strategy="mean"))}
    tuned, table = tune_models(models, X, y)

    assert table.loc["mean", "method"] == "none"
    assert table.loc["mean", "note"] == "nothing to tune"
    assert tuned["mean"] is models["mean"]


def test_search_optimises_the_headline_metric_not_a_proxy(heavy_tailed):
    """Ranking by Spearman while tuning on R2 optimises the wrong thing."""
    from sklearn.metrics import get_scorer

    X, y = heavy_tailed
    scorer = scoring_for("regression")
    assert scorer != "r2"

    # Perfect ordering, wrong level: Spearman 1.0, R2 poor.
    estimator = _ConstantFactor(3.0).fit(X, y)
    assert float(scorer(estimator, X, y)) == pytest.approx(1.0)
    assert float(get_scorer("r2")(estimator, X, y)) < 0.5

    assert scoring_for("classification") == "f1_macro"


@pytest.mark.parametrize("method", ["random", "grid"])
def test_search_methods_agree_on_a_tiny_space(method, heavy_tailed):
    X, y = heavy_tailed
    space = {"ridge on log(y)": {"model__alpha": [0.01, 1.0, 100.0]}}
    _, table = tune_models({"ridge on log(y)": _pipe(Ridge())}, X, y,
                           method=method, n_iter=3, spaces=space)
    assert table.loc["ridge on log(y)", "method"] == method
    assert np.isfinite(table.loc["ridge on log(y)", "cv_score"])


def test_every_task_has_spaces_for_its_tunable_models():
    for task in TASKS:
        tunable = [name for name in build_baselines(task)
                   if search_space_for(name, task)]
        assert tunable, f"{task}: no baseline has a search space"


# --------------------------------------------------------------------------- #
# 3. Over- and underfitting
# --------------------------------------------------------------------------- #


def test_a_memorising_model_is_diagnosed_as_overfitting(heavy_tailed):
    X, y = heavy_tailed
    memoriser = _pipe(DecisionTreeRegressor(max_depth=None, random_state=0))
    diagnosis = diagnose_fit(memoriser, X, y, name="deep tree")

    assert diagnosis["verdict"] == "overfitting"
    assert diagnosis["train_score"] > diagnosis["cv_score"]
    assert "capacity" in diagnosis["action"]


def test_a_constant_model_is_diagnosed_as_underfitting(heavy_tailed):
    X, y = heavy_tailed
    diagnosis = diagnose_fit(_pipe(DummyRegressor(strategy="mean")), X, y, name="mean")

    assert diagnosis["verdict"] == "underfitting"
    assert "add capacity" in diagnosis["action"]


def test_a_well_matched_model_is_diagnosed_as_balanced(heavy_tailed):
    """A linear model on a log-linear target: the right capacity for the job."""
    X, y = heavy_tailed
    logged = pd.DataFrame(np.log(X.to_numpy()), columns=X.columns)
    diagnosis = diagnose_fit(_pipe(LinearRegression()), logged, np.log(y), name="linear")

    assert diagnosis["verdict"] == "balanced"
    assert abs(diagnosis["relative_gap"]) < 0.05


def test_clustering_diagnosis_is_declared_not_applicable(heavy_tailed):
    X, _ = heavy_tailed
    diagnosis = diagnose_fit(None, X, None, task="clustering", name="k-means")
    assert diagnosis["verdict"] == "not applicable"


def test_learning_curve_shows_the_gap_closing(heavy_tailed):
    X, y = heavy_tailed
    curve = learning_curve_report(_pipe(KNeighborsRegressor(n_neighbors=5)), X, y,
                                  fractions=(0.3, 0.6, 1.0))
    assert len(curve) == 3
    assert curve["cv_score"].iloc[-1] >= curve["cv_score"].iloc[0]
    assert (curve["gap"] >= 0).all()


def test_validation_curve_finds_the_regularisation_trade(heavy_tailed):
    X, y = heavy_tailed
    curve = validation_curve_report(_pipe(Ridge()), X, y, param_name="model__alpha",
                                    param_range=(1e-3, 1.0, 1e3))
    assert list(curve.index) == [1e-3, 1.0, 1e3]
    assert {"train_score", "cv_score", "gap"} <= set(curve.columns)


# --------------------------------------------------------------------------- #
# The one-standard-error rule
# --------------------------------------------------------------------------- #


def test_one_se_rule_prefers_the_simpler_model_within_the_margin():
    scores = {"ridge on log(y)": 0.980, "gradient boosting": 0.985}
    errors = {"gradient boosting": 0.010, "ridge on log(y)": 0.004}
    choice, report = select_within_one_se(scores, errors, complexity=COMPLEXITY_RANK)

    assert report["best_by_score"] == "gradient boosting"
    assert choice == "ridge on log(y)", "0.980 is within one SE of 0.985"
    assert report["traded_accuracy"] == pytest.approx(0.005)


def test_one_se_rule_keeps_the_best_when_the_lead_is_real():
    scores = {"ridge on log(y)": 0.90, "gradient boosting": 0.985}
    errors = {"gradient boosting": 0.002, "ridge on log(y)": 0.002}
    choice, report = select_within_one_se(scores, errors, complexity=COMPLEXITY_RANK)

    assert choice == "gradient boosting"
    assert report["within_one_se"] == ["gradient boosting"]


def test_one_se_rule_handles_lower_is_better():
    scores = {"a": 0.10, "b": 0.12}
    choice, report = select_within_one_se(scores, {"a": 0.05}, complexity={"a": 5, "b": 1},
                                          higher_is_better=False)
    assert choice == "b"
    assert report["best_by_score"] == "a"


def test_one_se_rule_rejects_empty_input():
    with pytest.raises(ValueError, match="no scores"):
        select_within_one_se({}, {})
    with pytest.raises(ValueError, match="every score is NaN"):
        select_within_one_se({"a": float("nan")}, {})


# --------------------------------------------------------------------------- #
# 4. Final selection and validation
# --------------------------------------------------------------------------- #


def test_optimise_models_runs_all_four_steps(heavy_tailed):
    X, y = heavy_tailed
    models = {"mean": _pipe(DummyRegressor()), "ridge on log(y)": _pipe(Ridge()),
              "decision tree (depth 3)": _pipe(DecisionTreeRegressor(random_state=0))}
    results = optimise_models(models, X, y, n_iter=5)

    assert len(results.cv_folds) == 5
    assert set(results.tuning.index) == set(models)
    assert set(results.diagnosis["model"]) == set(models)
    assert results.final_name in models
    assert results.selection["chosen"] == results.final_name
    assert len(results.learning_curve) > 0


def test_final_validation_reports_optimism(heavy_tailed):
    X, y = heavy_tailed
    models = {"ridge on log(y)": _pipe(Ridge()), "mean": _pipe(DummyRegressor())}
    results = optimise_models(models, X, y, n_iter=4, learning_curves=False)

    cut = int(0.8 * len(X))
    validation = validate_final_model(results, X.iloc[:cut], y.iloc[:cut],
                                      X.iloc[cut:], y.iloc[cut:])

    assert validation["model"] == results.final_name
    assert np.isfinite(validation["test_score"])
    assert validation["optimism"] == pytest.approx(
        validation["cv_score"] - validation["test_score"], abs=1e-9)
    assert validation["verdict"]


def test_checks_flag_a_selection_that_overfitted_the_folds(heavy_tailed):
    """A fabricated optimism of +0.4 must not pass the final check."""
    X, y = heavy_tailed
    results = optimise_models({"ridge on log(y)": _pipe(Ridge())}, X, y, n_iter=3,
                              learning_curves=False)
    results.validation = {"model": "ridge on log(y)", "metric": "spearman",
                          "cv_score": 0.99, "test_score": 0.59, "optimism": 0.40,
                          "verdict": "fabricated"}
    failed = {c.name for c in check_optimization(results) if c.failed}
    assert "the held-out score is close to the cross-validated one" in failed


def test_checks_pass_on_a_healthy_run(heavy_tailed):
    X, y = heavy_tailed
    models = {"mean": _pipe(DummyRegressor()), "ridge on log(y)": _pipe(Ridge())}
    results = optimise_models(models, X, y, n_iter=4, learning_curves=False)
    cut = int(0.8 * len(X))
    validate_final_model(results, X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:])

    failures = [f"{c.name}: {c.detail}" for c in check_optimization(results) if c.failed]
    assert not failures, "\n".join(failures)


def test_select_final_model_can_be_called_on_its_own(heavy_tailed):
    X, y = heavy_tailed
    models = {"mean": _pipe(DummyRegressor()), "ridge on log(y)": _pipe(Ridge())}
    results = optimise_models(models, X, y, n_iter=3, learning_curves=False)

    # Re-selecting with a complexity ranking that prefers the mean must change it.
    name, model = select_final_model(results, X, y, complexity={"mean": 0,
                                                                "ridge on log(y)": 99})
    assert name in models and model is results.models[name]


def test_results_serialise_to_json_friendly_types(heavy_tailed):
    import json

    X, y = heavy_tailed
    results = optimise_models({"ridge on log(y)": _pipe(Ridge())}, X, y, n_iter=3,
                              learning_curves=False)
    results.checks = check_optimization(results)
    json.dumps(results.to_dict(), default=str)      # must not raise


class _ConstantFactor(BaseEstimator, RegressorMixin):
    """Predicts k * y: perfect ordering, wrong level."""

    def __init__(self, k: float = 1.0):
        self.k = k

    def fit(self, X, y):
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        return self.y_[: len(X)] * self.k
