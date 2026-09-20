"""Tests for src/models.py -- the baseline ladder, on all three task types.

``main.py`` runs the regression ladder on the CLV data every run. This file adds
what that cannot reach: a classification target, a clustering problem with no
target at all, and the properties that make a baseline trustworthy -- that a
naive floor really is naive, that nothing leaks, that the benchmark measures the
model rather than the preprocessing around it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models import (
    BASELINE_SPECS,
    HEADLINE_METRIC,
    METRIC_GUIDE,
    TASKS,
    BaselineResults,
    ColumnProductRegressor,
    LogTargetRegressor,
    analyse_baselines,
    benchmark_baselines,
    build_baselines,
    check_baselines,
    cross_validate_model,
    evaluate_baselines,
    infer_task,
    metric_guide,
    score_predictions,
    spec_table,
)
from src.preprocessing import TARGET, CLVPreprocessor, PreprocessConfig


# --------------------------------------------------------------------------- #
# Fixtures: one dataset per task
# --------------------------------------------------------------------------- #


@pytest.fixture
def classification_data(rng):
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    # 80/20 imbalance, so the majority-class baseline is informative about the metric
    y = pd.Series((X["a"] + 0.4 * rng.normal(size=n) > 0.85).astype(int))
    return X, y


@pytest.fixture
def clustering_data(rng):
    blob = lambda cx, cy, n: pd.DataFrame({  # noqa: E731 - three lines of test data
        "x": rng.normal(cx, 0.3, n), "y": rng.normal(cy, 0.3, n)})
    return pd.concat([blob(0, 0, 80), blob(4, 4, 80), blob(0, 5, 80)], ignore_index=True)


# --------------------------------------------------------------------------- #
# Task inference and the catalogue
# --------------------------------------------------------------------------- #


def test_infer_task(processed, classification_data, clustering_data):
    assert infer_task(processed["y_train"]) == "regression"
    assert infer_task(classification_data[1]) == "classification"
    assert infer_task(pd.Series(["a", "b", "a"])) == "classification"
    assert infer_task(None) == "clustering"


def test_every_task_has_a_floor_and_a_model_baseline():
    for task in TASKS:
        kinds = {s.kind for s in BASELINE_SPECS if s.task == task}
        assert kinds == {"naive", "model"}, task
        assert HEADLINE_METRIC[task][0] in METRIC_GUIDE


def test_every_baseline_documents_itself():
    for spec in BASELINE_SPECS:
        assert len(spec.rationale) > 40, spec.name
        assert spec.kind in {"naive", "model"}
    assert len(spec_table("regression")) == 8


def test_naive_baselines_skip_the_preprocessor():
    """A constant predictor ignores X; preprocessing it only distorts the benchmark."""
    models = build_baselines("regression", preprocessor=StandardScaler())
    assert not hasattr(models["mean"], "steps")            # not a Pipeline
    assert hasattr(models["decision tree (depth 3)"], "steps")


def test_unknown_task_and_unknown_baseline_are_rejected():
    with pytest.raises(ValueError, match="task must be one of"):
        build_baselines("ranking")
    with pytest.raises(ValueError, match="unknown baseline"):
        build_baselines("regression", include=["mean", "prophet"])


# --------------------------------------------------------------------------- #
# The heuristic estimator
# --------------------------------------------------------------------------- #


def test_column_product_rescales_to_the_training_mean(processed):
    X, y = processed["X_train_raw"], processed["y_train"]
    model = ColumnProductRegressor().fit(X, y)
    predictions = model.predict(X)
    assert model.usable_
    assert predictions.mean() == pytest.approx(float(y.mean()), rel=1e-6)


def test_column_product_falls_back_when_columns_are_missing(rng):
    X = pd.DataFrame({"unrelated": rng.normal(size=50)})
    y = pd.Series(rng.normal(100, 10, size=50))
    model = ColumnProductRegressor().fit(X, y)
    assert not model.usable_
    np.testing.assert_allclose(model.predict(X), float(y.mean()))


def test_log_target_regressor_round_trips(processed):
    from sklearn.linear_model import LinearRegression

    X = processed["X_train"].iloc[:, :5]
    y = processed["y_train"]
    model = LogTargetRegressor(LinearRegression()).fit(X, y)
    assert model.smearing_factor_ > 0
    np.testing.assert_allclose(model.predict(X),
                               np.exp(model.predict_log(X)) * model.smearing_factor_)


# --------------------------------------------------------------------------- #
# Regression ladder
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def regression_results(request) -> BaselineResults:
    processed = request.getfixturevalue("processed")
    return evaluate_baselines(
        processed["X_train_raw"], processed["y_train"],
        preprocessor=CLVPreprocessor(PreprocessConfig(poly_degree=1)),
        benchmark_repeats=1)


def test_regression_ladder_scores_every_baseline(regression_results):
    board = regression_results.board
    assert len(board) == 8
    assert board["spearman"].notna().all()
    assert regression_results.task == "regression"


def test_the_heuristic_reproduces_the_eda_number(regression_results):
    """EDA 9 reports Spearman 0.878 for purchases x AOV; the ladder must agree."""
    rho = regression_results.board.loc["heuristic (product of two columns)", "spearman"]
    assert rho == pytest.approx(0.878, abs=0.01)


def test_naive_floors_score_like_floors(regression_results):
    board = regression_results.board
    assert abs(board.loc["mean", "r2_raw"]) < 0.05
    assert board.loc["mean", "spearman"] < 0.1
    # The median wins on MAE and loses on R2: the metric choice is a business choice.
    assert board.loc["median", "mae_raw"] < board.loc["mean", "mae_raw"]


def test_model_baselines_beat_the_heuristic(regression_results):
    bar = regression_results.analysis["bar_for_a_real_model"]
    assert bar["baseline"] == "heuristic (product of two columns)"
    assert bar["model_baselines_beating_it"] >= 4


def test_checks_pass_on_a_healthy_ladder(regression_results):
    failures = [f"{c.name}: {c.detail}" for c in regression_results.checks if c.failed]
    assert not failures, "\n".join(failures)


def test_benchmark_shows_the_naive_baselines_are_cheaper(regression_results):
    bench = regression_results.benchmark
    assert set(bench.index) == set(regression_results.board.index)
    assert bench.loc["mean", "fit_ms"] < bench.loc["ridge on log(y)", "fit_ms"]
    assert (bench["model_kb"] > 0).all()


# --------------------------------------------------------------------------- #
# Classification ladder
# --------------------------------------------------------------------------- #


def test_classification_ladder(classification_data):
    X, y = classification_data
    results = evaluate_baselines(X, y, benchmark=False)

    assert results.task == "classification"
    assert len(results.board) == 6
    majority = float(y.value_counts(normalize=True).max())
    assert results.board.loc["most frequent class", "accuracy"] == pytest.approx(
        majority, abs=0.02)
    # Accuracy flatters the majority baseline; F1-macro does not. That is the point.
    assert results.board.loc["most frequent class", "f1_macro"] < 0.6
    assert results.board.loc["logistic regression", "f1_macro"] > \
        results.board.loc["most frequent class", "f1_macro"]
    assert not [c for c in results.checks if c.failed]


def test_binary_probability_metrics_are_reported(classification_data):
    X, y = classification_data
    results = evaluate_baselines(X, y, include=["logistic regression", "stratified random"],
                                 benchmark=False)
    assert results.board.loc["logistic regression", "roc_auc"] > 0.8
    assert results.board.loc["stratified random", "roc_auc"] == pytest.approx(0.5, abs=0.15)


# --------------------------------------------------------------------------- #
# Clustering ladder
# --------------------------------------------------------------------------- #


def test_clustering_ladder(clustering_data):
    results = evaluate_baselines(clustering_data, task="clustering", benchmark=False,
                                 n_clusters=3)

    assert results.task == "clustering"
    assert results.board.loc["k-means", "silhouette"] > 0.5      # three separated blobs
    assert results.board.loc["random labels", "silhouette"] < 0.1
    assert np.isnan(results.board.loc["one cluster", "silhouette"])
    assert results.board.loc["k-means", "n_clusters"] == 3
    assert not [c for c in results.checks if c.failed]


def test_single_cluster_metrics_are_nan_not_zero(clustering_data):
    row = score_predictions("trivial", "clustering",
                            y_pred=np.zeros(len(clustering_data)), X=clustering_data)
    assert row["n_clusters"] == 1
    assert np.isnan(row["silhouette"])


# --------------------------------------------------------------------------- #
# Analysis and leakage
# --------------------------------------------------------------------------- #


def test_analysis_names_the_disagreement(regression_results):
    analysis = regression_results.analysis
    assert analysis["headline_metric"] == "spearman"
    assert set(analysis["winner_by_metric"]) <= set(regression_results.board.columns)
    if not analysis["metrics_agree"]:
        assert "different winners" in analysis["disagreement"]


def test_lift_over_floor_is_relative_to_the_best(regression_results):
    lift = regression_results.analysis["lift_over_floor"]
    assert lift[regression_results.best] == pytest.approx(1.0)
    assert lift["mean"] == pytest.approx(0.0, abs=1e-9)


def test_cross_validation_is_out_of_fold(processed):
    """A model that memorises the training rows must not score well here."""
    from sklearn.neighbors import KNeighborsRegressor

    X, y = processed["X_train"].iloc[:, :4], processed["y_train"]
    memoriser = KNeighborsRegressor(n_neighbors=1)
    row, oof = cross_validate_model("1-NN", memoriser, X, y, task="regression")
    in_sample = memoriser.fit(X, y).predict(X)

    assert np.allclose(in_sample, y)                     # perfect in sample
    assert not np.allclose(oof, y)                       # not out of fold
    assert row["spearman"] < 0.999


def test_evaluate_baselines_rejects_an_empty_selection(processed):
    with pytest.raises(ValueError, match="unknown baseline"):
        evaluate_baselines(processed["X_train_raw"], processed["y_train"],
                           include=["nothing at all"])


def test_metric_guide_documents_every_reported_metric(regression_results):
    guide = metric_guide("regression")
    for column in regression_results.board.columns:
        assert column in guide.index, column
        assert guide.loc[column, "misleads"]


def test_analyse_handles_an_empty_board():
    empty = pd.DataFrame(columns=["spearman"]).set_index(pd.Index([], name="model"))
    assert analyse_baselines(empty, "regression") == {"headline_metric": "spearman",
                                                      "task": "regression"}


def test_benchmark_runs_for_clustering(clustering_data):
    models = build_baselines("clustering", n_clusters=3)
    bench = benchmark_baselines(models, clustering_data, task="clustering", repeats=1)
    assert set(bench.index) == set(models)
    assert (bench["fit_ms"] >= 0).all()


def test_check_baselines_reports_a_broken_ladder(regression_results):
    """Corrupt the predictions and the checks must notice."""
    broken = BaselineResults(
        task="regression",
        board=regression_results.board,
        predictions={"mean": np.full(3, np.nan)},        # wrong length and not finite
    )
    checks = check_baselines(broken, pd.DataFrame(np.zeros((800, 2))))
    failed = {c.name for c in checks if c.failed}
    assert "predictions are finite" in failed
    assert "one prediction per row" in failed
