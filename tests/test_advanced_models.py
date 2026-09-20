"""Tests for src/advanced_models.py -- the two extra architectures and the comparison.

The comparison is the part worth testing hardest: it is easy to write a function
that always says the new model won. These tests check that it says the *right*
thing on data engineered so that the answer is known in advance -- a problem where
trees must win, one where the linear baseline must win, and one where the
difference is pure noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from src.advanced_models import (
    ADVANCED_SPECS,
    RELIABILITY_MARGIN,
    paired_test,
    AdvancedResults,
    analyse_comparison,
    architecture_table,
    build_advanced_models,
    check_advanced_models,
    combined_leaderboard,
    compare_models,
    document_results,
    evaluate_advanced,
    fold_scores,
    run_comparison,
)
from src.models import BASELINE_SPECS, TASKS, evaluate_baselines, infer_task


# --------------------------------------------------------------------------- #
# Fixtures: problems whose right answer is known
# --------------------------------------------------------------------------- #


@pytest.fixture
def step_function_data(rng):
    """A target built from thresholds: trees should beat a linear model here."""
    n = 400
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(0, 10, n)})
    y = pd.Series(np.where(X["a"] > 5, 100.0, 10.0) + np.where(X["b"] > 7, 50.0, 0.0)
                  + rng.normal(0, 1, n))
    return X, y


@pytest.fixture
def smooth_multiplicative_data(rng):
    """The capstone's shape: a product of powers, which a linear-on-log model nails."""
    n = 400
    X = pd.DataFrame({
        "total_purchase_count": rng.lognormal(1.5, 0.9, n),
        "average_order_value": rng.lognormal(4.2, 0.5, n),
        "days_since_last_purchase": rng.uniform(1, 400, n),
    })
    y = pd.Series(np.exp(3.4) * X["total_purchase_count"] ** 0.3
                  * X["average_order_value"] ** 0.86
                  * X["days_since_last_purchase"] ** -0.32 * rng.lognormal(0, 0.05, n))
    return X, y


@pytest.fixture
def blobs(rng):
    def blob(cx, cy, n):
        return pd.DataFrame({"x": rng.normal(cx, 0.3, n), "y": rng.normal(cy, 0.3, n)})

    return pd.concat([blob(0, 0, 80), blob(4, 4, 80), blob(0, 5, 80)], ignore_index=True)


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #


def test_two_distinct_architectures_per_task():
    for task in TASKS:
        specs = [s for s in ADVANCED_SPECS if s.task == task]
        assert len(specs) == 2, task
        assert len({s.architecture for s in specs}) == 2, f"{task}: same architecture twice"
        assert len({s.name for s in specs}) == 2


def test_architectures_are_distinct_from_the_baselines():
    """An 'advanced' model that repeats a baseline measures nothing."""
    for task in TASKS:
        baseline_names = {s.name for s in BASELINE_SPECS if s.task == task}
        advanced_names = {s.name for s in ADVANCED_SPECS if s.task == task}
        assert not (baseline_names & advanced_names), task


def test_every_spec_documents_bias_and_cost():
    for spec in ADVANCED_SPECS:
        assert len(spec.architecture) > 20, spec.name
        assert len(spec.bias) > 40, spec.name
        assert len(spec.cost) > 20, spec.name
    assert len(architecture_table("regression")) == 2


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown advanced model"):
        build_advanced_models("regression", include=["transformer"])


def test_networks_are_given_a_scaler():
    """An MLP on unscaled inputs would measure the scaling, not the architecture."""
    models = build_advanced_models("regression")
    steps = dict(models["neural network (MLP)"].steps)
    assert "scale" in steps


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def test_fold_scores_returns_one_score_per_fold(smooth_multiplicative_data):
    from sklearn.linear_model import LinearRegression

    X, y = smooth_multiplicative_data
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    scores = fold_scores("linear", LinearRegression(), X, y, cv=cv, task="regression")
    assert len(scores) == 4
    assert np.isfinite(scores).all()


def test_fold_scores_for_clustering_measures_stability(blobs):
    from sklearn.cluster import KMeans

    scores = fold_scores("k-means", KMeans(n_clusters=3, n_init=10, random_state=0),
                         blobs, task="clustering", n_subsamples=4)
    assert len(scores) == 4
    assert scores.min() > 0.5                      # well-separated blobs, every subsample


def test_evaluate_advanced_runs_both_models(smooth_multiplicative_data):
    X, y = smooth_multiplicative_data
    results = evaluate_advanced(X, y, benchmark=False)
    assert set(results.board.index) == {"gradient boosting", "neural network (MLP)"}
    assert all(len(v) == 5 for v in results.fold_scores.values())


# --------------------------------------------------------------------------- #
# Tuning (the search itself is tested in tests/test_model_optimization.py)
# --------------------------------------------------------------------------- #


def test_tuning_is_recorded_in_the_verdict(step_function_data):
    """A 'the baseline won' claim must say whether the challengers were tuned."""
    X, y = step_function_data
    default = run_comparison(X, y, task="regression", benchmark_repeats=1)
    searched = run_comparison(X, y, task="regression", tune=True, n_iter=5,
                              benchmark_repeats=1)

    assert not default.tuned and "default settings" in default.analysis["verdict"]
    assert searched.tuned and "randomised search" in searched.analysis["verdict"]
    assert len(searched.tuning) == 2


# --------------------------------------------------------------------------- #
# The significance test
# --------------------------------------------------------------------------- #


def test_consistent_difference_is_significant():
    result = paired_test(np.array([0.010, 0.011, 0.009, 0.0105, 0.0095]))
    assert result["mean_difference"] == pytest.approx(0.01, abs=1e-9)
    assert result["p_value"] < 0.01
    assert result["significant"]


def test_noisy_difference_is_not_significant():
    result = paired_test(np.array([0.05, -0.04, 0.03, -0.02, 0.002]))
    assert abs(result["mean_difference"]) < 0.01
    assert result["p_value"] > 0.5
    assert not result["significant"]


def test_the_correction_is_more_conservative_than_a_plain_t_test():
    """The whole point: folds share training rows, so the naive test over-claims."""
    from scipy import stats

    d = np.array([0.004, 0.005, 0.003, 0.0045, 0.0035])
    corrected = paired_test(d)
    naive = float(stats.ttest_1samp(d, 0.0).pvalue)

    assert corrected["p_value"] > naive, "the correction must widen the interval"
    assert abs(corrected["t_stat"]) < abs(float(stats.ttest_1samp(d, 0.0).statistic))


def test_wilcoxon_floor_is_reported_at_five_folds():
    result = paired_test(np.array([0.01, 0.011, 0.009, 0.0105, 0.0095]))
    assert result["p_wilcoxon"] == pytest.approx(0.0625)
    assert "cannot reach 0.05" in result["note"]


def test_degenerate_inputs_do_not_raise():
    for values in ([], [0.01], np.zeros(5)):
        result = paired_test(np.asarray(values, dtype=float))
        assert not result["significant"]
        assert not np.isfinite(result["p_value"])


# --------------------------------------------------------------------------- #
# Comparison: does it say the right thing?
# --------------------------------------------------------------------------- #


def test_trees_win_on_a_step_function(step_function_data):
    """Thresholds are what the tree family is for; the comparison must notice.

    Note what the reference turns out to be: on a target made of two thresholds,
    the *depth-3 tree baseline* is already near-perfect, so the comparison holds
    boosting to that rather than to a linear model. That is the point of picking
    the strongest model baseline as the reference.
    """
    X, y = step_function_data
    results = run_comparison(X, y, task="regression", benchmark_repeats=1)

    combined = results.combined
    assert results.best == "gradient boosting"
    assert combined.index[0] == "gradient boosting"
    assert results.analysis["reference"] == "decision tree (depth 3)"

    # The architecture claim: tree-shaped models beat the linear family here.
    assert combined.loc["gradient boosting", "spearman"] > \
        combined.loc["linear regression", "spearman"]
    assert combined.loc["decision tree (depth 3)", "spearman"] > \
        combined.loc["ridge on log(y)", "spearman"]


def test_linear_wins_on_a_smooth_multiplicative_target(smooth_multiplicative_data):
    """The capstone's own finding: with the right representation, simple wins.

    The preprocessor matters here and the test says so explicitly: given *logged
    features*, a linear model represents a product exactly and neither advanced
    architecture can improve on it. Without the log the same comparison flips,
    which is exactly readme.md §16.2's point about representation.
    """
    from sklearn.preprocessing import FunctionTransformer

    X, y = smooth_multiplicative_data
    log_features = FunctionTransformer(np.log, feature_names_out="one-to-one")
    results = run_comparison(X, y, task="regression", preprocessor=log_features,
                             benchmark_repeats=1)

    assert results.analysis["reference"] in {"ridge on log(y)", "linear regression on log(y)"}
    assert not results.analysis["overall_best_is_advanced"]
    assert results.comparison["delta"].max() <= 0
    assert "Keep the baseline" in results.analysis["verdict"]


def test_a_noise_sized_lead_is_not_called_reliable():
    """Hand-built fold scores: a lead inside the fold spread must not be 'reliable'."""
    baselines = _baseline_stub(0.90)
    advanced = AdvancedResults(
        task="regression",
        board=pd.DataFrame({"spearman": [0.902]}, index=pd.Index(["gradient boosting"],
                                                                 name="model")),
        # These average to 0.902 -- a 0.002 lead inside a 0.04 spread.
        fold_scores={"gradient boosting": np.array([0.95, 0.85, 0.94, 0.86, 0.91])},
    )
    comparison = _comparison_from(advanced, baselines, reference_folds=np.full(5, 0.90))
    assert comparison.loc["gradient boosting", "delta"] == pytest.approx(0.002, abs=1e-9)
    assert not comparison.loc["gradient boosting", "reliable"]

    analysis = analyse_comparison(baselines, advanced, comparison)
    assert "not distinguishable from fold noise" in analysis["verdict"]


def test_a_consistent_lead_is_called_reliable():
    baselines = _baseline_stub(0.90)
    advanced = AdvancedResults(
        task="regression",
        board=pd.DataFrame({"spearman": [0.95]}, index=pd.Index(["gradient boosting"],
                                                                name="model")),
        fold_scores={"gradient boosting": np.array([0.95, 0.951, 0.949, 0.95, 0.95])},
    )
    comparison = _comparison_from(advanced, baselines,
                                  reference_folds=np.full(5, 0.90))
    assert comparison.loc["gradient boosting", "reliable"]
    assert comparison.loc["gradient boosting", "folds_won"] == 5

    analysis = analyse_comparison(baselines, advanced, comparison)
    assert "beats" in analysis["verdict"]
    assert analysis["error_reduction"] == pytest.approx(0.5, abs=1e-6)


def test_error_reduction_is_omitted_when_the_leader_trails():
    baselines = _baseline_stub(0.99)
    advanced = AdvancedResults(
        task="regression",
        board=pd.DataFrame({"spearman": [0.98]}, index=pd.Index(["gradient boosting"],
                                                                name="model")),
        fold_scores={"gradient boosting": np.full(5, 0.98)},
    )
    comparison = _comparison_from(advanced, baselines, reference_folds=np.full(5, 0.99))
    analysis = analyse_comparison(baselines, advanced, comparison)
    assert "error_reduction" not in analysis
    assert "No advanced architecture beat" in analysis["verdict"]


def test_the_reference_is_the_best_model_baseline_not_the_best_overall(
        smooth_multiplicative_data):
    X, y = smooth_multiplicative_data
    baselines = evaluate_baselines(X, y, benchmark=False)
    advanced = evaluate_advanced(X, y, benchmark=False)
    comparison = compare_models(baselines, advanced, X, y)

    reference = comparison.iloc[0]["reference"]
    naive = {s.name for s in BASELINE_SPECS if s.task == "regression" and s.kind == "naive"}
    assert reference not in naive


# --------------------------------------------------------------------------- #
# Leaderboard, checks and documentation
# --------------------------------------------------------------------------- #


def test_combined_leaderboard_tags_every_row(smooth_multiplicative_data):
    X, y = smooth_multiplicative_data
    baselines = evaluate_baselines(X, y, benchmark=False)
    advanced = evaluate_advanced(X, y, benchmark=False)
    combined = combined_leaderboard(baselines, advanced)

    assert len(combined) == len(baselines.board) + len(advanced.board)
    assert set(combined["kind"]) <= {"naive", "model", "advanced"}
    assert (combined["kind"] == "advanced").sum() == 2
    assert combined["spearman"].is_monotonic_decreasing


def test_checks_catch_a_model_that_lost_to_the_floor(smooth_multiplicative_data):
    X, y = smooth_multiplicative_data
    baselines = evaluate_baselines(X, y, benchmark=False)
    broken = AdvancedResults(
        task="regression",
        board=pd.DataFrame({"spearman": [-0.5, -0.4]},
                           index=pd.Index(["gradient boosting", "neural network (MLP)"],
                                          name="model")),
        fold_scores={"gradient boosting": np.full(5, -0.5),
                     "neural network (MLP)": np.full(5, -0.4)},
    )
    failed = {c.name for c in check_advanced_models(broken, baselines) if c.failed}
    assert "every advanced model beats the naive floor" in failed


def test_document_results_is_generated_from_the_numbers(tmp_path, step_function_data):
    X, y = step_function_data
    results = run_comparison(X, y, task="regression", benchmark_repeats=1,
                             report_path=tmp_path / "report.md")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "# Advanced models vs the baseline ladder (regression)" in report
    assert results.analysis["verdict"] in report
    assert "gradient boosting" in report
    assert "## 3. Paired comparison" in report
    assert "## 5. Fold-level spread" in report
    # Every check appears, so the report cannot claim more than the run verified.
    for check in results.checks:
        assert check.name in report


def test_clustering_comparison_runs(blobs):
    results = run_comparison(blobs, task="clustering", n_clusters=3, eps=0.9,
                             benchmark_repeats=1)
    assert set(results.board.index) == {"gaussian mixture", "DBSCAN"}
    assert results.board["silhouette"].max() > 0.5
    assert not [c for c in results.checks if c.failed]


def test_classification_comparison_runs(rng):
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + 0.4 * rng.normal(size=n) > 0.85).astype(int))
    results = run_comparison(X, y, benchmark_repeats=1)

    assert infer_task(y) == "classification"
    assert set(results.board.index) == {"gradient boosting", "neural network (MLP)"}
    assert not [c for c in results.checks if c.failed]


# --------------------------------------------------------------------------- #
# Helpers for the hand-built comparisons
# --------------------------------------------------------------------------- #


def _baseline_stub(score: float):
    """A BaselineResults with one model baseline at a known score."""
    from src.models import BaselineResults

    return BaselineResults(
        task="regression",
        board=pd.DataFrame({"spearman": [score]},
                           index=pd.Index(["ridge on log(y)"], name="model")),
    )


def _comparison_from(advanced: AdvancedResults, baselines, reference_folds: np.ndarray):
    """Build the comparison table directly, bypassing the refit."""
    rows = []
    for name, folds in advanced.fold_scores.items():
        paired = folds - reference_folds
        delta, spread = float(np.mean(paired)), float(np.std(paired))
        rows.append({
            "model": name,
            "score": float(advanced.board.loc[name, "spearman"]),
            "reference": "ridge on log(y)",
            "reference_score": float(baselines.board.loc["ridge on log(y)", "spearman"]),
            "fold_mean": float(np.mean(folds)),
            "reference_fold_mean": float(np.mean(reference_folds)),
            "delta": delta,
            "delta_sd": spread,
            "folds_won": int(np.sum(paired > 0)),
            "folds": int(len(paired)),
            "reliable": bool(spread > 0 and abs(delta) / spread > RELIABILITY_MARGIN),
            "p_value": float(paired_test(paired)["p_value"]),
            "significant": bool(paired_test(paired)["significant"]),
            "fit_time_ratio": float("nan"),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("delta", ascending=False)
