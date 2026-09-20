"""Tests for src/feature_engineering.py -- creation, selection, and the claim behind both.

``main.py`` runs :func:`~src.feature_engineering.check_feature_engineering` on real
data every run; this file adds the cases that need a fixture built to break --
a feature whose inputs are missing, a selection strategy with no ``y``, a column
that is constant only in training.

The test that matters most is ``test_catalogue_log_space_classification``: the
whole feature strategy rests on products and ratios being linear combinations of
the logged inputs, and that is measured here rather than assumed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    ALL_DERIVED,
    DEFAULT_DERIVED,
    FEATURE_SPECS,
    INDEPENDENT_DERIVED,
    DerivedFeatures,
    FeatureSelector,
    SafeLogTransformer,
    check_feature_engineering,
    feature_report,
    resolve_derived,
)
from src.preprocessing import TARGET, CLVPreprocessor, quiet


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #


def test_named_sets_resolve():
    assert resolve_derived("none") == ()
    assert resolve_derived("default") == DEFAULT_DERIVED
    assert set(resolve_derived("all")) == set(FEATURE_SPECS)
    assert resolve_derived("purchase_value,recency_span") == ("purchase_value", "recency_span")
    assert set(resolve_derived(INDEPENDENT_DERIVED)) == {"recency_span", "is_lapsed"}


def test_unknown_feature_is_rejected():
    with pytest.raises(ValueError, match="unknown derived feature"):
        resolve_derived("free_shipping_propensity")


def test_every_spec_declares_its_inputs(dirty_frame):
    """A spec that names a column it does not use, or vice versa, would mislead."""
    X = dirty_frame.drop(columns=[TARGET])
    for name, spec in FEATURE_SPECS.items():
        assert set(spec.inputs) <= set(X.columns), name
        built = spec.build(X)
        assert len(built) == len(X), name
        assert np.isfinite(built.to_numpy()).all(), name


@pytest.mark.parametrize("name", sorted(FEATURE_SPECS))
def test_catalogue_log_space_classification(processed, name):
    """Measure the claim: does this feature add rank to the logged inputs, or not?

    A product or ratio of the inputs is a linear combination of their logs, so it
    cannot raise the rank of the logged design matrix. A difference or a
    threshold can, and must.

    No floor is applied to either side: the claim is about the arithmetic. What
    the *pipeline* does to values below the positivity floor is the subject of
    ``test_floor_breaks_exact_collinearity`` below.
    """
    spec = FEATURE_SPECS[name]
    X = processed["X_train_raw"]
    numeric = [c for c in X.columns if X[c].dtype.kind in "fi"]
    built = spec.build(X).to_numpy(dtype=float)

    # recency_span is negative for the 52 customers whose last purchase precedes
    # their first, and log() has nothing to say about those rows.
    keep = np.ones(len(X), dtype=bool) if spec.kind == "threshold" else built > 0
    logs = np.log(X[numeric].to_numpy(dtype=float))[keep]
    column = built[keep] if spec.kind == "threshold" else np.log(built[keep])

    base_rank = np.linalg.matrix_rank(logs)
    with_feature = np.linalg.matrix_rank(np.column_stack([logs, column]))
    if spec.new_in_log_space:
        assert with_feature == base_rank + 1, f"{name} claims to be new but adds no rank"
    else:
        assert with_feature == base_rank, f"{name} claims to be redundant but adds rank"


def test_floor_breaks_exact_collinearity(processed):
    """The documented exception: a ratio clipped at the floor is no longer linear.

    ``purchase_rate`` drops below 1e-3 for a handful of customers with one order
    and years of tenure. Clipping those values is what the pipeline must do to
    keep ``log()`` finite, and it costs the exact redundancy for those rows --
    which is why ``check_feature_engineering`` excludes them and says so.
    """
    X = processed["X_train_raw"]
    numeric = [c for c in X.columns if X[c].dtype.kind in "fi"]
    logs = np.log(X[numeric].to_numpy(dtype=float))
    rate = FEATURE_SPECS["purchase_rate"].build(X)

    assert (rate < 1e-3).sum() > 0, "fixture no longer exercises the floor"
    floored = np.log(rate.clip(lower=1e-3).to_numpy(dtype=float))
    assert np.linalg.matrix_rank(np.column_stack([logs, floored])) == \
        np.linalg.matrix_rank(logs) + 1

    kept = (rate > 1e-3).to_numpy()
    unfloored = np.log(rate.to_numpy(dtype=float)[kept])
    assert np.linalg.matrix_rank(np.column_stack([logs[kept], unfloored])) == \
        np.linalg.matrix_rank(logs[kept])


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #


def test_derived_features_are_appended_not_replaced(dirty_frame):
    X = dirty_frame.drop(columns=[TARGET])
    out = DerivedFeatures(features=ALL_DERIVED).fit(X).transform(X)
    assert list(out.columns[:len(X.columns)]) == list(X.columns)
    assert set(ALL_DERIVED) <= set(out.columns)


def test_derived_features_skip_when_inputs_are_missing(dirty_frame):
    """src/evaluation.py builds reduced feature sets; a missing input must not crash."""
    X = dirty_frame.drop(columns=[TARGET, "average_order_value"])
    transformer = DerivedFeatures(features=ALL_DERIVED).fit(X)
    assert "purchase_value" not in transformer.derived_      # needs average_order_value
    assert "recency_span" in transformer.derived_            # does not

    out = transformer.transform(X)
    assert np.isfinite(out[transformer.derived_].to_numpy(dtype=float)).all()


def test_recency_span_is_floored_for_impossible_rows(dirty_frame):
    """Three fixture rows have a last purchase before their first (negative span)."""
    X = dirty_frame.drop(columns=[TARGET])
    out = DerivedFeatures(features=("recency_span",), floor=1e-3).fit(X).transform(X)
    assert (out["recency_span"] > 0).all()
    assert np.isfinite(np.log(out["recency_span"])).all()


def test_threshold_feature_is_binary_and_not_logged(dirty_frame):
    X = dirty_frame.drop(columns=[TARGET])
    derived = DerivedFeatures(features=("is_lapsed",)).fit(X)
    out = derived.transform(X)
    assert set(np.unique(out["is_lapsed"])) <= {0.0, 1.0}

    logged = SafeLogTransformer(columns=["total_purchase_count"]).fit(out).transform(out)
    assert "is_lapsed" in logged.columns          # untouched by the log step
    assert set(np.unique(logged["is_lapsed"])) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


@pytest.fixture
def selection_frame(rng) -> pd.DataFrame:
    """Four columns with known pathologies: a constant, a duplicate, two signals."""
    n = 120
    signal = rng.normal(size=n)
    return pd.DataFrame({
        "signal": signal,
        "copy_of_signal": signal * 2.0 + 1e-9,       # perfectly correlated
        "constant": np.ones(n),
        "noise": rng.normal(size=n),
    })


def test_variance_strategy_drops_only_the_constant(selection_frame):
    selector = FeatureSelector(strategy="variance").fit(selection_frame)
    assert selector.dropped_ == ["constant"]


def test_correlation_strategy_drops_the_duplicate(selection_frame):
    selector = FeatureSelector(strategy="correlation").fit(selection_frame)
    assert set(selector.dropped_) == {"constant", "copy_of_signal"}
    assert "signal" in selector.feature_names_out_


def test_supervised_strategies_keep_k(selection_frame):
    y = selection_frame["signal"] * 3 + 0.01 * selection_frame["noise"]
    for strategy in ("mutual_info", "model"):
        selector = FeatureSelector(strategy=strategy, k=2).fit(selection_frame, y)
        assert len(selector.feature_names_out_) == 2
        assert "signal" in selector.feature_names_out_, strategy


def test_supervised_strategy_without_y_falls_back(selection_frame):
    selector = FeatureSelector(strategy="mutual_info", k=2).fit(selection_frame)
    assert selector.strategy_ == "variance"
    assert selector.dropped_ == ["constant"]


def test_vif_strategy_drops_collinear_columns(selection_frame):
    selector = FeatureSelector(strategy="vif", vif_threshold=6.0).fit(selection_frame)
    assert "copy_of_signal" in selector.dropped_ or "signal" in selector.dropped_


def test_selector_never_returns_an_empty_matrix():
    constant = pd.DataFrame({"a": np.ones(10), "b": np.ones(10)})
    selector = FeatureSelector(strategy="variance").fit(constant)
    assert list(selector.feature_names_out_) == ["a", "b"]


def test_selection_is_fitted_on_training_rows_only(selection_frame):
    """A column constant in training is dropped, even if it varies later."""
    train = selection_frame.copy()
    train["seasonal"] = 0.0
    test = selection_frame.copy()
    test["seasonal"] = np.arange(len(test), dtype=float)

    selector = FeatureSelector(strategy="variance").fit(train)
    assert "seasonal" in selector.dropped_
    assert "seasonal" not in selector.transform(test).columns


# --------------------------------------------------------------------------- #
# Integration with the preprocessor, and the checks main.py runs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("feature_set,expected_first_order", [
    ("none", 7), ("default", 9), ("independent", 9), ("all", 15),
])
def test_feature_count_grows_with_the_set(processed, config, feature_set, expected_first_order):
    cfg = replace(config, poly_degree=1,
                  add_derived_features=feature_set != "none",
                  derived_features=feature_set)
    with quiet():
        prep = CLVPreprocessor(cfg).fit(processed["X_train_raw"])
    assert prep.n_features_out_ == expected_first_order


def test_check_feature_engineering_passes_on_a_real_fit(processed, config):
    cfg = replace(config, add_derived_features=True, derived_features="all")
    with quiet():
        prep = CLVPreprocessor(cfg).fit(processed["X_train_raw"],
                                        np.log(processed["y_train"].to_numpy()))
    checks = check_feature_engineering(prep, processed["X_train_raw"],
                                       processed["X_test_raw"],
                                       np.log(processed["y_train"].to_numpy()))
    failures = [f"{c.name}: {c.detail}" for c in checks if c.failed]
    assert not failures, "\n".join(failures)


def test_feature_report_covers_every_column(processed):
    report = feature_report(processed["X_train"], np.log(processed["y_train"].to_numpy()))
    assert len(report) == processed["X_train"].shape[1]
    assert {"std", "share_zero", "spearman_vs_target", "vif"} <= set(report.columns)
    assert report["abs_spearman"].is_monotonic_decreasing
