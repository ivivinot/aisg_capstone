"""pytest wrapper around the checks in src/preprocessing.py, plus unit tests.

``main.py`` already runs every check in ``src.preprocessing`` on real data at the end
of each run. This file exists so the same checks can run in CI, on fixtures that
contain the pathologies the real file happens *not* to have -- missing values, an
unseen category, a non-positive feature -- and so a refactor fails here rather
than in a production run.

    pytest tests/ -q                # fast: contract, behaviour, unit tests
    pytest tests/ -q -m slow        # the cross-validated checks as well
    pytest tests/ -q --cov=src
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.preprocessing import (
    TARGET,
    CLVPreprocessor,
    LogTargetTransformer,
    PreprocessConfig,
    audit_dataset,
    check_output_contract,
    check_statistics,
    check_transformer_behaviour,
    filter_rows,
    stratified_split,
)


def _assert_all_passed(checks) -> None:
    failures = [f"{c.name}: {c.detail}" for c in checks if c.failed]
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------- #
# The three groups main.py runs, as three tests
# --------------------------------------------------------------------------- #


def test_output_contract(processed):
    _assert_all_passed(check_output_contract(
        processed["X_train"], processed["X_test"],
        processed["y_train"], processed["y_test"], processed["prep"]))


def test_transformer_behaviour(processed):
    _assert_all_passed(check_transformer_behaviour(
        processed["prep"], processed["X_train_raw"], processed["X_test_raw"]))


@pytest.mark.slow
def test_statistical_validation(processed, config):
    _assert_all_passed(check_statistics(processed["train_raw"], config))


# --------------------------------------------------------------------------- #
# Unit tests on the dirty frame -- the cases the real file cannot exercise
# --------------------------------------------------------------------------- #


def test_missing_values_are_imputed_with_training_medians(dirty_frame, config):
    X = dirty_frame.drop(columns=[TARGET])
    holey = X.copy()
    holey.loc[0, "average_order_value"] = np.nan
    prep = CLVPreprocessor(config).fit(X)

    out = prep.transform(holey)
    median = float(X["average_order_value"].median())
    expected = prep.transform(X.assign(
        average_order_value=X["average_order_value"].where(X.index != 0, median)))
    assert np.isfinite(out.to_numpy()).all()
    np.testing.assert_allclose(out.to_numpy()[0], expected.to_numpy()[0])


def test_unseen_category_does_not_raise(fitted_on_dirty):
    prep, X = fitted_on_dirty
    odd = X.copy()
    odd.loc[0, "loyalty_program_membership"] = "Platinum"
    out = prep.transform(odd)
    assert np.isfinite(out.to_numpy()).all()


@pytest.mark.parametrize("value", [0.0, -5.0, 1e-12])
def test_log_floor_keeps_output_finite(fitted_on_dirty, value):
    prep, X = fitted_on_dirty
    floored = X.copy()
    floored.loc[0, "total_purchase_count"] = value
    assert np.isfinite(prep.transform(floored).to_numpy()).all()


def test_missing_feature_raises_value_error(fitted_on_dirty):
    prep, X = fitted_on_dirty
    with pytest.raises(ValueError, match="average_order_value"):
        prep.transform(X.drop(columns=["average_order_value"]))


def test_invalid_recency_is_flagged_not_repaired(dirty_frame, config):
    """policy='flag' keeps the value and adds a column; policy='clip' overwrites it."""
    X = dirty_frame.drop(columns=[TARGET])
    flagged = CLVPreprocessor(config).fit(X)
    assert "flag_invalid_recency" in flagged.feature_names_out_
    assert flagged.quality_report()["domain_rules"]["invalid_recency"] == 3

    from dataclasses import replace
    clipped = CLVPreprocessor(replace(config, recency_policy="clip")).fit(X)
    assert "flag_invalid_recency" not in clipped.feature_names_out_


@pytest.mark.parametrize("degree,expected", [(1, 7), (2, 35), (3, 119)])
def test_feature_count_by_degree(processed, config, degree, expected):
    from dataclasses import replace
    prep = CLVPreprocessor(replace(config, poly_degree=degree)).fit(processed["X_train_raw"])
    assert prep.n_features_out_ == expected


def test_single_row_equals_batch(fitted_on_dirty):
    """The property predict.py depends on: batch statistics must not exist."""
    prep, X = fitted_on_dirty
    batch = prep.transform(X)
    singles = pd.concat([prep.transform(X.iloc[[i]]) for i in range(len(X))])
    np.testing.assert_allclose(singles.to_numpy(), batch.to_numpy())


def test_clone_and_set_params_round_trip(fitted_on_dirty, config):
    """src/models.py tunes prep__config through clone/set_params."""
    prep, X = fitted_on_dirty
    twin = clone(prep)
    twin.set_params(config=config)
    twin.fit(X)
    np.testing.assert_allclose(twin.transform(X).to_numpy(), prep.transform(X).to_numpy())


def test_save_load_round_trip(fitted_on_dirty, tmp_path):
    prep, X = fitted_on_dirty
    reloaded = CLVPreprocessor.load(prep.save(tmp_path / "prep.joblib"))
    np.testing.assert_allclose(reloaded.transform(X).to_numpy(), prep.transform(X).to_numpy())


# --------------------------------------------------------------------------- #
# Target transform, audit, split
# --------------------------------------------------------------------------- #


def test_log_target_rejects_non_positive_values():
    with pytest.raises(ValueError, match="non-positive"):
        LogTargetTransformer().fit(np.array([10.0, 0.0, 5.0]))


def test_smearing_factor_corrects_the_level():
    """exp(E[log y]) under-states E[y]; Duan's factor puts the level back."""
    y = np.array([100.0, 200.0, 400.0, 1600.0])
    transformer = LogTargetTransformer().fit(y)
    residuals = np.array([0.2, -0.1, 0.05, -0.15])
    transformer.fit_smearing(residuals)
    assert transformer.smearing_factor_ == pytest.approx(float(np.mean(np.exp(residuals))))
    assert transformer.smearing_factor_ > 1.0


def test_audit_reports_the_known_violations(raw):
    report = audit_dataset(raw)
    violations = report["domain_violations"]
    assert report["rows"] == 1000
    assert report["target"]["zero_value_customers"] == 0
    assert violations["non_integer_purchase_count"] == 988
    assert violations["purchase_count_below_one"] == 143
    assert violations["last_purchase_before_first"] == 52


def test_filter_rows_removes_nothing_by_default(raw, config):
    kept, removed = filter_rows(raw, config=config)
    assert len(kept) == len(raw)
    assert removed == {}


def test_split_is_deterministic_and_stratified(raw, config):
    clean, _ = filter_rows(raw, config=config)
    first_train, first_test = stratified_split(clean, config)
    second_train, second_test = stratified_split(clean, config)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)
    assert len(first_test) == pytest.approx(len(clean) * config.test_size, abs=1)
    # Stratifying on value deciles is there to keep the whales on both sides.
    assert first_test[TARGET].median() == pytest.approx(first_train[TARGET].median(), rel=0.05)


def test_dropping_invalid_recency_is_opt_in(raw, config):
    kept, removed = filter_rows(raw, config=config, drop_invalid_recency=True)
    assert removed["invalid_recency"] == 52
    assert len(kept) == len(raw) - 52
