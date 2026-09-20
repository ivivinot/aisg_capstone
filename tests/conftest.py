"""Fixtures shared by the test modules.

Two kinds of data:

* ``dirty_frame`` -- twelve hand-written rows containing every pathology the EDA
  found (fractional counts, sub-1 counts, a last purchase before the first) plus
  three the real file does *not* have (a missing value, an unseen category, a
  non-positive value). Tests that assert behaviour use this, because every
  expectation is checkable by eye.
* ``processed`` -- the real CSV, filtered, split and transformed exactly as
  ``main.py`` does it. Tests that assert the pipeline still earns its numbers use
  this. Session-scoped: fitting it once costs about a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    CLVPreprocessor,
    PreprocessConfig,
    TARGET,
    filter_rows,
    load_raw,
    quiet,
    stratified_split,
)

DATA = ROOT / "data" / "synthetic_data_126.csv"


@pytest.fixture(scope="session")
def config() -> PreprocessConfig:
    return PreprocessConfig()


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return load_raw(DATA)


@pytest.fixture(scope="session")
def processed(raw, config) -> dict:
    """The real data, put through main.py's stages 1-5."""
    clean, _ = filter_rows(raw, config=config)
    train_raw, test_raw = stratified_split(clean, config)
    X_train_raw = train_raw.drop(columns=[config.target])
    X_test_raw = test_raw.drop(columns=[config.target])
    with quiet():
        prep = CLVPreprocessor(config).fit(X_train_raw)
        X_train, X_test = prep.transform(X_train_raw), prep.transform(X_test_raw)
    return {
        "prep": prep, "train_raw": train_raw,
        "X_train_raw": X_train_raw, "X_test_raw": X_test_raw,
        "X_train": X_train, "X_test": X_test,
        "y_train": train_raw[config.target], "y_test": test_raw[config.target],
    }


@pytest.fixture
def dirty_frame() -> pd.DataFrame:
    """Twelve rows carrying every quality violation the pipeline claims to handle."""
    return pd.DataFrame({
        "total_purchase_count": [12.44, 0.03, 5.0, 43.16, 0.7, 9.75,
                                 2.0, 88.0, 1.5, 0.5, 30.0, 7.0],
        "average_order_value": [40.71, 58.41, 72.0, 155.18, 58.4, 44.82,
                                90.0, 210.0, 33.0, 61.0, 120.0, 85.0],
        "days_since_first_purchase": [309.32, 298.51, 400.0, 810.47, 120.0, 220.62,
                                      95.0, 1200.0, 60.0, 30.0, 500.0, 275.0],
        # rows 4, 8 and 9: a last purchase *before* the first (52 such rows in the file)
        "days_since_last_purchase": [52.28, 203.12, 88.0, 82.61, 300.0, 37.3,
                                     20.0, 140.0, 75.0, 90.0, 45.0, 66.0],
        "product_category_diversity": [0.0317, 0.121, 0.26, 0.475, 0.09, 0.216,
                                       0.31, 0.62, 0.05, 0.4, 0.18, 0.29],
        "loyalty_program_membership": ["Not Enrolled", "Enrolled"] * 6,
        TARGET: [415.06, 186.94, 620.0, 1790.0, 240.0, 475.64,
                 330.0, 5200.0, 150.0, 210.0, 1450.0, 540.0],
    })


@pytest.fixture
def fitted_on_dirty(dirty_frame, config) -> tuple:
    """A preprocessor fitted on the dirty frame, and the frame it was fitted on."""
    X = dirty_frame.drop(columns=[TARGET])
    with quiet():
        return CLVPreprocessor(config).fit(X), X


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
