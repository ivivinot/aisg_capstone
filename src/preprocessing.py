"""Reusable preprocessing pipeline for the capstone CLV dataset.

Every transformer here answers a finding from ``eda.ipynb``; the section numbers
in the docstrings point at the analysis that motivated the choice.

    EDA finding                                  Strategy implemented here
    -------------------------------------------  ------------------------------------
    No missing cells, but the loader must not     FrameImputer: median for the skewed
    assume that holds for new data (2)            numerics, mode for the binary flag
    52 customers have a last purchase BEFORE      DomainRuleTransformer: the rule is
    their first -> negative recency (2, 6)        flagged, clipped or dropped, never
                                                  silently repaired
    143 customers with < 1 purchase, 988 with     DomainRuleTransformer: positivity
    fractional counts -> log() is unsafe (2)      floor, so log() cannot emit -inf
    Target skew 6.4, top 10% hold 38% of value    QuantileClipper with wide default
    -- the tail is the business, not noise (3)    bounds: an extrapolation guard rail,
                                                  NOT tail removal
    Target is a power law in the features:        SafeLogTransformer + PolynomialFeatures
    log-log linear gives R2 0.987, degree 3       + StandardScaler -- the representation
    gives 0.9998 (7, 10)                          that beat tuned XGBoost on every metric
    exp(E[log y]) under-states E[y] for a         LogTargetTransformer with Duan's
    skewed target (8)                             smearing estimator

Every default here was chosen by measurement, not by reflex. Running ``main.py``
with one flag changed gives (5-fold CV R2 on log(value), Ridge alpha=0.01, the
preprocessor refitted inside each fold):

    --no-log                      0.44211   without the log transform, nothing works
    --poly-degree 1               0.98661   log-linear; matches EDA 7's 0.9868
    defaults (degree 3)           0.99938   the representation the EDA selected
    --recency-policy clip         0.99579   repairing the 52 impossible rows hurts
    --outlier-quantiles .001 .999 0.99282   winsorizing the tail hurts more

The first two lines are why the log transform is the centre of this module. The
last two are why its "cleaning" is deliberately conservative: EDA 7 established
that the label is a deterministic function of the feature values *as supplied*,
so every edit to a feature deletes the input that produced its label. The
pipeline therefore flags and bounds rather than overwrites -- and the aggressive
options stay available behind flags, with their cost written down.

Design rules that keep the pipeline honest:

* Everything that *learns* a statistic (imputation fills, clip bounds, scaler
  means) is an sklearn transformer fitted on the training split only, so the
  whole object can be cross-validated without leakage.
* Anything that *removes rows* lives outside the pipeline, in
  :func:`filter_rows`, because sklearn transformers may not change the number of
  samples without breaking X/y alignment.
* :class:`CLVPreprocessor` is itself a scikit-learn transformer, so it drops
  straight into a ``Pipeline`` with any estimator.

Usage
-----
    from src.preprocessing import CLVPreprocessor, PreprocessConfig, load_raw

    df = load_raw("data/synthetic_data_126.csv")
    train, test = stratified_split(df)
    prep = CLVPreprocessor(PreprocessConfig()).fit(train.drop(columns=[TARGET]))
    X_train = prep.transform(train.drop(columns=[TARGET]))
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

__all__ = [
    "TARGET",
    "SKEWED_NUMERIC",
    "CATEGORICAL_MAP",
    "PreprocessConfig",
    "CLVPreprocessor",
    "LogTargetTransformer",
    "CategoricalEncoder",
    "DomainRuleTransformer",
    "DerivedFeatures",
    "FrameImputer",
    "QuantileClipper",
    "ConstantColumnDropper",
    "SafeLogTransformer",
    "load_raw",
    "audit_dataset",
    "filter_rows",
    "stratified_split",
    "get_logger",
    "quiet",
]

logger = logging.getLogger("capstone.preprocessing")


# --------------------------------------------------------------------------- #
# Dataset contract (eda.ipynb 2, readme.md 9)
# --------------------------------------------------------------------------- #

TARGET = "estimated_lifetime_value"

#: The five continuous features. All are strictly positive and right-skewed, and
#: all enter the model in logs (EDA 7).
SKEWED_NUMERIC: Tuple[str, ...] = (
    "total_purchase_count",
    "average_order_value",
    "days_since_first_purchase",
    "days_since_last_purchase",
    "product_category_diversity",
)

#: The one categorical column, and the binary column it becomes.
CATEGORICAL_MAP: Dict[str, Dict[str, int]] = {
    "loyalty_program_membership": {"Enrolled": 1, "Not Enrolled": 0},
}
BINARY_FEATURES: Tuple[str, ...] = ("loyalty",)

#: Columns whose pairing encodes a domain rule: a last purchase cannot predate a
#: first one (EDA 6 -- 52 rows violate this).
FIRST_PURCHASE_COL = "days_since_first_purchase"
LAST_PURCHASE_COL = "days_since_last_purchase"
COUNT_COL = "total_purchase_count"
AOV_COL = "average_order_value"


def get_logger(name: str = "capstone", level: int = logging.INFO) -> logging.Logger:
    """A console logger that is safe to call repeatedly.

    Writes to stdout, not stderr, so the log lines interleave with the report
    ``main.py`` prints instead of arriving in a block ahead of it.
    """
    import sys

    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s | %(message)s"))
        log.addHandler(handler)
    log.setLevel(level)
    return log


@contextmanager
def quiet(level: int = logging.WARNING):
    """Silence the module logger, e.g. across the folds of a cross-validation."""
    previous = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class PreprocessConfig:
    """Every knob of the pipeline, in one serialisable object.

    The defaults reproduce the representation that won model selection in
    ``eda.ipynb`` 9-11: logged numerics, degree-3 polynomial expansion,
    standardised, with the data-quality violations clipped rather than dropped.
    """

    # -- columns ----------------------------------------------------------- #
    target: str = TARGET
    numeric_features: Tuple[str, ...] = SKEWED_NUMERIC
    categorical_map: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {k: dict(v) for k, v in CATEGORICAL_MAP.items()}
    )

    # -- missing values (EDA 2: none in this file, assume nothing elsewhere) - #
    numeric_impute: str = "median"          # median | mean | knn
    binary_impute: str = "most_frequent"    # most_frequent | constant
    binary_fill_value: int = 0
    knn_neighbors: int = 5
    add_missing_indicators: bool = True

    # -- data quality rules (EDA 2, 6) -------------------------------------- #
    # Default "flag", not "clip": see the note on measured cost in
    # DomainRuleTransformer -- repairing the 52 impossible rows costs more than it
    # buys on this file.
    recency_policy: str = "flag"            # clip | flag | none  ("drop" -> filter_rows)
    add_quality_flags: bool = False
    positivity_floor: float = 1e-3          # keeps log() finite for sub-1 counts

    # -- outliers (EDA 3: the tail is real; guard, do not amputate) ---------- #
    # (0.0, 1.0) => bounds are the training min/max, so no training row is
    # altered and new data is still held inside the fitted range. See QuantileClipper.
    outlier_strategy: str = "clip"          # clip | flag | none  ("drop" -> filter_rows)
    outlier_quantiles: Tuple[float, float] = (0.0, 1.0)
    outlier_margin: float = 0.5             # widen the bounds by 50% before clipping

    # -- normalisation (EDA 7, 10) ------------------------------------------ #
    log_transform: bool = True
    poly_degree: int = 3
    scale: bool = True
    drop_constant_features: bool = True     # prune dead columns the expansion creates

    # -- optional feature engineering (off by default: keeps the feature set
    #    identical to the one the EDA validated) ---------------------------- #
    add_derived_features: bool = False

    # -- splitting (EDA 8) --------------------------------------------------- #
    test_size: float = 0.2
    n_strata: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        self.numeric_features = tuple(self.numeric_features)
        self.outlier_quantiles = tuple(self.outlier_quantiles)
        _choice("numeric_impute", self.numeric_impute, {"median", "mean", "knn"})
        _choice("binary_impute", self.binary_impute, {"most_frequent", "constant"})
        _choice("recency_policy", self.recency_policy, {"clip", "flag", "none"})
        _choice("outlier_strategy", self.outlier_strategy, {"clip", "flag", "none"})
        lo, hi = self.outlier_quantiles
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"outlier_quantiles must satisfy 0 <= lo < hi <= 1, got {(lo, hi)}")
        if self.outlier_margin < 0:
            raise ValueError("outlier_margin must be >= 0")
        if self.poly_degree < 1:
            raise ValueError("poly_degree must be >= 1")

    @property
    def binary_features(self) -> Tuple[str, ...]:
        """Names of the binary columns produced by categorical encoding."""
        return tuple(_binary_name(col) for col in self.categorical_map)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=list), encoding="utf-8")
        return path


def _choice(name: str, value: str, allowed: set) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")


def _binary_name(categorical_col: str) -> str:
    """``loyalty_program_membership`` -> ``loyalty`` (the EDA's own naming)."""
    return categorical_col.split("_")[0]


# --------------------------------------------------------------------------- #
# Loading and auditing
# --------------------------------------------------------------------------- #


def load_raw(path: str | Path, target: str = TARGET) -> pd.DataFrame:
    """Read the CSV and check the columns the pipeline depends on are present."""
    frame = pd.read_csv(path)
    required = set(SKEWED_NUMERIC) | set(CATEGORICAL_MAP) | {target}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {sorted(missing)}")
    logger.info("loaded %s -- %d rows x %d columns", path, frame.shape[0], frame.shape[1])
    return frame


def audit_dataset(frame: pd.DataFrame, target: str = TARGET) -> dict:
    """Reproduce the EDA 2 quality checks as a machine-readable report.

    Nothing is repaired here. The report is what justifies the strategies the
    pipeline then applies, and it is written to disk beside the processed data so
    a reviewer can see what the pipeline was handed.
    """
    numeric = [c for c in SKEWED_NUMERIC if c in frame.columns]
    report: dict = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells_total": int(frame.isna().sum().sum()),
        "missing_by_column": {c: int(n) for c, n in frame.isna().sum().items() if n},
    }

    if target in frame.columns:
        y = frame[target]
        report["target"] = {
            "min": float(y.min()),
            "median": float(y.median()),
            "max": float(y.max()),
            "skew": float(y.skew()),
            "skew_of_log": float(np.log(y[y > 0]).skew()),
            "zero_value_customers": int((y == 0).sum()),
            "non_positive": int((y <= 0).sum()),
            # EDA 3: the concentration that makes ranking metrics the headline.
            "top_decile_value_share": float(
                np.sort(y.to_numpy())[::-1][: max(1, len(y) // 10)].sum() / y.sum()
            ),
        }

    violations: dict = {}
    if {FIRST_PURCHASE_COL, LAST_PURCHASE_COL} <= set(frame.columns):
        violations["last_purchase_before_first"] = int(
            (frame[LAST_PURCHASE_COL] > frame[FIRST_PURCHASE_COL]).sum()
        )
    if COUNT_COL in frame.columns:
        violations["non_integer_purchase_count"] = int((frame[COUNT_COL] % 1 != 0).sum())
        violations["purchase_count_below_one"] = int((frame[COUNT_COL] < 1).sum())
    violations["non_positive_in_logged_columns"] = {
        c: int((frame[c] <= 0).sum()) for c in numeric if (frame[c] <= 0).any()
    }
    report["domain_violations"] = violations
    report["feature_skew"] = {c: float(frame[c].skew()) for c in numeric}
    return report


def filter_rows(
    frame: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
    drop_invalid_recency: bool = False,
    drop_non_positive_target: bool = True,
    drop_duplicates: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Row-level cleaning, applied *before* the split.

    Row removal cannot live inside an sklearn transformer -- ``transform`` must
    return one row per input row or X and y fall out of alignment. So the "drop"
    variants of the outlier and recency strategies are implemented here and are
    off by default: EDA 2 is explicit that the impossible rows are evidence about
    the generator, not errors to be quietly deleted.
    """
    config = config or PreprocessConfig()
    before = len(frame)
    out = frame
    removed: dict = {}

    if drop_duplicates:
        out = out.drop_duplicates()
        removed["duplicates"] = before - len(out)

    if drop_invalid_recency and {FIRST_PURCHASE_COL, LAST_PURCHASE_COL} <= set(out.columns):
        mask = out[LAST_PURCHASE_COL] <= out[FIRST_PURCHASE_COL]
        removed["invalid_recency"] = int((~mask).sum())
        out = out[mask]

    if drop_non_positive_target and config.target in out.columns:
        # log(y) is undefined at y <= 0; EDA 3 confirms the minimum here is 58.64.
        mask = out[config.target] > 0
        removed["non_positive_target"] = int((~mask).sum())
        out = out[mask]

    removed = {k: v for k, v in removed.items() if v}
    if removed:
        logger.warning("filter_rows removed %d of %d rows: %s", before - len(out), before, removed)
    return out.reset_index(drop=True), removed


def stratified_split(
    frame: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split stratified on target deciles (EDA 8).

    Stratifying on value deciles keeps the same share of high-value customers on
    both sides -- with skew 6.4 and 38% of value in the top decile, a plain
    random split of 1,000 rows can easily hand one side a disproportionate number
    of whales.

    Note the caveat the EDA insists on: a CLV *forecast* must be split by time.
    This file has no timestamps, so this is a cross-sectional split and the
    results must be read as such (readme.md 7.1, 9).
    """
    config = config or PreprocessConfig()
    y = frame[config.target]
    strata = pd.qcut(y, config.n_strata, labels=False, duplicates="drop")
    train, test = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=strata,
    )
    logger.info(
        "split -- train %d rows (median %.0f), test %d rows (median %.0f)",
        len(train), train[config.target].median(), len(test), test[config.target].median(),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Transformers
# --------------------------------------------------------------------------- #


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Map the categorical column(s) onto binary columns.

    One column, two levels (Enrolled 40% / Not Enrolled 60%), so an explicit
    mapping beats one-hot: it produces the single ``loyalty`` column the EDA used
    and keeps the polynomial expansion small. Unseen categories become NaN and
    are handed to the imputer rather than silently coded as 0.
    """

    def __init__(
        self,
        mapping: Optional[Dict[str, Dict[str, int]]] = None,
        drop_original: bool = True,
    ):
        self.mapping = mapping
        self.drop_original = drop_original

    def _map(self) -> Dict[str, Dict[str, int]]:
        return self.mapping if self.mapping is not None else CATEGORICAL_MAP

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.unseen_categories_: Dict[str, List[str]] = {}
        self.feature_names_out_ = self._names_out(list(X.columns))
        return self

    def _names_out(self, columns: Sequence[str]) -> np.ndarray:
        mapping = self._map()
        names = [c for c in columns if not (self.drop_original and c in mapping)]
        names += [_binary_name(c) for c in mapping if c in columns]
        return np.asarray(names, dtype=object)

    def transform(self, X) -> pd.DataFrame:
        X = _as_frame(X).copy()
        for col, levels in self._map().items():
            if col not in X.columns:
                continue
            unseen = sorted(set(X[col].dropna().unique()) - set(levels))
            if unseen:
                # Recorded, not guessed at: an unknown loyalty tier is missing
                # information, and the imputer decides what to do with it.
                known = set(self.unseen_categories_.get(col, []))
                self.unseen_categories_[col] = sorted(known | set(unseen))
                logger.warning("%s: unseen categories %s -> NaN", col, unseen)
            X[_binary_name(col)] = X[col].map(levels).astype(float)
            if self.drop_original:
                X = X.drop(columns=[col])
        return X[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_


class DomainRuleTransformer(BaseEstimator, TransformerMixin):
    """Enforce the physical constraints the EDA found violated.

    Two rules:

    1. **A last purchase cannot predate a first one.** 52 of 1,000 rows break
       this (EDA 2, 6).

       * ``flag`` (default) appends ``flag_invalid_recency`` and leaves the value
         untouched;
       * ``clip`` pulls ``days_since_last_purchase`` back to
         ``days_since_first_purchase`` -- the minimal edit that makes the implied
         recency non-negative;
       * ``none`` ignores the rule; dropping is in :func:`filter_rows`.

       **Why flag, and not clip, is the default here.** Measured on this file
       (5-fold CV R2 on log(value), Ridge alpha=0.01, everything else at its
       default), ``flag`` scores **0.99938** and ``clip`` **0.99579** -- clipping
       45 training rows costs six times the error. Flagging is also free:
       ``none`` scores 0.99937, so the extra indicator column pays for itself and
       the information is kept rather than thrown away.

       That ordering is not a quirk: EDA 7 showed the target
       is a deterministic function of the feature values *as supplied*, so
       overwriting a feature deletes the input that produced the label for that
       row. On real transactional data an impossible recency is a corrupted
       reading and repairing it would help; on a generated file it is the signal.
       Flagging keeps both -- the model sees the value and knows it is suspect.
       The measurement is why the default is what it is; run
       ``main.py --recency-policy clip`` to reproduce the comparison.

    2. **Logged columns must be strictly positive.** 143 customers have fewer
       than one purchase, and the floor guards against a value reaching 0 in new
       data, where ``log`` would emit ``-inf`` and poison the scaler.

    The violation counts seen during ``fit`` are kept on ``rule_report_`` so the
    caller can assert that training data looked like the EDA said it did.
    """

    def __init__(
        self,
        numeric_features: Sequence[str] = SKEWED_NUMERIC,
        recency_policy: str = "clip",
        positivity_floor: float = 1e-3,
        add_flags: bool = False,
    ):
        self.numeric_features = numeric_features
        self.recency_policy = recency_policy
        self.positivity_floor = positivity_floor
        self.add_flags = add_flags

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.rule_report_ = {
            "invalid_recency": int(_invalid_recency_mask(X).sum()),
            "non_positive": {
                c: int((X[c] <= 0).sum())
                for c in self.numeric_features
                if c in X.columns and (X[c] <= 0).any()
            },
        }
        if self.rule_report_["invalid_recency"]:
            logger.info(
                "domain rules: %d rows with last purchase before first -> policy=%s",
                self.rule_report_["invalid_recency"], self.recency_policy,
            )
        self.flag_names_ = self._flag_names(list(X.columns))
        self.feature_names_out_ = np.asarray(list(X.columns) + self.flag_names_, dtype=object)
        return self

    def _flag_names(self, columns: Sequence[str]) -> List[str]:
        """``recency_policy='flag'`` implies its own indicator; ``add_flags`` adds the rest."""
        names: List[str] = []
        recency_available = {FIRST_PURCHASE_COL, LAST_PURCHASE_COL} <= set(columns)
        if recency_available and (self.add_flags or self.recency_policy == "flag"):
            names.append("flag_invalid_recency")
        if self.add_flags and COUNT_COL in columns:
            names += ["flag_fractional_count", "flag_count_below_one"]
        return names

    def transform(self, X) -> pd.DataFrame:
        X = _as_frame(X).copy()
        invalid = _invalid_recency_mask(X)

        if "flag_invalid_recency" in self.flag_names_:
            X["flag_invalid_recency"] = invalid.astype(float)
        if "flag_fractional_count" in self.flag_names_:
            X["flag_fractional_count"] = (X[COUNT_COL] % 1 != 0).astype(float)
            X["flag_count_below_one"] = (X[COUNT_COL] < 1).astype(float)

        if self.recency_policy == "clip" and invalid.any():
            X.loc[invalid, LAST_PURCHASE_COL] = X.loc[invalid, FIRST_PURCHASE_COL]

        floor = float(self.positivity_floor)
        for col in self.numeric_features:
            if col in X.columns:
                X[col] = X[col].clip(lower=floor)

        return X[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_


class FrameImputer(BaseEstimator, TransformerMixin):
    """Per-column missing-value handling that preserves column names and order.

    The training file has zero missing cells (EDA 2), so on this data the imputer
    is a no-op -- which is exactly why it is here: a pipeline that only works on
    data as clean as its training file is not reusable. The strategies follow the
    shape of each column rather than a blanket default:

    * numerics are skewed (skew up to 6.4), so the **median** is the robust fill;
      ``knn`` is offered for the case where missingness is informative;
    * the binary loyalty flag gets its **mode**;
    * ``add_indicators`` appends a 0/1 column for any feature that was missing
      during training, so "this was imputed" stays visible to the model.
    """

    def __init__(
        self,
        numeric_features: Sequence[str] = SKEWED_NUMERIC,
        binary_features: Sequence[str] = BINARY_FEATURES,
        numeric_strategy: str = "median",
        binary_strategy: str = "most_frequent",
        binary_fill_value: int = 0,
        knn_neighbors: int = 5,
        add_indicators: bool = True,
    ):
        self.numeric_features = numeric_features
        self.binary_features = binary_features
        self.numeric_strategy = numeric_strategy
        self.binary_strategy = binary_strategy
        self.binary_fill_value = binary_fill_value
        self.knn_neighbors = knn_neighbors
        self.add_indicators = add_indicators

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.numeric_cols_ = [c for c in self.numeric_features if c in X.columns]
        self.binary_cols_ = [c for c in self.binary_features if c in X.columns]

        self.fill_values_: Dict[str, float] = {}
        for col in self.numeric_cols_:
            if self.numeric_strategy == "mean":
                self.fill_values_[col] = float(X[col].mean())
            else:  # median, and the fallback used by knn on an all-NaN column
                self.fill_values_[col] = float(X[col].median())
        for col in self.binary_cols_:
            if self.binary_strategy == "constant":
                self.fill_values_[col] = float(self.binary_fill_value)
            else:
                mode = X[col].mode(dropna=True)
                self.fill_values_[col] = (
                    float(mode.iloc[0]) if len(mode) else float(self.binary_fill_value)
                )

        self.knn_imputer_ = None
        if self.numeric_strategy == "knn" and self.numeric_cols_:
            self.knn_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
            self.knn_imputer_.fit(X[self.numeric_cols_])

        # Follow sklearn: indicators exist only for columns missing at fit time,
        # so the output width cannot depend on what a later batch happens to hold.
        self.indicator_cols_ = (
            [c for c in self.numeric_cols_ + self.binary_cols_ if X[c].isna().any()]
            if self.add_indicators
            else []
        )
        self.feature_names_out_ = np.asarray(
            list(X.columns) + [f"missing_{c}" for c in self.indicator_cols_], dtype=object
        )
        n_missing = int(X[self.numeric_cols_ + self.binary_cols_].isna().sum().sum())
        logger.info("imputer fitted -- %d missing cells in training data", n_missing)
        return self

    def transform(self, X) -> pd.DataFrame:
        X = _as_frame(X).copy()
        for col in self.indicator_cols_:
            X[f"missing_{col}"] = X[col].isna().astype(float)

        if self.knn_imputer_ is not None:
            block = pd.DataFrame(
                self.knn_imputer_.transform(X[self.numeric_cols_]),
                columns=self.numeric_cols_,
                index=X.index,
            )
            X[self.numeric_cols_] = block
        for col in self.numeric_cols_ + self.binary_cols_:
            if X[col].isna().any():
                X[col] = X[col].fillna(self.fill_values_[col])

        return X[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Outlier handling as a guard rail, not as tail removal.

    EDA 3 is unambiguous: the heavy tail *is* the business -- the top 10% of
    customers hold 38% of total value, and deleting them would destroy the thing
    the model exists to rank. The log transform, not row deletion, is what stops
    squared-error loss from chasing whales.

    What is still worth doing is bounding the feature space seen at *predict*
    time. The selected model is a degree-3 polynomial, which extrapolates
    violently outside its training range (EDA 12 makes the same point about
    permutation importance). Clipping new data to the training bounds keeps
    predictions inside the region the model was actually fitted on.

    **Why the default is quantiles (0.0, 1.0) plus a margin.** Four settings,
    measured on this file as 5-fold CV R2 on log(value) with Ridge alpha=0.01:

        quantiles (0.001, 0.999), no margin   0.99282   trims ~2 values/column
        quantiles (0, 1), no margin           0.99735   bounds = training min/max
        quantiles (0, 1), margin 0.5          0.99938   <- the default
        no clipping at all                    0.99971

    Winsorizing costs accuracy for the reason given in
    :class:`DomainRuleTransformer`: EDA 7 showed the label is a deterministic
    function of the supplied feature values, so altering a value deletes the
    input that produced it. Even bare min/max bounds cost something, because
    inside a CV fold the bounds come from 640 rows and the held-out extremes fall
    outside them -- and those extremes are the high-value customers the model
    exists to rank, so capping them biases exactly the wrong tail.

    The margin fixes that. Bounds are widened to ``[lo / (1 + margin),
    hi * (1 + margin)]``, so a value has to be 50% beyond the most extreme
    customer ever seen in training before it is touched. Ordinary new data passes
    through untouched; a corrupted or wildly out-of-range record still cannot
    push the degree-3 polynomial into the region where it extrapolates violently
    (EDA 12 makes the same point about permutation importance). The guard rail is
    kept, and it costs essentially nothing.

    Tighter winsorizing remains one flag away -- ``main.py --outlier-quantiles
    0.01 0.99`` -- for data where the tail really is contaminated rather than
    generated.

    Strategies: ``clip`` (default), ``flag`` (append 0/1 out-of-range columns and
    leave the values untouched), ``none``. Dropping is in :func:`filter_rows`.
    """

    def __init__(
        self,
        columns: Sequence[str] = SKEWED_NUMERIC,
        strategy: str = "clip",
        quantiles: Tuple[float, float] = (0.0, 1.0),
        margin: float = 0.5,
    ):
        self.columns = columns
        self.strategy = strategy
        self.quantiles = quantiles
        self.margin = margin

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = [c for c in self.columns if c in X.columns]
        lo_q, hi_q = self.quantiles
        # Bounds come from the TRAINING split only -- this is the leakage-critical
        # line in the whole module.
        self.raw_bounds_ = {
            c: (float(X[c].quantile(lo_q)), float(X[c].quantile(hi_q))) for c in self.columns_
        }
        self.bounds_ = {
            c: _widen(lo, hi, self.margin) for c, (lo, hi) in self.raw_bounds_.items()
        }
        self.n_clipped_at_fit_ = {
            c: int(((X[c] < lo) | (X[c] > hi)).sum()) for c, (lo, hi) in self.bounds_.items()
        }
        flags = [f"outlier_{c}" for c in self.columns_] if self.strategy == "flag" else []
        self.feature_names_out_ = np.asarray(list(X.columns) + flags, dtype=object)
        if self.strategy != "none":
            logger.info(
                "outlier bounds learned on train at quantiles %s -- %d values out of range",
                tuple(self.quantiles), sum(self.n_clipped_at_fit_.values()),
            )
        return self

    def transform(self, X) -> pd.DataFrame:
        X = _as_frame(X).copy()
        if self.strategy == "none":
            return X[list(self.feature_names_out_)]
        for col, (lo, hi) in self.bounds_.items():
            if self.strategy == "flag":
                X[f"outlier_{col}"] = ((X[col] < lo) | (X[col] > hi)).astype(float)
            else:
                X[col] = X[col].clip(lower=lo, upper=hi)
        return X[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_


class SafeLogTransformer(BaseEstimator, TransformerMixin):
    """Natural log of the skewed numerics -- the core normalisation step.

    EDA 7 established that the target is multiplicative in these features::

        value = exp(b0) * x1^b1 * ... * exp(bk * loyalty) * eps

    Taking logs turns that product into a sum, which is why a *linear* model on
    logged features reaches R2 0.987 and, with polynomial terms, 0.9998 -- beating
    tuned XGBoost and Random Forest on every metric (EDA 10). This transformer is
    where most of the predictive work in the whole pipeline happens.

    "Safe" means the floor from :class:`DomainRuleTransformer` is re-applied here,
    so a zero or negative arriving in new data becomes a small positive number
    instead of ``-inf``.
    """

    def __init__(
        self,
        columns: Sequence[str] = SKEWED_NUMERIC,
        floor: float = 1e-3,
        prefix: str = "log_",
    ):
        self.columns = columns
        self.floor = floor
        self.prefix = prefix

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = [c for c in self.columns if c in X.columns]
        self.feature_names_out_ = np.asarray(
            [f"{self.prefix}{c}" if c in self.columns_ else c for c in X.columns], dtype=object
        )
        return self

    def transform(self, X) -> pd.DataFrame:
        X = _as_frame(X).copy()
        n_floored = 0
        for col in self.columns_:
            n_floored += int((X[col] < self.floor).sum())
            X[col] = np.log(X[col].clip(lower=self.floor))
        if n_floored:
            logger.warning(
                "safe log: %d values raised to the floor %g before log()", n_floored, self.floor
            )
        X.columns = list(self.feature_names_out_)
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Optional feature engineering, appended before the log step.

    Off by default, so the pipeline's output matches the six-column feature set
    the EDA validated. Two additions are worth having when it is switched on:

    * ``purchase_value`` = purchases x order value -- the analyst heuristic that
      already reaches Spearman 0.88 (readme.md 7.3, EDA 9). Any model must beat
      it, so it is useful to carry as a column.
    * ``recency_span`` = days_since_first - days_since_last -- the BTYD
      ``recency`` (readme.md 3.2). Non-negative once the domain rule has run.

    Both are floored so they survive the log step.
    """

    def __init__(self, floor: float = 1e-3):
        self.floor = floor

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.derived_: List[str] = []
        if {COUNT_COL, AOV_COL} <= set(X.columns):
            self.derived_.append("purchase_value")
        if {FIRST_PURCHASE_COL, LAST_PURCHASE_COL} <= set(X.columns):
            self.derived_.append("recency_span")
        self.feature_names_out_ = np.asarray(list(X.columns) + self.derived_, dtype=object)
        return self

    def transform(self, X) -> pd.DataFrame:
        X = _as_frame(X).copy()
        if "purchase_value" in self.derived_:
            X["purchase_value"] = (X[COUNT_COL] * X[AOV_COL]).clip(lower=self.floor)
        if "recency_span" in self.derived_:
            X["recency_span"] = (X[FIRST_PURCHASE_COL] - X[LAST_PURCHASE_COL]).clip(lower=self.floor)
        return X[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_


class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    """Remove features that carry no information in the training split.

    Polynomial expansion turns one dead column into many: an indicator that is
    all-zero in training (because ``--drop-invalid-recency`` removed the rows it
    marks, or because the outlier bounds are wide enough that nothing trips them)
    produces a constant term for every monomial it appears in -- 36 dead columns
    from one flag at degree 3, and 335 from five. They cannot help any estimator,
    they make coefficient tables unreadable, and a zero-variance column is a
    division by zero waiting for any scaler that is less careful than sklearn's.

    Fitted per split, so inside cross-validation each fold drops exactly the
    columns that are dead *in that fold*.
    """

    def __init__(self, enabled: bool = True, tol: float = 0.0):
        self.enabled = enabled
        self.tol = tol

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if not self.enabled:
            self.dropped_ = []
        else:
            std = X.std(ddof=0)
            self.dropped_ = [str(c) for c in std.index[std <= self.tol]]
            # Never hand back an empty matrix, however degenerate the input.
            if len(self.dropped_) == X.shape[1]:
                self.dropped_ = []
        self.feature_names_out_ = np.asarray(
            [c for c in X.columns if c not in set(self.dropped_)], dtype=object
        )
        if self.dropped_:
            logger.info(
                "dropped %d constant feature(s) out of %d, e.g. %s",
                len(self.dropped_), X.shape[1], self.dropped_[:3],
            )
        return self

    def transform(self, X) -> pd.DataFrame:
        return _as_frame(X)[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_


# --------------------------------------------------------------------------- #
# Target-side transform
# --------------------------------------------------------------------------- #


class LogTargetTransformer(BaseEstimator, TransformerMixin):
    """log(y) for fitting, with Duan's smearing estimator for the way back.

    EDA 3: the target has skew 6.4 and is near-symmetric in logs, so models are
    fitted on ``log(value)``. EDA 8: converting back with a plain ``exp`` is
    biased low, because ``exp(E[log y]) < E[y]`` whenever the residuals have any
    spread. Duan's smearing factor -- ``mean(exp(residual))`` on the training
    data -- corrects the level without touching the ranking.

    The factor cannot be known until a model has produced residuals, so the
    modelling step calls :meth:`fit_smearing` with its in-fold residuals. Until
    it does, the factor is 1.0 and :meth:`inverse_transform` is a plain ``exp``.
    """

    def __init__(self, smearing: bool = True):
        self.smearing = smearing

    def fit(self, y, X=None):
        y = np.asarray(y, dtype=float).ravel()
        if (y <= 0).any():
            raise ValueError(
                f"{int((y <= 0).sum())} non-positive target values: log(y) is undefined. "
                "Run filter_rows(drop_non_positive_target=True) first."
            )
        self.smearing_factor_ = 1.0
        self.n_ = len(y)
        return self

    def transform(self, y) -> np.ndarray:
        return np.log(np.asarray(y, dtype=float).ravel())

    def fit_smearing(self, residuals_log) -> "LogTargetTransformer":
        """Set the smearing factor from a fitted model's log-scale residuals."""
        residuals_log = np.asarray(residuals_log, dtype=float).ravel()
        self.smearing_factor_ = float(np.mean(np.exp(residuals_log))) if self.smearing else 1.0
        logger.info("Duan smearing factor = %.4f", self.smearing_factor_)
        return self

    def inverse_transform(self, y_log) -> np.ndarray:
        factor = getattr(self, "smearing_factor_", 1.0)
        return np.exp(np.asarray(y_log, dtype=float).ravel()) * factor


# --------------------------------------------------------------------------- #
# The facade
# --------------------------------------------------------------------------- #


class CLVPreprocessor(BaseEstimator, TransformerMixin):
    """The full feature pipeline as a single scikit-learn transformer.

    Step order, and why each step sits where it does::

        1. CategoricalEncoder     loyalty_program_membership -> loyalty (0/1)
        2. DomainRuleTransformer  impossible values fixed BEFORE any statistic is
                                  learned from them
        3. FrameImputer           fills learned on train only
        4. DerivedFeatures        optional; must precede the log step so the new
                                  columns are logged like the rest
        5. QuantileClipper        bounds learned on train; monotone, so clipping
                                  before or after the log is equivalent -- doing
                                  it before keeps the bounds readable in dollars
        6. SafeLogTransformer     the multiplicative -> additive step (EDA 7)
        7. PolynomialFeatures     degree 3: the curvature the tuner rediscovered
        8. StandardScaler         puts the polynomial terms on one scale, which is
                                  what makes Ridge's single alpha meaningful
        9. ConstantColumnDropper  prunes the dead columns the expansion can create

    Because it is a transformer, it composes::

        Pipeline([("prep", CLVPreprocessor(cfg)), ("model", Ridge(alpha=0.01))])

    and can therefore be cross-validated with the fills, bounds and scaler
    statistics refitted inside every fold -- the only way to get an honest score.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config

    # -- construction ------------------------------------------------------- #

    def _build(self, cfg: PreprocessConfig) -> Pipeline:
        numeric = list(cfg.numeric_features)
        binary = list(cfg.binary_features)
        steps: List[Tuple[str, object]] = [
            ("encode", CategoricalEncoder(mapping=cfg.categorical_map)),
            (
                "rules",
                DomainRuleTransformer(
                    numeric_features=numeric,
                    recency_policy=cfg.recency_policy,
                    positivity_floor=cfg.positivity_floor,
                    add_flags=cfg.add_quality_flags,
                ),
            ),
            (
                "impute",
                FrameImputer(
                    numeric_features=numeric,
                    binary_features=binary,
                    numeric_strategy=cfg.numeric_impute,
                    binary_strategy=cfg.binary_impute,
                    binary_fill_value=cfg.binary_fill_value,
                    knn_neighbors=cfg.knn_neighbors,
                    add_indicators=cfg.add_missing_indicators,
                ),
            ),
        ]

        log_columns = list(numeric)
        if cfg.add_derived_features:
            steps.append(("derive", DerivedFeatures(floor=cfg.positivity_floor)))
            log_columns += ["purchase_value", "recency_span"]

        steps.append(
            (
                "outliers",
                QuantileClipper(
                    columns=log_columns,
                    strategy=cfg.outlier_strategy,
                    quantiles=cfg.outlier_quantiles,
                    margin=cfg.outlier_margin,
                ),
            )
        )
        if cfg.log_transform:
            steps.append(
                ("log", SafeLogTransformer(columns=log_columns, floor=cfg.positivity_floor))
            )
        if cfg.poly_degree > 1:
            steps.append(
                ("poly", PolynomialFeatures(degree=cfg.poly_degree, include_bias=False))
            )
        if cfg.scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("prune", ConstantColumnDropper(enabled=cfg.drop_constant_features)))
        return Pipeline(steps)

    # -- sklearn API -------------------------------------------------------- #

    def fit(self, X, y=None) -> "CLVPreprocessor":
        cfg = self.config or PreprocessConfig()
        self.config_ = cfg
        X = _as_frame(X)
        self.input_features_ = list(X.columns)
        self.pipeline_ = self._build(cfg)
        self.pipeline_.set_output(transform="pandas")
        self.pipeline_.fit(X)
        self.feature_names_out_ = [str(c) for c in self.pipeline_[-1].get_feature_names_out()]
        self.n_features_out_ = len(self.feature_names_out_)
        logger.info(
            "preprocessor fitted -- %d raw columns -> %d model features (poly degree %d)",
            len(self.input_features_), self.n_features_out_, cfg.poly_degree,
        )
        return self

    def transform(self, X) -> pd.DataFrame:
        _check_fitted(self)
        X = _as_frame(X)
        missing = [c for c in self.input_features_ if c not in X.columns]
        if missing:
            raise ValueError(f"transform() is missing columns seen during fit: {missing}")
        out = self.pipeline_.transform(X[self.input_features_])
        return pd.DataFrame(np.asarray(out), columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        _check_fitted(self)
        return np.asarray(self.feature_names_out_, dtype=object)

    # -- convenience -------------------------------------------------------- #

    def quality_report(self) -> dict:
        """What the fitted pipeline learned about the data it was given."""
        _check_fitted(self)
        report: dict = {
            "n_input_features": len(self.input_features_),
            "n_output_features": self.n_features_out_,
            "poly_degree": self.config_.poly_degree,
        }
        named = dict(self.pipeline_.named_steps)
        if "rules" in named:
            report["domain_rules"] = named["rules"].rule_report_
        if "impute" in named:
            report["imputation_fills"] = named["impute"].fill_values_
            report["missing_indicators"] = list(named["impute"].indicator_cols_)
        if "outliers" in named:
            report["outlier_bounds"] = {k: list(v) for k, v in named["outliers"].bounds_.items()}
            report["outlier_raw_bounds"] = {
                k: list(v) for k, v in named["outliers"].raw_bounds_.items()
            }
            report["outlier_margin"] = named["outliers"].margin
            report["outlier_values_out_of_range_at_fit"] = named["outliers"].n_clipped_at_fit_
        if "encode" in named:
            report["unseen_categories"] = named["encode"].unseen_categories_
        if "prune" in named:
            report["dropped_constant_features"] = named["prune"].dropped_
        return report

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("preprocessor saved -> %s", path)
        return path

    @staticmethod
    def load(path: str | Path) -> "CLVPreprocessor":
        import joblib

        return joblib.load(Path(path))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _widen(lo: float, hi: float, margin: float) -> Tuple[float, float]:
    """Widen a bound pair so the clip only bites on genuinely out-of-range values.

    Multiplicative for positive columns (all five numerics here are strictly
    positive and span orders of magnitude, so a proportional margin is the
    natural one); additive on the observed span otherwise.
    """
    if margin <= 0:
        return lo, hi
    span = hi - lo
    lo_out = lo / (1.0 + margin) if lo > 0 else lo - margin * span
    hi_out = hi * (1.0 + margin) if hi > 0 else hi + margin * span
    return float(lo_out), float(hi_out)


def _invalid_recency_mask(X: pd.DataFrame) -> pd.Series:
    if {FIRST_PURCHASE_COL, LAST_PURCHASE_COL} <= set(X.columns):
        return X[LAST_PURCHASE_COL] > X[FIRST_PURCHASE_COL]
    return pd.Series(False, index=X.index)


def _check_fitted(obj) -> None:
    if not hasattr(obj, "pipeline_"):
        raise RuntimeError(f"{type(obj).__name__} is not fitted; call fit() first.")
