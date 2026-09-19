"""Feature engineering: create the model's features, select among them, prove they earn it.

``src/preprocessing.py`` ends with clean, trustworthy *columns*. This module turns
those columns into the *features* a model sees, and it owns three questions that
the cleaning step has no opinion about:

    1. which new columns to create, and why (the strategy below);
    2. which of the resulting features to keep (:class:`FeatureSelector`);
    3. whether either of those actually helped (:func:`compare_feature_sets`,
       :func:`feature_report`, :func:`check_feature_engineering`).

The strategy, and the one fact that decides it
----------------------------------------------
EDA 7 established that the target is **multiplicative** in the six raw columns::

    value = exp(b0) * count^0.29 * aov^0.86 * last^-0.32 * first^-0.015 * ... * eps

which is why the pipeline logs everything: a product becomes a sum, and a linear
model on logged features reaches CV R2 0.9994. That single fact settles most
feature-engineering questions before any are asked, because **in log space a
product or ratio of existing columns is a linear combination of their logs**::

    log(count * aov)   = log(count) + log(aov)
    log(count / first) = log(count) - log(first)

A linear model cannot gain anything from a column it can already form as a
weighted sum of columns it has -- and a degree-3 polynomial expansion of the
logged inputs already contains every product of up to three of them. So the
usual RFM feature-engineering reflexes (spend per order, orders per day, value
per day of tenure) are, for this representation, **exactly redundant by
construction**. They are not useless: a tree has to approximate a product with
axis-aligned splits, so the same column can help a Random Forest measurably.

That divides the catalogue in two, and :data:`FEATURE_SPECS` records which side
each feature falls on in its ``new_in_log_space`` field:

* **products and ratios** (``purchase_value``, ``purchase_rate``,
  ``inter_purchase_days``, ``value_per_day``, ``recency_ratio``, ``dormancy``) --
  redundant for the linear family, potentially useful for trees;
* **differences and thresholds** (``recency_span``, ``is_lapsed``) -- *not*
  expressible as a linear combination of logs, so these are the only two that can
  add information to the model that actually won (EDA 10).

``recency_span = first - last`` is the BTYD ``recency`` (readme.md 3.2) and
``is_lapsed`` asks whether a customer has been silent for longer than twice their
own average inter-purchase interval -- the classic non-contractual churn signal,
and a threshold, so it survives the log.

One exception, found by the checks rather than by reasoning: the redundancy is
exact only above the positivity floor. ``purchase_rate`` falls below 1e-3 for 7
of the 800 training customers (a single order against four years of tenure), and
a floored value is no longer a linear combination of anything. So those rows --
and only those -- carry information the logged inputs do not.

The measured version of that argument is :func:`compare_feature_sets`, which
``python main.py --compare-features`` prints. Mean 5-fold CV R2 on log(value):

    feature set             features   Ridge (log, deg 3)   Random Forest (raw)
    -------------------------------------------------------------------------
    none (the 6 raw)             119              0.99936               0.95394
    default (2 derived)          219              0.99928               0.95710
    independent only             209              0.99929               0.95357
    all (8 derived)              799              0.99963               0.96525

Both predictions hold. Ridge does not move: three feature sets, four decimal
places, no ordering worth defending -- exactly what "redundant by construction"
looks like. The Random Forest gains **+0.011 R2** from the full set, a quarter of
its remaining error, and gains it from the *products and ratios*: the
independent-only row is the one set that leaves it where it started. The features
a linear model cannot use are precisely the ones a tree cannot build.

Selection
---------
Degree 3 on seven first-order terms is 119 features from 800 training rows, and
the polynomial basis is collinear by construction. :class:`FeatureSelector`
offers the usual strategies, with the default (``variance``) doing exactly what
the old ``ConstantColumnDropper`` did: remove columns that carry no information
in *this* fold. Everything stronger is available and off by default, because on
this dataset Ridge on the full basis is already at the ceiling (readme.md 16.2)
and selection can only cost accuracy.

**VIF is available and is the wrong tool here.** A polynomial basis is designed to
be collinear -- ``log_x`` and ``log_x^2`` have a VIF in the hundreds and both are
needed -- so dropping terms by VIF prunes the representation rather than cleaning
it. It is implemented because it is the standard answer to multicollinearity and
because its cost should be measurable rather than argued about. Ridge(0.01) on
the degree-3 basis, mean 5-fold CV R2 on log(value):

    strategy                 features    CV R2      reading
    ----------------------------------------------------------------------------
    none                          119    0.99936    the reference
    variance (default)            119    0.99936    nothing is constant here
    correlation (>= 0.999)        103    0.99936    16 near-duplicates, free to drop
    model (Ridge |coef|, k=40)     40    0.99919    a third of the basis, small cost
    mutual_info (k=40)             40    0.99780    the largest cost of the four
    vif (> 6.0)                    69    0.99917    drops 50 terms, buys nothing

Nothing here beats keeping everything, which is the honest answer on a dataset
whose representation is already at its ceiling. ``correlation`` is the one
strategy that is free, and it is the one to reach for first on a wider dataset.

Leakage
-------
Every transformer here is fitted, so it refits inside each cross-validation fold
along with the rest of the pipeline. That matters most for the supervised
strategies (``mutual_info``, ``model``): choosing features on the full dataset and
*then* cross-validating is one of the most common ways to publish an inflated
score, and it is structurally impossible here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocessing import (
    AOV_COL,
    COUNT_COL,
    FIRST_PURCHASE_COL,
    LAST_PURCHASE_COL,
    SKEWED_NUMERIC,
    Check,
    _as_frame,
    _Recorder,
    quiet,
)

logger = logging.getLogger("capstone.feature_engineering")

__all__ = [
    "FeatureSpec",
    "FEATURE_SPECS",
    "DEFAULT_DERIVED",
    "INDEPENDENT_DERIVED",
    "ALL_DERIVED",
    "resolve_derived",
    "DerivedFeatures",
    "SafeLogTransformer",
    "FeatureSelector",
    "SELECTION_STRATEGIES",
    "build_feature_steps",
    "feature_report",
    "compare_feature_sets",
    "check_feature_engineering",
]


# --------------------------------------------------------------------------- #
# 1. The catalogue -- what may be created, and what each column is for
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureSpec:
    """One engineered feature: how to build it, and what it is worth knowing about.

    ``new_in_log_space`` is the field that matters. ``False`` means the feature is
    a product or ratio of existing columns, so after the log step it is a linear
    combination of columns the model already has -- redundant for a linear model,
    possibly useful for a tree. ``True`` means it is a difference or a threshold,
    which no linear combination of logs can reproduce.
    """

    name: str
    inputs: Tuple[str, ...]
    formula: str
    kind: str                      # product | ratio | difference | threshold
    new_in_log_space: bool
    rationale: str
    build: Callable[[pd.DataFrame], pd.Series]


# Builders are module-level functions, not lambdas: a fitted pipeline is pickled
# to outputs/preprocessor.joblib, and a lambda would make that fail.

def _purchase_value(X: pd.DataFrame) -> pd.Series:
    return X[COUNT_COL] * X[AOV_COL]


def _purchase_rate(X: pd.DataFrame) -> pd.Series:
    return X[COUNT_COL] / X[FIRST_PURCHASE_COL]


def _inter_purchase_days(X: pd.DataFrame) -> pd.Series:
    return X[FIRST_PURCHASE_COL] / X[COUNT_COL]


def _value_per_day(X: pd.DataFrame) -> pd.Series:
    return (X[COUNT_COL] * X[AOV_COL]) / X[FIRST_PURCHASE_COL]


def _recency_ratio(X: pd.DataFrame) -> pd.Series:
    return X[LAST_PURCHASE_COL] / X[FIRST_PURCHASE_COL]


def _dormancy(X: pd.DataFrame) -> pd.Series:
    # Days since the last order, measured in the customer's own average
    # inter-purchase intervals: last / (first / count).
    return X[LAST_PURCHASE_COL] * X[COUNT_COL] / X[FIRST_PURCHASE_COL]


def _recency_span(X: pd.DataFrame) -> pd.Series:
    return X[FIRST_PURCHASE_COL] - X[LAST_PURCHASE_COL]


def _is_lapsed(X: pd.DataFrame) -> pd.Series:
    """Silent for longer than twice this customer's own purchase interval."""
    interval = X[FIRST_PURCHASE_COL] / X[COUNT_COL]
    return (X[LAST_PURCHASE_COL] > 2.0 * interval).astype(float)


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    spec.name: spec for spec in (
        FeatureSpec(
            "purchase_value", (COUNT_COL, AOV_COL), "count x aov", "product", False,
            "The analyst heuristic every model must beat: Spearman 0.88 on its own "
            "(readme.md 7.3, EDA 9).", _purchase_value),
        FeatureSpec(
            "purchase_rate", (COUNT_COL, FIRST_PURCHASE_COL), "count / tenure", "ratio", False,
            "Purchase frequency normalised by how long the customer has existed -- "
            "the F of RFM, per day.", _purchase_rate),
        FeatureSpec(
            "inter_purchase_days", (FIRST_PURCHASE_COL, COUNT_COL), "tenure / count",
            "ratio", False,
            "Average days between orders; the denominator of every churn heuristic.",
            _inter_purchase_days),
        FeatureSpec(
            "value_per_day", (COUNT_COL, AOV_COL, FIRST_PURCHASE_COL),
            "count x aov / tenure", "ratio", False,
            "Spend rate: separates a big spender from an old customer with the same total.",
            _value_per_day),
        FeatureSpec(
            "recency_ratio", (LAST_PURCHASE_COL, FIRST_PURCHASE_COL), "last / tenure",
            "ratio", False,
            "What fraction of the customer's life has passed since their last order.",
            _recency_ratio),
        FeatureSpec(
            "dormancy", (LAST_PURCHASE_COL, COUNT_COL, FIRST_PURCHASE_COL),
            "last / (tenure / count)", "ratio", False,
            "Silence measured in the customer's own purchase intervals -- the "
            "non-contractual churn signal of readme.md 3.", _dormancy),
        FeatureSpec(
            "recency_span", (FIRST_PURCHASE_COL, LAST_PURCHASE_COL), "tenure - last",
            "difference", True,
            "The BTYD recency (readme.md 3.2): a difference, so no combination of "
            "logs can reproduce it.", _recency_span),
        FeatureSpec(
            "is_lapsed", (LAST_PURCHASE_COL, FIRST_PURCHASE_COL, COUNT_COL),
            "last > 2 x (tenure / count)", "threshold", True,
            "Has the customer been silent for longer than twice their own interval? "
            "A threshold, so it is new information in log space.", _is_lapsed),
    )
}

#: What ``--derived-features`` switches on by default: the two the EDA named.
DEFAULT_DERIVED: Tuple[str, ...] = ("purchase_value", "recency_span")

#: The only two that are not linear combinations of the logged inputs.
INDEPENDENT_DERIVED: Tuple[str, ...] = tuple(
    name for name, spec in FEATURE_SPECS.items() if spec.new_in_log_space)

ALL_DERIVED: Tuple[str, ...] = tuple(FEATURE_SPECS)

_NAMED_SETS = {
    "none": (), "default": DEFAULT_DERIVED,
    "independent": INDEPENDENT_DERIVED, "all": ALL_DERIVED,
}


def resolve_derived(selection: str | Sequence[str] | None) -> Tuple[str, ...]:
    """``"all"``, ``"independent"``, ``"a,b"`` or a list -> validated feature names."""
    if selection is None:
        return ()
    if isinstance(selection, str):
        if selection in _NAMED_SETS:
            return _NAMED_SETS[selection]
        selection = [part.strip() for part in selection.split(",") if part.strip()]
    unknown = [name for name in selection if name not in FEATURE_SPECS]
    if unknown:
        raise ValueError(f"unknown derived feature(s) {unknown}; "
                         f"available: {sorted(FEATURE_SPECS)}")
    return tuple(selection)


# --------------------------------------------------------------------------- #
# 2. Creation
# --------------------------------------------------------------------------- #


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Append engineered columns, before the log step so they are logged too.

    Which columns is a configuration choice (:func:`resolve_derived`); what each
    one means is in :data:`FEATURE_SPECS`. A feature whose inputs are missing from
    the frame is skipped rather than failing, so a reduced feature set -- the
    drop-column importance in ``src/evaluation.py`` builds several -- still runs.

    Everything except the binary flags is floored at ``floor`` so it survives
    ``log()``: ``recency_span`` is negative for the 52 rows whose last purchase
    precedes their first (EDA 2), and the domain rule has deliberately *not*
    repaired those values.
    """

    def __init__(self, features: Sequence[str] = DEFAULT_DERIVED, floor: float = 1e-3):
        self.features = features
        self.floor = floor

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        available = set(X.columns)
        # Only names are stored, never the spec objects: the fitted pipeline is
        # pickled, and a stored callable would tie the artifact to this module.
        self.derived_: List[str] = [
            name for name in resolve_derived(self.features)
            if set(FEATURE_SPECS[name].inputs) <= available and name not in available
        ]
        skipped = [n for n in resolve_derived(self.features) if n not in self.derived_]
        if skipped:
            logger.info("derived features skipped (inputs unavailable): %s", skipped)
        self.feature_names_out_ = np.asarray(list(X.columns) + self.derived_, dtype=object)
        return self

    def transform(self, X) -> pd.DataFrame:
        X = _as_frame(X).copy()
        for name in self.derived_:
            spec = FEATURE_SPECS[name]
            values = spec.build(X)
            X[name] = values if spec.kind == "threshold" else values.clip(lower=self.floor)
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

    "Safe" means the floor from ``DomainRuleTransformer`` is re-applied here, so a
    zero or negative arriving in new data becomes a small positive number instead
    of ``-inf``.
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


# --------------------------------------------------------------------------- #
# 3. Selection
# --------------------------------------------------------------------------- #

SELECTION_STRATEGIES = ("none", "variance", "correlation", "mutual_info", "model", "vif")


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Keep a subset of the expanded features, chosen on the training fold only.

    ==============  ==========================================================
    strategy        what it removes
    ==============  ==========================================================
    ``none``        nothing
    ``variance``    columns whose training standard deviation is <= ``tol``.
                    The default, and with ``tol=0`` it is exactly the old
                    ``ConstantColumnDropper``: polynomial expansion turns one
                    dead indicator into 36 dead columns at degree 3.
    ``correlation`` one of every pair correlated above ``correlation_threshold``
                    -- near-duplicates, not merely collinear terms.
    ``mutual_info`` everything outside the top ``k`` by mutual information with
                    the target. Supervised, so it needs ``y`` at fit time.
    ``model``       everything outside the top ``k`` by ``|coefficient|`` of a
                    Ridge fitted on the standardised features. Supervised.
    ``vif``         iteratively the highest variance-inflation factor while any
                    exceeds ``vif_threshold``. Documented in this module's
                    docstring as the wrong tool for a polynomial basis; here so
                    that claim can be measured instead of asserted.
    ==============  ==========================================================

    A supervised strategy with no ``y`` (``prep.fit(X)`` with nothing to learn
    from) falls back to ``variance`` and says so, rather than silently keeping
    everything.
    """

    def __init__(
        self,
        strategy: str = "variance",
        k: Optional[int] = None,
        tol: float = 0.0,
        correlation_threshold: float = 0.999,
        vif_threshold: float = 6.0,
        max_vif_drops: int = 50,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.k = k
        self.tol = tol
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.max_vif_drops = max_vif_drops
        self.random_state = random_state

    # -- fitting ------------------------------------------------------------ #

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if self.strategy not in SELECTION_STRATEGIES:
            raise ValueError(f"strategy must be one of {SELECTION_STRATEGIES}, "
                             f"got {self.strategy!r}")

        strategy = self.strategy
        if strategy in ("mutual_info", "model") and y is None:
            logger.warning("%s selection needs y at fit time; falling back to variance",
                           strategy)
            strategy = "variance"
        self.strategy_ = strategy

        dropped = {
            "none": lambda: [],
            "variance": lambda: self._by_variance(X),
            "correlation": lambda: self._by_correlation(X),
            "mutual_info": lambda: self._by_score(X, self._mutual_info(X, y)),
            "model": lambda: self._by_score(X, self._ridge_importance(X, y)),
            "vif": lambda: self._by_vif(X),
        }[strategy]()

        # Never hand back an empty matrix, however degenerate the input.
        if len(dropped) == X.shape[1]:
            logger.warning("%s selection would drop every column; keeping all", strategy)
            dropped = []

        self.dropped_ = [str(c) for c in dropped]
        self.feature_names_out_ = np.asarray(
            [c for c in X.columns if c not in set(self.dropped_)], dtype=object)
        if self.dropped_:
            logger.info("%s selection: kept %d of %d features (dropped e.g. %s)",
                        strategy, len(self.feature_names_out_), X.shape[1], self.dropped_[:3])
        return self

    def transform(self, X) -> pd.DataFrame:
        return _as_frame(X)[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_

    def selection_report(self) -> dict:
        """What this selector did, for the run report."""
        return {
            "strategy": getattr(self, "strategy_", self.strategy),
            "requested_strategy": self.strategy,
            "n_in": int(len(getattr(self, "feature_names_in_", []))),
            "n_out": int(len(getattr(self, "feature_names_out_", []))),
            "n_dropped": int(len(getattr(self, "dropped_", []))),
            "dropped": list(getattr(self, "dropped_", []))[:20],
        }

    # -- strategies --------------------------------------------------------- #

    def _by_variance(self, X: pd.DataFrame) -> List[str]:
        std = X.std(ddof=0)
        return [str(c) for c in std.index[std <= self.tol]]

    def _by_correlation(self, X: pd.DataFrame) -> List[str]:
        keep_alive = self._by_variance(X)                       # constants first
        candidates = [c for c in X.columns if c not in set(keep_alive)]
        corr = X[candidates].corr().abs().to_numpy()
        dropped = set(keep_alive)
        for i in range(len(candidates)):
            if candidates[i] in dropped:
                continue
            for j in range(i + 1, len(candidates)):
                if candidates[j] in dropped:
                    continue
                if corr[i, j] >= self.correlation_threshold:
                    dropped.add(candidates[j])
        return sorted(dropped, key=list(X.columns).index)

    def _by_score(self, X: pd.DataFrame, scores: pd.Series) -> List[str]:
        k = self.k or max(1, X.shape[1] // 2)
        keep = set(scores.sort_values(ascending=False).head(k).index)
        return [str(c) for c in X.columns if c not in keep]

    def _mutual_info(self, X: pd.DataFrame, y) -> pd.Series:
        from sklearn.feature_selection import mutual_info_regression

        values = mutual_info_regression(X, np.asarray(y).ravel(),
                                        random_state=self.random_state)
        return pd.Series(values, index=X.columns)

    def _ridge_importance(self, X: pd.DataFrame, y) -> pd.Series:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(X)
        model = Ridge(alpha=1.0).fit(scaled, np.asarray(y).ravel())
        return pd.Series(np.abs(model.coef_).ravel(), index=X.columns)

    def _by_vif(self, X: pd.DataFrame) -> List[str]:
        dropped = list(self._by_variance(X))
        keep = [c for c in X.columns if c not in set(dropped)]
        for _ in range(self.max_vif_drops):
            vif = _vif_series(X[keep])
            worst = vif.idxmax()
            if vif[worst] <= self.vif_threshold:
                break
            dropped.append(str(worst))
            keep.remove(worst)
            if len(keep) <= 2:
                break
        return dropped


def _vif_series(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factors from the inverse correlation matrix.

    ``VIF_j = 1 / (1 - R2_j)`` is the j-th diagonal entry of the inverse
    correlation matrix, which is one pseudo-inverse instead of one regression per
    column -- the difference between seconds and minutes at 119 features.

    A *perfectly* collinear column needs care: the matrix is then singular, and
    ``pinv`` quietly returns a small finite diagonal entry where the true VIF is
    infinite -- so the column that most deserves dropping would look like the
    safest one. Rank-deficient columns are therefore identified with a
    pivoted QR and reported as ``inf`` instead.
    """
    values = X.to_numpy(dtype=float)
    corr = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
    corr = np.atleast_2d(corr)
    np.fill_diagonal(corr, 1.0)
    vif = np.abs(np.diag(np.linalg.pinv(corr)))

    rank = int(np.linalg.matrix_rank(corr))
    if rank < corr.shape[0]:
        from scipy.linalg import qr

        centred = values - values.mean(axis=0, keepdims=True)
        _, _, pivots = qr(centred, mode="economic", pivoting=True)
        vif[list(pivots[rank:])] = np.inf
    return pd.Series(vif, index=X.columns)


# --------------------------------------------------------------------------- #
# 4. The steps CLVPreprocessor plugs in
# --------------------------------------------------------------------------- #


def build_feature_steps(cfg) -> List[Tuple[str, object]]:
    """The feature half of the pipeline: derive -> log -> expand -> scale -> select.

    Called by ``CLVPreprocessor._build``; ``cfg`` is a
    :class:`~src.preprocessing.PreprocessConfig`. Ordering is not arbitrary:

    * derived columns come **before** the log so they are logged like the rest,
      which is what makes a product become a sum;
    * the polynomial expansion comes after the log, so its terms are products of
      logs (EDA 7's curvature) rather than products of raw dollars;
    * scaling comes after the expansion, so Ridge's single alpha means the same
      thing for every term;
    * selection comes last, when the features it chooses between are the ones the
      model will actually see.
    """
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    steps: List[Tuple[str, object]] = []
    log_columns = list(cfg.numeric_features)

    derived = resolve_derived(cfg.derived_features) if cfg.add_derived_features else ()
    if derived:
        steps.append(("derive", DerivedFeatures(features=derived, floor=cfg.positivity_floor)))
        # Thresholds are already 0/1 and must not be logged.
        log_columns += [name for name in derived
                        if FEATURE_SPECS[name].kind != "threshold"]

    if cfg.log_transform:
        steps.append(("log", SafeLogTransformer(columns=log_columns, floor=cfg.positivity_floor)))
    if cfg.poly_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=cfg.poly_degree, include_bias=False)))
    if cfg.scale:
        steps.append(("scale", StandardScaler()))

    strategy = cfg.feature_selection if cfg.drop_constant_features else "none"
    steps.append(("select", FeatureSelector(
        strategy=strategy,
        k=cfg.select_k,
        correlation_threshold=cfg.correlation_threshold,
        vif_threshold=cfg.vif_threshold,
        random_state=cfg.random_state,
    )))
    return steps


# --------------------------------------------------------------------------- #
# 5. Validation -- did any of this help?
# --------------------------------------------------------------------------- #


def feature_report(X: pd.DataFrame, y=None, top: Optional[int] = None) -> pd.DataFrame:
    """Per-feature diagnostics on the processed matrix.

    ``spearman`` is against the target and is the only column that says anything
    about usefulness; ``vif`` is included for completeness and should be read with
    this module's docstring in mind -- on a polynomial basis a high VIF is the
    design, not a defect.
    """
    X = _as_frame(X)
    report = pd.DataFrame(index=X.columns)
    report["std"] = X.std(ddof=0).to_numpy()
    report["share_zero"] = (X == 0).mean().to_numpy()

    if y is not None:
        from scipy import stats

        y = np.asarray(y, dtype=float).ravel()
        report["spearman_vs_target"] = [
            float(stats.spearmanr(X[c].to_numpy(), y)[0]) if X[c].std(ddof=0) > 0 else np.nan
            for c in X.columns
        ]
        report["abs_spearman"] = report["spearman_vs_target"].abs()

    report["vif"] = _vif_series(X).to_numpy()
    sort_key = "abs_spearman" if "abs_spearman" in report else "std"
    report = report.sort_values(sort_key, ascending=False)
    return report.head(top) if top else report


def compare_feature_sets(
    train_raw: pd.DataFrame,
    config,
    cv_folds: int = 5,
    variants: Optional[Sequence[Tuple[str, dict]]] = None,
) -> pd.DataFrame:
    """Measure the strategy: does each feature set help the linear model, or the trees?

    Cross-validated R2 on ``log(value)`` for each feature set under two
    representations -- Ridge on the logged, degree-3 basis (the model that won)
    and a Random Forest on the raw columns (the model that did not). The
    prediction this module's docstring makes is that products and ratios move the
    second column and not the first.
    """
    from dataclasses import replace

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import Pipeline

    from .preprocessing import CLVPreprocessor          # local: avoids an import cycle

    X = train_raw.drop(columns=[config.target])
    y_log = np.log(train_raw[config.target].to_numpy())
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=config.random_state)

    variants = variants or (
        ("none (6 raw columns)", {"add_derived_features": False}),
        ("default (2 derived)", {"add_derived_features": True,
                                 "derived_features": DEFAULT_DERIVED}),
        ("independent only", {"add_derived_features": True,
                              "derived_features": INDEPENDENT_DERIVED}),
        ("all (8 derived)", {"add_derived_features": True,
                             "derived_features": ALL_DERIVED}),
    )

    families = {
        "ridge_log_poly3": ({"log_transform": True, "poly_degree": 3, "scale": True},
                            Ridge(alpha=0.01)),
        "random_forest_raw": ({"log_transform": False, "poly_degree": 1, "scale": False},
                              RandomForestRegressor(n_estimators=200,
                                                    random_state=config.random_state,
                                                    n_jobs=-1)),
    }

    rows = []
    for label, overrides in variants:
        row = {"feature_set": label}
        for family, (shape, estimator) in families.items():
            cfg = replace(config, **{**overrides, **shape})
            pipe = Pipeline([("prep", CLVPreprocessor(cfg)), ("model", estimator)])
            with quiet(logging.ERROR):
                score = cross_validate(pipe, X, y_log, cv=cv, scoring="r2")["test_score"].mean()
            row[family] = float(score)
            if family == "ridge_log_poly3":
                with quiet(logging.ERROR):
                    row["n_features"] = int(CLVPreprocessor(cfg).fit(X).n_features_out_)
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("feature_set")
    return frame[["n_features", "ridge_log_poly3", "random_forest_raw"]]


def check_feature_engineering(
    prep,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_log=None,
    sample: int = 25,
) -> List[Check]:
    """The checks ``main.py`` runs on the created and selected features.

    The interesting one is the collinearity check: it does not trust
    :data:`FEATURE_SPECS`, it *measures* whether each product/ratio feature really
    is a linear combination of the logged inputs, by comparing matrix ranks. If a
    feature the catalogue calls redundant turned out to add rank, the catalogue
    would be wrong -- and so would the strategy built on it.
    """
    rec = _Recorder("feature engineering")
    config = prep.config_
    named = dict(prep.pipeline_.named_steps)
    rows = X_test_raw.head(sample)
    requested = resolve_derived(config.derived_features) if config.add_derived_features else ()

    # 1. Creation ----------------------------------------------------------- #
    if not requested:
        rec.skip("derived features created", "no derived features requested")
        rec.skip("derived features are finite and positive", "no derived features requested")
    else:
        derive = named.get("derive")
        created = list(getattr(derive, "derived_", []))
        rec.record("derived features created", set(created) == set(requested),
                   f"{created}")

        def created_values():
            frame = rows.copy()
            frame = _apply_until(prep, "derive", frame)
            values = frame[created]
            positive = [c for c in created
                        if FEATURE_SPECS[c].kind != "threshold"]
            ok = bool(np.isfinite(values.to_numpy()).all()
                      and (values[positive] > 0).to_numpy().all())
            return ok, f"{len(created)} columns, min {values.to_numpy().min():.4g}"

        rec.guard("derived features are finite and positive", created_values)

    # 2. The claim the whole strategy rests on ------------------------------ #
    def collinearity():
        if not requested or not config.log_transform:
            raise _SkipCheck("needs derived features and the log transform")
        redundant = [n for n in requested if not FEATURE_SPECS[n].new_in_log_space]
        independent = [n for n in requested if FEATURE_SPECS[n].new_in_log_space]

        # The rank argument is about the *unfloored* values: a ratio that falls
        # below the positivity floor is clipped, and a clipped column is no longer
        # a linear combination of anything. Those rows are excluded here and
        # counted in the detail, because the exception is a property of the floor,
        # not of the catalogue.
        raw = _apply_until(prep, "impute", rows.copy())
        floor = config.positivity_floor
        built = {name: FEATURE_SPECS[name].build(raw) for name in requested}
        unfloored = np.logical_and.reduce(
            [(values > floor).to_numpy() for name, values in built.items()
             if FEATURE_SPECS[name].kind != "threshold"]
            or [np.ones(len(raw), dtype=bool)])
        n_floored = int((~unfloored).sum())
        if unfloored.sum() < len(raw) // 2:
            raise _SkipCheck(f"{n_floored} of {len(raw)} sample rows hit the floor")

        logs = np.log(raw[list(config.numeric_features)].to_numpy()[unfloored])
        base_rank = np.linalg.matrix_rank(logs)
        verdicts = []
        for name in requested:
            spec = FEATURE_SPECS[name]
            values = built[name].to_numpy()[unfloored]
            column = values if spec.kind == "threshold" else np.log(values)
            with_extra = np.linalg.matrix_rank(np.column_stack([logs, column]))
            verdicts.append(with_extra == base_rank + (1 if spec.new_in_log_space else 0))

        detail = (f"{len(redundant)} product/ratio feature(s) add no rank, "
                  f"{len(independent)} difference/threshold feature(s) do")
        if n_floored:
            detail += f" ({n_floored} floored row(s) excluded)"
        return all(verdicts), detail

    _guard_skippable(rec, "the catalogue's log-space classification is correct", collinearity)

    # 3. Selection ---------------------------------------------------------- #
    selector = named.get("select")
    if selector is None:
        rec.skip("selection is a subset of its input", "no selection step")
    else:
        report = selector.selection_report()
        rec.record("selection is a subset of its input",
                   set(selector.feature_names_out_) <= set(selector.feature_names_in_),
                   f"{report['strategy']}: kept {report['n_out']} of {report['n_in']}")

        def deterministic():
            from sklearn.base import clone as _clone

            # include=False: the selector must be refitted on what it was *given*,
            # not on what it returned -- feeding it its own output would hand it an
            # already-selected frame and fail for every strategy that drops anything.
            frame = _apply_until(prep, "select", X_train_raw.copy(), include=False)
            twin = _clone(selector)
            with quiet(logging.ERROR):
                twin.fit(frame, np.asarray(y_train_log) if y_train_log is not None else None)
            return list(twin.feature_names_out_) == list(selector.feature_names_out_), \
                "refitting on the same rows selects the same columns"

        rec.guard("selection is deterministic", deterministic)

        rec.record("selection applies unchanged to unseen rows",
                   list(prep.transform(rows).columns) == list(selector.feature_names_out_),
                   f"{len(selector.feature_names_out_)} columns on test rows")

    # 4. Width contract ----------------------------------------------------- #
    rec.record("output width matches the fitted feature names",
               prep.transform(rows).shape[1] == prep.n_features_out_,
               f"{prep.n_features_out_} features")
    return rec.checks


class _SkipCheck(Exception):
    """Raised inside a check body when the check does not apply to this config."""


def _guard_skippable(rec: _Recorder, name: str, fn) -> None:
    try:
        ok, detail = fn()
    except _SkipCheck as reason:
        rec.skip(name, str(reason))
    except Exception as exc:                          # noqa: BLE001 - reported, not swallowed
        rec.checks.append(Check(name, "FAIL", f"raised {type(exc).__name__}: {exc}", rec.group))
    else:
        rec.record(name, ok, detail)


def _apply_until(prep, step_name: str, frame: pd.DataFrame,
                 include: bool = True) -> pd.DataFrame:
    """Run the fitted pipeline up to one named step.

    Checks need to look at the features *between* steps -- after ``derive``, after
    ``log``, or (with ``include=False``) at what a step was handed, which is what
    refitting that step on the same input requires.
    """
    names = [name for name, _ in prep.pipeline_.steps]
    if step_name not in names:
        raise _SkipCheck(f"no {step_name!r} step in this configuration")
    index = names.index(step_name) + (1 if include else 0)
    with quiet(logging.ERROR):
        return _as_frame(prep.pipeline_[:index].transform(frame[prep.input_features_]))
