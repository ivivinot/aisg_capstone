"""The scorecard every model in this project is judged by (eda.ipynb 8).

`readme.md` 7.2 sets out the metric set the CLV literature actually uses, and the
order matters: marketing budget is allocated by *ordering* customers, so rank
metrics lead and error metrics follow.

    Spearman rho          are customers ranked in the right order?
    normalized Gini       how much of the perfect ordering is captured? (1 = perfect)
    top-decile capture    share of real value inside the model's predicted top 10%
    decile MAPE           are predicted *levels* right, decile by decile? (calibration)
    MAE / R2 (raw)        error in dollars, for a reader who wants one number
    R2 / RMSE (log)       variance explained where the model is actually fitted

Two of these need care and are therefore implemented here rather than taken from
scikit-learn:

* **Normalized Gini** is the model's Gini divided by the Gini of a perfect
  ordering, so 1.0 means "ranked exactly right" whatever the value distribution.
* **Decile MAPE** compares *decile means*, never individual customers. Plain MAPE
  explodes on near-zero values (`readme.md` 8); the decile version is what
  Wang et al. report.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "normalized_gini",
    "decile_mape",
    "top_decile_capture",
    "scorecard",
    "leaderboard",
    "value_tiers",
    "SCORE_COLUMNS",
    "TIER_EDGES",
    "TIER_LABELS",
]

#: Column order used for every printed table, ranking metrics first.
SCORE_COLUMNS: Sequence[str] = (
    "spearman", "norm_gini", "top_decile_capture", "decile_mape",
    "mae_raw", "r2_raw", "r2_log", "rmse_log",
)

#: The tiering the business actually acts on (eda.ipynb 13).
TIER_EDGES: Sequence[float] = (0, 0.5, 0.8, 0.95, 1.0)
TIER_LABELS: Sequence[str] = (
    "Standard (bottom 50%)", "Growth (50-80%)", "High value (80-95%)", "VIP (top 5%)",
)


def _spearman(y_pred, y_true) -> float:
    """Rank correlation, or NaN for a constant prediction.

    The mean baseline predicts one number for everyone, so it has no ordering to
    correlate: that is a property of the baseline, not an error, and scipy's
    warning about it is suppressed rather than printed once per fold.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if np.all(y_pred == y_pred[0]):
        return float("nan")
    return float(stats.spearmanr(y_pred, y_true)[0])


def normalized_gini(y_true, y_pred) -> float:
    """Gini of the predicted ordering divided by the Gini of the perfect ordering."""
    y_true = np.asarray(y_true, dtype=float)

    def gini(truth: np.ndarray, scores) -> float:
        order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
        cumulative = np.cumsum(truth[order]) / truth.sum()
        return float(cumulative.sum() / len(truth) - (len(truth) + 1) / (2 * len(truth)))

    return gini(y_true, y_pred) / gini(y_true, y_true)


def decile_mape(y_true, y_pred, n_deciles: int = 10) -> float:
    """Mean absolute percentage error of decile means -- calibration, not point accuracy.

    Customers are sorted by *prediction* into deciles; within each, the mean
    prediction is compared with the mean actual value. This is the calibration
    measure of `readme.md` 7.2, and it is well defined even when individual
    values are small.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float),
                          "p": np.asarray(y_pred, dtype=float)})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), n_deciles, labels=False)
    summary = frame.groupby("decile")[["y", "p"]].mean()
    return float(np.mean(np.abs(summary["p"] - summary["y"]) / summary["y"]))


def top_decile_capture(y_true, y_pred) -> float:
    """Share of total actual value held by the customers the model ranks in its top 10%.

    The number a CRM team acts on: if the budget only reaches the top decile,
    this is how much of the value it reaches. Its ceiling is the concentration of
    the data itself -- 0.381 on this file (eda.ipynb 3), not 1.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    k = max(1, len(y_true) // 10)
    top = np.argsort(-np.asarray(y_pred, dtype=float), kind="mergesort")[:k]
    return float(y_true[top].sum() / y_true.sum())


def scorecard(
    name: str,
    y_true_raw,
    y_pred_raw,
    y_true_log=None,
    y_pred_log=None,
) -> dict:
    """One row of the leaderboard: every metric for one set of predictions.

    ``y_*_raw`` are dollars, ``y_*_log`` the log-scale values the model was
    actually fitted on. The log columns are optional so that baselines which have
    no log-space prediction can still be scored.
    """
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    row = {
        "model": name,
        "spearman": _spearman(y_pred_raw, y_true_raw),
        "norm_gini": normalized_gini(y_true_raw, y_pred_raw),
        "top_decile_capture": top_decile_capture(y_true_raw, y_pred_raw),
        "decile_mape": decile_mape(y_true_raw, y_pred_raw),
        "mae_raw": float(mean_absolute_error(y_true_raw, y_pred_raw)),
        "r2_raw": float(r2_score(y_true_raw, y_pred_raw)),
    }
    if y_true_log is not None and y_pred_log is not None:
        y_true_log = np.asarray(y_true_log, dtype=float)
        y_pred_log = np.asarray(y_pred_log, dtype=float)
        row["r2_log"] = float(r2_score(y_true_log, y_pred_log))
        row["rmse_log"] = float(mean_squared_error(y_true_log, y_pred_log) ** 0.5)
    return row


def leaderboard(rows: Sequence[dict], sort_by: str = "spearman") -> pd.DataFrame:
    """Scorecard rows -> a table indexed by model, best first.

    Sorted on rank quality, which is the primary metric of `readme.md` 7.2. The
    constant baseline has an undefined Spearman by construction and is sorted
    last rather than dropped.
    """
    frame = pd.DataFrame(list(rows)).set_index("model")
    columns = [c for c in SCORE_COLUMNS if c in frame.columns]
    return frame[columns].sort_values(sort_by, ascending=False, na_position="last")


def value_tiers(
    actual,
    predicted,
    edges: Sequence[float] = TIER_EDGES,
    labels: Optional[Sequence[str]] = TIER_LABELS,
) -> pd.DataFrame:
    """Cut customers into value tiers by *prediction*, and report their *actual* value.

    This is the operational output of the whole pipeline (eda.ipynb 13): tiers
    come from the model, the value shown is what those customers really turned
    out to be worth, so the table doubles as a fair test of the ranking.
    """
    frame = pd.DataFrame({"actual": np.asarray(actual, dtype=float),
                          "predicted": np.asarray(predicted, dtype=float)})
    frame["tier"] = pd.qcut(frame["predicted"].rank(method="first"),
                            list(edges), labels=list(labels) if labels else None)
    report = (frame.groupby("tier", observed=True)
              .agg(customers=("actual", "size"),
                   mean_actual=("actual", "mean"),
                   mean_predicted=("predicted", "mean"),
                   total_actual=("actual", "sum")))
    report["share_of_value"] = report["total_actual"] / report["total_actual"].sum()
    return report


# --------------------------------------------------------------------------- #
# Smoke test: python src/metrics.py
# --------------------------------------------------------------------------- #


def _smoke_test() -> int:
    """Score three predictors of a known target, so each metric shows its range."""
    print("=" * 78)
    print("src/metrics.py -- SCORECARD SMOKE TEST")
    print("=" * 78)

    rng = np.random.default_rng(42)
    actual = pd.Series(rng.lognormal(6.0, 1.0, 500))
    predictors = {
        "perfect": actual.to_numpy(),
        "good (10% noise)": actual.to_numpy() * rng.lognormal(0, 0.1, 500),
        "ranking only (wrong level)": actual.to_numpy() * 3.0,
        "constant (the mean)": np.full(len(actual), actual.mean()),
        "random": rng.permutation(actual.to_numpy()),
    }
    board = leaderboard([scorecard(name, actual, pred) for name, pred in predictors.items()])
    print(board.round(4).to_string())

    print("\n  Read the rows against each other:")
    print("   * 'ranking only' has a perfect Spearman and a ruinous decile MAPE -- ranking")
    print("     and calibration are different questions (readme.md 7.2).")
    print("   * the constant predictor has an undefined Spearman by construction, and its")
    print("     top-decile capture is the share a random tenth of customers holds.")
    print("   * top-decile capture tops out at the data's own concentration, not at 1.0:")
    print(f"     {board.loc['perfect', 'top_decile_capture']:.1%} here.")

    tiers = value_tiers(actual, predictors["good (10% noise)"])
    print("\n  Value tiers from the 10%-noise predictor:")
    print(tiers.round(3).to_string())

    checks = {
        "perfect ranking scores 1": abs(board.loc["perfect", "spearman"] - 1) < 1e-9,
        "perfect Gini is 1": abs(board.loc["perfect", "norm_gini"] - 1) < 1e-9,
        "a constant has no ranking": bool(np.isnan(board.loc["constant (the mean)", "spearman"])),
        "a wrong level keeps its ranking": abs(
            board.loc["ranking only (wrong level)", "spearman"] - 1) < 1e-9,
        "a wrong level is badly calibrated":
            board.loc["ranking only (wrong level)", "decile_mape"] > 1.0,
        "random ranking scores about zero": abs(board.loc["random", "spearman"]) < 0.15,
    }
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("\n" + "=" * 78)
    if not all(checks.values()):
        print("FAILED -- a metric did not behave as documented.")
        return 1
    print("Every metric behaved as its docstring claims.")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(_smoke_test())
