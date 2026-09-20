"""Held-out evaluation, feature importance and value tiers (eda.ipynb 11-13).

Everything in here runs **once**, after model selection has finished on the
training split. The test split is opened at :func:`evaluate_on_test` and nowhere
else, which is the whole point of having reserved it (EDA 8).

Three things are produced:

* the **test scorecard**, next to the cross-validated estimate it should match;
* **two importance measures**, because on this model they disagree and the
  disagreement is the finding -- drop-column importance refits without a feature
  and is honest, permutation importance shuffles a column of a degree-3
  polynomial and sends the fitted surface into regions the data never visits;
* the **value tiers** a CRM team would act on, built from predictions and scored
  against what those customers were actually worth.

Figures are written as PNGs rather than shown, so the pipeline stays runnable
without a display.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_validate

# Relative when imported as part of the package, absolute when this file is run
# directly (``python src/evaluation.py``), where there is no parent package for
# the leading dot to resolve against. The __main__ block at the bottom is a
# self-contained smoke test.
if __package__ in (None, ""):                                    # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.metrics import TIER_EDGES, TIER_LABELS, scorecard, value_tiers
    from src.models import LogTargetRegressor
    from src.preprocessing import PreprocessConfig, quiet
else:
    from .metrics import TIER_EDGES, TIER_LABELS, scorecard, value_tiers
    from .models import LogTargetRegressor
    from .preprocessing import PreprocessConfig, quiet

#: A library logger: it inherits the handler main.py installs on "capstone",
#: so importing this module does not start printing on its own.
logger = logging.getLogger("capstone.evaluation")

__all__ = [
    "fit_final_model",
    "evaluate_on_test",
    "drop_column_importance",
    "permutation_scores",
    "tier_report",
    "save_bundle",
    "load_bundle",
    "write_figures",
]

# The EDA's palette: one accent, one contrast, recessive everything else.
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"


# --------------------------------------------------------------------------- #
# Final model
# --------------------------------------------------------------------------- #


def fit_final_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> LogTargetRegressor:
    """Refit the selected pipeline on every training row, target in logs."""
    final = LogTargetRegressor(model).fit(X_train, y_train)
    logger.info("final model refitted on %d rows (smearing factor %.4f)",
                len(X_train), final.smearing_factor_)
    return final


def evaluate_on_test(
    final: LogTargetRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Score the final model on the held-out split -- the one and only look at it."""
    pred_raw = final.predict(X_test)
    pred_log = final.predict_log(X_test)
    y_test_log = np.log(np.asarray(y_test, dtype=float))
    row = scorecard(f"{name} -- TEST", y_test, pred_raw, y_test_log, pred_log)
    return row, pred_raw, pred_log


# --------------------------------------------------------------------------- #
# Importance (EDA 12)
# --------------------------------------------------------------------------- #


def drop_column_importance(
    model,
    X: pd.DataFrame,
    y_log: np.ndarray,
    cv: Optional[KFold] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Refit the pipeline without each column; report the loss in CV R2.

    Slow but honest: it asks what a feature is *worth*, not what happens when its
    values are scrambled. Dropping a column means dropping it from the data *and*
    from the preprocessor's config, so the reduced pipeline is genuinely the same
    recipe over fewer inputs rather than the same recipe with a hole in it.
    """
    cv = cv or KFold(n_splits=5, shuffle=True, random_state=42)
    config: PreprocessConfig = model.named_steps["prep"].config or PreprocessConfig()
    columns = list(columns or X.columns)

    with quiet():
        full = float(cross_validate(model, X, y_log, cv=cv, scoring="r2")["test_score"].mean())

    rows: List[dict] = []
    for column in columns:
        reduced_config = _config_without(config, column)
        if reduced_config is None:
            continue
        reduced = clone(model)
        reduced.set_params(prep__config=reduced_config)
        with quiet():
            score = float(cross_validate(reduced, X.drop(columns=[column]), y_log,
                                         cv=cv, scoring="r2")["test_score"].mean())
        rows.append({"feature": column, "cv_r2_without": score, "loss": full - score})

    frame = (pd.DataFrame(rows).sort_values("loss", ascending=False)
             .set_index("feature"))
    frame.attrs["full_cv_r2"] = full
    return frame


def _config_without(config: PreprocessConfig, column: str) -> Optional[PreprocessConfig]:
    """The same representation, minus one input column."""
    if column in config.numeric_features:
        return replace(config,
                       numeric_features=tuple(c for c in config.numeric_features if c != column))
    if column in config.categorical_map:
        return replace(config,
                       categorical_map={k: v for k, v in config.categorical_map.items()
                                        if k != column})
    logger.warning("%s is not a modelled column -- skipped in drop-column importance", column)
    return None


def permutation_scores(
    model,
    X_test: pd.DataFrame,
    y_test_log: np.ndarray,
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance on the test split, for contrast with drop-column.

    Kept because the comparison is instructive, not because it is trustworthy
    here: shuffling one column of a high-degree polynomial fabricates feature
    combinations that never occur together, and the fitted surface extrapolates
    wildly there. EDA 12 shows it ranking the date columns above
    ``average_order_value``, which drop-column importance flatly contradicts.
    """
    with quiet():
        result = permutation_importance(model, X_test, y_test_log, scoring="r2",
                                        n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": list(X_test.columns),
                          "mean_drop_in_r2": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("mean_drop_in_r2", ascending=False)
            .set_index("feature"))


# --------------------------------------------------------------------------- #
# Tiers (EDA 13)
# --------------------------------------------------------------------------- #


def tier_report(
    y_test: pd.Series,
    pred_test: np.ndarray,
    edges: Sequence[float] = TIER_EDGES,
    labels: Sequence[str] = TIER_LABELS,
) -> pd.DataFrame:
    """The operational output: predicted tier vs the value those customers held."""
    return value_tiers(y_test, pred_test, edges=edges, labels=labels)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_bundle(
    path: str | Path,
    final: LogTargetRegressor,
    metadata: dict,
) -> Path:
    """Persist the fitted model plus everything needed to interpret its output.

    One file, not two: a model without its smearing factor, its feature contract
    and the scores it earned is not deployable, and a bundle that can drift apart
    from its metadata will.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, **metadata}, path)
    logger.info("model bundle saved -> %s", path)
    return path


def load_bundle(path: str | Path) -> dict:
    """Read a bundle written by :func:`save_bundle`."""
    import joblib

    return joblib.load(Path(path))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def write_figures(
    outdir: str | Path,
    leaderboard: pd.DataFrame,
    y_test: pd.Series,
    pred_test: np.ndarray,
    pred_test_log: np.ndarray,
    importance: pd.DataFrame,
    tiers: pd.DataFrame,
    baseline_spearman: Optional[float] = None,
) -> List[Path]:
    """Write the four charts that carry the result, and return their paths."""
    import matplotlib

    matplotlib.use("Agg")  # no display needed; the pipeline may run headless
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8, "figure.dpi": 110,
    })
    written: List[Path] = []

    # 1. the leaderboard ---------------------------------------------------- #
    board = leaderboard.dropna(subset=["spearman"]).sort_values("spearman")
    board = board[~board.index.str.startswith("Baseline")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)
    axes[0].barh(board.index, board["spearman"], color=BLUE, height=0.62)
    for i, v in enumerate(board["spearman"]):
        axes[0].text(v + 0.0015, i, f"{v:.4f}", va="center", fontsize=8.5)
    if baseline_spearman is not None:
        axes[0].axvline(baseline_spearman, color=MUTED, ls="--", lw=1.2)
        axes[0].text(baseline_spearman + 0.002, len(board) - 0.4,
                     f"heuristic baseline {baseline_spearman:.3f}",
                     color=MUTED, fontsize=8, va="top")
    axes[0].set(title="5-fold CV Spearman (ranking quality)", xlim=(0.87, 1.005))
    r2 = board["r2_log"].sort_values()
    axes[1].barh(r2.index, r2, color=ORANGE, height=0.62)
    for i, v in enumerate(r2):
        axes[1].text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8.5)
    axes[1].set(title="5-fold CV R2 on log(value)", xlim=(0.8, 1.03))
    axes[1].set_yticklabels([])
    for ax in axes:
        ax.grid(axis="y", visible=False)
    written.append(_save(fig, outdir / "model_selection.png"))

    # 2. test diagnostics --------------------------------------------------- #
    y_test = pd.Series(np.asarray(y_test, dtype=float))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(pred_test, y_test, s=14, alpha=0.5, color=BLUE, edgecolors="none")
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    axes[0].plot(lims, lims, color=ORANGE, lw=1.5)
    axes[0].set(title="Predicted vs actual (test)", xlabel="predicted value",
                ylabel="actual value", xscale="log", yscale="log")

    frame = pd.DataFrame({"y": y_test.to_numpy(), "p": pred_test})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), 10, labels=False) + 1
    summary = frame.groupby("decile")[["y", "p"]].mean()
    x = np.arange(len(summary))
    axes[1].bar(x - 0.19, summary["y"], width=0.38, color=BLUE, label="actual")
    axes[1].bar(x + 0.19, summary["p"], width=0.38, color=ORANGE, label="predicted")
    axes[1].set(xticks=x, xticklabels=summary.index, xlabel="predicted-value decile",
                ylabel="mean value", title="Calibration by decile")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="x", visible=False)

    resid = np.log(y_test.to_numpy()) - pred_test_log
    axes[2].scatter(pred_test_log, resid, s=14, alpha=0.5, color=BLUE, edgecolors="none")
    axes[2].axhline(0, color=ORANGE, lw=1.5)
    axes[2].set(title=f"Test residuals (log), std = {resid.std():.3f}",
                xlabel="predicted log value", ylabel="residual")
    fig.tight_layout()
    written.append(_save(fig, outdir / "test_diagnostics.png"))

    # 3. drop-column importance --------------------------------------------- #
    plot = importance.sort_values("loss")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    # A log axis reads the four orders of magnitude between the features, but it
    # cannot show a loss of zero or less, so every bar carries its real number.
    floor = 1e-6
    ax.barh(plot.index, plot["loss"].clip(lower=floor), color=BLUE, height=0.6)
    for i, value in enumerate(plot["loss"]):
        ax.text(max(value, floor) * 1.15, i, f"{value:+.5f}", va="center", fontsize=8.5,
                color=INK if value > floor else MUTED)
    ax.set(title="Drop-column importance (CV)",
           xlabel="loss in CV R2 when removed (bars below 1e-6 are at the axis floor)",
           xscale="log", xlim=(floor * 0.7, float(plot["loss"].max()) * 6))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    written.append(_save(fig, outdir / "feature_importance.png"))

    # 4. value tiers -------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.barh([str(i) for i in tiers.index], tiers["share_of_value"], color=BLUE, height=0.6)
    for i, (v, n) in enumerate(zip(tiers["share_of_value"], tiers["customers"])):
        ax.text(v + 0.006, i, f"{v:.0%} of value   ({n} customers)", va="center", fontsize=9)
    ax.set(title="Share of actual value by predicted tier (test set)",
           xlabel="share of total value", xlim=(0, float(tiers["share_of_value"].max()) * 1.45))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    written.append(_save(fig, outdir / "value_tiers.png"))

    return written


def _save(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Smoke test: python src/evaluation.py
# --------------------------------------------------------------------------- #


def _smoke_test() -> int:
    """Fit, score, explain, tier, persist and plot -- on synthetic data, end to end.

    Everything this module does, in the order ``train.py`` does it, without
    needing the dataset or a fitted pipeline from anywhere else. Figures and the
    bundle are written to a temporary directory and deleted with it.
    """
    import tempfile

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if __package__ in (None, ""):                                # pragma: no cover
        from src.preprocessing import TARGET
    else:
        from .preprocessing import TARGET

    print("=" * 78)
    print("src/evaluation.py -- EVALUATION SMOKE TEST")
    print("=" * 78)

    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "purchases": rng.lognormal(1.5, 0.9, n),
        "order_value": rng.lognormal(4.2, 0.5, n),
        "recency": rng.uniform(1, 400, n),
    })
    y = pd.Series(np.exp(3.4) * X["purchases"] ** 0.3 * X["order_value"] ** 0.86
                  * X["recency"] ** -0.32 * rng.lognormal(0, 0.1, n), name=TARGET)
    X_train, X_test = X.iloc[:400], X.iloc[400:].reset_index(drop=True)
    y_train, y_test = y.iloc[:400], y.iloc[400:].reset_index(drop=True)

    model = Pipeline([("prep", StandardScaler()), ("model", Ridge(alpha=0.01))])
    logged = Pipeline([("log", _LogColumns()), ("rest", model)])

    print(f"  {len(X_train)} train / {len(X_test)} test rows, "
          "ridge on logged features\n")

    print("1. FINAL MODEL AND HELD-OUT SCORE")
    print("-" * 78)
    final = fit_final_model(logged, X_train, y_train)
    row, pred_test, pred_test_log = evaluate_on_test(final, X_test, y_test, "ridge on logs")
    print(pd.DataFrame([row]).set_index("model").round(4).to_string())

    print("\n2. DROP-COLUMN IMPORTANCE")
    print("-" * 78)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    importance = _generic_drop_column(logged, X_train, np.log(y_train.to_numpy()), cv)
    print(importance.round(5).to_string())

    print("\n3. PERMUTATION IMPORTANCE (for contrast)")
    print("-" * 78)
    fitted = clone(logged).fit(X_train, np.log(y_train.to_numpy()))
    print(permutation_scores(fitted, X_test, np.log(y_test.to_numpy()),
                             n_repeats=5).round(4).to_string())

    print("\n4. VALUE TIERS")
    print("-" * 78)
    tiers = tier_report(y_test, pred_test)
    print(tiers.round(3).to_string())

    print("\n5. BUNDLE AND FIGURES")
    print("-" * 78)
    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = save_bundle(Path(tmp) / "model.joblib", final,
                                  {"name": "ridge on logs", "test_scorecard": row})
        reloaded = load_bundle(bundle_path)
        same = bool(np.allclose(reloaded["model"].predict(X_test), pred_test))
        print(f"  bundle round-trip: {'identical predictions' if same else 'MISMATCH'}")

        board = pd.DataFrame([{**row, "model": "ridge on logs", "r2_log": row["r2_log"]}]
                             ).set_index("model")
        figures = write_figures(Path(tmp) / "figures", board, y_test, pred_test,
                                pred_test_log, importance, tiers)
        print(f"  figures written: {', '.join(p.name for p in figures)}")

    print("\n" + "=" * 78)
    if not same:
        print("FAILED -- the saved bundle did not reproduce its predictions.")
        return 1
    print("Fit, score, explain, tier, persist and plot all ran.")
    print("In the project this module is used through train.py (stages 12-15).")
    return 0


class _LogColumns(BaseEstimator, TransformerMixin):
    """Tiny log transformer, so the smoke test does not need CLVPreprocessor."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(pd.DataFrame(X).clip(lower=1e-3))


def _generic_drop_column(model, X, y_log, cv) -> pd.DataFrame:
    """Drop-column importance for a plain estimator (no PreprocessConfig involved)."""
    with quiet():
        full = float(cross_validate(model, X, y_log, cv=cv, scoring="r2")["test_score"].mean())
    rows = []
    for column in X.columns:
        with quiet():
            score = float(cross_validate(clone(model), X.drop(columns=[column]), y_log,
                                         cv=cv, scoring="r2")["test_score"].mean())
        rows.append({"feature": column, "cv_r2_without": score, "loss": full - score})
    frame = pd.DataFrame(rows).sort_values("loss", ascending=False).set_index("feature")
    frame.attrs["full_cv_r2"] = full
    return frame


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(_smoke_test())
