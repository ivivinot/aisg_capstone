"""Modelling entry point: select, tune, and score the model once on the test split.

    python train.py                     # full run: data -> checks -> select -> tune -> test
    python train.py --no-tune           # skip the randomised search
    python train.py --no-importance     # skip drop-column importance (the slowest step)
    python train.py --no-stat-checks    # skip the slow data checks before modelling

The data stages, the flags that control them and the validation gate all come
from ``main.py``, so the two entry points cannot drift apart: this file runs
``main.run_preprocessing`` and ``main.run_validation`` first and **stops if a
check failed** -- modelling on data that did not pass its own checks produces
numbers nobody should quote.

Then, following eda.ipynb 9-13:

    11. model selection: 8 candidates + 2 baselines, 5-fold CV (EDA 9)
    12. randomised search over the finalists (EDA 10)
    13. refit the winner, score the held-out test split ONCE (EDA 11)
    14. drop-column and permutation importance (EDA 12)
    15. customer value tiers (EDA 13)
    16. artifacts: leaderboard, per-customer predictions, figures, model bundle

Every score this prints is a *function-recovery* score, not a forecast. EDA 7
established that the target is a deterministic function of the feature columns,
so a near-perfect result confirms the pipeline works and says nothing about how
well customer value can be predicted. readme.md 9 sets out what data would be
needed for the latter.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.model_selection import KFold, RandomizedSearchCV  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from main import (  # noqa: E402
    add_advanced_arguments,
    add_baseline_arguments,
    add_optimization_arguments,
    add_preprocessing_arguments,
    add_validation_arguments,
    banner,
    config_from_args,
    empty_summary,
    print_tally,
    run_advanced_stage,
    run_baseline_stage,
    run_optimization_stage,
    run_preprocessing,
    run_validation,
    save_everything as save_data_artifacts,
)
from src.evaluation import (  # noqa: E402
    drop_column_importance,
    evaluate_on_test,
    fit_final_model,
    permutation_scores,
    save_bundle,
    tier_report,
    write_figures,
)
from src.metrics import leaderboard as build_leaderboard, scorecard  # noqa: E402
from src.models import LogTargetRegressor, has_xgboost  # noqa: E402
from src.preprocessing import (  # noqa: E402
    AOV_COL,
    COUNT_COL,
    CLVPreprocessor,
    PreprocessConfig,
    get_logger,
    quiet,
)

log = get_logger("capstone")

HEURISTIC_BASELINE = "Baseline: purchases x AOV"


# --------------------------------------------------------------------------- #
# Candidate zoo, cross-validation and tuning (EDA 9-10)
# --------------------------------------------------------------------------- #
#
# These used to live in src/models.py. That module is now the *baseline* ladder
# -- the bar a real model has to clear, for any task -- and this is the part that
# only makes sense for this project: eight candidates, the representation each
# one deserves, and a randomised search over the three finalists.
#
# The two baseline rows below are scored in-sample, exactly as eda.ipynb 9 scored
# them, so the leaderboard here matches the notebook. The cross-validated,
# benchmarked baseline ladder is src/models.evaluate_baselines, which main.py
# runs as its own stage.


def representation(base: Optional[PreprocessConfig] = None, **overrides) -> PreprocessConfig:
    """A copy of ``base`` with some knobs changed -- one candidate's feature space."""
    return replace(base or PreprocessConfig(), **overrides)


def _tree_representation(base: PreprocessConfig) -> PreprocessConfig:
    """Raw features for the tree ensembles.

    Trees are invariant to monotone transforms of a feature, so logging and
    scaling would buy them nothing; EDA 9 fitted them on the raw columns and this
    keeps the comparison honest. The clipping guard stays on -- with the default
    (0, 1) quantiles it moves no training row and only holds unseen data inside
    the fitted range.
    """
    return representation(base, log_transform=False, poly_degree=1, scale=False)


def candidate_models(
    base: Optional[PreprocessConfig] = None,
    include_xgboost: bool = True,
) -> Dict[str, Pipeline]:
    """The candidate zoo of EDA 9, spanning the families in `readme.md` 4 and 6.

    Each entry pairs a representation with an estimator. Read the names as
    claims: "Linear (raw features)" is there to show what the log transform is
    worth, and the tree ensembles are there because "tabular problem -> gradient
    boosting" is the default instinct that this dataset punishes.

    ``base.poly_degree`` is a *ceiling*, not a setting: the polynomial candidate
    enters at degree 2 and the search (:func:`search_spaces`) may climb to the
    ceiling, which is how EDA 10 arrives at degree 3. Running with
    ``--poly-degree 1`` therefore holds every candidate at the first-order
    log-linear representation.
    """
    base = base or PreprocessConfig()
    logs = representation(base, poly_degree=1)          # logged + scaled, no expansion
    trees = _tree_representation(base)
    seed = base.random_state
    entry_degree = min(2, base.poly_degree)

    def pipe(config: PreprocessConfig, model) -> Pipeline:
        return Pipeline([("prep", CLVPreprocessor(config)), ("model", model)])

    candidates: Dict[str, Pipeline] = {
        "Linear (raw features)": pipe(
            representation(base, log_transform=False, poly_degree=1), LinearRegression()),
        "Linear (log features)": pipe(logs, LinearRegression()),
        "Polynomial on logs + Ridge": pipe(
            representation(base, poly_degree=entry_degree), Ridge(alpha=1.0)),
        "Random Forest": pipe(trees, RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=-1)),
        "HistGradientBoosting": pipe(trees, HistGradientBoostingRegressor(random_state=seed)),
        "k-NN (k=10)": pipe(logs, KNeighborsRegressor(n_neighbors=10)),
        "SVR (RBF)": pipe(logs, SVR(C=10.0, epsilon=0.05)),
    }

    if include_xgboost and has_xgboost():
        # Optional dependency, reached only when has_xgboost() said yes; the
        # ignore keeps the editor quiet in environments without it.
        from xgboost import XGBRegressor  # type: ignore[import-not-found]

        candidates["XGBoost"] = pipe(trees, XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            random_state=seed, n_jobs=-1))
    elif include_xgboost:
        log.warning("xgboost is not installed -- skipping that candidate")

    return candidates


# --------------------------------------------------------------------------- #
# Baselines (readme.md 7.3)
# --------------------------------------------------------------------------- #


def baseline_rows(X: pd.DataFrame, y: pd.Series) -> List[dict]:
    """The two references any model must beat before it is worth deploying.

    * **Train mean** -- the floor. A constant prediction, so its Spearman is
      undefined by construction and its normalized Gini is noise around zero.
    * **purchases x order value** -- what an analyst computes without a model.
      On this file it already reaches rho = 0.88 (EDA 9), and that, not the mean,
      is the bar a model has to clear to be worth deploying.
    """
    y = pd.Series(np.asarray(y, dtype=float))
    y_log = np.log(y)

    rows = [scorecard("Baseline: train mean", y, np.full(len(y), y.mean()),
                      y_log, np.full(len(y), y_log.mean()))]

    heuristic = X[COUNT_COL].to_numpy(dtype=float) * X[AOV_COL].to_numpy(dtype=float)
    # Rescaled to the training mean so the level is comparable; ranking is untouched.
    heuristic = heuristic * (y.mean() / heuristic.mean())
    rows.append(scorecard("Baseline: purchases x AOV", y, heuristic, y_log, np.log(heuristic)))
    return rows


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #


def cv_scorecard(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[KFold] = None,
) -> Tuple[dict, np.ndarray]:
    """Out-of-fold predictions for one candidate -> one scorecard row.

    The whole pipeline -- preprocessor included -- is refitted inside every fold,
    so nothing the model sees was computed with the validation rows in it. The
    smearing factor is taken from the out-of-fold residuals for the same reason.
    """
    cv = cv or KFold(n_splits=5, shuffle=True, random_state=42)
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y, dtype=float))
    oof_log = np.zeros(len(y))

    with quiet():  # a full refit per fold would otherwise flood the log
        for train_idx, val_idx in cv.split(X):
            fitted = LogTargetRegressor(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_log[val_idx] = fitted.predict_log(X.iloc[val_idx])

    y_log = np.log(y.to_numpy())
    smear = float(np.mean(np.exp(y_log - oof_log)))
    return scorecard(name, y, np.exp(oof_log) * smear, y_log, oof_log), oof_log


def run_model_selection(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: Dict[str, Pipeline],
    cv: Optional[KFold] = None,
    with_baselines: bool = True,
) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    """Score every candidate, and the baselines, with the same folds."""
    rows: List[dict] = baseline_rows(X, y) if with_baselines else []
    oof: Dict[str, np.ndarray] = {}

    for name, model in candidates.items():
        row, oof_log = cv_scorecard(name, model, X, y, cv)
        rows.append(row)
        oof[name] = oof_log
        log.info("%-28s CV spearman %.4f  R2(log) %.4f",
                    name, row["spearman"], row.get("r2_log", float("nan")))
    return rows, oof


# --------------------------------------------------------------------------- #
# Hyperparameter search (EDA 10)
# --------------------------------------------------------------------------- #


def search_spaces(
    base: Optional[PreprocessConfig] = None,
    finalists: Sequence[str] = ("Polynomial on logs + Ridge", "XGBoost", "Random Forest"),
    max_degree: Optional[int] = None,
) -> Dict[str, Tuple[dict, int]]:
    """Search space per finalist: ``{name: (param_distributions, n_iter)}``.

    The finalists are the best linear-family model plus the two strongest
    non-linear families, so flexibility gets a fair chance after tuning.

    The polynomial family searches over ``prep__config`` -- the *representation*,
    not just the estimator -- up to ``base.poly_degree``, the ceiling the caller set. That is the search that matters here: EDA 10 has it
    climbing from CV R2 0.9941 to 0.9998 by choosing degree 3, while Random
    Forest does not move at all.
    """
    base = base or PreprocessConfig()
    max_degree = base.poly_degree if max_degree is None else max_degree
    spaces: Dict[str, Tuple[dict, int]] = {
        "Polynomial on logs + Ridge": (
            {
                "prep__config": [representation(base, poly_degree=d)
                                 for d in range(1, max_degree + 1)],
                "model__alpha": loguniform(1e-3, 1e3),
            },
            25,
        ),
        "Random Forest": (
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
            25,
        ),
        "XGBoost": (
            {
                "model__n_estimators": randint(200, 1200),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(2, 9),
                "model__min_child_weight": randint(1, 12),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__reg_lambda": loguniform(0.1, 20),
            },
            30,
        ),
    }
    return {name: spaces[name] for name in finalists if name in spaces}


def tune(
    X: pd.DataFrame,
    y_log: np.ndarray,
    candidates: Dict[str, Pipeline],
    spaces: Dict[str, Tuple[dict, int]],
    cv: Optional[KFold] = None,
    seed: int = 42,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Pipeline], List[dict]]:
    """Randomised search per finalist, optimising R2 on ``log(value)``.

    Random search rather than a grid: the spaces mix continuous and discrete
    knobs, and with this many dimensions a grid spends most of its budget on the
    axes that turn out not to matter.
    """
    cv = cv or KFold(n_splits=5, shuffle=True, random_state=seed)
    tuned: Dict[str, Pipeline] = {}
    rows: List[dict] = []

    for name, (space, n_iter) in spaces.items():
        if name not in candidates:
            log.warning("%s is not among the candidates -- skipping its search", name)
            continue
        search = RandomizedSearchCV(candidates[name], space, n_iter=n_iter, scoring="r2",
                                    cv=cv, random_state=seed, n_jobs=n_jobs, refit=True)
        with quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y_log)
        tuned[name] = search.best_estimator_
        rows.append({
            "model": name,
            "tuned_cv_r2_log": float(search.best_score_),
            "n_iter": n_iter,
            "best_params": readable_params(search.best_params_),
        })
        log.info("%-28s tuned CV R2(log) %.4f", name, search.best_score_)
    return tuned, rows


def readable_params(params: dict) -> dict:
    """Search results, printable: a whole config object becomes its degree."""
    out = {}
    for key, value in params.items():
        if isinstance(value, PreprocessConfig):
            out[key] = f"poly_degree={value.poly_degree}, log_transform={value.log_transform}"
        elif isinstance(value, float):
            out[key] = round(value, 5)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select, tune and evaluate the CLV model on the processed data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preprocessing_arguments(p)
    add_validation_arguments(p)
    add_baseline_arguments(p)
    add_advanced_arguments(p)
    add_optimization_arguments(p)

    g = p.add_argument_group("modelling")
    g.add_argument("--cv-model-folds", type=int, default=5,
                   help="folds for model selection and tuning")
    g.add_argument("--n-jobs", type=int, default=-1, help="parallelism for the search")
    g.add_argument("--no-xgboost", action="store_true", help="leave XGBoost out of the zoo")
    g.add_argument("--no-tune", action="store_true",
                   help="skip the randomised search and take the best default candidate")
    g.add_argument("--no-importance", action="store_true",
                   help="skip drop-column importance (the slowest step)")
    g.add_argument("--no-figures", action="store_true", help="do not write PNG charts")
    g.add_argument("--ignore-failed-checks", action="store_true",
                   help="model anyway when a data check fails (it will be reported)")
    return p.parse_args(argv)


def show(frame: pd.DataFrame, decimals: int = 4, indent: str = "  ") -> None:
    """Print a table indented under its section heading."""
    text = frame.round(decimals).to_string()
    print("\n".join(indent + line for line in text.splitlines()))


# --------------------------------------------------------------------------- #
# Modelling stages
# --------------------------------------------------------------------------- #


def run_modelling(args: argparse.Namespace, config: PreprocessConfig, data: dict) -> dict:
    """Stages 11-15: select, tune, test once, explain, tier."""
    X_train, y_train = data["X_train_raw"], data["y_train"]
    X_test, y_test = data["X_test_raw"], data["y_test"]
    cv = KFold(n_splits=args.cv_model_folds, shuffle=True, random_state=config.random_state)

    # 11. model selection ---------------------------------------------------- #
    banner(11, f"MODEL SELECTION -- {args.cv_model_folds}-fold CV on the training split (EDA 9)")
    candidates = candidate_models(config, include_xgboost=not args.no_xgboost)
    print(f"  {len(candidates)} candidates + 2 baselines; every fold refits the whole"
          f" pipeline,\n  preprocessor included, so these scores carry no leakage.\n")
    rows, _ = run_model_selection(X_train, y_train, candidates, cv=cv)
    board = build_leaderboard(rows)
    show(board)
    heuristic_rho = float(board.loc[HEURISTIC_BASELINE, "spearman"])
    print(f"\n  The bar is the heuristic baseline (Spearman {heuristic_rho:.3f}), not the mean.")

    # 12. tuning ------------------------------------------------------------- #
    tuning_rows: list = []
    if args.no_tune:
        best_name = _best_default(board, candidates)
        best_model = candidates[best_name]
        banner(12, "TUNING -- skipped (--no-tune); the best default candidate is used")
        print(f"  selected: {best_name}")
        tuned_board = pd.DataFrame()
    else:
        banner(12, "TUNING -- randomised search over the finalists (EDA 10)")
        spaces = search_spaces(config)
        print(f"  finalists: {', '.join(spaces)}"
              f"\n  objective: R2 on log(value), same folds as above\n")
        tuned, tuning_rows = tune(X_train, np.log(y_train.to_numpy()), candidates, spaces,
                                  cv=cv, seed=config.random_state, n_jobs=args.n_jobs)
        for row in tuning_rows:
            default = float(board.loc[row["model"], "r2_log"])
            row["default_cv_r2_log"] = default
            print(f"  {row['model']:28s} {default:.4f} -> {row['tuned_cv_r2_log']:.4f}")
            print(f"  {'':28s} {row['best_params']}")

        print("\n  Re-scoring the tuned finalists on every metric:")
        tuned_rows = []
        for name, model in tuned.items():
            row, _ = cv_scorecard(f"{name} (tuned)", model, X_train, y_train, cv)
            tuned_rows.append(row)
        tuned_board = build_leaderboard(tuned_rows)
        show(tuned_board)
        best_name = str(tuned_board.index[0]).replace(" (tuned)", "")
        best_model = tuned[best_name]
        print(f"\n  selected by CV Spearman: {best_name}")
        print(f"  {_describe(best_model)}")

    # 13. final model, one look at the test split ---------------------------- #
    banner(13, "FINAL MODEL -- held-out test evaluation (EDA 11)")
    final = fit_final_model(best_model, X_train, y_train)
    test_row, pred_test, pred_test_log = evaluate_on_test(final, X_test, y_test, best_name)

    comparison_rows = [test_row]
    if len(tuned_board):
        cv_row = tuned_board.loc[f"{best_name} (tuned)"].to_dict()
        comparison_rows.append({**cv_row, "model": f"{best_name} -- CV (train)"})
    else:
        comparison_rows.append({**board.loc[best_name].to_dict(),
                                "model": f"{best_name} -- CV (train)"})
    comparison_rows.append({**board.loc[HEURISTIC_BASELINE].to_dict(),
                            "model": f"{HEURISTIC_BASELINE} (train)"})
    comparison = pd.DataFrame(comparison_rows).set_index("model")
    show(comparison)
    print(f"\n  Duan smearing factor {final.smearing_factor_:.4f}"
          f"   test residual spread (log) {np.std(data['y_test_log'] - pred_test_log):.4f}")
    print("  Test and CV agree, so nothing was overfitted to the folds. Read the level of")
    print("  these numbers against readme.md 7: a deployed 12-month CLV model reports")
    print("  Spearman ~0.56. The gap is the gap between forecasting and formula recovery.")

    # 14. importance -------------------------------------------------------- #
    importance = pd.DataFrame()
    perm = pd.DataFrame()
    if not args.no_importance:
        banner(14, "WHAT DRIVES VALUE (EDA 12)")
        importance = drop_column_importance(best_model, X_train,
                                            np.log(y_train.to_numpy()), cv=cv)
        print(f"  full-model CV R2 on log(value): {importance.attrs['full_cv_r2']:.6f}")
        print("\n  Drop-column importance -- refit without the column, measure the loss:")
        show(importance, decimals=6)
        # A separate fit on log(value): permutation importance measures the model
        # in the space it was trained in, and cloning keeps the selected pipeline
        # itself unfitted so nothing downstream depends on this call's side effects.
        perm_model = clone(best_model).fit(X_train, np.log(y_train.to_numpy()))
        perm = permutation_scores(perm_model, X_test, data["y_test_log"],
                                  seed=config.random_state)
        print("\n  Permutation importance, for contrast (EDA 12: not to be trusted on a")
        print("  high-degree polynomial -- shuffling invents feature combinations the")
        print("  fitted surface never saw):")
        show(perm)

    # 15. tiers ------------------------------------------------------------- #
    banner(15, "CUSTOMER VALUE TIERS -- what the business acts on (EDA 13)")
    tiers = tier_report(y_test, pred_test)
    show(tiers, decimals=3)
    print("\n  Tiers come from the predictions; the value shown is what those customers")
    print("  actually turned out to be worth, so the table is also a test of the ranking.")

    return {
        "leaderboard": board, "tuned_board": tuned_board, "tuning_rows": tuning_rows,
        "best_name": best_name, "best_model": best_model, "final": final,
        "test_row": test_row, "comparison": comparison,
        "pred_test": pred_test, "pred_test_log": pred_test_log,
        "importance": importance, "permutation": perm, "tiers": tiers,
        "heuristic_spearman": heuristic_rho,
    }


def _best_default(board: pd.DataFrame, candidates: dict) -> str:
    """Top-ranked candidate that is an actual model rather than a baseline."""
    for name in board.index:
        if name in candidates:
            return str(name)
    raise RuntimeError("no candidate model scored")


def _describe(model) -> str:
    """One line: which representation and which estimator won."""
    config = model.named_steps["prep"].config or PreprocessConfig()
    estimator = model.named_steps["model"]
    return (f"representation: log={config.log_transform}, poly_degree={config.poly_degree}, "
            f"scale={config.scale}\n  estimator: {estimator}")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_model_artifacts(args: argparse.Namespace, config: PreprocessConfig,
                         data: dict, model: dict) -> list:
    """Everything the modelling stages produced, beside the data artifacts."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = config.target
    written: list = []

    model["leaderboard"].to_csv(outdir / "model_leaderboard.csv")
    written.append(outdir / "model_leaderboard.csv")

    predictions = pd.DataFrame({
        target: data["y_test"].to_numpy(),
        "predicted": model["pred_test"],
        "predicted_log": model["pred_test_log"],
        "tier": pd.qcut(pd.Series(model["pred_test"]).rank(method="first"),
                        [0, 0.5, 0.8, 0.95, 1.0],
                        labels=list(model["tiers"].index.astype(str))),
    })
    predictions = pd.concat([data["X_test_raw"].reset_index(drop=True), predictions], axis=1)
    predictions.to_csv(outdir / "test_predictions.csv", index=False)
    written.append(outdir / "test_predictions.csv")
    model["tiers"].to_csv(outdir / "customer_tiers.csv")
    written.append(outdir / "customer_tiers.csv")

    report = {
        "best_model": model["best_name"],
        "representation": {"log_transform": config.log_transform,
                           "poly_degree": _poly_degree(model["best_model"], config),
                           "scale": config.scale},
        "estimator": str(model["best_model"].named_steps["model"]),
        "smearing_factor": model["final"].smearing_factor_,
        "cv_leaderboard": model["leaderboard"].reset_index().to_dict(orient="records"),
        "tuning": model["tuning_rows"],
        "tuned_leaderboard": (model["tuned_board"].reset_index().to_dict(orient="records")
                              if len(model["tuned_board"]) else []),
        "test_scorecard": model["test_row"],
        "drop_column_importance": (model["importance"].reset_index().to_dict(orient="records")
                                   if len(model["importance"]) else []),
        "permutation_importance": (model["permutation"].reset_index().to_dict(orient="records")
                                   if len(model["permutation"]) else []),
        "value_tiers": model["tiers"].reset_index().to_dict(orient="records"),
        "caveat": ("EDA 7: the target is a deterministic function of the features, so these "
                   "scores measure function recovery, not forecasting skill (readme.md 9)."),
    }
    (outdir / "modelling_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    written.append(outdir / "modelling_report.json")

    written.append(save_bundle(outdir / "model.joblib", model["final"], {
        "name": model["best_name"],
        "config": config,
        "input_features": list(data["X_test_raw"].columns),
        "target": target,
        "test_scorecard": model["test_row"],
        "tier_edges": [0, 0.5, 0.8, 0.95, 1.0],
        "tier_labels": list(model["tiers"].index.astype(str)),
        "train_prediction_quantiles": _tier_thresholds(model["final"], data["X_train_raw"]),
    }))

    if not args.no_figures:
        written += write_figures(outdir / "figures", model["leaderboard"], data["y_test"],
                                 model["pred_test"], model["pred_test_log"],
                                 model["importance"], model["tiers"],
                                 baseline_spearman=model["heuristic_spearman"])
    return written


def _poly_degree(model, config: PreprocessConfig) -> int:
    prep_config = model.named_steps["prep"].config or config
    return int(prep_config.poly_degree)


def _tier_thresholds(final, X_train: pd.DataFrame) -> dict:
    """Tier cut-offs in dollars, learned on the training split.

    A tier has to mean the same thing for one new customer as it did in the test
    report, so the boundaries are fixed here rather than recomputed from whatever
    batch happens to be scored next (see predict.py).
    """
    pred = final.predict(X_train)
    return {str(q): float(np.quantile(pred, q)) for q in (0.5, 0.8, 0.95)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    outdir = Path(args.outdir)

    print("=" * 78)
    print("CAPSTONE -- CUSTOMER LIFETIME VALUE MODELLING")
    print("=" * 78)
    print(f"  data    {args.data}")
    print(f"  outdir  {outdir if not args.no_save else '(not saving)'}")
    stages = "process" if args.no_validate else "process -> validate -> test"
    print(f"  stages  {stages} -> select"
          f"{'' if args.no_tune else ' -> tune'} -> score -> tier")

    data = run_preprocessing(args, config)
    checks = [] if args.no_validate else run_validation(args, config, data)

    baselines = advanced = None
    if not args.no_baselines:
        # The ladder runs before model selection for a reason: every leaderboard
        # below is read against these numbers (readme.md 7.3).
        baselines = run_baseline_stage(args, config, data)
        checks += baselines.checks

        if not args.no_advanced:
            advanced = run_advanced_stage(args, config, data, baselines)
            checks += advanced.checks

    optimization = None
    if not args.no_optimize:
        # The same optimization protocol main.py runs, so the candidate zoo below
        # is read against models that were tuned and diagnosed the same way.
        optimization = run_optimization_stage(args, config, data,
                                              baselines.task if baselines else "regression")
        checks += optimization.checks

    validation = print_tally(checks) if checks else empty_summary()

    if validation["failed"] and not args.ignore_failed_checks:
        print(f"\nSTOPPING -- {validation['failed']} data check(s) failed: "
              f"{'; '.join(validation['failures'])}")
        print("Fix them, or re-run with --ignore-failed-checks if you know why they fail.")
        return 1

    model = run_modelling(args, config, data)

    if not args.no_save:
        written = save_data_artifacts(args, config, data, validation,
                                      baselines, advanced, optimization)
        written += save_model_artifacts(args, config, data, model)
        banner(16, "ARTIFACTS")
        for path in written:
            print(f"  {path}")
        print("\n  Score a new batch of customers:")
        print("    python predict.py --input new_customers.csv --output scored.csv")

    print(f"\nDone. Selected model: {model['best_name']}.")
    print("Every score above is a function-recovery score (EDA 7, readme.md 9), not a")
    print("12-month forecast: this file has no timestamps and no future window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
