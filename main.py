"""Entry point: process the data, then test and validate what came out.

    python main.py                          # defaults, reproduces the EDA set-up
    python main.py --poly-degree 1          # plain log-linear representation
    python main.py --outlier-strategy flag  # keep raw values, append 0/1 columns
    python main.py --drop-invalid-recency   # remove the 52 impossible rows instead
    python main.py --derived-features all   # add every engineered column
    python main.py --compare-features       # measure what each feature set is worth
    python main.py --feature-selection vif  # and what the wrong selection costs
    python main.py --metric-guide           # what each metric answers, and how it lies
    python main.py --baseline-catalogue     # the baselines for every task, then exit
    python main.py --architectures          # what each advanced model assumes, then exit
    python main.py --no-advanced            # skip the architecture comparison
    python main.py --cv-strategy repeated   # shrink fold noise at a linear cost
    python main.py --search-method halving  # successive halving instead of random
    python main.py --no-optimize            # skip tuning, diagnosis and selection
    python main.py --no-baselines           # skip the baseline ladder
    python main.py --no-stat-checks         # skip the slow cross-validated checks
    python main.py --no-validate            # process only, check nothing

What it does, in order:

    1. audit the raw CSV -- the EDA 2 quality checks, as a JSON report
    2. row-level filtering (off by default; see src/preprocessing.filter_rows)
    3. split 80/20 stratified on value deciles, then fit the preprocessor on the
       TRAINING split only and transform both (EDA 8)
    4. feature engineering: the created columns and the features kept
       (src/feature_engineering.py)
    5. validate the output contract: usable, finite, leakage-free
    6. test the transformer: single row == batch, order and index invariance,
       dirty-data edge cases, save/load and clone round-trips
    7. test the feature engineering: what was created, what the selection kept,
       and whether the catalogue's log-space claims actually hold
    8. validate statistically: feature count per degree, the CV R2 the EDA
       measured, and a label-shuffle test whose score must collapse to zero
    9. run the baseline ladder (src/models.py): select, train, evaluate, analyse
       the metrics and benchmark the cost of the models a real one must beat
   10. run two further architectures (src/advanced_models.py) on the same folds,
       compare them with the ladder fold by fold, and write the comparison up
   11. optimise every developed model (src/model_optimization.py): fold scheme,
       hyperparameter search, over/underfit diagnosis, one-standard-error
       selection, and one look at the held-out split
   12. write the processed data, the fitted pipeline and the reports to outputs/

Checks are collected, not raised, so one run reports every failure -- and the
process exits non-zero if anything failed, which makes this usable as a gate.

The test split is transformed but must not be looked at again until a final model
is evaluated once (EDA 8). Modelling lives in ``train.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict, replace
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    Check,
    CLVPreprocessor,
    LogTargetTransformer,
    PreprocessConfig,
    audit_dataset,
    check_output_contract,
    check_statistics,
    check_transformer_behaviour,
    filter_rows,
    get_logger,
    load_raw,
    stratified_split,
    summarise,
)
from src.models import (  # noqa: E402
    BaselineResults,
    build_baselines,
    evaluate_baselines,
    infer_task,
    metric_guide,
    spec_table,
)
from src.advanced_models import ADVANCED_SPECS, build_advanced_models, run_comparison  # noqa: E402
from src.model_optimization import (  # noqa: E402
    CV_STRATEGIES,
    SEARCH_METHODS,
    search_space_for,
    check_optimization,
    optimise_models,
    validate_final_model,
)
from src.feature_engineering import (  # noqa: E402
    FEATURE_SPECS,
    SELECTION_STRATEGIES,
    check_feature_engineering,
    compare_feature_sets,
    feature_report,
    resolve_derived,
)

log = get_logger("capstone")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_preprocessing_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The data-side flags. ``train.py`` reuses these so the two cannot drift."""
    p.add_argument("--data", default=str(ROOT / "data" / "synthetic_data_126.csv"),
                   help="raw CSV to preprocess")
    p.add_argument("--outdir", default=str(ROOT / "outputs"),
                   help="where processed data, the fitted pipeline and reports are written")

    g = p.add_argument_group("missing values")
    g.add_argument("--numeric-impute", choices=["median", "mean", "knn"], default="median",
                   help="fill for the skewed numerics; median is robust to the 6.4 skew")
    g.add_argument("--knn-neighbors", type=int, default=5, help="k, when --numeric-impute knn")
    g.add_argument("--no-missing-indicators", action="store_true",
                   help="do not append missing_<col> columns for features imputed at fit time")

    g = p.add_argument_group("data quality (EDA 2, 6)")
    g.add_argument("--recency-policy", choices=["clip", "flag", "none"], default="flag",
                   help="the 52 rows whose last purchase precedes their first: flag them "
                        "(default, non-destructive), clip the value, or ignore the rule")
    g.add_argument("--drop-invalid-recency", action="store_true",
                   help="remove those rows entirely, before the split")
    g.add_argument("--quality-flags", action="store_true",
                   help="also append flag_fractional_count / flag_count_below_one")

    g = p.add_argument_group("outliers (EDA 3)")
    g.add_argument("--outlier-strategy", choices=["clip", "flag", "none"], default="clip",
                   help="clip bounds the feature space; the heavy tail is never deleted")
    g.add_argument("--outlier-quantiles", type=float, nargs=2, default=(0.0, 1.0),
                   metavar=("LO", "HI"),
                   help="the training quantiles used as bounds; 0 1 means min/max, so no "
                        "training row moves and only unseen data is held in range")
    g.add_argument("--outlier-margin", type=float, default=0.5,
                   help="widen those bounds by this fraction, so the clip bites only on values "
                        "well beyond anything seen in training; 0 disables the margin")

    g = p.add_argument_group("feature engineering (src/feature_engineering.py)")
    g.add_argument("--derived-features", nargs="?", const="default", default="none",
                   metavar="SET",
                   help="engineered columns to add: none | default | independent | all, "
                        "or a comma-separated list of names")
    g.add_argument("--feature-selection", choices=list(SELECTION_STRATEGIES), default="variance",
                   help="how to select among the expanded features; variance is the old "
                        "constant-column pruning, vif is measurably the wrong tool here")
    g.add_argument("--select-k", type=int, default=None,
                   help="how many features the supervised strategies keep")

    g = p.add_argument_group("normalisation (EDA 7, 10)")
    g.add_argument("--no-log", action="store_true", help="skip the log transform (not recommended)")
    g.add_argument("--poly-degree", type=int, default=3,
                   help="polynomial expansion of the logged features; the tuner chose 3")
    g.add_argument("--no-scale", action="store_true", help="skip standardisation")

    g = p.add_argument_group("split and run")
    g.add_argument("--test-size", type=float, default=0.2)
    g.add_argument("--n-strata", type=int, default=10, help="value bins used for stratification")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--no-save", action="store_true", help="run everything but write nothing")
    return p


def add_validation_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The check-side flags, so ``train.py`` can run the same gate before modelling."""
    g = p.add_argument_group("tests and validation")
    g.add_argument("--no-validate", action="store_true",
                   help="skip every check: process the data and write it out")
    g.add_argument("--no-tests", action="store_true",
                   help="skip the transformer behaviour tests (stage 6)")
    g.add_argument("--no-stat-checks", action="store_true",
                   help="skip the cross-validated checks (stage 8), the slowest part")
    g.add_argument("--cv-folds", type=int, default=5,
                   help="folds used by the statistical checks")
    g.add_argument("--no-feature-tests", action="store_true",
                   help="skip the feature-engineering checks (stage 7)")
    g.add_argument("--feature-report", action="store_true",
                   help="print and save per-feature diagnostics")
    g.add_argument("--compare-features", action="store_true",
                   help="measure each feature set against Ridge and a Random Forest "
                        "(a few seconds)")
    return p


def add_baseline_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The baseline ladder's flags (src/models.py)."""
    g = p.add_argument_group("baseline models (src/models.py)")
    g.add_argument("--no-baselines", action="store_true",
                   help="skip the baseline ladder (stage 9)")
    g.add_argument("--task", choices=("auto", "regression", "classification", "clustering"),
                   default="auto",
                   help="which ladder to run; auto reads it off the target")
    g.add_argument("--baseline-poly-degree", type=int, default=1,
                   help="polynomial degree for the baselines' representation; 1 keeps a "
                        "baseline a baseline")
    g.add_argument("--no-benchmark", action="store_true",
                   help="skip the fit-time / throughput / model-size benchmark")
    g.add_argument("--benchmark-repeats", type=int, default=3,
                   help="fits per baseline when timing")
    g.add_argument("--metric-guide", action="store_true",
                   help="print what each metric answers and how it misleads")
    g.add_argument("--baseline-catalogue", action="store_true",
                   help="print the baseline catalogue for every task and exit")
    return p


def add_advanced_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The advanced-architecture flags (src/advanced_models.py)."""
    g = p.add_argument_group("advanced models (src/advanced_models.py)")
    g.add_argument("--no-advanced", action="store_true",
                   help="skip the second and third architectures (stage 10)")
    g.add_argument("--no-report", action="store_true",
                   help="do not write the Markdown comparison report")
    g.add_argument("--architectures", action="store_true",
                   help="print what each advanced architecture assumes, then exit")
    g.add_argument("--tune-advanced", action="store_true",
                   help="give each advanced architecture a bounded randomised search "
                        "before comparing, so a baseline win is not a win over defaults")
    g.add_argument("--search-iter", type=int, default=20,
                   help="draws per architecture when --tune-advanced is on")
    return p


def add_optimization_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The optimization flags (src/model_optimization.py)."""
    g = p.add_argument_group("model optimization (src/model_optimization.py)")
    g.add_argument("--no-optimize", action="store_true",
                   help="skip cross-validated tuning, the over/underfit diagnosis and "
                        "the final selection (stage 11)")
    g.add_argument("--cv-strategy", choices=list(CV_STRATEGIES), default="auto",
                   help="fold scheme; auto stratifies regression folds on value deciles")
    g.add_argument("--search-method", choices=list(SEARCH_METHODS), default="random",
                   help="how the hyperparameter space is explored")
    g.add_argument("--optimize-iter", type=int, default=15,
                   help="draws per model for the optimization stage")
    g.add_argument("--no-learning-curve", action="store_true",
                   help="skip the learning curve of the selected model")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess the capstone CLV dataset, then test and validate the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preprocessing_arguments(p)
    add_validation_arguments(p)
    add_baseline_arguments(p)
    add_advanced_arguments(p)
    add_optimization_arguments(p)
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        numeric_impute=args.numeric_impute,
        knn_neighbors=args.knn_neighbors,
        add_missing_indicators=not args.no_missing_indicators,
        recency_policy=args.recency_policy,
        add_quality_flags=args.quality_flags,
        outlier_strategy=args.outlier_strategy,
        outlier_quantiles=tuple(args.outlier_quantiles),
        outlier_margin=args.outlier_margin,
        log_transform=not args.no_log,
        poly_degree=args.poly_degree,
        scale=not args.no_scale,
        add_derived_features=bool(resolve_derived(args.derived_features)),
        derived_features=args.derived_features,
        feature_selection=args.feature_selection,
        select_k=args.select_k,
        test_size=args.test_size,
        n_strata=args.n_strata,
        random_state=args.seed,
    )


def banner(number: int, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"{number}. {title}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_audit(report: dict) -> None:
    """Show the quality findings that the pipeline's strategies are answering."""
    banner(1, "DATA AUDIT -- what the raw file actually contains")
    print(f"  rows x columns          {report['rows']} x {report['columns']}")
    print(f"  duplicate rows          {report['duplicate_rows']}")
    print(f"  missing cells           {report['missing_cells_total']}"
          f"{'  ' + str(report['missing_by_column']) if report['missing_by_column'] else ''}")

    tgt = report.get("target")
    if tgt:
        print(f"\n  target: min {tgt['min']:,.1f} | median {tgt['median']:,.1f} "
              f"| max {tgt['max']:,.1f}")
        print(f"          skew {tgt['skew']:.2f} -> {tgt['skew_of_log']:.2f} after log  "
              f"(this is why models are fitted on log(value))")
        print(f"          zero-value customers {tgt['zero_value_customers']}  "
              f"(no zeros -> no hurdle/ZILN machinery needed)")
        print(f"          top decile holds {tgt['top_decile_value_share']:.0%} of total value  "
              f"(the tail is the business: guard it, do not delete it)")

    v = report["domain_violations"]
    print("\n  domain-rule violations (EDA 2, 6):")
    print(f"    last purchase before first    {v.get('last_purchase_before_first', 0)}"
          f"   -> impossible recency; handled by --recency-policy")
    print(f"    non-integer purchase counts   {v.get('non_integer_purchase_count', 0)}"
          f"   -> evidence of a generated file; kept as-is")
    print(f"    purchase count below one      {v.get('purchase_count_below_one', 0)}"
          f"   -> log() needs a positivity floor")
    if v.get("non_positive_in_logged_columns"):
        print(f"    non-positive in logged cols   {v['non_positive_in_logged_columns']}")

    print("\n  feature skew (all logged before modelling):")
    for col, skew in report["feature_skew"].items():
        print(f"    {col:32s} {skew:+.2f}")


def print_checks(checks: List[Check]) -> None:
    """One aligned line per check, so a failure is impossible to miss."""
    for check in checks:
        detail = f"  -- {check.detail}" if check.detail else ""
        print(f"  [{check.status:4s}] {check.name}{detail}")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def run_preprocessing(args: argparse.Namespace, config: PreprocessConfig) -> dict:
    """Stages 1-5: audit, filter, split, fit on train only, transform both."""
    raw = load_raw(args.data, target=config.target)
    audit = audit_dataset(raw, target=config.target)
    print_audit(audit)

    clean, removed = filter_rows(
        raw,
        config=config,
        drop_invalid_recency=args.drop_invalid_recency,
        drop_non_positive_target=True,
    )
    banner(2, "ROW FILTERING -- removal is opt-in and always reported")
    print(f"  rows in {len(raw)} -> rows out {len(clean)}"
          f"{('  removed: ' + str(removed)) if removed else '  (nothing removed)'}")
    if not args.drop_invalid_recency:
        n_bad = audit["domain_violations"].get("last_purchase_before_first", 0)
        print(f"  the {n_bad} impossible-recency rows were kept and will be handled in-pipeline"
              f" (policy={config.recency_policy})")

    train_raw, test_raw = stratified_split(clean, config)
    X_train_raw = train_raw.drop(columns=[config.target])
    X_test_raw = test_raw.drop(columns=[config.target])
    y_train, y_test = train_raw[config.target], test_raw[config.target]

    # The target is prepared first: the supervised selection strategies in
    # src/feature_engineering.py are fitted on log(value), like every model here.
    target_tf = LogTargetTransformer().fit(y_train)
    y_train_log = target_tf.transform(y_train)
    y_test_log = target_tf.transform(y_test)

    prep = CLVPreprocessor(config).fit(X_train_raw, y_train_log)
    X_train = prep.transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    banner(3, "PREPROCESSING -- fitted on the training split only")
    report = prep.quality_report()
    print(f"  {len(prep.input_features_)} raw columns -> {prep.n_features_out_} model features")
    print(f"  steps: {' -> '.join(name for name, _ in prep.pipeline_.steps)}")
    print(f"  rows: train {X_train.shape[0]}, test {X_test.shape[0]}")
    n_invalid = report["domain_rules"]["invalid_recency"]
    action = {"flag": "flagged, value left intact", "clip": "REPAIRED (value overwritten)",
              "none": "ignored"}[config.recency_policy]
    print(f"  impossible-recency rows in train: {n_invalid} -> {action}")
    print(f"  missing indicators added: {report['missing_indicators'] or 'none (no missing cells)'}")
    if config.outlier_strategy != "none":
        print(f"\n  outlier bounds learned on the training split (quantiles "
              f"{config.outlier_quantiles}, widened by {config.outlier_margin:.0%}):")
        for col, (lo, hi) in report["outlier_bounds"].items():
            n = report["outlier_values_out_of_range_at_fit"][col]
            raw_lo, raw_hi = report["outlier_raw_bounds"][col]
            print(f"    {col:32s} [{lo:10,.3f}, {hi:12,.3f}]"
                  f"   from [{raw_lo:,.3f}, {raw_hi:,.3f}], {n} train values beyond")
    print(f"\n  first 8 model features: {prep.feature_names_out_[:8]}")

    print_feature_engineering(args, config, prep, X_train, y_train_log, train_raw)

    return {
        "audit": audit, "removed": removed, "prep": prep, "prep_report": report,
        "train_raw": train_raw,
        "X_train_raw": X_train_raw, "X_test_raw": X_test_raw,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_train_log": y_train_log, "y_test_log": y_test_log,
    }


def show(frame: pd.DataFrame, decimals: int = 4, indent: str = "  ") -> None:
    """Print a table indented under its section heading."""
    print("\n".join(indent + line for line in frame.round(decimals).to_string().splitlines()))


def print_feature_engineering(args: argparse.Namespace, config: PreprocessConfig,
                              prep: CLVPreprocessor, X_train: pd.DataFrame,
                              y_train_log, train_raw: pd.DataFrame) -> None:
    """Stage 4: what was created, what was kept, and -- on request -- what it was worth."""
    banner(4, "FEATURE ENGINEERING -- the columns created and the features kept")
    report = prep.quality_report()
    created = report.get("derived_features", [])

    if not created:
        print("  derived features: none")
        print("  The six raw columns are the feature set the EDA validated. Add engineered")
        print("  ones with --derived-features default|independent|all, and measure what")
        print("  they are worth with --compare-features (readme.md 16.3).")
    else:
        print(f"  derived features ({len(created)}), created before the log step:\n")
        print(f"    {'name':22s} {'formula':26s} {'kind':11s} new in log space?")
        for name in created:
            spec = FEATURE_SPECS[name]
            verdict = "yes" if spec.new_in_log_space else "no -- a linear combination of logs"
            print(f"    {spec.name:22s} {spec.formula:26s} {spec.kind:11s} {verdict}")
        print("\n  A product or ratio of logged columns is a weighted sum of columns the model")
        print("  already has, so it cannot help the linear family -- but a tree cannot form a")
        print("  product, so the same column can help the ensembles (--compare-features).")

    selection = report.get("feature_selection")
    if selection:
        dropped = (f", dropped e.g. {selection['dropped'][:3]}"
                   if selection["n_dropped"] else "")
        print(f"\n  selection: {selection['strategy']} -- kept {selection['n_out']} of "
              f"{selection['n_in']} features{dropped}")
        if selection["strategy"] != selection["requested_strategy"]:
            print(f"  (requested {selection['requested_strategy']}; it fell back, see the log)")

    if args.feature_report:
        print("\n  per-feature diagnostics, top 12 by |Spearman| against the target:\n")
        show(feature_report(X_train, y_train_log, top=12))
        print("\n  A high VIF here is the polynomial basis working as designed, not a defect.")

    if args.compare_features:
        print("\n  measured: mean 5-fold CV R2 on log(value) for each feature set\n")
        show(compare_feature_sets(train_raw, config), decimals=5)
        print("\n  Read the two model columns against each other: products and ratios move")
        print("  the Random Forest and leave Ridge where it was, which is exactly what a")
        print("  multiplicative target in log space predicts.")


def run_validation(args: argparse.Namespace, config: PreprocessConfig, data: dict) -> dict:
    """Stages 5-8: the check groups from src/preprocessing.py and src/feature_engineering.py."""
    checks: List[Check] = []

    banner(5, "OUTPUT VALIDATION -- the processed data must be usable and leakage-free")
    contract = check_output_contract(data["X_train"], data["X_test"],
                                     data["y_train"], data["y_test"], data["prep"])
    print_checks(contract)
    checks += contract

    if args.no_tests:
        banner(6, "TRANSFORMER TESTS -- skipped (--no-tests)")
    else:
        banner(6, "TRANSFORMER TESTS -- the same rows, put through the pipeline sideways")
        print("  Each case runs on raw test rows the preprocessor was not fitted on. The")
        print("  first one is the one that matters most: if a single row does not score")
        print("  exactly as it does inside a batch, predict.py disagrees with the")
        print("  evaluation and nothing downstream reveals it.\n")
        behaviour = check_transformer_behaviour(data["prep"], data["X_train_raw"],
                                                data["X_test_raw"], data["y_train_log"],
                                                seed=config.random_state)
        print_checks(behaviour)
        checks += behaviour

    if args.no_feature_tests:
        banner(7, "FEATURE-ENGINEERING TESTS -- skipped (--no-feature-tests)")
    else:
        banner(7, "FEATURE-ENGINEERING TESTS -- the created columns and the selection")
        print("  The third check does not trust the catalogue in src/feature_engineering.py,")
        print("  it measures it: a product or ratio must add no rank to the logged inputs,")
        print("  and a difference or threshold must add exactly one.\n")
        features = check_feature_engineering(data["prep"], data["X_train_raw"],
                                             data["X_test_raw"], data["y_train_log"])
        print_checks(features)
        checks += features

    if args.no_stat_checks:
        banner(8, "STATISTICAL VALIDATION -- skipped (--no-stat-checks)")
    else:
        banner(8, "STATISTICAL VALIDATION -- is this still the pipeline the EDA measured?")
        print(f"  {args.cv_folds}-fold CV with the preprocessor refitted inside every fold.")
        print("  The last check refits it on a shuffled target, where any score above")
        print("  zero would mean information about y had reached the features.\n")
        statistics = check_statistics(data["train_raw"], config,
                                      cv_folds=args.cv_folds, seed=config.random_state)
        print_checks(statistics)
        checks += statistics

    return checks


def print_tally(checks: List[Check]) -> dict:
    """The one tally for the whole run, printed after the last stage that adds to it."""
    summary = summarise(checks)
    print(f"\n  {summary['passed']} passed, {summary['failed']} failed, "
          f"{summary['skipped']} skipped")
    if summary["failed"]:
        print("  FAILED: " + "; ".join(summary["failures"]))
    return summary


def run_baseline_stage(args: argparse.Namespace, config: PreprocessConfig,
                       data: dict) -> BaselineResults:
    """Stage 9: the baseline ladder from src/models.py.

    Baselines run on the **first-order** representation -- logged and scaled, with
    no polynomial expansion -- even when the pipeline is configured for degree 3.
    That is deliberate: the expansion is the modelling choice ``train.py`` is
    supposed to *make*, and a baseline that borrows it is no longer a baseline.
    ``--baseline-poly-degree`` overrides it.
    """
    banner(9, "BASELINE MODELS -- the bar a real model has to clear (readme.md 7.3)")
    baseline_config = replace(config, poly_degree=args.baseline_poly_degree)
    task = args.task if args.task != "auto" else infer_task(data["y_train"])

    print(f"  task: {task} (inferred from the target)" if args.task == "auto"
          else f"  task: {task} (forced with --task)")
    print(f"  representation: log + scale, poly_degree={args.baseline_poly_degree}"
          " -- a baseline may not borrow the expansion that model selection exists to choose")
    print(f"  protocol: {args.cv_folds}-fold out-of-fold predictions on the training split;"
          "\n  naive baselines skip the preprocessor entirely (they ignore X)\n")

    results = evaluate_baselines(
        data["X_train_raw"], data["y_train"], task=task,
        preprocessor=CLVPreprocessor(baseline_config),
        cv=KFold(n_splits=args.cv_folds, shuffle=True, random_state=config.random_state),
        benchmark=not args.no_benchmark, benchmark_repeats=args.benchmark_repeats,
    )

    print("  Leaderboard, sorted by the headline metric for this task:\n")
    show(results.board)

    analysis = results.analysis
    bar = analysis.get("bar_for_a_real_model")
    print(f"\n  best baseline: {results.best}")
    if bar:
        print(f"  the bar: {bar['baseline']} at {analysis['headline_metric']} "
              f"{bar['score']:.4f} -- {bar['model_baselines_beating_it']} of {bar['of']} "
              "model baselines beat it")
    if not analysis.get("metrics_agree", True):
        print(f"  the metrics disagree: {analysis['disagreement']}")
        print("  Ranking and calibration are different questions (readme.md 7.2), so a")
        print("  single sorted column is not a verdict.")

    if results.benchmark is not None:
        print("\n  What each baseline costs (median of"
              f" {args.benchmark_repeats} fits on the full training split):\n")
        show(results.benchmark, decimals=3)

    if args.metric_guide:
        print("\n  What each metric is actually telling you:\n")
        show(metric_guide(task).drop(columns=["direction"]), decimals=3)

    print("\n  Checks:")
    print_checks(results.checks)
    return results


def run_optimization_stage(args: argparse.Namespace, config: PreprocessConfig,
                           data: dict, task: str):
    """Stage 11: cross-validate, tune, diagnose the fit, select, and validate once.

    Every developed model goes in together -- the baselines from ``src/models.py``
    and the two architectures from ``src/advanced_models.py`` -- because a
    protocol applied to only half the field cannot be used to choose between
    them. The held-out split is opened here for the final number, and nowhere
    else in ``main.py``.
    """
    banner(11, "MODEL OPTIMIZATION -- folds, search, fit diagnosis, final choice")
    preprocessor = CLVPreprocessor(replace(config, poly_degree=args.baseline_poly_degree))
    models = {**build_baselines(task, preprocessor=preprocessor),
              **build_advanced_models(task, preprocessor=preprocessor)}
    tunable = [name for name in models if search_space_for(name, task)]
    print(f"  {len(models)} developed models, {len(tunable)} with something to tune")
    print(f"  cross-validation: {args.cv_strategy}   search: {args.search_method}, "
          f"{args.optimize_iter} draws\n")

    results = optimise_models(models, data["X_train_raw"], data["y_train"], task=task,
                              strategy=args.cv_strategy, method=args.search_method,
                              n_iter=args.optimize_iter,
                              learning_curves=not args.no_learning_curve,
                              random_state=config.random_state)

    print(f"  1. FOLDS -- {results.cv!r}\n")
    show(results.cv_folds)

    print("\n  2. SEARCH -- best score per model\n")
    searched = results.tuning[results.tuning["method"] != "none"]
    show(searched[["method", "n_candidates", "cv_score", "cv_score_sd", "best_params"]])

    if len(results.diagnosis):
        print("\n  3. FIT DIAGNOSIS -- training score against cross-validated score\n")
        show(results.diagnosis[["model", "train_score", "cv_score", "gap", "verdict"]]
             .set_index("model"))
        verdicts = results.diagnosis["verdict"].value_counts().to_dict()
        print(f"\n  {verdicts}")
        for _, row in results.diagnosis.iterrows():
            if row["verdict"] != "balanced":
                print(f"    {row['model']}: {row['action']}")

    selection = results.selection
    print(f"\n  4. SELECTION -- one-standard-error rule")
    print(f"     best by score:  {selection['best_by_score']} "
          f"({selection['best_score']:.4f} +/- {selection['standard_error']:.4f})")
    print(f"     within 1 SE:    {', '.join(selection['within_one_se'])}")
    print(f"     chosen:         {results.final_name}, trading "
          f"{selection['traded_accuracy']:+.4f} for a simpler model")

    if len(results.learning_curve):
        print("\n     learning curve for the chosen model:\n")
        show(results.learning_curve)

    validation = validate_final_model(results, data["X_train_raw"], data["y_train"],
                                      data["X_test_raw"], data["y_test"])
    print(f"\n  5. HELD-OUT VALIDATION\n     {validation['verdict']}")

    results.checks = check_optimization(results)
    print("\n  Checks:")
    print_checks(results.checks)
    return results


def run_advanced_stage(args: argparse.Namespace, config: PreprocessConfig,
                       data: dict, baselines: BaselineResults):
    """Stage 10: two further architectures, compared with the ladder fold by fold.

    The advanced models get **the same preprocessing and the same folds** as the
    baselines. Anything else would measure the features or the split rather than
    the architecture, which is the only thing this stage is asking about.
    """
    banner(10, "ADVANCED MODELS -- two more architectures, and whether they earn it")
    task = baselines.task
    for spec in ADVANCED_SPECS:
        if spec.task == task:
            print(f"  {spec.name:22s} {spec.architecture}")
            print(f"  {'':22s} bias: {spec.bias.split('.')[0]}.")
    print(f"  hyperparameters: {'bounded randomised search, ' + str(args.search_iter) + ' draws each' if args.tune_advanced else 'defaults (--tune-advanced to search)'}")
    print()

    report_path = (None if args.no_save or args.no_report
                   else Path(args.outdir) / "advanced_models_report.md")
    results = run_comparison(
        data["X_train_raw"], data["y_train"], task=task,
        preprocessor=CLVPreprocessor(replace(config, poly_degree=args.baseline_poly_degree)),
        cv=KFold(n_splits=args.cv_folds, shuffle=True, random_state=config.random_state),
        baselines=baselines, benchmark_repeats=args.benchmark_repeats,
        tune=args.tune_advanced, n_iter=args.search_iter,
        report_path=report_path,
    )

    print("  Every model from both ladders, same folds, same metrics:\n")
    show(results.combined)

    if len(results.tuning):
        print("\n  What the search chose (scored on the headline metric, not a proxy):\n")
        show(results.tuning)

    if len(results.comparison):
        print("\n  Paired against the best model baseline, fold by fold:\n")
        show(results.comparison[["reference", "delta", "delta_sd", "folds_won",
                                 "p_value", "significant", "fit_time_ratio"]])
        print("\n  'delta' is the mean difference on the SAME folds and 'delta_sd' its")
        print("  spread across them. 'p_value' is the CORRECTED paired t-test: folds share")
        print("  training rows, so a plain t-test would report fold noise as significance.")

    analysis = results.analysis
    if analysis.get("pooling_note"):
        print(f"\n  NOTE: {analysis['pooling_note']}")
    if "error_reduction" in analysis:
        print(f"\n  The leader closes {analysis['error_reduction']:.1%} of the distance "
              "between the reference and a perfect score.")
    print(f"\n  VERDICT: {analysis.get('verdict', 'n/a')}")

    print("\n  Checks:")
    print_checks(results.checks)
    return results


def save_everything(args: argparse.Namespace, config: PreprocessConfig,
                    data: dict, validation: dict,
                    baselines: "BaselineResults | None" = None,
                    advanced=None, optimization=None) -> list:
    """Write the processed data, the fitted pipeline and the reports."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = config.target
    written: list = []

    train_out = data["X_train"].assign(**{target: data["y_train"].to_numpy(),
                                          f"log_{target}": data["y_train_log"]})
    test_out = data["X_test"].assign(**{target: data["y_test"].to_numpy(),
                                        f"log_{target}": data["y_test_log"]})
    train_out.to_csv(outdir / "processed_train.csv", index=False)
    test_out.to_csv(outdir / "processed_test.csv", index=False)
    written += [outdir / "processed_train.csv", outdir / "processed_test.csv"]
    written.append(data["prep"].save(outdir / "preprocessor.joblib"))
    written.append(config.to_json(outdir / "preprocess_config.json"))

    (outdir / "data_audit.json").write_text(json.dumps(data["audit"], indent=2), encoding="utf-8")
    written.append(outdir / "data_audit.json")

    (outdir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, default=str), encoding="utf-8")
    written.append(outdir / "validation_report.json")

    if args.feature_report:
        feature_report(data["X_train"], data["y_train_log"]).to_csv(
            outdir / "feature_report.csv")
        written.append(outdir / "feature_report.csv")
    if args.compare_features:
        compare_feature_sets(data["train_raw"], config).to_csv(
            outdir / "feature_comparison.csv")
        written.append(outdir / "feature_comparison.csv")

    (outdir / "preprocessing_report.json").write_text(
        json.dumps({"config": asdict(config), "rows_removed": data["removed"],
                    "fitted_pipeline": data["prep_report"],
                    "validation": {k: v for k, v in validation.items() if k != "checks"}},
                   indent=2, default=str),
        encoding="utf-8")
    written.append(outdir / "preprocessing_report.json")

    if baselines is not None:
        baselines.board.to_csv(outdir / "baseline_leaderboard.csv")
        written.append(outdir / "baseline_leaderboard.csv")
        if baselines.benchmark is not None:
            baselines.benchmark.to_csv(outdir / "baseline_benchmark.csv")
            written.append(outdir / "baseline_benchmark.csv")
        (outdir / "baseline_report.json").write_text(
            json.dumps(baselines.to_dict(), indent=2, default=str), encoding="utf-8")
        written.append(outdir / "baseline_report.json")

    if advanced is not None:
        advanced.board.to_csv(outdir / "advanced_leaderboard.csv")
        written.append(outdir / "advanced_leaderboard.csv")
        if len(advanced.combined):
            advanced.combined.to_csv(outdir / "model_comparison.csv")
            written.append(outdir / "model_comparison.csv")
        if len(advanced.comparison):
            advanced.comparison.to_csv(outdir / "paired_comparison.csv")
            written.append(outdir / "paired_comparison.csv")
        (outdir / "advanced_report.json").write_text(
            json.dumps(advanced.to_dict(), indent=2, default=str), encoding="utf-8")
        written.append(outdir / "advanced_report.json")
        if not args.no_report:
            written.append(outdir / "advanced_models_report.md")

    if optimization is not None:
        optimization.tuning.to_csv(outdir / "hyperparameter_search.csv")
        written.append(outdir / "hyperparameter_search.csv")
        if len(optimization.diagnosis):
            optimization.diagnosis.to_csv(outdir / "fit_diagnosis.csv", index=False)
            written.append(outdir / "fit_diagnosis.csv")
        if len(optimization.learning_curve):
            optimization.learning_curve.to_csv(outdir / "learning_curve.csv")
            written.append(outdir / "learning_curve.csv")
        (outdir / "optimization_report.json").write_text(
            json.dumps(optimization.to_dict(), indent=2, default=str), encoding="utf-8")
        written.append(outdir / "optimization_report.json")
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def empty_summary() -> dict:
    return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": [], "checks": []}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.baseline_catalogue:
        print("=" * 78)
        print("BASELINE CATALOGUE -- src/models.py")
        print("=" * 78)
        print("  A naive floor learns one number and exists so the metrics can be read.")
        print("  A model baseline is a real but simple estimator -- the bar a complicated")
        print("  model actually has to clear before it is worth its complexity.")
        for task in ("regression", "classification", "clustering"):
            table = spec_table(task)
            print(f"\n{task.upper()}")
            for name, row in table.iterrows():
                print(f"\n  {name}  [{row['kind']}]")
                print(textwrap.fill(row["why it is here"], width=74,
                                    initial_indent="    ", subsequent_indent="    "))
        return 0

    if args.architectures:
        print("=" * 78)
        print("ADVANCED ARCHITECTURES -- src/advanced_models.py")
        print("=" * 78)
        print("  Two per task, chosen to have different inductive biases: when one wins,")
        print("  the bias is what won, and that is a finding rather than a tuning result.")
        for task in ("regression", "classification", "clustering"):
            print(f"\n{task.upper()}")
            for spec in ADVANCED_SPECS:
                if spec.task == task:
                    print(f"\n  {spec.name}")
                    for label, text in (("architecture", spec.architecture),
                                        ("bias", spec.bias), ("cost", spec.cost)):
                        print(textwrap.fill(f"{label}: {text}", width=74,
                                            initial_indent="    ",
                                            subsequent_indent="      "))
        return 0

    config = config_from_args(args)
    outdir = Path(args.outdir)

    print("=" * 78)
    print("CAPSTONE -- CLV DATA PREPROCESSING PIPELINE")
    print("=" * 78)
    print(f"  data    {args.data}")
    print(f"  outdir  {outdir if not args.no_save else '(not saving)'}")
    print(f"  stages  {'process' if args.no_validate else 'process -> validate -> test'}")

    data = run_preprocessing(args, config)
    checks = [] if args.no_validate else run_validation(args, config, data)

    baselines = None
    advanced = None
    if not args.no_baselines:
        baselines = run_baseline_stage(args, config, data)
        checks += baselines.checks

        if not args.no_advanced:
            advanced = run_advanced_stage(args, config, data, baselines)
            checks += advanced.checks

    optimization = None
    if not args.no_optimize:
        optimization = run_optimization_stage(args, config, data,
                                              baselines.task if baselines else
                                              infer_task(data["y_train"]))
        checks += optimization.checks

    validation = print_tally(checks) if checks else empty_summary()

    if not args.no_save:
        written = save_everything(args, config, data, validation, baselines, advanced,
                                  optimization)
        banner(12, "ARTIFACTS")
        for path in written:
            print(f"  {path}")
        print("\n  Reuse on new customers:")
        print("    from src.preprocessing import CLVPreprocessor")
        print(f"    prep = CLVPreprocessor.load(r'{outdir / 'preprocessor.joblib'}')")
        print("    X_new = prep.transform(new_customers_df)")

    if validation["failed"]:
        print(f"\nFAILED -- {validation['failed']} of {validation['total']} checks did not pass.")
        print("Nothing was written (--no-save)." if args.no_save else
              "The processed data was still written; fix the failures before modelling.")
        return 1

    print("\nDone. The test split is transformed but must stay untouched until a final")
    print("model is scored once (EDA 8) -- that is what train.py does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
