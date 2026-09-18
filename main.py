"""Entry point: process the data, then test and validate what came out.

    python main.py                          # defaults, reproduces the EDA set-up
    python main.py --poly-degree 1          # plain log-linear representation
    python main.py --outlier-strategy flag  # keep raw values, append 0/1 columns
    python main.py --drop-invalid-recency   # remove the 52 impossible rows instead
    python main.py --no-stat-checks         # skip the slow cross-validated checks
    python main.py --no-validate            # process only, check nothing

What it does, in order:

    1. load the raw CSV
    2. audit it -- the EDA 2 quality checks, as a JSON report
    3. row-level filtering (off by default; see src/preprocessing.filter_rows)
    4. split 80/20, stratified on value deciles (EDA 8)
    5. fit the preprocessor on the TRAINING split only, transform both splits
    6. validate the output contract: usable, finite, leakage-free
    7. test the transformer itself: single row == batch, order and index
       invariance, dirty-data edge cases, save/load and clone round-trips
    8. validate statistically: feature count per degree, the CV R2 the EDA
       measured, and a label-shuffle test whose score must collapse to zero
    9. write the processed data, the fitted pipeline and the reports to outputs/

Checks are collected, not raised, so one run reports every failure -- and the
process exits non-zero if anything failed, which makes this usable as a gate.

The test split is transformed but must not be looked at again until a final model
is evaluated once (EDA 8). Modelling lives in ``train.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

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

    g = p.add_argument_group("normalisation (EDA 7, 10)")
    g.add_argument("--no-log", action="store_true", help="skip the log transform (not recommended)")
    g.add_argument("--poly-degree", type=int, default=3,
                   help="polynomial expansion of the logged features; the tuner chose 3")
    g.add_argument("--no-scale", action="store_true", help="skip standardisation")
    g.add_argument("--derived-features", action="store_true",
                   help="add purchase_value and recency_span before the log step")

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
                   help="skip the transformer behaviour tests (stage 5)")
    g.add_argument("--no-stat-checks", action="store_true",
                   help="skip the cross-validated checks (stage 6), the slowest part")
    g.add_argument("--cv-folds", type=int, default=5,
                   help="folds used by the statistical checks")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess the capstone CLV dataset, then test and validate the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preprocessing_arguments(p)
    add_validation_arguments(p)
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
        add_derived_features=args.derived_features,
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

    prep = CLVPreprocessor(config).fit(X_train_raw)
    X_train = prep.transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    target_tf = LogTargetTransformer().fit(y_train)
    y_train_log = target_tf.transform(y_train)
    y_test_log = target_tf.transform(y_test)

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

    return {
        "audit": audit, "removed": removed, "prep": prep, "prep_report": report,
        "train_raw": train_raw,
        "X_train_raw": X_train_raw, "X_test_raw": X_test_raw,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_train_log": y_train_log, "y_test_log": y_test_log,
    }


def run_validation(args: argparse.Namespace, config: PreprocessConfig, data: dict) -> dict:
    """Stages 6-8: the three groups of checks defined in src/preprocessing.py."""
    checks: List[Check] = []

    banner(4, "OUTPUT VALIDATION -- the processed data must be usable and leakage-free")
    contract = check_output_contract(data["X_train"], data["X_test"],
                                     data["y_train"], data["y_test"], data["prep"])
    print_checks(contract)
    checks += contract

    if args.no_tests:
        banner(5, "TRANSFORMER TESTS -- skipped (--no-tests)")
    else:
        banner(5, "TRANSFORMER TESTS -- the same rows, put through the pipeline sideways")
        print("  Each case runs on raw test rows the preprocessor was not fitted on. The")
        print("  first one is the one that matters most: if a single row does not score")
        print("  exactly as it does inside a batch, predict.py disagrees with the")
        print("  evaluation and nothing downstream reveals it.\n")
        behaviour = check_transformer_behaviour(data["prep"], data["X_train_raw"],
                                                data["X_test_raw"], seed=config.random_state)
        print_checks(behaviour)
        checks += behaviour

    if args.no_stat_checks:
        banner(6, "STATISTICAL VALIDATION -- skipped (--no-stat-checks)")
    else:
        banner(6, "STATISTICAL VALIDATION -- is this still the pipeline the EDA measured?")
        print(f"  {args.cv_folds}-fold CV with the preprocessor refitted inside every fold.")
        print("  The last check refits it on a shuffled target, where any score above")
        print("  zero would mean information about y had reached the features.\n")
        statistics = check_statistics(data["train_raw"], config,
                                      cv_folds=args.cv_folds, seed=config.random_state)
        print_checks(statistics)
        checks += statistics

    summary = summarise(checks)
    print(f"\n  {summary['passed']} passed, {summary['failed']} failed, "
          f"{summary['skipped']} skipped")
    if summary["failed"]:
        print("  FAILED: " + "; ".join(summary["failures"]))
    return summary


def save_everything(args: argparse.Namespace, config: PreprocessConfig,
                    data: dict, validation: dict) -> list:
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

    (outdir / "preprocessing_report.json").write_text(
        json.dumps({"config": asdict(config), "rows_removed": data["removed"],
                    "fitted_pipeline": data["prep_report"],
                    "validation": {k: v for k, v in validation.items() if k != "checks"}},
                   indent=2, default=str),
        encoding="utf-8")
    written.append(outdir / "preprocessing_report.json")
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def empty_summary() -> dict:
    return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": [], "checks": []}


def main(argv=None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    outdir = Path(args.outdir)

    print("=" * 78)
    print("CAPSTONE -- CLV DATA PREPROCESSING PIPELINE")
    print("=" * 78)
    print(f"  data    {args.data}")
    print(f"  outdir  {outdir if not args.no_save else '(not saving)'}")
    print(f"  stages  {'process' if args.no_validate else 'process -> validate -> test'}")

    data = run_preprocessing(args, config)
    validation = empty_summary() if args.no_validate else run_validation(args, config, data)

    if not args.no_save:
        written = save_everything(args, config, data, validation)
        banner(7, "ARTIFACTS")
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
