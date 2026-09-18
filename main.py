"""Entry point: run the CLV preprocessing pipeline end to end.

    python main.py                          # defaults, reproduces the EDA set-up
    python main.py --poly-degree 1          # plain log-linear representation
    python main.py --outlier-strategy flag  # keep raw values, append 0/1 columns
    python main.py --drop-invalid-recency   # remove the 52 impossible rows instead
    python main.py --no-validate            # skip the smoke test

What it does, in order:

    1. load the raw CSV
    2. audit it -- the EDA 2 quality checks, as a JSON report
    3. row-level filtering (off by default; see src/preprocessing.filter_rows)
    4. split 80/20, stratified on value deciles (EDA 8)
    5. fit the preprocessor on the TRAINING split only, transform both splits
    6. assert the outputs are usable and leakage-free
    7. optional smoke test: Ridge on the processed features, 5-fold CV
    8. write the processed data, the fitted pipeline and the reports to outputs/

The test split is transformed but must not be looked at again until a final
model is evaluated once (EDA 8).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    CLVPreprocessor,
    LogTargetTransformer,
    PreprocessConfig,
    audit_dataset,
    filter_rows,
    get_logger,
    load_raw,
    quiet,
    stratified_split,
)

log = get_logger("capstone")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess the capstone CLV dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
                   help="training quantiles used as the bounds; 0 1 means min/max, so no "
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
    g.add_argument("--no-validate", action="store_true", help="skip the Ridge smoke test")
    g.add_argument("--no-save", action="store_true", help="run everything but write nothing")
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


# --------------------------------------------------------------------------- #
# Reporting and checks
# --------------------------------------------------------------------------- #


def print_audit(report: dict) -> None:
    """Show the quality findings that the pipeline's strategies are answering."""
    print("\n" + "=" * 78)
    print("1. DATA AUDIT -- what the raw file actually contains")
    print("=" * 78)
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


def validate_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    prep: CLVPreprocessor,
) -> dict:
    """Fail loudly on anything that would quietly break the modelling step."""
    checks: dict = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = bool(ok)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
        if not ok:
            raise AssertionError(f"pipeline validation failed: {name} {detail}")

    print("\n" + "=" * 78)
    print("4. VALIDATION -- the pipeline output must be usable and leakage-free")
    print("=" * 78)

    check("column parity between splits", list(X_train.columns) == list(X_test.columns),
          f"{X_train.shape[1]} features")
    check("row parity with targets",
          len(X_train) == len(y_train) and len(X_test) == len(y_test),
          f"train {len(X_train)}, test {len(X_test)}")
    check("no missing values after imputation",
          not X_train.isna().any().any() and not X_test.isna().any().any())
    check("all values finite (no -inf from log)",
          bool(np.isfinite(X_train.to_numpy()).all() and np.isfinite(X_test.to_numpy()).all()))
    check("no constant columns", int((X_train.std(ddof=0) == 0).sum()) == 0,
          f"{int((X_train.std(ddof=0) == 0).sum())} constant")

    if prep.config_.scale:
        # The scaler is fitted on train, so train is standardised exactly and test
        # is merely close. A test mean far from 0 would mean the splits differ.
        check("train features standardised",
              bool(np.allclose(X_train.mean(), 0, atol=1e-8)
                   and np.allclose(X_train.std(ddof=0), 1, atol=1e-8)))
        check("test features not re-standardised (no leakage)",
              not np.allclose(X_test.mean(), 0, atol=1e-8),
              f"test mean in [{X_test.mean().min():+.3f}, {X_test.mean().max():+.3f}]")

    check("target strictly positive (log is defined)",
          bool((y_train > 0).all() and (y_test > 0).all()))
    return checks


def smoke_test(train_raw: pd.DataFrame, config: PreprocessConfig) -> dict:
    """Cross-validate a Ridge on top of the preprocessor, inside the folds.

    This is a pipeline check, not a modelling result. Because the preprocessor is
    a real sklearn transformer, every fold refits the imputation fills, the clip
    bounds and the scaler -- so the score is honest and any leakage would show up
    as a gap against the EDA's numbers. Degree 3 should land near the EDA's
    CV R2 of ~0.9998 on log(value) (EDA 10).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    from src.preprocessing import TARGET

    X = train_raw.drop(columns=[TARGET])
    y_log = np.log(train_raw[TARGET].to_numpy())

    pipe = Pipeline([("prep", CLVPreprocessor(config)), ("ridge", Ridge(alpha=0.01))])
    cv = KFold(n_splits=5, shuffle=True, random_state=config.random_state)
    with quiet():  # five folds x a full refit would otherwise flood the report
        oof_log = cross_val_predict(pipe, X, y_log, cv=cv)

    residual = y_log - oof_log
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
    r2_log = 1 - ss_res / ss_tot

    # Duan's smearing on the out-of-fold residuals, then back to dollars (EDA 8).
    target_tf = LogTargetTransformer().fit(train_raw[TARGET])
    target_tf.fit_smearing(residual)
    pred_raw = target_tf.inverse_transform(oof_log)
    actual = train_raw[TARGET].to_numpy()

    spearman = float(pd.Series(pred_raw).corr(pd.Series(actual), method="spearman"))
    mae = float(np.mean(np.abs(pred_raw - actual)))

    result = {
        "cv_r2_log": r2_log,
        "cv_spearman": spearman,
        "cv_mae_raw": mae,
        "residual_std_log": float(residual.std()),
        "smearing_factor": target_tf.smearing_factor_,
    }

    print("\n" + "=" * 78)
    print("5. SMOKE TEST -- Ridge(alpha=0.01) on the processed features, 5-fold CV")
    print("=" * 78)
    print(f"  CV R2 on log(value)     {r2_log:.5f}")
    print(f"  CV Spearman             {spearman:.5f}")
    print(f"  CV MAE (dollars)        {mae:,.2f}")
    print(f"  residual spread (log)   {residual.std():.4f}"
          f"  ~ {100 * (np.exp(residual.std()) - 1):.1f}% around the fit")
    print(f"  Duan smearing factor    {target_tf.smearing_factor_:.4f}")
    print("\n  The preprocessor is refitted inside every fold, so this score carries no")
    print("  leakage. It measures the representation, not customer behaviour: EDA 7 showed")
    print("  the target is a deterministic formula in these features, so a near-perfect")
    print("  score here confirms the pipeline works -- it is not a forecasting result.")
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    outdir = Path(args.outdir)

    print("=" * 78)
    print("CAPSTONE -- CLV DATA PREPROCESSING PIPELINE")
    print("=" * 78)
    print(f"  data    {args.data}")
    print(f"  outdir  {outdir if not args.no_save else '(not saving)'}")

    # 1-2. load and audit ---------------------------------------------------- #
    raw = load_raw(args.data, target=config.target)
    audit = audit_dataset(raw, target=config.target)
    print_audit(audit)

    # 3. row-level filtering ------------------------------------------------- #
    clean, removed = filter_rows(
        raw,
        config=config,
        drop_invalid_recency=args.drop_invalid_recency,
        drop_non_positive_target=True,
    )
    print("\n" + "=" * 78)
    print("2. ROW FILTERING -- removal is opt-in and always reported")
    print("=" * 78)
    print(f"  rows in {len(raw)} -> rows out {len(clean)}"
          f"{('  removed: ' + str(removed)) if removed else '  (nothing removed)'}")
    if not args.drop_invalid_recency:
        n_bad = audit["domain_violations"].get("last_purchase_before_first", 0)
        print(f"  the {n_bad} impossible-recency rows were kept and will be handled in-pipeline"
              f" (policy={config.recency_policy})")

    # 4. split --------------------------------------------------------------- #
    train_raw, test_raw = stratified_split(clean, config)
    X_train_raw = train_raw.drop(columns=[config.target])
    X_test_raw = test_raw.drop(columns=[config.target])
    y_train, y_test = train_raw[config.target], test_raw[config.target]

    # 5. fit on train only, transform both ----------------------------------- #
    prep = CLVPreprocessor(config).fit(X_train_raw)
    X_train = prep.transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    target_tf = LogTargetTransformer().fit(y_train)
    y_train_log = target_tf.transform(y_train)
    y_test_log = target_tf.transform(y_test)

    print("\n" + "=" * 78)
    print("3. PREPROCESSING -- fitted on the training split only")
    print("=" * 78)
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

    # 6. validation ---------------------------------------------------------- #
    checks = validate_outputs(X_train, X_test, y_train, y_test, prep)

    # 7. smoke test ---------------------------------------------------------- #
    smoke = smoke_test(train_raw, config) if not args.no_validate else None

    # 8. persist ------------------------------------------------------------- #
    if not args.no_save:
        outdir.mkdir(parents=True, exist_ok=True)
        train_out = X_train.assign(**{config.target: y_train.to_numpy(),
                                      f"log_{config.target}": y_train_log})
        test_out = X_test.assign(**{config.target: y_test.to_numpy(),
                                    f"log_{config.target}": y_test_log})
        train_out.to_csv(outdir / "processed_train.csv", index=False)
        test_out.to_csv(outdir / "processed_test.csv", index=False)
        prep.save(outdir / "preprocessor.joblib")
        config.to_json(outdir / "preprocess_config.json")
        (outdir / "data_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8")
        (outdir / "preprocessing_report.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "rows_removed": removed,
                    "fitted_pipeline": report,
                    "validation": checks,
                    "smoke_test": smoke,
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("6. ARTIFACTS")
        print("=" * 78)
        for name in ("processed_train.csv", "processed_test.csv", "preprocessor.joblib",
                     "preprocess_config.json", "data_audit.json", "preprocessing_report.json"):
            print(f"  {outdir / name}")
        print("\n  Reuse on new customers:")
        print("    from src.preprocessing import CLVPreprocessor")
        print(f"    prep = CLVPreprocessor.load(r'{outdir / 'preprocessor.joblib'}')")
        print("    X_new = prep.transform(new_customers_df)")

    print("\nDone. The test split is transformed but must stay untouched until a final")
    print("model is scored once (EDA 8).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
