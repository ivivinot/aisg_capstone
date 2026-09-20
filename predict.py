"""Score new customers with the model `train.py` fitted.

    python predict.py --input new_customers.csv --output scored.csv
    python predict.py --input new_customers.csv --top 20        # print, do not write

The input needs the six raw customer columns -- the same ones the training file
has, in any order, with any extra columns carried through untouched::

    total_purchase_count, average_order_value, days_since_first_purchase,
    days_since_last_purchase, product_category_diversity,
    loyalty_program_membership

Everything else travels inside ``outputs/model.joblib``: the fitted preprocessor
(imputation fills, clip bounds, scaler, polynomial expansion), the estimator, the
Duan smearing factor, and the tier thresholds in dollars.

**Tier thresholds are the ones learned at training time**, not quantiles of the
batch being scored. Re-deriving them per batch would mean a customer's tier
depended on who else happened to be scored that day, and "VIP" has to mean the
same thing every run for a retention budget to be attached to it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import load_bundle  # noqa: E402
from src.preprocessing import get_logger  # noqa: E402

log = get_logger("capstone.predict")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score customers with the fitted CLV model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="CSV of customers to score")
    p.add_argument("--output", default=None, help="where to write the scored CSV")
    p.add_argument("--model", default=str(ROOT / "outputs" / "model.joblib"),
                   help="bundle written by train.py")
    p.add_argument("--top", type=int, default=10,
                   help="how many of the highest-value customers to print")
    return p.parse_args(argv)


def assign_tiers(predictions: np.ndarray, thresholds: dict, labels: list) -> pd.Categorical:
    """Map dollar predictions onto the training-time tier boundaries."""
    edges = [-np.inf] + [thresholds[str(q)] for q in (0.5, 0.8, 0.95)] + [np.inf]
    return pd.cut(predictions, bins=edges, labels=labels, include_lowest=True)


def score(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Raw customer rows in, the same rows plus a prediction and a tier out."""
    features = list(bundle["input_features"])
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    predictions = bundle["model"].predict(frame[features])
    out = frame.copy()
    out["predicted_clv"] = predictions
    out["tier"] = assign_tiers(predictions, bundle["train_prediction_quantiles"],
                               list(bundle["tier_labels"]))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    bundle = load_bundle(args.model)
    frame = pd.read_csv(args.input)
    log.info("scoring %d customers with %s", len(frame), bundle["name"])

    scored = score(frame, bundle)

    print("=" * 78)
    print("CLV PREDICTIONS")
    print("=" * 78)
    print(f"  model        {bundle['name']}  (test Spearman "
          f"{bundle['test_scorecard']['spearman']:.4f})")
    print(f"  customers    {len(scored)}")
    print(f"  predicted    median {scored['predicted_clv'].median():,.0f}  "
          f"mean {scored['predicted_clv'].mean():,.0f}  "
          f"total {scored['predicted_clv'].sum():,.0f}")

    counts = scored["tier"].value_counts().reindex(bundle["tier_labels"], fill_value=0)
    value = scored.groupby("tier", observed=False)["predicted_clv"].sum()
    print("\n  tier breakdown (thresholds fixed at training time):")
    for label in bundle["tier_labels"]:
        share = value.get(label, 0.0) / max(scored["predicted_clv"].sum(), 1e-9)
        print(f"    {label:24s} {counts[label]:5d} customers   {share:6.1%} of predicted value")

    if args.top:
        columns = [c for c in ("customer_id", *bundle["input_features"]) if c in scored.columns]
        top = scored.nlargest(args.top, "predicted_clv")[columns + ["predicted_clv", "tier"]]
        print(f"\n  top {args.top} customers by predicted value:")
        print("\n".join("  " + line for line in top.round(2).to_string().splitlines()))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        print(f"\n  written -> {args.output}")

    print("\nThese are predictions of the dataset's `estimated_lifetime_value` column,")
    print("which is itself an estimate, not observed future spend (readme.md 9).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
