# Advanced models vs the baseline ladder (regression)

Headline metric: **spearman** -- Are the rows ranked in the right order?

> **Verdict.** No advanced architecture beat ridge on log(y) at their default settings: the best of them (gradient boosting) is 0.0297 behind, and the gap is significant (corrected paired t-test p = 0.0411), at 7.3x the fit time. Keep the baseline.


## 1. The architectures compared

| Model | Architecture | Inductive bias | Cost |
|---|---|---|---|
| gradient boosting | additive ensemble: shallow trees fitted in sequence on the residual | Piecewise-constant and axis-aligned. Assumes the response is built from thresholds and interactions rather than smooth curvature, so it excels on rules and struggles to represent a product of continuous features. | Hundreds of trees to store and traverse; fast to fit on tabular data; many hyperparameters, and an optional dependency when XGBoost is used. |
| neural network (MLP) | dense feed-forward network, two hidden layers (64, 32), ReLU | A smooth, continuously differentiable surface with no axis alignment. Assumes enough data to learn the shape and inputs on a comparable scale; in exchange it bends in every direction at once. | Iterative fitting with early stopping, sensitive to scaling and to the seed; small to store, and the least interpretable model here. |

## 2. Every model, same folds, same metrics

| model | kind | spearman | norm_gini | top_decile_capture | decile_mape | mae_raw | r2_raw | r2_log | rmse_log |
|---|---|---|---|---|---|---|---|---|---|
| ridge on log(y) | model | 0.9929 | 0.9949 | 0.3694 | 0.0196 | 75.0303 | 0.965 | 0.9866 | 0.0994 |
| linear regression on log(y) | model | 0.9929 | 0.9949 | 0.3694 | 0.0191 | 75.1294 | 0.965 | 0.9866 | 0.0994 |
| k-NN (k=10) | model | 0.9693 | 0.9793 | 0.3643 | 0.0883 | 165.699 | 0.7235 | 0.9206 | 0.2417 |
| linear regression | model | 0.9675 | 0.9805 | 0.3661 | 0.7845 | 356.2616 | 0.562 | nan | nan |
| gradient boosting | advanced | 0.9654 | 0.9712 | 0.3632 | 0.042 | 166.151 | 0.7802 | nan | nan |
| neural network (MLP) | advanced | 0.9307 | 0.9574 | 0.3569 | 0.1691 | 198.7817 | 0.7455 | 0.6764 | 0.4881 |
| heuristic (product of two columns) | naive | 0.8771 | 0.9116 | 0.3401 | 0.81 | 800.2403 | -29.2574 | -3.0229 | 1.7211 |
| decision tree (depth 3) | model | 0.6984 | 0.8012 | 0.3123 | 0.0551 | 355.7987 | 0.5355 | 0.5592 | 0.5697 |
| mean | naive | -0.0411 | -0.1029 | 0.106 | 0.143 | 570.791 | -0.0026 | -0.2299 | 0.9516 |
| median | naive | -0.0758 | -0.0929 | 0.1013 | 0.3265 | 510.5304 | -0.0635 | -0.0039 | 0.8597 |

## 3. Paired comparison against the best model baseline

Each row compares the advanced model with the reference **on the same folds**, so the difference is paired rather than two separate averages.

| model | score | reference | reference_score | fold_mean | reference_fold_mean | delta | delta_sd | folds_won | folds | reliable | p_value | significant | fit_time_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gradient boosting | 0.9654 | ridge on log(y) | 0.9929 | 0.9616 | 0.9913 | -0.0297 | 0.0133 | 0 | 5 | True | 0.0411 | True | 7.2707 |
| neural network (MLP) | 0.9307 | ridge on log(y) | 0.9929 | 0.9425 | 0.9913 | -0.0488 | 0.0416 | 0 | 5 | True | 0.1932 | False | 5.6051 |

## 4. What the advanced models cost

| model | fit_ms | predict_us_per_row | model_kb |
|---|---|---|---|
| neural network (MLP) | 138.66 | 10.222 | 78.698 |
| gradient boosting | 179.865 | 17.107 | 310.98 |

## 5. Fold-level spread

|  | mean | sd | min | max |
|---|---|---|---|---|
| gradient boosting | 0.9616 | 0.0126 | 0.9441 | 0.9744 |
| neural network (MLP) | 0.9425 | 0.0429 | 0.8808 | 0.9885 |

A model whose spread across folds is wider than its lead over the reference has not been shown to be better.

## 6. Checks

* `PASS` both advanced architectures ran -- ['gradient boosting', 'neural network (MLP)']
* `PASS` predictions are finite -- 2 prediction sets checked
* `PASS` every model has a score for each fold -- {'gradient boosting': 5, 'neural network (MLP)': 5}
* `PASS` the reference is a model baseline, not a naive one -- compared against ridge on log(y)
* `PASS` the paired difference is consistent with the leaderboard -- delta -0.0297 = fold mean - reference fold mean
* `PASS` every advanced model beats the naive floor -- floor 0.8771, worst advanced 0.9307
* `PASS` the comparison carries a significance test -- corrected paired t-test p = 0.0411 (significant)
* `PASS` the verdict states the cost as well as the gain -- No advanced architecture beat ridge on log(y) at their default setting...
