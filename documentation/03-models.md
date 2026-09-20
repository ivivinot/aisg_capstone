# 3. Models — what each one assumes, and when it wins

Three tiers, deliberately: a **floor** that makes metrics readable, **simple models** that a
complicated one must beat, and **distinct architectures** that test whether a different assumption
about the data pays. Every tier is evaluated on the same folds with the same metrics.

## Tier 1 — naive floors

They learn one number. Their job is to make every other row interpretable.

| Task | Baseline | What it does | Why it is there |
|---|---|---|---|
| regression | mean | one number for everyone | R² ≈ 0 by construction; out of fold each fold predicts its own training mean, so Spearman is noise around zero |
| | median | the robust twin | beats the mean on MAE and loses on R², which is the first sign that a metric choice is a business choice |
| | **heuristic** (`purchases × AOV`) | multiply two columns, rescale | **the real bar**: Spearman 0.877 on this data. Beating only the mean proves nothing |
| classification | most-frequent | the majority class | its accuracy *is* the majority share — why accuracy is the wrong headline on imbalanced data |
| | stratified / uniform random | draws from the prior / ignores it | separates "learned the prior" from "learned nothing"; AUC ≈ 0.5 whatever the imbalance |
| clustering | one cluster | no structure at all | internal indices are *undefined* here, not zero — a useful reminder that they are relative |
| | random labels | random assignment at the same k | a silhouette of 0.25 means nothing until you know random scores ≈ 0 |

Naive baselines **bypass the preprocessor** (`raw_input=True`). A constant predictor ignores `X`,
so preprocessing it would only make the benchmark measure the preprocessor; and the heuristic is
defined on the raw columns — fed a logged, expanded matrix it silently degrades to the mean.

## Tier 2 — model baselines

Real estimators, deliberately simple, at fixed readable settings.

| Task | Models |
|---|---|
| regression | linear; **linear on log(y)** with Duan smearing; ridge on log(y); decision tree (depth 3); k-NN (k=10) |
| classification | logistic regression; decision tree (depth 3); k-NN (k=10) |
| clustering | k-means; agglomerative (ward) |

On the capstone data these are the story:

| Baseline | Spearman | Decile MAPE | MAE |
|---|---|---|---|
| ridge on log(y) | **0.9925** | 0.0196 | $75 |
| linear on log(y) | 0.9923 | **0.0191** | $75 |
| k-NN (k=10) | 0.9714 | 0.0883 | $166 |
| linear (raw target) | 0.9675 | 0.7845 | $356 |
| heuristic | 0.8771 | 0.8100 | $800 |
| decision tree (depth 3) | 0.9388 | 0.0551 | $356 |

**One line of preprocessing beats every estimator swap.** Linear regression on `log(y)` scores
0.9923 against 0.9675 for the same estimator on the raw target, and its decile MAPE is 41× better.

## Tier 3 — two distinct architectures

Adding a second gradient booster to a ladder that has one measures hyperparameters. These are
chosen for *different inductive biases*, so a win says something.

| Task | Model 2 | Model 3 |
|---|---|---|
| regression / classification | **Gradient boosting** — additive ensemble of shallow trees fitted in sequence on the residual. Piecewise-constant, axis-aligned; excellent at thresholds and interactions, clumsy on a smooth product. | **Neural network (MLP)** — dense feed-forward (64, 32), ReLU. A smooth, continuously differentiable surface with no axis alignment; needs scaled inputs and more data. |
| clustering | **Gaussian mixture** — elliptical components with full covariance, *soft* membership: a point can be 70% one cluster and 30% another. | **DBSCAN** — density-based, no k, arbitrary shapes, and the only model here allowed to call a point *noise*. |

Implementation note: gradient boosting is XGBoost when installed and scikit-learn's
`HistGradientBoosting` otherwise — same architecture, different parameter names, which the tuner
translates (`n_estimators` → `max_iter`) and reports.

### The result, and why it is the project's central finding

| Model | delta vs ridge | fold sd | Folds won | p (corrected) | Fit cost |
|---|---|---|---|---|---|
| gradient boosting | **−0.0088** | 0.0023 | 0 of 5 | **0.0070** | 3.5× |
| neural network (MLP) | −0.0488 | 0.0416 | 0 of 5 | 0.1932 | 5.3× |

Tuned (10 draws each), boosting closes to −0.0076 with p = 0.0000 — the gap narrows and becomes
*more* certain, because the search bought consistency rather than accuracy.

> **Verdict.** No advanced architecture beat ridge on log(y): the best of them is 0.0088 behind,
> and the gap is significant. Keep the baseline.

A target that is a smooth product of powers is exactly what axis-aligned steps cannot represent
and what a linear model on logged features represents *exactly*. The MLP, which can bend smoothly,
still loses because it has to **learn** the shape that the log transform hands the linear model for
free.

## How the comparison is made honest

**Paired, not parallel.** Per-fold scores for every model, differenced on the *same* folds.

**A corrected significance test.** CV folds share training rows, so their differences are
correlated and a plain paired t-test is anti-conservative — it reports fold noise as significance.
`paired_test()` uses the corrected resampled t-test (Nadeau & Bengio), inflating the variance by
`1/k + 1/(k−1)`. Wilcoxon is reported beside it with the caveat that at five folds it cannot go
below 0.0625 whatever the data does.

**Cost beside accuracy.** Fit time, prediction throughput and serialised size. The naive rows fit
~250× faster than ridge, and almost all of ridge's 26 ms is preprocessing.

## Optimization: over- and underfitting

Every model's training score is compared with its cross-validated one, and the gap is named:

| Model | Train | CV | Gap | Verdict |
|---|---|---|---|---|
| linear on log(y) | 0.9931 | 0.9929 | 0.0001 | balanced |
| gradient boosting | 0.9969 | 0.9854 | 0.0115 | balanced |
| **k-NN (k=10)** | **1.0000** | 0.9722 | 0.0278 | balanced |
| **neural network (MLP)** | 0.9964 | 0.9351 | **0.0613** | **overfitting** |
| mean, median | n/a | < 0 | — | underfitting |

k-NN scores a perfect 1.0000 on its own training rows — memorisation the training score alone
would have hidden entirely. Each verdict carries its action: *reduce effective capacity* for the
MLP, *add capacity* for the floors, where on this project "capacity" means a better representation,
not a bigger estimator.

**Selection** then applies the one-standard-error rule — among models within one SE of the best,
take the simplest:

```
best by score:  ridge on log(y)              0.9925 ± 0.0006
within 1 SE:    linear regression on log(y), ridge on log(y)
chosen:         linear regression on log(y), trading +0.0002 for a simpler model
```

**Validation** opens the held-out split once: cross-validated 0.9923, held-out 0.9917,
**optimism +0.0006**. Optimism — not the test score — is the number to read, because tuning and
selection both consumed the folds.

## Choosing a model for a new problem

| If your data looks like | Reach for | Because |
|---|---|---|
| a product of drivers, heavy right tail | linear on **logged** features | logs make the product additive; the estimator barely matters |
| thresholds, rules, interactions | gradient boosting | axis-aligned splits are exactly the right shape |
| smooth but not multiplicative, plenty of rows | MLP | bends in every direction; needs the data to pay for it |
| segments of similar size and shape | k-means | fast, and the silhouette is meaningful |
| overlapping or elliptical segments | Gaussian mixture | soft membership, full covariance |
| unknown k, arbitrary shapes, outliers to exclude | DBSCAN | density, and it can say "no segment" |

And the standing advice this project earns: **fix the representation before reaching for a bigger
estimator.** On this dataset that was worth an order of magnitude in error; the architecture change
was worth nothing at all.
