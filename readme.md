# Capstone — 12-Month Customer Lifetime Value Prediction for E-commerce

**Objective.** Predict the value a customer will generate over the **next 12 months** in a
non-contractual e-commerce setting, and use that prediction to target marketing spend.

This document is the **literature review and research synthesis** for the capstone: what the
field has established, which methods win in which data regime, how such models are evaluated,
where they usually go wrong, and how all of that maps onto the dataset in `data/`.

> **Status.** Research synthesis only. `eda.ipynb` is still empty, and §9 explains why the
> dataset as supplied cannot support a genuine 12-month forecast without a change of scope.

---

## Contents

1. [Defining the problem](#1-defining-the-problem)
2. [Why 12-month CLV is hard](#2-why-12-month-clv-is-hard)
3. [Method family A — probabilistic "buy-till-you-die" models](#3-method-family-a--probabilistic-buy-till-you-die-models)
4. [Method family B — feature-based machine learning](#4-method-family-b--feature-based-machine-learning)
5. [Method family C — deep and distributional models](#5-method-family-c--deep-and-distributional-models)
6. [Which family wins, and when](#6-which-family-wins-and-when)
7. [Evaluation: the protocol the literature actually uses](#7-evaluation-the-protocol-the-literature-actually-uses)
8. [Pitfalls that invalidate results](#8-pitfalls-that-invalidate-results)
9. [Reading the supplied dataset against this literature](#9-reading-the-supplied-dataset-against-this-literature)
10. [Proposed approach for this capstone](#10-proposed-approach-for-this-capstone)
11. [Public datasets and tooling](#11-public-datasets-and-tooling)
12. [References](#12-references)

---

## 1. Defining the problem

### 1.1 The quantity being predicted

For customer *i*, with a **feature cut-off date** *t₀* and a **horizon** *H* = 12 months:

```
CLV_i(12m) = Σ  (margin on purchases by customer i in (t₀, t₀ + 12 months])
```

Three modelling choices sit inside that formula, and papers differ on all three:

| Choice | Options seen in the literature | Notes |
|---|---|---|
| **Value measure** | gross revenue · revenue net of returns · gross margin · contribution after marketing cost | ASOS uses *"sales, net of returns, of a customer over a one year period"* [[6]](#ref6). Returns matter in fashion e-commerce; ignoring them inflates CLV for serial returners. |
| **Horizon** | 12 months is the industry standard for e-commerce; academic BTYD work often projects to an infinite horizon with discounting | A finite horizon is measurable and therefore falsifiable. Infinite-horizon CLV needs a discount rate and cannot be validated directly. |
| **Discounting** | usually omitted at 12 months; Gupta et al. [[1]](#ref1) discount for longer horizons | At *H* = 12 months, discounting changes ranking almost not at all. |

### 1.2 Historic CLV vs predicted CLV

A recurring confusion worth settling in the capstone report:

* **Historic (realised) value** — what a customer *has already* spent. Descriptive, no model needed.
* **Predicted CLV** — expected spend in a *future* window. This is a forecasting problem and
  requires a time-separated target.

Only the second is decision-useful: marketing budget is spent on future behaviour.

### 1.3 Contractual vs non-contractual

The single most important structural distinction in this literature [[1]](#ref1):

* **Contractual** (subscriptions, insurance): churn is *observed* — the customer cancels. CLV reduces
  to survival analysis × expected spend.
* **Non-contractual** (e-commerce, retail): churn is **latent**. A customer who has not purchased for
  five months may be gone, or may simply buy twice a year. Every model below exists to handle this
  ambiguity. **E-commerce capstones are always in this setting.**

---

## 2. Why 12-month CLV is hard

| Property | Consequence for modelling |
|---|---|
| **Zero inflation** — a large share of customers buy nothing in the next 12 months | A single regression on all customers spends its capacity on predicting zeros. The KDD Cup 1998 benchmark is ~95% zeros [[7]](#ref7). |
| **Extreme right skew** — a small minority generates most of the value | Squared-error loss chases a few whales. ZILN was designed precisely because "MSE … can be sensitive to extremely large LTVs from top spenders" [[7]](#ref7). |
| **Latent churn** | No label for "still a customer"; it must be inferred (BTYD) or side-stepped (direct forward-value regression). |
| **Right-censoring** | The most recent cohorts have not yet lived a full 12 months, so they cannot supply a complete label. Training data is therefore always *older* than the deployment population. |
| **Non-stationarity** | Promotions, seasonality, pandemics and price changes shift the mapping from features to value; a model trained on 2023 cohorts may mis-calibrate for 2025 ones. |
| **Ranking ≠ calibration** | Budget allocation needs correct *ordering* of customers; financial planning needs correct *levels*. These are different objectives and are measured separately (§7). |

---

## 3. Method family A — probabilistic "buy-till-you-die" models

BTYD models describe a customer's purchasing as two latent processes: a purchase process while
"alive", and an unobserved "death" (churn) event. They are fitted on transaction summaries only.

### 3.1 The core models

| Model | Source | Assumptions |
|---|---|---|
| **Pareto/NBD** | Schmittlein, Morrison & Colombo, 1987 [[2]](#ref2) | Purchases ~ Poisson(λ) while alive; lifetime ~ exponential(μ); λ and μ gamma-distributed across customers. Churn can occur at any time. |
| **BG/NBD** | Fader, Hardie & Lee, 2005 [[3]](#ref3) | Same purchase process, but churn happens only *immediately after a purchase* (beta-geometric). Far easier to estimate; near-identical accuracy. The workhorse. |
| **MBG/NBD** | Batislam et al. / Hoppe & Wagner | Allows a customer to churn without ever repeating — fixes a BG/NBD edge case for one-time buyers. |
| **Gamma-Gamma** | Fader, Hardie & Lee [[4]](#ref4) | Models *spend per transaction*, assumed independent of purchase frequency. Multiplied with the frequency model to give monetary CLV. |
| **EP/NBD** (extended Pareto/NBD) | Jasek et al., 2018 [[5]](#ref5) | Best performer across six Czech/Slovak online retailers. |

### 3.2 Why this family matters even in an ML project

**RFM is a sufficient statistic.** Fader, Hardie & Lee [[4]](#ref4) show that under the
Pareto/NBD + Gamma-Gamma framework, a customer's entire transaction history can be compressed to
**recency, frequency, monetary value and age (T)** with no loss of information for forecasting,
and they use *iso-value curves* to group customers with different histories but identical future
value. This result is the reason RFM aggregates remain the default feature set — and it is
directly relevant here, because the supplied dataset is exactly such a summary (§9).

Standard BTYD input definitions (the nomenclature used by PyMC-Marketing [[19]](#ref19)):

| Variable | Meaning |
|---|---|
| `frequency` | number of **repeat** purchases (total − 1) |
| `T` | customer age: time from first purchase to the end of the observation window |
| `recency` | time from first purchase to the **most recent** purchase (not time since last purchase) |
| `monetary_value` | average value of repeat purchases |

### 3.3 Strengths and limits

**Strengths:** few parameters, interpretable, calibrated at the aggregate level, needs only a
transaction log, works with small samples, and gives P(alive) and expected transactions as
by-products. Comparative studies find probabilistic models "stable, with significant lifts over
a status-quo baseline" across many retail datasets [[5]](#ref5).

**Limits, which motivate families B and C:**
* Cannot use covariates in their classic form — no browsing behaviour, no marketing touches, no
  product mix, no demographics. Chamberlain et al. [[6]](#ref6) note it is "difficult to incorporate
  the vast majority of customer data available to modern e-commerce companies into the RFM/BTYD
  framework, particularly automatically learned or highly sparse features."
* The Gamma-Gamma independence assumption (spend ⟂ frequency) is frequently violated.
* Stationary purchase rates: no seasonality, no trend, no promotion response.

---

## 4. Method family B — feature-based machine learning

The industrial mainstream since about 2016: treat 12-month forward value as a supervised learning
target and throw every available feature at it.

### 4.1 Groupon — the two-stage template

Vanderveld, Pandey, Han & Parekh (KDD 2016) [[8]](#ref8) built Groupon's CLV system as a
**two-stage random forest**:

1. **Stage 1 (classification):** will this user purchase in the window at all?
2. **Stage 2 (regression):** conditional on purchasing, how much will they spend?

with `CLV = P(purchase) × E[spend | purchase]`. This "hurdle" decomposition directly targets the
zero-inflation problem of §2, and it separates the *drivers* of the two effects — engagement
features (visits, email opens, app sessions) dominate stage 1, while monetary history dominates
stage 2. The paper's core claim is that **engagement data, not just transactions, carries the signal**.

The pattern remains standard practice. A recent hurdle study reports a two-stage CatBoost reaching
R² = 0.522 versus 0.385 for the best single-stage MSE model and 0.309 for a Tweedie-loss baseline
[[9]](#ref9) — one study on one dataset, but the direction matches Groupon's rationale.

### 4.2 ASOS — 12-month CLV at scale, with learned representations

Chamberlain, Cardoso, Liu, Pagliari & Deisenroth (KDD 2017) [[6]](#ref6) describe the deployed
system at ASOS (~12.5M active customers). Directly relevant to this capstone:

* **Target:** net spend over the **next 12 months** — the exact framing of our objective.
* **Baseline:** a random forest over **132 handcrafted features**.
* **Innovation:** customer **embeddings** learned with skip-gram with negative sampling (SGNS)
  over product-view sequences, rather than aggregating product embeddings — chosen because fashion
  catalogues turn over too fast for per-product features to generalise.
* **Results:** Spearman ρ = **0.56** over all customers (0.46 excluding zero-value customers) and
  churn AUC = **0.798**; embeddings gave significant uplift over handcrafted features alone, with
  32–128 dimensions optimal.

Two lessons worth carrying into any capstone write-up: (i) a Spearman of ~0.5 is what a
well-engineered production system achieves, so wildly higher numbers on a first attempt indicate
leakage, not skill; and (ii) the paper reports rank correlation, not R².

### 4.3 Model classes and features in practice

| Model class | Typical role |
|---|---|
| **Gradient-boosted trees** (XGBoost, LightGBM, CatBoost) | The default for tabular CLV; strong with modest data and mixed feature types |
| **Random forest** | The classic baseline of [[8]](#ref8) and [[6]](#ref6) |
| **Regularised linear / GLM** (log-link, Tweedie, Gamma) | Interpretable baseline; Tweedie handles the zero-plus-positive-continuous structure natively |
| **Quantile regression** | When the decision needs an interval, not a point |
| **Meta-learner stacking** | Combines specialised sub-models [[10]](#ref10) |

Feature groups repeatedly found useful: **RFM aggregates** (still the strongest single block),
**engagement/behaviour** (sessions, views, cart adds, email response), **basket composition**
(category breadth, discount share, returns rate), **marketing exposure**, **acquisition channel and
first-order characteristics** (the only features available for brand-new customers), and
**tenure/seasonality**.

---

## 5. Method family C — deep and distributional models

This family accepts the ML framing but redesigns the **loss function** around the shape of the
CLV distribution — the most transferable idea in the recent literature.

### 5.1 ZILN — zero-inflated lognormal (the key reference)

Wang, Liu & Miao (2019, Google) [[7]](#ref7) model CLV as a **mixture of a point mass at zero and a
lognormal distribution**, with a network emitting three parameters:

* `p` — probability of returning (sigmoid),
* `μ` — lognormal mean (identity),
* `σ` — lognormal scale (softplus),

so the loss decomposes into **cross-entropy for churn + lognormal likelihood for positive spend**.
One model replaces the two-stage pipeline, and the output is a full predictive *distribution*
rather than a point estimate.

**Evaluated on** the Kaggle *Acquire Valued Shoppers* challenge and *KDD Cup 1998*, using
**Spearman's ρ, normalized Gini, decile-level MAPE and AUPRC**. Reported gains over MSE loss:
Spearman +23.9% (linear) and +48.0% (DNN); normalized Gini +28.6% (linear) and +11.4% (DNN). On
KDD Cup 1998, Gini 0.190 vs 0.184 and decile MAPE 0.176 vs 0.210, with total profit $15,498
against the competition winner's $14,712 (+5%). Reference implementation:
[google/lifetime_value](https://github.com/google/lifetime_value) [[18]](#ref18).

**Why it matters here:** the ZILN *loss* can be dropped into any model, including a small tabular
one. It is the cheapest available upgrade when a CLV target is zero-inflated and heavy-tailed.

### 5.2 Industrial-scale successors

| Work | Idea |
|---|---|
| **ODMN + MDME**, Kuaishou, CIKM 2022 [[11]](#ref11) | Models *ordered dependencies* between LTVs at different horizons (30/60/90 days…), and splits the severely imbalanced distribution into balanced sub-distributions handled by multiple experts. Introduces **Mutual Gini** (a Lorenz-curve metric). Beats ZILN and two-stage XGBoost on Kuaishou data. |
| **OptDist**, CIKM 2024 [[12]](#ref12) | Instead of one distribution for everyone, learns several candidate sub-distributions and *selects* the right one per customer; validated in deployed acquisition campaigns. |
| **Seq2seq CLV**, Bauer & Jannach, TKDD 2021 [[13]](#ref13) | An encoder-decoder RNN over purchase sequences learns periodicity, trend and seasonality that hand-built RFM features miss, combined with feature-based models. |

The trajectory is consistent: **the modelling frontier is about the target's distribution and its
temporal structure, not about bigger models.**

### 5.3 Surveys and comparative studies

* **Gupta et al., JSR 2006** [[1]](#ref1) — the canonical taxonomy (RFM, probability models,
  econometric, persistence, computer science, diffusion/growth); still the best framing reference.
* **Dogan, Hiziroglu, Pisirgen & Seymen, WIREs DMKD 2025** [[14]](#ref14) — recent systematic
  overview of business-analytics approaches to CLV.
* **Jasek et al., Informatics 2018** [[5]](#ref5) — head-to-head of EP/NBD, Markov chain, VAR and
  status-quo models across six online retailers; **EP/NBD wins on most metrics**.
* **Jasek et al., Prague Economic Papers 2019** [[15]](#ref15) — adds *non-financial* data to CLV
  models in e-commerce.

---

## 6. Which family wins, and when

Synthesising [[5]](#ref5), [[6]](#ref6), [[7]](#ref7), [[8]](#ref8), [[11]](#ref11):

| Your situation | Use this | Why |
|---|---|---|
| Transaction log only; modest data; need interpretability and calibrated aggregates | **BG/NBD + Gamma-Gamma** (or EP/NBD) | Few parameters, strong and stable, no feature engineering [[5]](#ref5) |
| Transaction log **plus** behavioural/marketing features; tens of thousands of customers upward | **Two-stage GBDT** (propensity × value) | Absorbs arbitrary covariates; the industrial default [[8]](#ref8) |
| Heavy zero-inflation and a long tail; want one model and a predictive distribution | **ZILN loss** on a linear model or DNN | Purpose-built loss; +24–48% Spearman over MSE in [[7]](#ref7) |
| Rich sequences, seasonality, very large scale | **Seq2seq / multi-expert distributional nets** | Learns temporal structure [[13]](#ref13); handles imbalance [[11]](#ref11), [[12]](#ref12) |
| **Only a cross-sectional RFM summary, no timestamps** (this capstone, §9) | Supervised regression on log-value + **BTYD as a reference model** | Nothing else is identifiable from the data |

A consistent empirical thread: **ML beats BTYD when, and only when, it has extra features to
exploit.** With RFM alone, the gap is small and BTYD is better calibrated; with engagement data,
ML wins clearly [[6]](#ref6), [[8]](#ref8).

---

## 7. Evaluation: the protocol the literature actually uses

### 7.1 Temporal design — the non-negotiable part

```
|<---- calibration window ---->|<---- holdout window (12 months) ---->|
       features built here            target measured here
                              t₀ = feature cut-off
```

* Features use **only** data before *t₀*; the label is the realised value after *t₀*.
* Split **by time** (or by customer cohort), never at random. A random split over customer-periods
  leaks the future into training.
* Customers must be **eligible at t₀** (acquired before it) and observed for a full 12 months after
  it, otherwise the label is censored.

### 7.2 Metrics

| Goal | Metric | Notes |
|---|---|---|
| **Ranking** (budget allocation) | **Spearman ρ**, **normalized Gini** | The primary metrics in [[6]](#ref6), [[7]](#ref7), [[11]](#ref11). Gini = 1 perfect ordering, 0 random. Report both overall and on positive-value customers only. |
| **Calibration** (financial planning) | **Decile chart**, **decile-level MAPE** | Sort by prediction, compare predicted vs realised mean per decile [[7]](#ref7). |
| **Concentration** | **Top-decile lift**, share of realised value captured in the top decile | The number a CRM team actually acts on. |
| **Error magnitude** | MAE / RMSE, usually on log or after winsorising | Raw RMSE is dominated by a handful of whales. |
| **Churn sub-task** | AUC / AUPRC | ASOS reports churn AUC 0.798 [[6]](#ref6). |
| **Business** | Incremental profit, campaign ROI | [[7]](#ref7) reports profit on KDD Cup 1998; [[12]](#ref12) reports deployed campaign results. |

### 7.3 Baselines a CLV model must beat

1. **Persistence** — next 12 months = last 12 months' spend. Surprisingly strong; many models fail to beat it.
2. **Last order value / first order value** — the naive ranking baseline used in [[7]](#ref7).
3. **RFM decile scoring** — the pre-model industry standard.
4. **BG/NBD + Gamma-Gamma** — the statistical reference point.

**Reporting a CLV metric without at least one of these next to it is uninterpretable.**

---

## 8. Pitfalls that invalidate results

| Pitfall | How it shows up | Guard |
|---|---|---|
| **Target leakage through the feature window** | Features computed over a period overlapping the label window; R² looks superb | Hard cut-off at *t₀*; re-derive every feature from data before it |
| **Random train/test split** | Optimistic and unreproducible in production | Split by time or cohort |
| **Survivorship selection** | Training only on customers who stayed active | Define the eligible population at *t₀*, keep the zeros |
| **Dropping zero-value customers** | Inflates every metric; changes the business question | Keep them; model them explicitly ([[7]](#ref7), [[8]](#ref8)) |
| **MSE on a heavy tail** | Predictions collapse toward the mean; whales dominate gradients | ZILN / Tweedie / log-target / two-stage |
| **MAPE on near-zero values** | Explodes; meaningless | Decile-level MAPE instead [[7]](#ref7) |
| **Confusing historic with predicted value** | "Model" merely restates past spend | Check feature timing; compare against the persistence baseline |
| **Believing a single split** | One lucky window | Multiple time origins / rolling-origin evaluation |
| **Treating an already-estimated target as ground truth** | Model learns the formula that produced the label, not customer behaviour | See §9 — this is the live risk in this capstone |

---

## 9. Reading the supplied dataset against this literature

`data/synthetic_data_126.csv` — **1,000 rows × 7 columns**, one row per customer:

| Column | Type | Range / note | BTYD analogue |
|---|---|---|---|
| `total_purchase_count` | float | 0.03 – 844 | `frequency` |
| `average_order_value` | float | 17.9 – 848 | `monetary_value` |
| `days_since_first_purchase` | float | 7.5 – 1,494 | `T` (customer age) |
| `days_since_last_purchase` | float | 1.6 – 450 | `T − recency` |
| `product_category_diversity` | float | 0.005 – 0.78 | — (basket breadth) |
| `loyalty_program_membership` | categorical | Enrolled 40% / Not Enrolled 60% | — |
| `estimated_lifetime_value` | float | 58.6 – 16,797 | **target** |

Its shape matches the RFM sufficient-statistic result of §3.2 — which is encouraging. But five
findings from a first pass materially affect what can be claimed:

1. **The target is not a 12-month forward value.** It is named `estimated_lifetime_value` and carries
   no time window. There is no *t₀*, no transaction log and no holdout period, so the temporal
   design of §7.1 **cannot be constructed from this file**. Any model trained on it is a
   cross-sectional regression onto an existing estimate, not a forecast.
2. **The target is close to a deterministic function of the features.** A gradient-boosting
   regressor on log(target) reaches **CV R² ≈ 0.97** (5-fold). Rank correlation between
   `total_purchase_count × average_order_value` and the target is ρ = 0.88. In other words the label
   was very likely *generated* from these columns by formula. Excellent scores will be easy and
   will measure formula recovery, not predictive skill.
3. **No zeros** (minimum 58.64). The zero-inflation machinery of [[7]](#ref7) and [[8]](#ref8) is
   therefore not needed — worth stating explicitly, since it is the first thing a reviewer will
   look for. The heavy tail is real though: skew 6.4, and **the top 10% of customers hold 38% of
   total value** (top 20% → 54%).
4. **52 rows have `days_since_last_purchase` > `days_since_first_purchase`** — a last purchase
   before the first. Under the BTYD mapping this yields a negative recency, which is impossible.
   These need a documented decision (drop, clip, or treat as a generator artefact).
5. **`total_purchase_count` is non-integer for 988 of 1,000 rows** (e.g. 12.44 purchases). Real
   counts are integers, and BG/NBD requires integer frequencies. A BTYD model can only be fitted
   here after rounding, and should be presented as illustrative rather than as inference.

Also worth noting: `loyalty_program_membership` correlates **negatively** with value
(Spearman −0.34), the opposite of the usual loyalty finding — another sign of synthetic generation
rather than observed behaviour.

### 9.1 What this means for scope

Three honest options, in descending order of research value:

| Option | What it gives | Cost |
|---|---|---|
| **A. Re-frame as a real forecasting task on transactional data** — use Online Retail II, Olist, or Acquire Valued Shoppers (§11), build the calibration/holdout design of §7.1, and apply BTYD + two-stage GBDT + ZILN | The genuine 12-month CLV problem, with every method in this review applicable and comparable | New dataset; more pipeline work |
| **B. Keep this dataset, state the limitation plainly** — cross-sectional regression on an estimated target, with the heavy-tail and ranking metrics of §7.2 | Satisfies a "predict CLV" brief; clean, small, quick | Cannot demonstrate forecasting; near-deterministic target caps what is learned |
| **C. Hybrid (recommended)** — use this dataset as the modelling deliverable, and add a transactional case study for the 12-month design | Both the deliverable and the methodological point | Moderate extra work |

Whichever is chosen, §9's findings belong in the EDA, not in a footnote: recognising that a target
is synthetic and near-deterministic is itself a legitimate analytical result.

---

## 10. Proposed approach for this capstone

A baseline ladder, in the order the literature suggests:

1. **Naive baselines** — mean, median, and `frequency × AOV` (ρ = 0.88 here, so it is a *strong*
   baseline and must be reported).
2. **Regularised linear model on log(value)** — interpretable reference.
3. **Gradient-boosted trees on log(value)** — the tabular default; primary candidate.
4. **ZILN loss** [[7]](#ref7)[[18]](#ref18) — even without zeros, the lognormal head models the tail
   and yields a predictive distribution; include for methodological completeness.
5. **BG/NBD + Gamma-Gamma** [[3]](#ref3)[[4]](#ref4)[[19]](#ref19) — fitted on the RFM mapping in
   §9, presented as the domain-standard reference model with its caveats stated.

**Metrics:** Spearman ρ and normalized Gini (ranking), decile chart and decile MAPE (calibration),
top-decile value capture (business), MAE on log-value (magnitude). **Validation:** stratified K-fold
on value deciles for this cross-section — with an explicit note that a *temporal* split is the
correct design and is impossible here (§9).

---

## 11. Public datasets and tooling

### Datasets

| Dataset | Content | Fit for a 12-month CLV task |
|---|---|---|
| **UCI Online Retail II** [[16]](#ref16) | ~1M transactions, UK gift retailer, Dec 2009 – Dec 2011 | Two full years → a clean 12-month calibration / 12-month holdout split. **Best fit.** |
| **Olist Brazilian E-commerce** [[17]](#ref17) | 100k orders, 2016–2018, multi-marketplace, with reviews and logistics | Rich covariates; but most customers order once, so CLV is dominated by zeros |
| **Kaggle Acquire Valued Shoppers** | 311k customers, basket-level histories; task = value in the year after first purchase | Used by [[7]](#ref7); the closest public analogue to a 12-month CLV benchmark |
| **KDD Cup 1998** | ~200k lapsed donors, ~95% zeros | The classic zero-inflation benchmark [[7]](#ref7) |

### Tooling

| Tool | Use |
|---|---|
| **PyMC-Marketing** [[19]](#ref19) | Maintained Bayesian BTYD: BG/NBD, Pareto/NBD, Modified BG/NBD, Gamma-Gamma, BG/BB, Shifted BG |
| **lifetimes** (CamDavidsonPilon) [[20]](#ref20) | The historical standard, now **archived** — no new features or issue support. Fine to read, not to build on |
| **btyd** (ColtAllen) | Community successor to `lifetimes` |
| **google/lifetime_value** [[18]](#ref18) | Reference ZILN loss and evaluation code (Spearman, Gini, decile charts) |
| **XGBoost / LightGBM / CatBoost** | Two-stage and single-stage tabular models |

---

## 12. References

<a id="ref1"></a>[1] Gupta, S., Hanssens, D., Hardie, B., Kahn, W., Kumar, V., Lin, N., Ravishanker, N. & Sriram, S. (2006). *Modeling Customer Lifetime Value*. **Journal of Service Research** 9(2), 139–155. [SAGE](https://journals.sagepub.com/doi/10.1177/1094670506293810) · [PDF](https://www.anderson.ucla.edu/sites/default/files/documents/areas/fac/marketing/JSR2006(0).pdf)

<a id="ref2"></a>[2] Schmittlein, D., Morrison, D. & Colombo, R. (1987). *Counting Your Customers: Who Are They and What Will They Do Next?* **Management Science** 33(1). — the Pareto/NBD model.

<a id="ref3"></a>[3] Fader, P., Hardie, B. & Lee, K. L. (2005). *"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model*. **Marketing Science** 24(2), 275–284. — the BG/NBD model. [brucehardie.com](https://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)

<a id="ref4"></a>[4] Fader, P., Hardie, B. & Lee, K. L. (2005). *RFM and CLV: Using Iso-Value Curves for Customer Base Analysis*. **Journal of Marketing Research** 42(4), 415–430. [PDF](https://www.brucehardie.com/papers/rfm_clv_2005-02-16.pdf) · [SAGE](https://journals.sagepub.com/doi/abs/10.1509/jmkr.2005.42.4.415)

<a id="ref5"></a>[5] Jasek, P., Vrana, L., Sperkova, L., Smutny, Z. & Kobulsky, M. (2018). *Modeling and Application of Customer Lifetime Value in Online Retail*. **Informatics** 5(1), 2. [DOI](https://doi.org/10.3390/informatics5010002)

<a id="ref6"></a>[6] Chamberlain, B. P., Cardoso, A., Liu, C. H. B., Pagliari, R. & Deisenroth, M. P. (2017). *Customer Lifetime Value Prediction Using Embeddings*. **KDD '17**, 1753–1762. [arXiv:1703.02596](https://arxiv.org/abs/1703.02596) · [ACM](https://dl.acm.org/doi/10.1145/3097983.3098123)

<a id="ref7"></a>[7] Wang, X., Liu, T. & Miao, J. (2019). *A Deep Probabilistic Model for Customer Lifetime Value Prediction*. [arXiv:1912.07753](https://arxiv.org/abs/1912.07753)

<a id="ref8"></a>[8] Vanderveld, A., Pandey, A., Han, A. & Parekh, R. (2016). *An Engagement-Based Customer Lifetime Value System for E-commerce*. **KDD '16**. [Semantic Scholar](https://www.semanticscholar.org/paper/An-Engagement-Based-Customer-Lifetime-Value-System-Vanderveld-Pandey/cbeebb558867e9315a68d4155b5f17661cb4472a)

<a id="ref9"></a>[9] *A Two-Stage Hurdle Gradient-Boosting Framework for Zero-Inflated Customer Lifetime Value Prediction and Segmentation*. **Applied Sciences**. [DOI](https://doi.org/10.3390/app16136550)

<a id="ref10"></a>[10] *A Meta-learning based Stacked Regression Approach for Customer Lifetime Value Prediction*. [arXiv:2308.08502](https://arxiv.org/abs/2308.08502)

<a id="ref11"></a>[11] Li, K., Shao, G., Yang, N., Fang, X. & Song, Y. (2022). *Billion-user Customer Lifetime Value Prediction: An Industrial-scale Solution from Kuaishou*. **CIKM 2022**. [arXiv:2208.13358](https://arxiv.org/abs/2208.13358)

<a id="ref12"></a>[12] *OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction*. **CIKM 2024**. [arXiv:2408.08585](https://arxiv.org/abs/2408.08585)

<a id="ref13"></a>[13] Bauer, J. & Jannach, D. (2021). *Improved Customer Lifetime Value Prediction with Sequence-to-Sequence Learning and Feature-Based Models*. **ACM TKDD** 15(5). [ACM](https://dl.acm.org/doi/10.1145/3441444) · [PDF](https://web-ainf.aau.at/pub/jannach/files/Journal_TKDD_2021.pdf)

<a id="ref14"></a>[14] Dogan, O., Hiziroglu, A., Pisirgen, A. & Seymen, O. F. (2025). *Business Analytics in Customer Lifetime Value: An Overview Analysis*. **WIREs Data Mining and Knowledge Discovery**. [DOI](https://doi.org/10.1002/widm.1571)

<a id="ref15"></a>[15] Jasek, P. et al. (2019). *Predictive Performance of Customer Lifetime Value Models in E-Commerce and the Use of Non-Financial Data*. **Prague Economic Papers**. [Link](https://pep.vse.cz/artkey/pep-201906-0002_predictive-performance-of-customer-lifetime-value-models-in-e-commerce-and-the-use-of-non-financial-data.php)

<a id="ref16"></a>[16] Chen, D. *Online Retail II*. UCI Machine Learning Repository. [Link](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

<a id="ref17"></a>[17] Olist. *Brazilian E-Commerce Public Dataset*. Kaggle. [Link](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

<a id="ref18"></a>[18] Google. *lifetime_value* — reference implementation of the ZILN loss. [GitHub](https://github.com/google/lifetime_value)

<a id="ref19"></a>[19] PyMC Labs. *PyMC-Marketing — CLV models*. [Docs](https://www.pymc-marketing.io/en/stable/notebooks/clv/clv_quickstart.html)

<a id="ref20"></a>[20] Davidson-Pilon, C. *lifetimes* (archived). [GitHub](https://github.com/CamDavidsonPilon/lifetimes)

---

### Note on sourcing

References [1]–[8], [11]–[20] were checked against the publisher, arXiv or repository page. [9] and
[10] are cited from abstracts and search results rather than a full read — their specific numbers
(for example the R² = 0.522 two-stage result) should be verified before being quoted in the final
report. Numbers attributed to the supplied dataset in §9 were computed directly from
`data/synthetic_data_126.csv`.
