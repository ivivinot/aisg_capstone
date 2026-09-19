# Capstone — 12-Month Customer Lifetime Value Prediction for E-commerce

**Objective.** Predict the value a customer will generate over the **next 12 months** in a
non-contractual e-commerce setting, and use that prediction to target marketing spend.

This document has two parts. **§1–§12 are the literature review and research synthesis**: what
the field has established, which methods win in which data regime, how such models are
evaluated, where they usually go wrong, and how all of that maps onto the dataset in `data/`.
**§13–§17 are the project report**: the pipeline built from that analysis, how to run it, what
it produced, and what may honestly be claimed from the result.

> **Status.** Complete. `eda.ipynb` holds the analysis; `main.py` processes the data and runs
> the 25 checks that validate it; `train.py` models it; `outputs/` holds what the last run
> produced.
>
> **Read §9 and §17 before quoting any score.** The supplied target is an *already estimated*
> lifetime value and is a near-deterministic function of the feature columns, so the pipeline's
> near-perfect metrics measure **recovery of a generating formula**, not forecasting skill. A
> deployed 12-month CLV system reports Spearman ≈ 0.56 (§4.2).

**Quick start**

```bash
pip install -r requirements.txt
python main.py        # process the data, then test and validate it   (~2 s)
python train.py       # the same, then select, tune and score a model (~22 s)
pytest tests/ -q      # the same checks as a suite, on dirty fixtures
python predict.py --input new_customers.csv --output scored.csv
```

---

## Contents

**Part I — research synthesis**

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

**Part II — project report**

13. [The pipeline](#13-the-pipeline)
14. [How to run it](#14-how-to-run-it)
15. [Results](#15-results)
16. [Design decisions, and what they cost](#16-design-decisions-and-what-they-cost)  
    · [16.3 Engineered features](#163-engineered-features-the-strategy-and-what-it-was-worth) · [16.4 Feature selection](#164-feature-selection-and-what-each-strategy-costs)
17. [Deployment considerations and limits](#17-deployment-considerations-and-limits)

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

> **This plan was executed.** `eda.ipynb` works through the ladder and §13–§15 report what the
> pipeline built from it produces. Two items changed on contact with the data: the ZILN loss was
> dropped (there are no zero-value customers to justify it) and BG/NBD was not fitted (§15.1,
> EDA §6). The winner is item 3's neighbour rather than item 3 itself — a **degree-3 polynomial
> on logged features with Ridge**, which beat tuned gradient boosting on every metric.

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

---

# Part II — project report

## 13. The pipeline

`eda.ipynb` is the analysis; this is the same analysis as code that runs unattended, end to end,
in about 20 seconds.

Two entry points. `main.py` owns the data and refuses to hand it on quietly if a
check fails; `train.py` runs `main.py`'s stages first, stops on a failed check, and
only then models.

```
                    python main.py                         python train.py
data/synthetic_data_126.csv
        │
        ▼
  1  load + audit ──────────────► outputs/data_audit.json          ▲
        │                          (the EDA §2 checks, as JSON)    │
        ▼                                                          │
  2  row filtering                (opt-in; nothing removed by default)
        │                                                          │
        ▼                                                          │
  3  80/20 split, stratified on value deciles   ── the test split is now sealed
        │                                                          │
        ▼                                                          │
  4  CLVPreprocessor.fit(train)  ─► preprocessor.joblib, processed_*.csv
        │   encode → domain rules → impute → clip → log → poly(3) → scale → prune
        ▼                                                          │
  5  FEATURE ENGINEERING         created columns + selection       │
        │                        (src/feature_engineering.py)       │
        ▼                                                          │
  6  OUTPUT VALIDATION            8 contract checks                 │
  7  TRANSFORMER TESTS           12 invariance / edge-case checks   │  the gate:
  8  FEATURE-ENGINEERING TESTS    7 creation / selection checks     │  train.py stops
  9  STATISTICAL VALIDATION       5 golden-number + leakage checks  │  if any failed
        │                        ─► outputs/validation_report.json  │
        ▼                                                          │
 10  BASELINE LADDER (src/models.py)                                │
        │   select → train → evaluate → analyse → benchmark         │
        │   6 more checks        ─► baseline_leaderboard.csv         │
        ▼                                                          │
 11  artifacts ──────────────────────────────────────────────────────┘
        │
        ▼                                                  ┌──────────────────────
  9  model selection: 8 candidates + 2 baselines, 5-fold CV ─► model_leaderboard.csv
        │                                                  │
 10  randomised search over 3 finalists                    ─► modelling_report.json
        │                                                  │
 11  refit winner, score the test split ONCE               ─► test_predictions.csv
        │                                                  │
 12  importance, value tiers, figures                      ─► customer_tiers.csv
        │                                                  │
 13  model bundle                                          ─► model.joblib → predict.py
```

### 13.1 File structure

| Path | What it is |
|---|---|
| `main.py` | **Data entry point.** Audit → filter → split → preprocess → validate → test. Writes the processed data and the reports; exits non-zero if a check fails. |
| `train.py` | **Modelling entry point.** Reuses `main.py`'s stages, its validation gate and its baseline ladder, then runs the candidate zoo, the randomised search, the single test-set evaluation and the tiering. |
| `predict.py` | Scores a CSV of new customers from `outputs/model.joblib`. |
| `src/preprocessing.py` | The data contract, the audit, the split, the cleaning transformers, the `CLVPreprocessor` facade — **and the checks that test and validate what it produced**. |
| `src/feature_engineering.py` | The feature half: the catalogue of engineered columns, `DerivedFeatures`, the log/expand/scale steps, `FeatureSelector`, and the measurements that say whether any of it helped. |
| `src/metrics.py` | The scorecard of §7.2: Spearman, normalized Gini, top-decile capture, decile MAPE, MAE, R². |
| `src/models.py` | **The baseline ladder, for any task.** Regression, classification and clustering baselines with their rationale, the cross-validated evaluation, the metric guide, the cost benchmark and the checks. |
| `src/evaluation.py` | Test-set scoring, both importance measures, value tiers, figures, the model bundle. |
| `tests/`, `pytest.ini` | The same checks as a pytest suite, plus unit tests on a 12-row fixture carrying every quality violation. |
| `eda.ipynb` | The analysis this is built from — 14 sections, every decision argued. |
| `data/` | `synthetic_data_126.csv`, 1,000 customers × 7 columns. |
| `outputs/` | Everything the last run produced (see §14.4). |

### 13.2 The three architectural decisions

**1. The preprocessor is a scikit-learn transformer, not a script.** Imputation fills, clip
bounds, scaler statistics and the polynomial expansion are all *fitted*, so the whole object
drops into a `Pipeline` and is **refitted inside every cross-validation fold**. Any leakage
would have to come from the split itself, not from a statistic computed over all the rows. The
alternative — transform once, then cross-validate — silently reports scores that cannot be
reproduced in production.

**2. A candidate is a `(representation, estimator)` pair.** EDA §7 found the target to be
multiplicative, and EDA §9–§10 found that **representation beat model choice, decisively**. So
`src/models.py` pairs each estimator with the feature space it deserves: logged and expanded for
the linear family, raw for the tree ensembles (trees are invariant to monotone transforms). The
hyperparameter search then ranges over `prep__config` as well as the estimator's own knobs,
which is how it rediscovers degree 3 on its own.

**3. The target is handled on the way out, not just on the way in.** Every model is fitted on
`log(value)`; predictions come back through **Duan's smearing estimator**, because
`exp(E[log y])` under-states `E[y]` for a right-skewed target. The factor moves calibration and
leaves ranking untouched.

### 13.3 The fourth decision: the checks ship with the pipeline

The 38 checks live beside the code they check — 25 at the bottom of `src/preprocessing.py`, 7 in
`src/feature_engineering.py`, 6 in `src/models.py` — and `main.py` runs all of them on every run rather than leaving them in a suite someone
remembers to invoke. `train.py` treats them as a **gate**: a
failed check stops the run before a model is fitted, because a score computed on data that did
not pass its own checks is worse than no score. `tests/` is a thin pytest wrapper around the
same functions, so CI and the pipeline can never disagree about what "valid" means.

What is actually being checked, and why each group exists, is §14.3.

---

## 14. How to run it

```bash
pip install -r requirements.txt

python main.py        # process the data, then test and validate it   (~2 s)
python train.py       # the same, then select, tune and score a model (~22 s)
```

`main.py` prints seven numbered sections — audit, row filtering, preprocessing, output
validation, transformer tests, statistical validation, artifacts — and **exits non-zero if any
check failed**, so it works as a CI gate. `train.py` continues into model selection, tuning,
the single test-set evaluation, importance and tiers.

### 14.1 Flags

Both entry points share the data and validation flags:

| Flag | Effect |
|---|---|
| `--derived-features SET` | Create engineered columns: `default`, `independent`, `all`, or a comma-separated list. |
| `--feature-selection STRATEGY` | `none`, `variance` (default), `correlation`, `mutual_info`, `model`, `vif`. |
| `--select-k K` | How many features the supervised strategies keep. |
| `--feature-report` / `--compare-features` | Per-feature diagnostics / measure each feature set against Ridge and a Random Forest. |
| `--task auto\|regression\|classification\|clustering` | Which baseline ladder to run; `auto` reads it off the target. |
| `--baseline-catalogue` / `--metric-guide` | Print the baselines for every task / what each metric answers and how it misleads. |
| `--no-baselines` / `--no-benchmark` | Skip the ladder / skip its timing benchmark. |
| `--baseline-poly-degree N` | Representation for the baselines (default 1 — a baseline may not borrow the expansion). |
| `--poly-degree 1` | Hold the representation at the first-order log-linear form. |
| `--recency-policy clip` | Repair the 52 impossible-recency rows instead of flagging them. |
| `--drop-invalid-recency` | Remove those rows before the split. |
| `--outlier-quantiles .001 .999` | Winsorise the tail instead of only guarding against extrapolation. |
| `--no-log` | Drop the log transform (instructive, not recommended — see §16.2). |
| `--no-tests` | Skip the transformer behaviour checks. |
| `--no-stat-checks` | Skip the cross-validated checks — the slow part. |
| `--no-validate` | Process the data and check nothing. |
| `--seed`, `--test-size`, `--cv-folds`, `--no-save` | The usual reproducibility knobs. |

`train.py` adds: `--no-tune`, `--no-importance`, `--no-figures`, `--no-xgboost`, `--n-jobs`,
and `--ignore-failed-checks` for the case where a check fails and you know why.
`--help` lists everything with its default.

### 14.2 Running the checks as a test suite

```bash
pip install pytest
pytest tests/ -q                # 77 tests, about 4 s
pytest tests/ -q -m slow        # the cross-validated checks as well
```

The suite runs the same three groups `main.py` runs, plus unit tests on a 12-row fixture that
carries the pathologies the real file happens *not* to have — a missing value, an unseen
category, a non-positive feature — so a refactor fails in CI instead of in a production run.

### 14.3 What is validated, and why

**Group A — output contract (8 checks).** The processed splits are usable and leakage-free:
column parity, row parity with the targets, no missing values, all values finite (no `-inf`
from `log`), no constant columns, train standardised, **test not re-standardised**, target
strictly positive. The seventh is the one that matters: if the test split came back with mean
exactly 0, the scaler had seen it.

**Group B — transformer behaviour (12 checks).** Run on raw test rows the preprocessor was not
fitted on:

| Check | What breaking it would cost |
|---|---|
| single row == batch | `predict.py` would disagree with the evaluation, silently |
| row order, column order, index, unknown columns | a reordered export or an extra column would change predictions |
| a missing feature raises `ValueError` | a truncated file would be scored instead of rejected |
| unseen category, injected `NaN`, zero and negative values | one dirty row would crash a batch or emit `-inf` |
| extreme value clipped to the fitted bound | an absurd input would be extrapolated instead of bounded |
| `save`/`load` and `clone`/`set_params` round-trips | the tuner would optimise a different pipeline than the one saved |

**Group C — feature engineering (7 checks).** What was created, whether every derived column is finite and positive, whether the catalogue's log-space classification survives a rank test, and whether the selection is a deterministic subset that applies unchanged to unseen rows. §16.3 and §16.4 are the strategy these enforce.

**Group D — baseline models (6 checks).** Every baseline scored, predictions finite and one per
row, the naive floor behaving like a floor (R² ≈ 0 for the mean, accuracy = the majority share for
the most-frequent class, undefined indices for a single cluster), at least one model baseline
beating the strongest naive one, and a benchmark covering every row.

**Group E — statistical validation (5 checks).** Feature count per polynomial degree
(7 / 35 / 119), the CV R² the representation is supposed to earn (0.99936 ± 5e-4), and the
**label-shuffle test**: refit the whole pipeline on a permuted target, where the score must
collapse. It lands at **−0.374** — comfortably below zero, which is what an honest preprocessor
looks like. A pipeline that had leaked anything about `y` into the features would still score
above chance here, and no amount of reading the code would prove otherwise.

Checks are collected rather than raised, so one run reports every failure at once, and the
results are written to `outputs/validation_report.json` alongside the data.

### 14.4 Scoring new customers

```bash
python predict.py --input new_customers.csv --output scored.csv
```

The input needs the six raw customer columns, in any order; anything else is carried through.
Tier boundaries are **the ones learned at training time**, not quantiles of the batch being
scored — otherwise a customer's tier would depend on who else happened to be scored that day.

### 14.5 What a run writes

| Artifact | Written by | Contents |
|---|---|---|
| `data_audit.json` | `main.py` | The EDA §2 quality checks on the raw file. |
| `preprocess_config.json`, `preprocessing_report.json` | `main.py` | Every knob used, plus what the fitted pipeline learned: imputation fills, clip bounds, flagged rows, the validation summary. |
| `validation_report.json` | `main.py` | Every check, its status and its detail. |
| `processed_train.csv`, `processed_test.csv` | `main.py` | The 119 model features plus the target and its log. |
| `preprocessor.joblib` | `main.py` | The fitted feature pipeline on its own. |
| `baseline_leaderboard.csv`, `baseline_benchmark.csv`, `baseline_report.json` | `main.py` | The baseline ladder: scores, cost, the metric analysis and its checks. |
| `model_leaderboard.csv`, `modelling_report.json` | `train.py` | Cross-validated scores for every candidate, the search results, the test scorecard, both importance tables, the tiers. |
| `test_predictions.csv`, `customer_tiers.csv` | `train.py` | Per-customer predictions with tier, and the tier summary. |
| `model.joblib` | `train.py` | The fitted model with its smearing factor, feature contract, tier thresholds and test scores — what `predict.py` loads. |
| `figures/*.png` | `train.py` | Model selection, test diagnostics, drop-column importance, value tiers. |

---

## 15. Results

All numbers below are from a default run (seed 42) and match `eda.ipynb` to within fold noise.

### 15.1 What the audit found

| Check | Result | Consequence |
|---|---|---|
| Rows × columns | 1,000 × 7 | — |
| Missing cells, duplicates | 0, 0 | Imputation is kept anyway, for new data. |
| Zero-value customers | **0** (minimum 58.64) | No hurdle model, no ZILN (§5.1) — the zero-inflation machinery solves a problem this file does not have. |
| Target skew | **6.40**, → 0.31 after log | Everything is fitted on `log(value)`. |
| Value concentration | top 10% hold **38%** of value | Ranking metrics lead; the tail is guarded, never trimmed. |
| Non-integer purchase counts | **988 / 1,000** | The file is generated, not observed; BG/NBD cannot be fitted (§3.2, EDA §6). |
| Purchase count < 1 | 143 | `log()` needs a positivity floor. |
| Last purchase before first | **52** | Impossible recency; flagged, not silently repaired. |

### 15.2 The baseline ladder — what a real model has to beat

`src/models.py` is the baseline module, and `main.py` runs it as stage 9 on every run. Baselines
use the **first-order** representation (log + scale, no polynomial expansion) even when the
pipeline is configured for degree 3: the expansion is the modelling choice `train.py` exists to
*make*, and a baseline that borrows it is no longer a baseline. Naive baselines skip the
preprocessor entirely, because a constant predictor ignores `X` and preprocessing it would only
distort the benchmark.

Out-of-fold, 5-fold, on the training split:

| Baseline | Kind | Spearman ρ | Decile MAPE | MAE ($) | R² (log) |
|---|---|---|---|---|---|
| ridge on log(y) | model | **0.9929** | 0.0196 | 75.03 | 0.9866 |
| linear regression on log(y) | model | 0.9929 | **0.0191** | 75.13 | 0.9866 |
| k-NN (k=10) | model | 0.9693 | 0.0883 | 165.70 | 0.9206 |
| linear regression | model | 0.9675 | 0.7845 | 356.26 | n/a |
| **heuristic (purchases × AOV)** | naive | **0.8771** | 0.8100 | 800.24 | −3.02 |
| decision tree (depth 3) | model | 0.6984 | 0.0551 | 355.80 | 0.5592 |
| mean | naive | −0.0411 | 0.1430 | 570.79 | −0.2299 |
| median | naive | −0.0758 | 0.3265 | 510.53 | −0.0039 |

Four readings, each of which a single score column would have hidden:

* **The bar is 0.8771, not zero.** The heuristic reproduces the EDA's ρ = 0.878 exactly, and
  4 of 5 model baselines clear it. Any later number is read against this row, not against the mean.
* **The metrics disagree on a winner** — ridge leads on ranking, plain linear-on-log on decile
  calibration. Ranking and calibration are different questions (§7.2), so a sorted column is not a
  verdict.
* **One line of preprocessing beats every estimator swap.** Linear regression on `log(y)` scores
  0.9929 against 0.9675 for the same model on the raw target, and the log version's decile MAPE is
  41× better. That is the whole of §16.2 in one pair of rows.
* **The shallow tree is the worst model baseline** and the best-calibrated one after the linear
  pair — a smooth multiplicative surface approximated by three splits, which is the same finding
  the tuned ensembles run into in §15.3.

**Cost, not just accuracy** (median of 3 fits, 800 rows):

| Baseline | Fit (ms) | Predict (µs/row) | Size (KB) |
|---|---|---|---|
| heuristic (purchases × AOV) | 0.09 | 0.08 | 0.17 |
| mean | 0.10 | 0.03 | 0.55 |
| linear regression | 18.5 | 9.2 | 6.4 |
| k-NN (k=10) | 18.9 | 12.5 | 67.6 |
| ridge on log(y) | 26.0 | 9.6 | 6.8 |

The naive rows are ~250× cheaper to fit, and almost all of the 26 ms belongs to the preprocessing,
not the estimator. On this data the accuracy gap is worth paying for; on a problem where ridge
bought 0.5% the table would say to ship the heuristic.

`python main.py --metric-guide` prints what each metric answers and how it misleads;
`--baseline-catalogue` prints the ladder for all three task types.

### 15.3 Model selection — 5-fold CV on the training split

| Model | Spearman ρ | Norm. Gini | Decile MAPE | MAE ($) | R² (log) |
|---|---|---|---|---|---|
| **Polynomial on logs + Ridge** | **0.9965** | 0.9976 | 0.0073 | 47.63 | **0.9940** |
| SVR (RBF, on logs) | 0.9955 | 0.9944 | 0.0214 | 67.26 | 0.9861 |
| Linear (log features) | 0.9929 | 0.9949 | 0.0191 | 75.12 | 0.9866 |
| XGBoost | 0.9872 | 0.9898 | 0.0306 | 110.34 | 0.9727 |
| HistGradientBoosting | 0.9853 | 0.9879 | 0.0288 | 125.75 | 0.9663 |
| Random Forest | 0.9796 | 0.9844 | 0.0492 | 136.50 | 0.9546 |
| k-NN (k = 10, on logs) | 0.9674 | 0.9779 | 0.0901 | 172.26 | 0.9154 |
| Linear (raw features) | 0.9622 | 0.9685 | 0.2301 | 500.50 | 0.8494 |
| *Baseline: purchases × AOV* | *0.8781* | *0.9122* | *0.7843* | *726.23* | *−3.0384* |
| *Baseline: train mean* | *n/a* | *−0.0756* | *0.1695* | *569.73* | *0.0000* |

Three things to read off it:

* **The log-feature models win, and the polynomial version leads** — exactly what EDA §7
  predicted. Getting the representation right beat model flexibility.
* **The tree ensembles lose, and they lose specifically**: a smooth multiplicative surface has to
  be approximated with axis-aligned steps, which costs most in the thin tails. "Tabular problem →
  gradient boosting" would have been the wrong instinct here.
* **The heuristic baseline is the bar, not the mean.** `purchases × AOV` already ranks at
  ρ = 0.88 (§7.3); a model that only beat the mean would prove nothing.

### 15.4 Tuning the finalists

`RandomizedSearchCV`, optimising R² on `log(value)` over the same folds.

| Finalist | CV R² (log), default → tuned | What the search chose |
|---|---|---|
| **Polynomial on logs + Ridge** | 0.9940 → **0.9994** | **degree 3**, α ≈ 0.0019 — it rediscovered EDA §7's curvature on its own |
| XGBoost | 0.9727 → 0.9824 | depth 2, 837 trees, lr 0.032 |
| Random Forest | 0.9546 → 0.9545 | *no improvement at all* |

Extra capacity cannot buy back the wrong representation.

### 15.5 Final model — held-out test split, opened once

**Degree-3 polynomial on logged features + Ridge (α ≈ 0.0019).**

| | Spearman | Norm. Gini | Top-decile capture | Decile MAPE | MAE ($) | R² (log) |
|---|---|---|---|---|---|---|
| **Test (200 customers)** | **0.9999** | 1.0000 | 0.4177 | **0.0019** | **5.98** | 0.9999 |
| CV estimate (train) | 0.9999 | 1.0000 | 0.3708 | 0.0012 | 7.93 | 0.9994 |
| Baseline: purchases × AOV | 0.8781 | 0.9122 | 0.3416 | 0.7843 | 726.23 | −3.0384 |

Test and CV agree, so nothing was overfitted to the folds. The test residual spread is **0.7% in
log space**, against 10.3% for the first-order power law of EDA §7 — on unseen rows, the
generating function has been recovered almost exactly. **That is the sentence that must travel
with the number**: ASOS's production 12-month model reports Spearman 0.56 (§4.2), and the gap
between 0.56 and 1.000 is the gap between forecasting behaviour and recovering a formula.

### 15.6 What drives value

Drop-column importance — refit the whole pipeline without a column, measure the loss in CV R²
(full-model CV R² = 0.999403):

| Feature | Loss in CV R² | Reading |
|---|---|---|
| `average_order_value` | **0.1553** | Dominant. Elasticity 0.86: a 1% larger basket is worth ~0.86% more value. |
| `days_since_last_purchase` | 0.0146 | Recency decay — the classic RFM effect (elasticity −0.32). |
| `days_since_first_purchase` | 0.0108 | Matters through *interactions* despite a first-order elasticity of −0.015. |
| `total_purchase_count` | 0.0075 | Positive with strongly diminishing returns (elasticity 0.29). |
| `product_category_diversity` | 0.0016 | Marginal. |
| `loyalty_program_membership` | **−0.00002** | Nothing. Removing it is free. |

Permutation importance ranks the two date columns *far above* `average_order_value` — the
reverse ordering. It is wrong here, and instructively so: shuffling one column of a degree-3
polynomial invents feature combinations that never occur, and the fitted surface extrapolates
wildly there. **With a high-degree model, prefer refit-based importance.** The pipeline computes
both so the disagreement stays visible.

### 15.7 Customer tiers — the operational output

Tiers are cut from *predictions*; the value shown is what those customers *actually* held, so the
table doubles as a fair test of the ranking.

| Tier | Customers | Mean actual value | Share of test value |
|---|---|---|---|
| VIP (top 5%) | 10 | $5,214 | **29.5%** |
| High value (80–95%) | 30 | $1,608 | 27.3% |
| Growth (50–80%) | 60 | $761 | 25.9% |
| Standard (bottom 50%) | 100 | $305 | 17.3% |

Retention budget and service level follow the tier; §15.6 says which lever to pull inside a tier
(basket size first, lapse prevention second).

---

## 16. Design decisions, and what they cost

### 16.1 Per-feature decisions

| Feature | Raw state | Action | Why |
|---|---|---|---|
| `total_purchase_count` | Skew 10.8; 988 non-integer, 143 below 1 | Positivity floor → log → polynomial | Multiplicative in the target (EDA §7); the floor keeps `log()` finite for sub-1 counts. Counts are **not** rounded: rounding would edit the input that produced the label. |
| `average_order_value` | Skew 3.1 | log → polynomial | The dominant driver, elasticity 0.86. |
| `days_since_first_purchase` | Skew 1.2 | log → polynomial | Near-zero on its own, real through interactions (§15.6). |
| `days_since_last_purchase` | Skew 1.5; 52 rows exceed the first-purchase date | log → polynomial, **plus** `flag_invalid_recency` | The impossible rows are evidence about the generator, not errors: flagged, value left intact. |
| `product_category_diversity` | Skew 0.6 | log → polynomial | Marginal but kept; removing it costs a little. |
| `loyalty_program_membership` | Categorical, 40% enrolled | Mapped to 0/1 | Contributes nothing (§15.6), kept because the negative result is part of the finding (EDA §5). |
| `estimated_lifetime_value` (target) | Skew 6.4, no zeros | `log()` for fitting, Duan smearing on the way back | Symmetric in logs; `exp(E[log y])` would under-state the level. |
| All numerics | Heavy right tail | Clip to the training min/max **widened by 50%** | An extrapolation guard for unseen data, not tail removal: at these bounds no training row moves. |

### 16.2 The defaults were measured, not assumed

Mean 5-fold CV R² on `log(value)`, Ridge(α = 0.01), the preprocessor refitted inside each fold,
one knob changed at a time:

| Representation | CV R² | Reading |
|---|---|---|
| `--no-log` | 0.47884 | Without the log transform, nothing works. |
| `--poly-degree 1` | 0.98642 | Log-linear; matches EDA §7's 0.9868. |
| **defaults (degree 3)** | **0.99936** | The representation the EDA selected. |
| `--outlier-quantiles .001 .999` | 0.99732 | Winsorising the tail costs accuracy. |
| `--recency-policy clip` | 0.99579 | Repairing the 52 impossible rows costs more. |

The last two lines are why the cleaning is deliberately conservative. EDA §7 established that the
label is a deterministic function of the feature values **as supplied**, so every "repair" deletes
the input that produced its label. The pipeline flags and bounds rather than overwrites, and the
aggressive options stay available behind flags with their cost written down.

### 16.3 Engineered features: the strategy, and what it was worth

`src/feature_engineering.py` owns feature creation and selection. Its strategy follows from one
property of this dataset rather than from habit: EDA §7 showed the target is **multiplicative**,
which is why everything is logged — and in log space a product or ratio of existing columns is a
*linear combination of their logs*:

```
log(count × aov)   = log(count) + log(aov)
log(count / tenure) = log(count) − log(tenure)
```

A linear model cannot gain from a column it can already form as a weighted sum of columns it has,
and the degree-3 expansion already contains every product of up to three logged inputs. So the
usual RFM feature-engineering reflexes are **redundant by construction here** — for the linear
family. A tree cannot form a product at all, so the same columns can help an ensemble.

The catalogue records which side each feature falls on:

| Feature | Formula | Kind | New in log space? |
|---|---|---|---|
| `purchase_value` | count × aov | product | no |
| `purchase_rate` | count / tenure | ratio | no |
| `inter_purchase_days` | tenure / count | ratio | no |
| `value_per_day` | count × aov / tenure | ratio | no |
| `recency_ratio` | last / tenure | ratio | no |
| `dormancy` | last / (tenure / count) | ratio | no |
| **`recency_span`** | tenure − last | difference | **yes** — the BTYD recency (§3.2) |
| **`is_lapsed`** | last > 2 × (tenure / count) | threshold | **yes** — the churn signal of §3 |

**The measurement** (`python main.py --compare-features`, mean 5-fold CV R² on `log(value)`):

| Feature set | Features | Ridge (log, degree 3) | Random Forest (raw) |
|---|---|---|---|
| none (the 6 raw columns) | 119 | **0.99936** | 0.95394 |
| default (2 derived) | 219 | 0.99928 | 0.95710 |
| independent only | 209 | 0.99929 | 0.95357 |
| all (8 derived) | 799 | 0.99963 | **0.96525** |

Both predictions hold. Ridge does not move — four decimal places apart across three feature sets,
which is what "redundant by construction" looks like. The Random Forest gains **+0.011 R²** from
the full set, a quarter of its remaining error, and it gains that from the *products and ratios*:
the independent-only row leaves it exactly where it started. **The features a linear model cannot
use are precisely the ones a tree cannot build.**

Derived features are therefore **off by default**: they cost 680 extra columns and buy the
selected model nothing. They are one flag away for anyone modelling with trees.

One exception, found by a check rather than by reasoning: the redundancy is exact only *above the
positivity floor*. `purchase_rate` falls below 1e-3 for 7 of 800 training customers — a single
order against four years of tenure — and a floored value is no longer a linear combination of
anything. `check_feature_engineering` excludes those rows from its rank test and says how many it
excluded, rather than quietly passing or quietly failing.

### 16.4 Feature selection, and what each strategy costs

119 features from 800 rows, on a basis that is collinear by design. Ridge(0.01), degree 3, mean
5-fold CV R² on `log(value)`:

| Strategy | Features kept | CV R² | Reading |
|---|---|---|---|
| `none` | 119 | 0.99936 | the reference |
| **`variance`** (default) | 119 | 0.99936 | nothing is constant on this data; this is the old constant-column pruning, which matters when a flag is all-zero and the expansion turns it into 36 dead columns |
| `correlation` (≥ 0.999) | 103 | 0.99936 | 16 near-duplicates, free to drop — the one strategy that costs nothing |
| `model` (Ridge \|coef\|, k=40) | 40 | 0.99919 | a third of the basis for a small, measurable cost |
| `mutual_info` (k=40) | 40 | 0.99780 | the largest cost of the four |
| `vif` (> 6.0) | 69 | 0.99917 | drops 50 terms and buys nothing |

**Nothing beats keeping everything**, which is the honest answer on a representation already at
its ceiling. That is also the answer to the obvious review question — "where is the VIF step?" —
with a number attached instead of an argument: on a polynomial basis a high VIF *is* the design,
and `--feature-selection vif` shows what removing it costs.

Selection is a fitted pipeline step, so it is refitted **inside every cross-validation fold**.
Choosing features once on the whole dataset and then cross-validating is one of the most common
ways to publish an inflated score, and it is structurally impossible here.

### 16.5 What was deliberately left out

| Not used | Why |
|---|---|
| **BG/NBD + Gamma-Gamma** (§3) | 988/1,000 non-integer frequencies, 52 negative recencies, and — decisively — no timestamps and no future window to forecast into. Fitting it would describe the rounding, not the customers (EDA §6). |
| **Two-stage hurdle / ZILN loss** (§4.1, §5.1) | Both exist to handle zero-inflation. This file has **no** zero-value customers. |
| **SMOTE and friends** | Regression task; nothing to resample. |
| **VIF-based feature pruning** *(as a default)* | The winning representation is a degree-3 polynomial expansion, whose terms are collinear *by construction*. It is implemented and one flag away, and §16.4 measures what it costs: 50 terms dropped, CV R² 0.99936 to 0.99917. |
| **Uncertainty intervals** | Out of scope here, and the obvious next addition (§5.1 covers the distributional approaches). |

---

## 17. Deployment considerations and limits

### 17.1 If this were shipped

* **Serving.** `outputs/model.joblib` carries the fitted preprocessor, the estimator, the
  smearing factor, the input-column contract and the tier thresholds. `predict.py` is a batch
  scorer; the same bundle sits behind an API unchanged.
* **Tier stability.** Thresholds are fixed at training time, so "VIP" means the same thing in
  every batch. Recomputing them per batch would make a customer's tier depend on their cohort.
* **Unseen-data safety.** The clipper bounds every numeric at the training range widened by 50%,
  so an absurd input degrades into a bounded prediction instead of an extrapolated one. The
  encoder records unseen categories rather than failing.
* **Retraining trigger.** Retrain when the audit's drift signals move: feature medians or skews
  outside their training range, a rise in clipped values at scoring time, or decile calibration
  drifting beyond a few percent on a labelled sample. On real data, retrain per cohort anyway.
* **Monitoring.** Log the share of scored rows hitting a clip bound, the predicted-value
  distribution against training, and — once labels arrive — decile MAPE and Spearman on a
  holdout. Rank metrics and calibration metrics fail independently and both need watching.
* **Interpretability.** Elasticities from the first-order model (§15.6) are the explanation a
  business will act on. Permutation importance is *not* safe on this model; drop-column
  importance is, and it is what the pipeline reports first.

### 17.2 The limits that matter

1. **This is not a 12-month forecast.** No cut-off date, no transaction log, no future window, so
   the calibration/holdout design of §7.1 cannot be built. The model predicts an existing
   estimate, not future spend.
2. **The split is random, not temporal.** Correct for a cross-section, wrong for a forecasting
   claim (§8).
3. **The data is synthetic**, with impossible values, so no finding transfers to real customers —
   including the loyalty result, which describes the generator, not a programme.
4. **No uncertainty quantification.** One point prediction per customer.
5. **The metrics sit at their ceiling**, which makes them useless for separating a good model from
   a great one. On real data the same ladder would spread out.

### 17.3 What to do next

1. **Re-run this pipeline on transactional data.** UCI Online Retail II (§11) has two full years,
   which gives a clean 12-month calibration window and a 12-month holdout. Everything here
   transfers: `CLVPreprocessor`, the log-target handling, the metric set, the baseline ladder. The
   split function is the one piece that must change — from stratified-random to time-based.
2. **Add the reference models this file cannot support**: BG/NBD + Gamma-Gamma, and a two-stage
   hurdle model once zero-value customers exist.
3. **Keep this project as the specification-sensitivity case study.** EDA §5 and §7 make a point
   about analytical practice — that an effect estimated from a mis-specified model is not an
   effect either — that a cleaner dataset would not have made as vividly.
