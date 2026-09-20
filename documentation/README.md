# Capstone documentation

This folder is the project's **manual**: how the code is organised, what each public function
does, what each model assumes, and how to run the thing in production.

The **report** — the objective, the literature review, the quick start, the findings and every
number — is the root [`../readme.md`](../readme.md). It is the single entry point, and it carries
this index too; nothing here is repeated there.

| Page | Read it when |
|---|---|
| [01-architecture.md](01-architecture.md) | You want the module map, the data flow, and why the code is split the way it is. |
| [02-api-reference.md](02-api-reference.md) | You are calling the package from your own code. All 104 public symbols, by module. |
| [03-models.md](03-models.md) | You want to know what each model is, what it assumes, and when it wins. |
| [04-deployment.md](04-deployment.md) | You are putting this behind an API, a batch job, or a scheduler. |
| [05-operations.md](05-operations.md) | It is running and something needs monitoring, retraining or debugging. |
| [06-review.md](06-review.md) | You are assessing the project, or checking what is verified versus claimed. |

**Before quoting any number from these pages**, read
[`../readme.md` §9](../readme.md#9-reading-the-supplied-dataset-against-this-literature): the
supplied target is already an estimate and a near-deterministic function of the feature columns,
so every score here measures recovery of a generating formula, not forecasting skill. The long
version is in [06-review.md](06-review.md#what-the-numbers-do-and-do-not-mean).
