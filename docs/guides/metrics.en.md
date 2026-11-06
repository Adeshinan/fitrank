# Choosing a metric

What each metric measures, when to use it, its traps, and how the choice
changes a FitRank run. Every number on this page was computed with
scikit-learn.

There is no "best metric" in general. The right choice answers one question:

> **Which mistake costs you the most?**

## Where the names come from

- **The formulas** (F1, RMSE, R²…) are statistical tools, older than
  scikit-learn. The F1 comes from C. J. van Rijsbergen's "F-measure"
  (information retrieval, 1979).
- **The names** you write in the YAML (`f1_weighted`,
  `neg_root_mean_squared_error`…) are scikit-learn's scorer names. To list them:
  ```bash
  python -c "from sklearn.metrics import get_scorer_names; print(get_scorer_names())"
  ```
- **FitRank** invents no name. It checks that the name exists (and suggests a
  fix for a typo), chooses the defaults (`f1_weighted` for classification, the
  RMSE for regression) and shows errors as positive numbers.
- The **`neg_`** prefix is a scikit-learn convention: it always looks for the
  highest score, so errors are stored negative. An RMSE of 55 is stored as −55;
  FitRank shows 55.

### One primary metric

```yaml
evaluation:
  primary_metric: f1_macro                        # one: ranks the models
  metrics: [accuracy, recall_macro, roc_auc_ovr]  # several: shown for information
```

The primary metric is used **twice**:

1. inside each candidate, to choose the best parameters (the `refit` of
   `GridSearchCV`);
2. between the candidates, to choose the best model.

There is only one because with two criteria there is often no winner. If A is
better on F1 and B on accuracy, only you know which matters more.

## Classification

### The running example: 100 patients, 10 sick, 90 healthy

Three models are compared:

- **Our model**: finds 6 sick patients, misses 4, raises 3 false alarms.
- **Lazy**: always says "healthy".
- **Alarmist**: always says "sick".

| Metric | Our model | Lazy | Alarmist |
|---|---|---|---|
| `accuracy` | 0.93 | **0.90** ⚠️ | 0.10 |
| `balanced_accuracy` | 0.78 | 0.50 | 0.50 |
| `precision` | 0.67 | 0.00 | 0.10 |
| `recall` | 0.60 | 0.00 | **1.00** ⚠️ |
| `f1` | 0.63 | 0.00 | 0.18 |
| `f1_macro` | 0.80 | 0.47 | 0.09 |
| `f1_weighted` | 0.93 | **0.85** ⚠️ | 0.02 |
| `matthews_corrcoef` | 0.59 | 0.00 | 0.00 |

⚠️ = the metric is fooled by a useless model.

### `accuracy`: the share of right answers

```text
accuracy = right answers / total
```

- **Good for:** **balanced** classes (iris: 50/50/50, handwritten digits) where
  every mistake costs the same.
- **Trap:** with unbalanced classes, the lazy model scores 0.90 by doing
  nothing.
- **If badly chosen:** a model that ignores the rare class can rank first. On a
  fraud dataset with 8 % of frauds, a model that never flags a fraud still
  reaches 0.92.

### `balanced_accuracy`: accuracy, corrected

```text
balanced_accuracy = mean of the recall of each class
                  = (6/10 sick found + 87/90 healthy found) / 2 = 0.78
```

- **Good for:** **unbalanced** classes, when you want a number as easy to read
  as accuracy.
- **Landmark:** 0.50 = chance (2 classes). The lazy and the alarmist models
  both fall to 0.50.

### `precision`: "when it says yes, is it right?"

```text
precision = true positives / everything it flagged = 6 / 9 = 0.67
```

- **Good when a false alarm is expensive:**
    - a spam filter: sending a real email to spam is worse than letting an ad through;
    - accusing someone of fraud, recommending an expensive product.
- **Trap:** a model that flags almost nothing can have a perfect precision.
  Never use it alone as the primary metric.

### `recall`: "of the real cases, how many did it find?"

```text
recall = true positives / all real cases = 6 / 10 = 0.60
```

- **Good when missing a case is expensive:**
    - cancer screening: an undetected patient may die, a false alarm costs one more test;
    - fraud detection, a defective part on a production line.
- **Trap:** the alarmist has a recall of 1.00 by calling everyone sick. Never
  use it alone either.

### The precision / recall trade-off

The two pull against each other: a cautious model gains precision and loses
recall, an eager model does the opposite.

```text
Spam filter      →  favour precision  (never block a real email)
Cancer           →  favour recall     (never miss a patient)
Both matter      →  F1
```

### The F1 family: both at once

```text
F1 = 2 × precision × recall / (precision + recall)
```

It is a **harmonic** mean, which punishes imbalance: the alarmist (precision
0.10, recall 1.00) has a plain mean of 0.55, but an F1 of 0.18.

The formula is always the same. What changes is **which class** it is computed
on and **how the classes are combined**:

| Variant | Computation | Patients |
|---|---|---|
| `f1` | class `1` (sick) only | 0.63 |
| `f1_macro` | plain mean of the per-class F1: (0.63 + 0.96) / 2 | 0.80 |
| `f1_weighted` | mean weighted by the number of rows: (0.63 × 10 + 0.96 × 90) / 100 | 0.93 |
| `f1_micro` | all cases pooled, equal to accuracy | 0.93 |

| Variant | When | Trap |
|---|---|---|
| `f1` | 2 classes (0 / 1), **one** of them matters | fails with 3 classes or more, and with a text target |
| `f1_macro` | **unbalanced** classes, **all** of them matter | a small class weighs as much as a big one |
| `f1_weighted` | **balanced** classes, overall grade | **hides** rare classes: 0.85 for the lazy model |
| `f1_micro` | never in FitRank | identical to accuracy |

`f1_weighted` is FitRank's default because it **never fails** (2 classes or
more, any class names). It is **safe**, not **better**.

With 3 classes (A: 80 rows, B: 15, C: 5, badly recognised):

```text
F1 per class: A = 0.96   B = 0.90   C = 0.33

f1_macro    = 0.73   ← sees that C is badly recognised
f1_weighted = 0.92   ← C weighs only 5 %
f1          = error (3 classes)
```

### `matthews_corrcoef` (MCC): the most honest

A coefficient between −1 and +1 that uses the 4 cells of the confusion matrix
(found, missed, false alarms, rightly rejected).

```text
+1 = perfect     0 = chance or useless model     −1 = always wrong
```

- **Good for:** **very unbalanced** classes, when you want one number that
  cannot be fooled. The lazy and the alarmist models both get 0.
- **Limit:** harder to explain.

### `roc_auc`: how well it sorts, without a threshold

A model often gives a probability ("sick at 73 %"), then cuts at 50 %. The AUC
answers another question:

> If I pick one sick and one healthy patient at random, does the model give
> the sick one a higher probability?

```text
Truth:          healthy  healthy  healthy  sick  sick
Probabilities:  0.10     0.20     0.30     0.40  0.45

accuracy (50 % threshold) = 0.60   → no sick patient detected
roc_auc                   = 1.00   → the sorting is perfect, the threshold is misplaced
```

- **Good for:** judging a model before choosing the threshold (a credit risk
  score, where the bank picks the score above which it refuses).
- **Variants:** `roc_auc` (2 classes, works with `predict_proba` or
  `decision_function`, so with `svm` too), `roc_auc_ovr` / `roc_auc_ovo`
  (3 classes or more, need `predict_proba`).

### `neg_log_loss`: punishes confident mistakes

Two models with the same accuracy (0.75) on 4 cases:

```text
                 case 1   case 2   case 3   case 4 (truth = 1)
Confident:       0.99     0.01     0.99     0.01   ← wrong with 99 % certainty
Cautious:        0.80     0.20     0.80     0.40   ← wrong while hesitating

log_loss: confident = 1.16     cautious = 0.40   (lower is better)
```

- **Good when you use the probabilities themselves**, not only the class. For
  example an insurer that sets a price from "12 % risk of an accident".
- **Needs** `predict_proba`: on `svm`, FitRank skips it with a note when it is
  secondary, and the candidate fails when it is primary.

## Regression

### The running example: the price of 5 houses (in k€)

```text
True price : 200   250   300   350   400
Model A    : 210   240   310   340   410    ← always 10 off
Model B    : 200   250   300   350   450    ← perfect, except ONE big error of 50
```

| Metric | Model A | Model B | Winner |
|---|---|---|---|
| MAE (mean absolute error) | 10.0 | 10.0 | tie |
| **RMSE** (root mean squared error) | **10.0** | **22.4** | **A** |
| MedAE (median absolute error) | 10.0 | **0.0** | **B** |
| max_error (worst error) | 10 | 50 | A |
| MAPE (error in %) | 3.5 % | 2.5 % | B |
| R² | 0.98 | 0.90 | A |

Same data: depending on the metric, A or B wins.

### `neg_mean_absolute_error` (MAE): the mean error, in real units

```text
MAE = mean of |true − predicted|      → "on average, I am 10 k€ off"
```

- **Good when every error costs in proportion**: an error of 50 costs 5 times
  an error of 10, no more.
- **Good with outliers** (a €5M house in a €300k neighbourhood) that should not
  dominate everything.
- **The easiest to explain** to someone outside data science.

### `neg_root_mean_squared_error` (RMSE): FitRank's default

```text
RMSE = square root of the mean of the SQUARED errors
       the error of 50 → 2500: it crushes the others
```

- **Good when a big error is serious**: electricity demand forecasts (a large
  underestimate causes a blackout), drug dosage, warehouse stock.
- Same unit as the target, so easy to read.
- **Trap:** a single outlier can blow up the RMSE and sway the whole ranking.

### `neg_median_absolute_error` (MedAE): ignores the exceptions

```text
Errors of B: 0, 0, 0, 0, 50   → median = 0
```

- **Good when the data has known outliers** (typing mistakes, exceptional
  cases) and you judge the typical case.
- **Trap:** it ignores big errors. B gets a perfect score although it was 50 k€
  off.

### `neg_max_error`: the worst error

- **Good when one error above a threshold is unacceptable**: machining
  tolerance, safety.
- **Trap:** it looks at one row and ignores all the others.

### `neg_mean_absolute_percentage_error` (MAPE): the error in %

```text
MAPE = mean of |error| / true     → "I am 3.5 % off on average"
```

- **Good for comparing different scales**: being €100 off on €1,000 (10 %) is
  worse than on €100,000 (0.1 %), for example the sales of a small shop and a
  hypermarket.
- **Big trap: small values.**
  ```text
  True      :  1     100    1000
  Predicted :  3     102    1002      ← 2 off everywhere

  MAE  = 2
  MAPE = 67 %     (the "1 → 3" counts as a 200 % error)
  ```
  When the target can be 0, the MAPE becomes infinite.

### `r2` (R²): how much of the variation is explained

```text
R² = 1     → perfect
R² = 0     → no better than always predicting the mean
R² < 0     → worse than predicting the mean
```

- **Good for communicating** ("the model explains 90 % of the variation in
  price") and for comparing problems on different scales.
- **Trap:** it does not say how far off you are in real units. Use it next to
  the RMSE or the MAE.

### `neg_root_mean_squared_log_error` (RMSLE): relative errors on spread-out data

```text
True     : 100    1,000    100,000
Model 1  : 200    1,000    100,000     ← doubles a small price
Model 2  : 100    1,000    100,100     ← +100 on a large price (0.1 %)

RMSE  : 57.7 and 57.7        (identical: same absolute error of 100)
RMSLE : 0.397 and 0.0006     (doubling a price is serious, 0.1 % is not)
```

- **Good for very spread-out, positive targets**: house prices across a whole
  country, salaries, video views.
- **Needs** values ≥ 0.

## How to choose

### The decision tree

```text
CLASSIFICATION
│
├─ Are the classes balanced?
│     YES → f1_weighted (default) or accuracy
│     NO  ↓
│
├─ Does only one class matter (0/1 target)?
│     YES → What costs the most?
│            • missing a case     → recall    (+ f1 as secondary)
│            • a false alarm      → precision (+ f1 as secondary)
│            • both               → f1
│     NO  → f1_macro, balanced_accuracy or matthews_corrcoef
│
└─ Will you use the probabilities (scores, threshold chosen later)?
      → roc_auc (sorting) or neg_log_loss (accurate probabilities)


REGRESSION
│
├─ Is a big error serious?                    YES → RMSE (default)
├─ Outliers that should not weigh too much?   YES → MAE
├─ Very different scales?
│     • values never close to 0  → MAPE
│     • very spread out, > 0     → RMSLE
└─ To communicate                             → r2 as secondary
```

### The impact of the choice, at three levels

1. **On the ranking.** Another model can come first: the model that never
   flags a fraud can be first on accuracy and last on `f1_macro`.
2. **On the chosen parameters.** The primary metric also drives
   `GridSearchCV` when it picks C, max_depth… With `recall`, it tends to prefer
   `class_weight: balanced`, which pushes the model to flag more rare cases.
   The same model comes out different.
3. **On the behaviour in production.** The winning model optimises what was
   measured. Measuring accuracy on fraud means deploying a model that does not
   detect fraud, and finding out only once it is in service.

### What counts, and what does not, in the FitRank ranking

The ranking rests on one number: the **mean cross-validation score** of the
primary metric (the `cv_…` column), measured under identical conditions for
all (same split, same folds, same `random_state`).

| Column | Role |
|---|---|
| `cv_…` | **ranks** the models |
| `test_…` | **checks** the winner, never chooses it |
| `cv_std` | stability across folds, for you to interpret |
| secondary metrics | understand where the score comes from |
| `time` | information, not part of the ranking |

Limits to keep in mind:

- **Small gaps often mean nothing.** On the Diabetes example, lasso (RMSE
  55.36) and ridge (55.37) are 0.02 apart, while the folds vary by about ±2.35.
  They are equally good: take the simpler one.
- **Look at `cv_std`**: 0.90 ± 0.01 is more reliable than 0.91 ± 0.08.
- **Compare `cv` and `test`**: a much lower test score signals overfitting, or
  a test set that is too small.
- **On little data, everything is noisy**: the ranking can change with
  `random_state`.

### YAML examples by case

```yaml
# Iris, handwritten digits: balanced classes
evaluation:
  primary_metric: f1_weighted
  metrics: [accuracy]

# Screening for a disease (0/1 target): miss no one
evaluation:
  primary_metric: recall
  metrics: [precision, f1, balanced_accuracy]

# Spam filter (0/1 target): never block a real email
evaluation:
  primary_metric: precision
  metrics: [recall, f1]

# Fraud, churn: unbalanced classes, all of them matter
evaluation:
  primary_metric: f1_macro
  metrics: [balanced_accuracy, matthews_corrcoef, roc_auc]

# House prices in a city: a big error is expensive
evaluation:
  primary_metric: neg_root_mean_squared_error
  metrics: [neg_mean_absolute_error, r2]

# Sales of shops of very different sizes
evaluation:
  primary_metric: neg_mean_absolute_percentage_error
  metrics: [neg_mean_absolute_error, r2]
```

### Three rules to remember

1. **Choose the metric BEFORE looking at the results.** Switching metrics until
   your favourite model wins is fooling yourself.
2. **One primary metric, several secondary ones.** The primary decides, the
   secondary ones check there is no trap (a recall of 1.00 with a precision of
   0.10, for example).
3. **Always compare with the lazy model.** An accuracy of 0.90 means nothing if
   90 % of the rows are in the same class.

!!! note "A current limit"
    FitRank checks that a metric name **exists**, not that it **fits the
    task**: `task: classification` with `primary_metric: r2` is accepted
    today, and so are clustering scores such as `adjusted_rand_score`. A check
    of the metric against the task may come in a later version.
