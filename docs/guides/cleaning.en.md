# Measure the effect of a data cleaning

You suspect that some labels in your training data are wrong, and you have a
way to remove the suspicious rows. Two questions follow:

1. **Do the wrong labels change which model wins?**
2. **Does the cleaning bring the right ranking back?**

FitRank answers both with the same experiment run several times, changing only
the training rows. This guide shows the protocol, how to read the numbers and
the traps. The notebook
[label_noise_demo.ipynb](https://colab.research.google.com/github/Adeshinan/fitrank/blob/main/examples/label_noise_demo.ipynb)
runs it end to end in about a minute.

## The protocol

Every run must differ by **the training rows only**: same models, same folds,
same metric, same test set.

```python
from fitrank import Benchmark, compare_rankings

setup = {
    "target": "Class",
    "task": "classification",
    "models": [
        "logistic_regression",
        "knn",
        "random_forest",
        {"name": "svm", "calibrate": "sigmoid"},
    ],
    "evaluation": {"cv": 3, "primary_metric": "f1_macro"},
}

clean = Benchmark(data=train, test_data=test, **setup).run(export=False)
noisy = Benchmark(data=train_noisy, test_data=test, **setup).run(export=False)
cleaned = Benchmark(data=train_cleaned, test_data=test, **setup).run(export=False)

before = compare_rankings(clean, noisy)
after = compare_rankings(clean, cleaned)
```

1. **Split once.** Cut your data into `train` and `test` one time, before
   anything else, and keep both files.
2. **Keep the test set clean and fixed.** Give it with `test_data`: every run
   is scored on exactly the same rows, and nothing from it is used for
   training. Never clean the test set with the method you are evaluating:
   it would grade its own work.
3. **Change only the training rows**: `train` (the reference), `train_noisy`
   (with wrong labels, real or added on purpose) and `train_cleaned` (after
   your cleaning).
4. **Compare each ranking with the reference** using `compare_rankings`.

With real data, you have no clean version: compare the ranking before and
after the cleaning, `compare_rankings(noisy, cleaned)`. It tells you whether
the cleaning changes the decision, not whether it is right.

## Finding the suspicious rows

`oof_proba` gives, for every training row, the probabilities predicted by a
model that never saw it. The simplest rule removes the rows whose observed
label gets the lowest probability (*low confidence*):

```python
import numpy as np

proba = noisy.oof_proba("svm (calibrated: sigmoid)")
labels = noisy.y_train.loc[proba.index]
confidence = proba.to_numpy()[range(len(proba)), proba.columns.get_indexer(labels)]

suspects = proba.index[np.argsort(confidence)[: int(0.2 * len(proba))]]  # the 20 % least believed
train_cleaned = train_noisy.drop(index=suspects)
```

The same probabilities feed other methods, e.g.
`cleanlab.filter.find_label_issues(labels, proba.to_numpy())`. A model with no
probabilities (`svm`) needs `calibrate`: see
[Calibrated models](../configuration.md#calibrated-models).

## Reading the numbers

| Number | Meaning | Good sign after cleaning |
|---|---|---|
| `kendall_tau` | Agreement on the whole order: 1 = same, 0 = unrelated, -1 = reversed. | Closer to 1 than before. |
| `top1_agree` | The same best model on both sides? | `True` |
| `regret` | Test score lost, on the clean side, by choosing the other side's winner. | Closer to 0. |
| `table` | Each candidate's rank on both sides (`rank_shift`). | Small shifts. |

In the notebook (Dry Bean, 5,000 beans, 20 % wrong labels, 6 models):

| | `kendall_tau` | `top1_agree` | `regret` |
|---|---|---|---|
| clean vs noisy | 0.73 | True | 0.000 |
| clean vs cleaned | 1.00 | True | 0.000 |

The wrong labels swapped three models in the middle of the ranking; the
cleaned data gave the clean order back.

## The traps

**The size of the noise.** With 5 to 8 models, tau takes few values: one swap
between neighbours already moves it by 0.1 to 0.2. Before reading a change as
an effect of the cleaning, measure what changes when nothing changes: the tau
between two seeds on the same data.

```python
other_seed = Benchmark(data=train, test_data=test, project={"random_state": 1}, **setup)
baseline = compare_rankings(clean, other_seed.run(export=False))
print(baseline.kendall_tau)  # differences smaller than this are noise
```

Then repeat the whole experiment with several seeds (the split, the noise and
`random_state`) and report the mean and the spread, not one run.

**Cross-validation scores lie on noisy data.** They are computed on the noisy
labels too, so they drop for every model. The test scores, on clean labels,
show the real effect. `compare_rankings(..., by="test")` ranks on them.

**Removing rows removes good ones too.** Count how many removed rows were
really wrong when you can (noise added on purpose), try several thresholds,
and watch the rare classes: they lose rows first.

**The detector was tuned on the same folds.** The best parameters behind
`oof_proba` were chosen with every row. `oof_proba(..., params="default")` is
stricter.

**Calibration.** The low-confidence rule trusts the probabilities. `sigmoid`
suits small datasets; `isotonic` is more flexible but can overfit below about
1,000 rows. Add `neg_brier_score` to `evaluation.metrics` to see which one
gives better probabilities.
