# FAQ

## Why does FitRank not fill missing values or encode text itself?

Because a silent choice changes your results without telling you. Filling
ages with the mean or the median, dropping rows, encoding categories as
numbers: each choice is a decision about your data. FitRank reports the
problem, with the YAML that fixes it, and you decide. What you declared is
what ran, and the saved `config.yaml` shows it.

## Why are models ranked on the cross-validation score, not the test score?

The test set must stay a fair estimate of what happens on new data. If it
chose the winner, it would become part of the training, and its score would be
too optimistic. The test score is shown so you can check that the winner holds
up, never used to pick it.

## Two models are almost tied. Which one should I pick?

Look at `cv std`. When two scores are closer than their cv std, the data
cannot tell them apart: another `random_state` could swap them. Pick the
simpler or faster one, or the one whose test score holds up best.

## How do I get the same results again?

Keep the same `random_state` (42 by default), the same data and the same
library versions. Every results folder records the configuration as FitRank
read it (`config.yaml`) and the versions used (`metadata.json`). `n_jobs`
does not change the results, only the time.

## `n_jobs: -1` is slower on my data. Is that normal?

Yes, on small datasets. Starting the workers costs more than the work they
share. It pays off when there are many combinations or many rows: the Titanic
example goes from 26 s to 15 s on 12 cores. Keep the default `1` otherwise.

## Can I use XGBoost, LightGBM or my own estimator?

Yes, if it follows the scikit-learn API. Give its class path and allow its
package:

```yaml
project:
  allowed_modules: [xgboost]
models:
  - class: xgboost.XGBClassifier
```

See [Models](models.md#any-scikit-learn-model-by-its-class).

## A model failed. What happens?

The others go on. The failed model is ranked last, marked `failed`, with its
error in the leaderboard. The run fails (exit code `3`) only when every
candidate failed.

## FitRank warns that my target "looks categorical". Is it an error?

No, the run goes on. The target has text, true/false or two values, but the
task is `regression`: you probably want `task: classification`. The opposite
warning appears for a classification on a number with many distinct values.
You decide: the warning never blocks.

## Which metric should I choose?

The default ranks classification with `f1_weighted` (it works with any number
of classes, but hides a rare class) and regression with the RMSE (an error in
the unit of the target).
Choose the one that matches the mistake that costs you the most, for example
`recall_weighted` when missing a positive case is worse than a false alarm.
Any [scikit-learn scorer name](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
works. The guide [Choosing a metric](guides/metrics.md) compares them, with
examples.

## Does FitRank handle images or free text?

Not yet: FitRank works on tabular data (CSV, Excel, Parquet, DataFrame). Leave
free-text columns out with `data.exclude`, or turn them into numeric columns
before.
