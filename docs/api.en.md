# Python API

Everything the YAML file does is available from Python, from a file or a
pandas DataFrame. The keyword arguments of `Benchmark` are the YAML sections,
and both ways give the same results.

```python
import pandas as pd

from fitrank import Benchmark

df = pd.read_csv("iris.csv")  # the file from the quick start
results = Benchmark(
    data=df,
    target="species",
    task="classification",
    models=["logistic_regression", "svm", {"name": "knn", "params": {"n_neighbors": [3, 5]}}],
    evaluation={"cv": 5, "primary_metric": "f1_macro"},
).run()

print(results.leaderboard()[["rank", "model", "cv_f1_macro", "test_f1_macro"]])
model = results.best_model()  # a fitted scikit-learn Pipeline
print(model.predict(df.drop(columns="species").head()))
```

A YAML file runs the same way:

```python
results = Benchmark.from_yaml("fitrank.yaml").run()
```

User errors raise `ConfigError` or `DataError`, both `FitRankError`, with
the same messages as the command line.

## Compare two rankings

`compare_rankings` measures how much two runs agree on the order of the
models, e.g. on clean and on noisy training data, scored on the same test set:

```python
from fitrank import Benchmark, compare_rankings

clean = Benchmark(data=train_clean, test_data=test, **setup).run(export=False)
noisy = Benchmark(data=train_noisy, test_data=test, **setup).run(export=False)

cmp = compare_rankings(clean, noisy)
cmp.kendall_tau  # 1 = same order, 0 = unrelated, -1 = reversed
cmp.top1_agree  # the same best model on both sides?
cmp.regret  # test score lost, on the clean side, by taking the noisy winner
cmp.table  # model, reduction, reference_rank, other_rank, rank_shift, scores
cmp.notes  # candidates left out, ties at the top...
```

- Candidates are matched by (model, reduction). One that failed, or that
  exists on one side only, is left out with a note.
- Both sides must use the same primary metric. They are ranked on the CV
  score, as in the leaderboard; `by="test"` ranks them on the test score.
- `kendall_tau` is Kendall's tau-b: tied scores count as ties.
- A tie at the top is broken by the leaderboard order, with a note:
  `top1_agree` and `regret` depend on it.
- Each side can also be a leaderboard DataFrame, or the `leaderboard.csv` /
  `leaderboard.json` of a run folder.

With 5 to 8 models, tau takes few values and moves from one `random_state`
to another. Before reading a change of tau as an effect of the data, compute
the tau between two seeds on the same data: that is the size of the noise.

## Out-of-fold probabilities

`results.oof_proba(model)` gives, for every training row, the class
probabilities predicted by a model that never saw that row: the candidate's
pipeline (preprocessing, reduction, model) with its best parameters, refitted
on the other folds of the run's own cross-validation. They feed label-error
detection (e.g. `cleanlab.filter.find_label_issues`), error analysis or
stacking:

```python
results = Benchmark(data=train, test_data=test, **setup).run(export=False)

proba = results.oof_proba("random_forest")  # one row per training row, one column per class
labels = results.y_train.loc[proba.index]  # the observed labels, in the same order
confidence = proba.to_numpy()[range(len(proba)), proba.columns.get_indexer(labels)]
suspects = proba.index[confidence < 0.2]  # rows the model hardly believes
```

- The index is the one of the training rows (`results.X_train`): with
  `test_size`, the rows not put aside; with `test_data`, all of `data`.
- Classification only. A model without `predict_proba` (`svm`) is refused
  with a clear error.
- With several reductions, name one: `oof_proba("knn", reduction="pca")`.
- Nothing is computed by `run()`: the first call fits the folds, later calls
  return a copy of the same result.

The rows are not fully out-of-sample: each row is predicted by a model that
did not see it, but the best parameters were chosen with these same folds,
so with every row. This is common practice for label-error detection.
`params="default"` is stricter: parameters given a single value in `params`
(fixed in advance) are kept, the searched ones use the scikit-learn defaults.

## Reference

::: fitrank.Benchmark

::: fitrank.Results

::: fitrank.compare_rankings

::: fitrank.RankingComparison

::: fitrank.load_model

::: fitrank.register_model

::: fitrank.register_reduction

::: fitrank.FitRankError

::: fitrank.ConfigError

::: fitrank.DataError

::: fitrank.FitRankWarning
