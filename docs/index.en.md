# FitRank

**Reproducible, config-driven model comparison for scikit-learn. No hidden preprocessing.**

FitRank trains several scikit-learn models on your tabular data, tunes each one,
and ranks them in a leaderboard. You describe the whole experiment in one YAML
file: run it again next month, or send it to a colleague, and you get the same
ranking.

```yaml
data:
  path: iris.csv
  target: species         # the column to predict
task: classification
models:
  - name: logistic_regression
  - name: decision_tree
  - name: svm
```

```bash
pip install fitrank
fitrank run fitrank.yaml
```

Or try it in your browser, nothing to install:
[open the notebook in Colab](https://colab.research.google.com/github/Adeshinan/fitrank/blob/main/examples/try_fitrank.ipynb) and run all the cells (about 2 minutes).

## What FitRank guarantees

- **The same conditions for every model.** Same train/test split, same
  cross-validation folds, same metric. Models are ranked on the
  cross-validation score; the test score is shown, never used to choose.
- **Nothing happens to your data unless you declare it.** Missing values and
  text columns are reported with the YAML that fixes them; FitRank never fills,
  drops or encodes anything on its own.
- **No data leakage.** Preprocessing lives inside the pipeline, so it is fitted
  on the training folds only.
- **A failing model never stops the run.** It is marked `failed` in the
  leaderboard, with its error.
- **Clear errors.** Each message says what is wrong, where (YAML key or
  column) and how to fix it, without a traceback.

## Why FitRank?

| | FitRank | [LazyPredict](https://github.com/shankarpandala/lazypredict) | [PyCaret](https://pycaret.org) | AutoML ([auto-sklearn](https://automl.github.io/auto-sklearn/), [FLAML](https://microsoft.github.io/FLAML/), [AutoGluon](https://auto.gluon.ai)…) |
|---|---|---|---|---|
| Ranks many models in one run | ✅ | ✅ | ✅ | ✅ (aims at one final model or ensemble) |
| Hyperparameter search per model | ✅ every model | optional, top models after the ranking | ✅ separate step | ✅ |
| Whole experiment in one YAML file | ✅ | ❌ | ❌ | ❌ |
| Preprocessing only when declared | ✅ | ❌ (always imputes, scales and encodes) | ❌ (imputes and encodes by default) | ❌ |
| Who decides the pipeline | you | the library | you or the library | the library |

**Use something else if** you want the best possible score with no effort
(→ an AutoML tool) or a full low-code platform with deployment (→ PyCaret).

## Installation

```bash
pip install fitrank
```

Python 3.10 to 3.13. Excel files need `pip install "fitrank[excel]"`, Parquet
files `pip install "fitrank[parquet]"`.

## Where next

- [Quick start](quickstart.md): a first run, the leaderboard and the results folder.
- [Configuration reference](configuration.md): every YAML key, its type and default.
- [Models](models.md): the built-in models, any scikit-learn class, your own models.
- [Command line](cli.md) and [Python API](api.md).
- Guides: [Classification or regression?](guides/task.md) and [Choosing a metric](guides/metrics.md).
- [FAQ](faq.md).

FitRank is open source (MIT). The code, the issues and the changelog are on
[GitHub](https://github.com/Adeshinan/fitrank).
