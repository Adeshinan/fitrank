# FitRank

[![CI](https://github.com/Adeshinan/fitrank/actions/workflows/ci.yml/badge.svg)](https://github.com/Adeshinan/fitrank/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fitrank)](https://pypi.org/project/fitrank/)
[![Python](https://img.shields.io/pypi/pyversions/fitrank)](https://pypi.org/project/fitrank/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Reproducible, config-driven model comparison for scikit-learn. No hidden preprocessing.**

English · [Français](README.fr.md)

FitRank trains several scikit-learn models on your tabular data, tunes each one,
and ranks them in a leaderboard. You describe the whole experiment in one YAML
file: run it again next month, or send it to a colleague, and you get the same
ranking.

## Why FitRank?

FitRank compares scikit-learn models the same way every time, from a YAML file
you can version and share. Nothing happens to your data unless you declare it.

| | FitRank | [LazyPredict](https://github.com/shankarpandala/lazypredict) | [PyCaret](https://pycaret.org) | AutoML ([auto-sklearn](https://automl.github.io/auto-sklearn/), [FLAML](https://microsoft.github.io/FLAML/), [AutoGluon](https://auto.gluon.ai)…) |
|---|---|---|---|---|
| Ranks many models in one run | ✅ | ✅ | ✅ | ✅ (aims at one final model or ensemble) |
| Hyperparameter search per model | ✅ every model | optional, top models after the ranking | ✅ separate step | ✅ |
| Whole experiment in one YAML file | ✅ | ❌ | ❌ | ❌ |
| Preprocessing only when declared | ✅ | ❌ (always imputes, scales and encodes) | ❌ (imputes and encodes by default) | ❌ |
| Machine learning libraries installed | scikit-learn | scikit-learn (boosters optional) | scikit-learn, LightGBM and many more | boosters and more |
| Who decides the pipeline | you | the library | you or the library | the library |

<sub>Checked in each project's documentation on 2026-11-01 (LazyPredict 0.4.0, PyCaret 3.3.2,
auto-sklearn 0.15.0, FLAML 2.7.0, AutoGluon 1.6.3). Projects evolve: corrections are welcome.</sub>

**Use something else if** you want the best possible score with no effort
(→ an AutoML tool) or a full low-code platform with deployment (→ PyCaret).

## Installation

```bash
pip install fitrank
```

Python 3.10 to 3.13. Excel files need `pip install "fitrank[excel]"`, Parquet
files `pip install "fitrank[parquet]"`.

## Quick start

Save some data (here the iris flowers that ship with scikit-learn):

```bash
python -c "from sklearn.datasets import load_iris; load_iris(as_frame=True).frame.to_csv('iris.csv', index=False)"
```

Describe the experiment in `fitrank.yaml`:

```yaml
data:
  path: iris.csv
  target: target          # the column to predict
task: classification
models:
  - name: logistic_regression
  - name: decision_tree
  - name: svm
```

Run it:

```bash
fitrank run fitrank.yaml
```

Each model is tuned on the same cross-validation folds, ranked, then scored on
a test set kept aside. The results are saved in `runs/fitrank_<date>_<time>/`.

## The leaderboard

The [Iris example](examples/iris.yaml) compares 4 models, with and without a
PCA, after scaling the columns. Run from the repository root:

```bash
fitrank run examples/iris.yaml
```

```text
                                                        iris: cv f1_weighted, higher is better
+----------------------------------------------------------------------------------------------------------------------------------------------------+
| # | model               | reduction | cv f1_weighted | cv std | test f1_weighted |  time | best params / error                                     |
|---+---------------------+-----------+----------------+--------+------------------+-------+---------------------------------------------------------|
| 1 | svm                 | none      |         0.9749 | 0.0205 |           0.9333 | 1.3 s | C=0.1, class_weight=None, kernel=linear                 |
| 2 | knn                 | none      |         0.9665 | 0.0167 |           0.9327 | 0.8 s | n_neighbors=3, weights=uniform                          |
| 3 | logistic_regression | none      |         0.9663 | 0.0316 |           1.0000 | 1.1 s | C=10, class_weight=None                                 |
| 4 | random_forest       | none      |         0.9580 | 0.0268 |           0.9666 | 2.2 s | max_depth=3, n_estimators=100                           |
| 5 | svm                 | pca       |         0.9248 | 0.0313 |           0.8997 | 1.3 s | C=0.1, class_weight=None, kernel=linear, n_components=2 |
| 6 | logistic_regression | pca       |         0.9247 | 0.0410 |           0.8997 | 1.1 s | C=10, class_weight=None, n_components=2                 |
| 7 | random_forest       | pca       |         0.9080 | 0.0167 |           0.8997 | 2.2 s | max_depth=None, n_estimators=100, n_components=2        |
| 8 | knn                 | pca       |         0.8911 | 0.0203 |           0.8997 | 0.7 s | n_neighbors=11, weights=distance, n_components=2        |
+----------------------------------------------------------------------------------------------------------------------------------------------------+
Best model: svm + none (f1_weighted = 0.9749)
Results saved to runs/iris_2026-11-01_15-36-06/
```

Models are ranked on the cross-validation score (`cv f1_weighted`): the test
score is shown, never used to choose. The results folder holds the leaderboard
(CSV, JSON, Markdown), the configuration as FitRank read it, the versions used,
and the best model, ready for new data:

```bash
# use the folder name your run printed
fitrank predict runs/iris_2026-11-01_15-36-06/best_model.joblib new_flowers.csv -o predictions.csv
```

[`examples/iris.yaml`](examples/iris.yaml) explains every key of the
configuration; [`examples/diabetes.yaml`](examples/diabetes.yaml) is a
regression example.

## Other commands

```bash
fitrank validate fitrank.yaml        # check the config and the data, train nothing
fitrank list models --task regression
fitrank list reductions
fitrank --quiet run fitrank.yaml     # only the leaderboard, for scripts
fitrank --verbose run fitrank.yaml   # one line per model as soon as it is done
```

Exit codes: `0` success, `1` configuration error, `2` data error, `3` every model failed.

## Use it from Python

The same experiment, from a DataFrame:

```python
from sklearn.datasets import load_iris

from fitrank import Benchmark

df = load_iris(as_frame=True).frame
results = Benchmark(
    data=df,
    target="target",
    task="classification",
    models=["logistic_regression", "svm", {"name": "knn", "params": {"n_neighbors": [3, 5]}}],
    evaluation={"cv": 5, "primary_metric": "f1_macro"},
).run()

print(results.leaderboard()[["rank", "model", "cv_f1_macro", "test_f1_macro"]])
model = results.best_model()  # a fitted scikit-learn Pipeline
print(model.predict(df.drop(columns="target").head()))
```

`Benchmark.from_yaml("fitrank.yaml").run()` runs a YAML file. The keyword
arguments are the YAML sections, and both ways give the same results.

## Add your own model

```python
from sklearn.ensemble import ExtraTreesClassifier

from fitrank import Benchmark, register_model


@register_model("extra_trees", task="classification")
def extra_trees():
    # A fresh estimator, and the grid searched when the config gives no params.
    return ExtraTreesClassifier(), {"n_estimators": [100, 300]}


results = Benchmark(
    data="iris.csv", target="target", task="classification", models=["extra_trees", "svm"]
).run()
```

A registered model is used by name, like the built-in ones. Reductions are
added the same way with `register_reduction`. In v1.0 your own models are
available from Python; loading them from the YAML file is planned for v1.1.

## Available models and steps

| Task | Models |
|---|---|
| classification | `logistic_regression`, `svm`, `knn`, `decision_tree`, `random_forest`, `gradient_boosting`, `naive_bayes` |
| regression | `linear_regression`, `ridge`, `lasso`, `polynomial_regression`, `svr`, `knn`, `decision_tree`, `random_forest`, `gradient_boosting` |

| Section | Names |
|---|---|
| `reduction` | `none`, `pca`, `select_k_best`, `variance_threshold`, `polynomial_features` |
| `preprocessing` steps | `imputer` (mean, median, most_frequent, constant), `scaler` (standard, minmax, robust), `encoder` (onehot, ordinal) |

Each model comes with a small default grid; `params` in the YAML replaces it.
Metrics are the [scikit-learn scorer names](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter):
`f1_weighted` ranks classification by default, the RMSE ranks regression.

## Contributing

Bug reports and ideas are welcome in the [issues](https://github.com/Adeshinan/fitrank/issues).
To work on the code:

```bash
git clone https://github.com/Adeshinan/fitrank.git
cd fitrank
pip install -e ".[dev]"
pre-commit install
pytest                       # add -m "not e2e" to skip the slower end-to-end tests
```

Pull requests run ruff, mypy (strict) and the tests on Linux, macOS and Windows,
Python 3.10 to 3.13.

## License

[MIT](LICENSE) © 2026 Faïd Adéshinan ALAGBE
