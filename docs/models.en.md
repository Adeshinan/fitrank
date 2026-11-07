# Models

There are three ways to put a model in the comparison: by its FitRank name, by
its scikit-learn class path, or by registering your own.

## Built-in models

Each model comes with a small default grid, searched when the config gives no
`params`. `fitrank list models --task classification` shows the names.

### Classification

| Name | scikit-learn class | Default grid |
|---|---|---|
| `decision_tree` | `DecisionTreeClassifier` | `{class_weight: [null, balanced], max_depth: [null, 5, 10]}` |
| `gradient_boosting` | `GradientBoostingClassifier` | `{learning_rate: [0.05, 0.1], n_estimators: [100, 200]}` |
| `knn` | `KNeighborsClassifier` | `{n_neighbors: [3, 5, 11], weights: [uniform, distance]}` |
| `logistic_regression` | `LogisticRegression` | `{C: [0.1, 1, 10], class_weight: [null, balanced]}` |
| `naive_bayes` | `GaussianNB` | `{var_smoothing: [1.0e-09, 1.0e-08, 1.0e-07]}` |
| `random_forest` | `RandomForestClassifier` | `{class_weight: [null, balanced], max_depth: [null, 10], n_estimators: [100, 300]}` |
| `svm` | `SVC` | `{C: [0.1, 1, 10], class_weight: [null, balanced], kernel: [linear, rbf]}` |

### Regression

| Name | scikit-learn class | Default grid |
|---|---|---|
| `decision_tree` | `DecisionTreeRegressor` | `{max_depth: [null, 5, 10]}` |
| `gradient_boosting` | `GradientBoostingRegressor` | `{learning_rate: [0.05, 0.1], n_estimators: [100, 200]}` |
| `knn` | `KNeighborsRegressor` | `{n_neighbors: [3, 5, 11], weights: [uniform, distance]}` |
| `lasso` | `Lasso` | `{alpha: [0.01, 0.1, 1]}` |
| `linear_regression` | `LinearRegression` | `{fit_intercept: [true, false]}` |
| `polynomial_regression` | `Pipeline` (`PolynomialFeatures` + `LinearRegression`) | `{poly__degree: [2, 3]}` |
| `random_forest` | `RandomForestRegressor` | `{max_depth: [null, 10], n_estimators: [100, 300]}` |
| `ridge` | `Ridge` | `{alpha: [0.1, 1, 10]}` |
| `svr` | `SVR` | `{C: [0.1, 1, 10], kernel: [linear, rbf]}` |

`params` in the YAML replaces the default grid, it is never merged with it:

```yaml
models:
  - name: random_forest
    params: {n_estimators: [500]}     # only n_estimators=500 is tried
```

## Any scikit-learn model, by its class

A model that FitRank does not name can be given by its class path:

```yaml
models:
  - name: random_forest
  - class: sklearn.ensemble.ExtraTreesClassifier
    params: {n_estimators: [100, 300]}   # without params: its default parameters
```

It appears in the leaderboard as `ExtraTreesClassifier`. FitRank checks that
the class exists, is a scikit-learn estimator, matches the task (a regressor is
refused for classification) and can be created without arguments.

!!! warning "Only trusted modules are imported"
    A YAML file must never be able to run code. FitRank only imports classes
    from `sklearn`: any other path (`os.system`, `builtins.eval`...) is refused
    **before anything is imported**. To use another scikit-learn compatible
    package that you trust, list it in `project.allowed_modules`:

    ```yaml
    project:
      allowed_modules: [xgboost, my_package.models]   # must be installed or importable
    models:
      - class: xgboost.XGBClassifier
    ```

## Register your own model

From Python, `register_model` adds a model under a name, with its default
grid. It is then used by name, like the built-in ones:

```python
from sklearn.ensemble import ExtraTreesClassifier

from fitrank import Benchmark, register_model


@register_model("extra_trees", task="classification")
def extra_trees():
    # A fresh estimator, and the grid searched when the config gives no params.
    return ExtraTreesClassifier(), {"n_estimators": [100, 300]}


results = Benchmark(
    data="iris.csv", target="species", task="classification", models=["extra_trees", "svm"]
).run()
```

The factory is called once per candidate, so each one gets a fresh estimator.
The same name can be registered once per task. From the YAML file, give your
model by its class path instead, with its package in `project.allowed_modules`.

## Register your own reduction

`register_reduction` works the same way. The factory receives the task and
returns a transformer and its default grid:

```python
from sklearn.decomposition import TruncatedSVD

from fitrank import register_reduction


@register_reduction("svd")
def svd(task):
    return TruncatedSVD(), {"n_components": [2, 5]}
```

```yaml
reduction:
  - none
  - svd: {n_components: [3]}
```
