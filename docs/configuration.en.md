# Configuration reference

A FitRank configuration is a YAML file made of sections. Only `data`, `task`
and `models` are required; every other key has a default. An unknown key is an
error that lists the allowed keys, so a typo never goes unnoticed.

```yaml
project:        # optional: names, folders, reproducibility
data:           # required: the file, the target, the features
task:           # required: classification or regression
preprocessing:  # optional: what to do to which columns
reduction:      # optional: dimensionality reductions to try
models:         # required: the models to compare
evaluation:     # optional: split, folds, search, metrics
output:         # optional: what is saved
```

`fitrank validate config.yaml` checks a file without training anything. The
configuration as FitRank read it, defaults included, is saved in each results
folder as `config.yaml`.

## project

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | text | `fitrank` | Names the results folder: `runs/<name>_<date>_<time>/`. |
| `output_dir` | path | `runs` | Where the results folders are created. |
| `random_state` | whole number | `42` | The same number gives the same split, the same folds and the same results. |
| `n_jobs` | whole number | `1` | CPU cores for the search; `-1` uses all of them. `0` and values below `-1` are refused. |
| `allowed_modules` | list of module names | `[]` | Packages that [`models[].class`](models.md#any-scikit-learn-model-by-its-class) may import, besides `sklearn`. |

```yaml
project:
  name: titanic
  random_state: 42
  n_jobs: -1
```

`n_jobs: -1` gives the same results, faster on large searches (the Titanic
example goes from 26 s to 15 s on 12 cores). On small datasets, starting the
workers costs more than it saves: keep `1`. A model that sets `n_jobs` in its
`params` while `project.n_jobs` is not `1` is refused, so cores are never used
twice.

## data

| Key | Type | Default | Description |
|---|---|---|---|
| `path` | path | required | A `.csv`, `.xlsx` or `.parquet` file. A relative path starts from the folder the command runs in. |
| `target` | column name | required | The column to predict. |
| `features` | `all` or list of columns | `all` | The columns to train on. `all` means every column except the target. |
| `exclude` | list of columns | `[]` | With `features: all`, the columns to leave out (ids, free text...). |

```yaml
data:
  path: data/titanic.csv
  target: Survived
  features: all
  exclude: [PassengerId, Name, Ticket]
```

Use either a list of `features` or `exclude`, not both. A column that does not
exist is reported with the available columns and the closest name. Rows with a
missing target are reported, never dropped.

## task

`classification` (predict a category: survived or not, a species...) or
`regression` (predict a number: a price, a duration...). You decide: FitRank
only warns, without stopping, when the target does not look like the task,
for example a regression on a target with two values.

## preprocessing

A mapping of groups. Each group applies its `steps`, in order, to its
`columns`. Columns outside every group are passed through unchanged.

```yaml
preprocessing:
  numeric:                       # any group name
    columns: [Age, Fare]
    steps:
      - imputer: {strategy: median}
      - scaler: standard
  categorical:
    columns: [Sex, Embarked]
    steps:
      - imputer: most_frequent
      - encoder: onehot
```

| Step | Written as | What it does |
|---|---|---|
| `imputer` | `imputer` (mean), `imputer: median`, `imputer: {strategy: constant, fill_value: 0}` | Fills missing values. Strategies: `mean`, `median`, `most_frequent`, `constant`. Other keys are [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) options. |
| `scaler` | `scaler` (standard), `scaler: minmax`, `scaler: robust` | Puts numbers on the same scale. |
| `encoder` | `encoder` (onehot), `encoder: ordinal` | Turns categories into numbers. `onehot` ignores categories unseen in training; `ordinal` codes them `-1`. |

Rules:

- A column belongs to one group only, and a step appears once per group.
- A column with missing values needs an `imputer`, a text column needs an
  `encoder`: otherwise `fitrank validate` lists them, with a YAML snippet.
- The steps are fitted inside each cross-validation fold, on the training rows
  only: the scores are not inflated by data leakage.

## reduction

A list of dimensionality reductions; each one is combined with each model.
Default: `[none]`.

```yaml
reduction:
  - none                              # the columns as they are
  - pca: {n_components: [3, 5]}       # values in [] are searched, like params
```

| Name | Default grid | What it does |
|---|---|---|
| `none` | — | Keeps the columns as they are. |
| `pca` | `{n_components: [0.95]}` | Projects the columns onto fewer axes (0.95 = keep 95 % of the variance). |
| `select_k_best` | `{k: [5, 10]}` | Keeps the k columns most related to the target. |
| `variance_threshold` | `{threshold: [0.0, 0.01]}` | Drops columns that barely vary. |
| `polynomial_features` | `{degree: [2]}` | Adds products and powers of the columns. |

Options written after the name replace the default grid. With 2 reductions
and 4 models, FitRank evaluates 8 candidates.

## models

A list of at least one model. Each model has a `name` or a `class`, and
optional `params`.

```yaml
models:
  - name: logistic_regression                   # default grid
  - name: random_forest
    params: {n_estimators: [100, 300], max_depth: [null, 10]}
  - class: sklearn.ensemble.ExtraTreesClassifier
    params: {n_estimators: [100]}
```

| Key | Type | Description |
|---|---|---|
| `name` | model name | A FitRank model: see [Models](models.md). |
| `class` | class path | Any scikit-learn estimator, e.g. `sklearn.svm.LinearSVC`. |
| `params` | mapping of lists | The values to try for each parameter. Replaces the default grid; `null` is Python's `None`. |

Without `params`, a named model searches its small default grid, and a class
uses its default parameters. A parameter the model does not have is refused,
with the closest valid name.

## evaluation

| Key | Type | Default | Description |
|---|---|---|---|
| `test_size` | number between 0 and 1 | `0.2` | Share of the rows kept aside for the final test score. |
| `cv` | whole number ≥ 2 | `5` | Number of cross-validation folds on the other rows. |
| `search` | `grid` or `random` | `grid` | `grid` tries every combination of `params`; `random` tries `n_iter` of them. |
| `n_iter` | whole number ≥ 1 | `20` | With `search: random`, combinations tried per candidate (at most the grid size). |
| `primary_metric` | scorer name | `f1_weighted` / `neg_root_mean_squared_error` | Ranks the models (classification / regression default). |
| `metrics` | list of scorer names | see below | Also reported on the test set. |

The metrics are the
[scikit-learn scorer names](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
Without `metrics`, classification reports `accuracy`, `precision_weighted` and
`recall_weighted`; regression reports `r2` and `neg_mean_absolute_error`.

- Errors (names starting with `neg_`) are shown positive, and lower is better:
  the leaderboard says `root_mean_squared_error`.
- For classification, the split and the folds keep the share of each class.
- A metric that needs a method the model lacks (e.g. `roc_auc` and
  `predict_proba`) is skipped for that model, with a note.

## output

| Key | Type | Default | Description |
|---|---|---|---|
| `formats` | list of `csv`, `json`, `markdown` | all three | Leaderboard files written in the results folder. |
| `save_best_model` | true or false | `true` | Saves `best_model.joblib`, for `fitrank predict`. |

## A complete example

```yaml
project:
  name: titanic
  random_state: 42
  n_jobs: -1

data:
  path: examples/data/titanic.csv
  target: Survived
  features: all
  exclude: [PassengerId, Name, Ticket, Cabin, HomeDest]

task: classification

preprocessing:
  numeric:
    columns: [Age, Fare, SibSp, Parch]
    steps:
      - imputer: {strategy: median}
      - scaler: standard
  categorical:
    columns: [Sex, Pclass, Embarked]
    steps:
      - imputer: {strategy: most_frequent}
      - encoder: onehot

reduction:
  - none
  - pca: {n_components: [3, 5]}

models:
  - name: logistic_regression
    params: {C: [0.1, 1, 10]}
  - name: random_forest
    params: {n_estimators: [100, 300], max_depth: [null, 10]}
  - name: svm
    params: {C: [1, 10], kernel: [linear, rbf]}
  - name: decision_tree

evaluation:
  test_size: 0.2
  cv: 5
  primary_metric: f1_weighted
  metrics: [accuracy, precision_weighted, recall_weighted, roc_auc]
```
