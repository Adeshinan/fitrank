# Quick start

## 1. Get some data

Any CSV, Excel (`.xlsx`) or Parquet file with one row per example works. Here,
the iris flowers that ship with scikit-learn:

```bash
python -c "from sklearn.datasets import load_iris; load_iris(as_frame=True).frame.to_csv('iris.csv', index=False)"
```

## 2. Describe the experiment

Save this as `fitrank.yaml`:

```yaml
data:
  path: iris.csv
  target: target          # the column to predict
task: classification      # predict a category (regression: predict a number)
models:
  - name: logistic_regression
  - name: decision_tree
  - name: svm
```

Or let FitRank write a commented file from your data:

```bash
fitrank init --data iris.csv --target target
```

It writes `config.yaml` with your columns, four fast models and every option
explained in a comment.

## 3. Check, then run

```bash
fitrank validate fitrank.yaml   # checks the config and the data, trains nothing
fitrank run fitrank.yaml
```

`validate` is instant: use it after each change. It reports every problem at
once, for example an unknown model name with the closest valid one, or a
column with missing values and the `preprocessing` YAML that handles it.

## 4. Read the leaderboard

This is the [Iris example](https://github.com/Adeshinan/fitrank/blob/main/examples/iris.yaml),
which compares 4 models with and without a PCA:

```text
| # | model               | reduction | cv f1_weighted | cv std | test f1_weighted |  time | best params / error                     |
|---+---------------------+-----------+----------------+--------+------------------+-------+-----------------------------------------|
| 1 | svm                 | none      |         0.9749 | 0.0205 |           0.9333 | 1.3 s | C=0.1, class_weight=None, kernel=linear |
| 2 | knn                 | none      |         0.9665 | 0.0167 |           0.9327 | 0.8 s | n_neighbors=3, weights=uniform          |
| 3 | logistic_regression | none      |         0.9663 | 0.0316 |           1.0000 | 1.1 s | C=10, class_weight=None                 |
| 4 | random_forest       | none      |         0.9580 | 0.0268 |           0.9666 | 2.2 s | max_depth=3, n_estimators=100           |
...
Best model: svm + none (f1_weighted = 0.9749)
Results saved to runs/iris_2026-11-01_15-36-06/
```

- **cv f1_weighted**: the mean score over the cross-validation folds, with the
  best parameters found. The ranking uses this column only.
- **cv std**: how much the score varies from one fold to another. When two
  models are closer than their cv std, they are equally good on this data.
- **test f1_weighted**: the score on rows kept aside from the start (20 % by
  default). It tells you what to expect on new data; it never decides the rank.
- **best params / error**: the parameters chosen by the search, or why the
  model failed.

## 5. The results folder

Each run gets its own folder, so nothing is ever overwritten:

```text
runs/iris_2026-11-01_15-36-06/
├── config.yaml          the configuration as FitRank read it, defaults included
├── leaderboard.csv      ┐
├── leaderboard.json     ├ the formats listed in output.formats
├── leaderboard.md       ┘
├── best_model.joblib    if output.save_best_model is true
└── metadata.json        versions, date, duration, best candidate
```

The best model is a fitted scikit-learn `Pipeline`, preprocessing included.
Use it on new data with the same columns:

```bash
fitrank predict runs/iris_2026-11-01_15-36-06/best_model.joblib new_flowers.csv -o predictions.csv
```

## 6. Real-world data

Real data is rarely clean. The
[Titanic example](https://github.com/Adeshinan/fitrank/blob/main/examples/titanic.yaml)
has missing ages, fares and ports, and text columns. FitRank does not fix them
silently: without a `preprocessing` section, `fitrank validate` lists each
problem and the YAML that answers it. The example declares it all:

```yaml
data:
  path: examples/data/titanic.csv
  target: Survived
  features: all
  exclude: [PassengerId, Name, Ticket, Cabin, HomeDest]   # ids and free text
preprocessing:
  numeric:
    columns: [Age, Fare, SibSp, Parch]
    steps: [{imputer: {strategy: median}}, scaler]
  categorical:
    columns: [Sex, Pclass, Embarked]
    steps: [{imputer: {strategy: most_frequent}}, {encoder: onehot}]
```

The [configuration reference](configuration.md) explains every key.
