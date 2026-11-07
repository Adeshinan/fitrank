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

## Reference

::: fitrank.Benchmark

::: fitrank.Results

::: fitrank.load_model

::: fitrank.register_model

::: fitrank.register_reduction

::: fitrank.FitRankError

::: fitrank.ConfigError

::: fitrank.DataError

::: fitrank.FitRankWarning
