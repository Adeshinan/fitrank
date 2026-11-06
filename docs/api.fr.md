# API Python

Tout ce que fait le fichier YAML est disponible depuis Python, à partir d'un
fichier ou d'un DataFrame pandas. Les arguments nommés de `Benchmark` sont les
sections du YAML, et les deux façons donnent les mêmes résultats.

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
model = results.best_model()  # un Pipeline scikit-learn entraîné
print(model.predict(df.drop(columns="target").head()))
```

Un fichier YAML se lance de la même façon :

```python
results = Benchmark.from_yaml("fitrank.yaml").run()
```

Les erreurs d'utilisation lèvent `ConfigError` ou `DataError`, deux
`FitRankError`, avec les mêmes messages qu'en ligne de commande.

## Référence

La référence ci-dessous est générée depuis le code ; elle est en anglais.

::: fitrank.Benchmark

::: fitrank.Results

::: fitrank.load_model

::: fitrank.register_model

::: fitrank.register_reduction

::: fitrank.FitRankError

::: fitrank.ConfigError

::: fitrank.DataError

::: fitrank.FitRankWarning
