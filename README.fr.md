# FitRank

[![CI](https://github.com/Adeshinan/fitrank/actions/workflows/ci.yml/badge.svg)](https://github.com/Adeshinan/fitrank/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fitrank)](https://pypi.org/project/fitrank/)
[![Python](https://img.shields.io/pypi/pyversions/fitrank)](https://pypi.org/project/fitrank/)
[![Licence : MIT](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

**Comparer des modèles scikit-learn de façon reproductible, à partir d'un fichier de configuration. Aucun prétraitement caché.**

[English](README.md) · Français

FitRank entraîne plusieurs modèles scikit-learn sur vos données tabulaires,
règle les hyperparamètres de chacun et les classe dans un tableau. Toute
l'expérience tient dans un seul fichier YAML : relancez-le le mois prochain ou
envoyez-le à un collègue, vous obtiendrez le même classement.

## Pourquoi FitRank ?

FitRank compare les modèles scikit-learn toujours de la même manière, à partir
d'un fichier YAML que l'on peut versionner et partager. Vos données ne subissent
aucune transformation que vous n'avez pas déclarée.

| | FitRank | [LazyPredict](https://github.com/shankarpandala/lazypredict) | [PyCaret](https://pycaret.org) | AutoML ([auto-sklearn](https://automl.github.io/auto-sklearn/), [FLAML](https://microsoft.github.io/FLAML/), [AutoGluon](https://auto.gluon.ai)…) |
|---|---|---|---|---|
| Classe de nombreux modèles en une fois | ✅ | ✅ | ✅ | ✅ (vise un modèle final ou un ensemble) |
| Recherche d'hyperparamètres pour chaque modèle | ✅ tous les modèles | en option, pour les meilleurs après le classement | ✅ étape séparée | ✅ |
| Toute l'expérience dans un fichier YAML | ✅ | ❌ | ❌ | ❌ |
| Prétraitement uniquement s'il est déclaré | ✅ | ❌ (impute, normalise et encode toujours) | ❌ (impute et encode par défaut) | ❌ |
| Bibliothèques de machine learning installées | scikit-learn | scikit-learn (boosters en option) | scikit-learn, LightGBM et beaucoup d'autres | boosters et plus |
| Qui décide du pipeline | vous | la bibliothèque | vous ou la bibliothèque | la bibliothèque |

<sub>Vérifié dans la documentation de chaque projet le 1er novembre 2026 (LazyPredict 0.4.0,
PyCaret 3.3.2, auto-sklearn 0.15.0, FLAML 2.7.0, AutoGluon 1.6.3). Ces projets évoluent :
les corrections sont les bienvenues.</sub>

**Choisissez plutôt un autre outil si** vous voulez le meilleur score possible
sans effort (→ un outil d'AutoML) ou une plateforme low-code complète, jusqu'au
déploiement (→ PyCaret).

## Installation

```bash
pip install fitrank
```

Python 3.10 à 3.13. Pour lire des fichiers Excel : `pip install "fitrank[excel]"` ;
des fichiers Parquet : `pip install "fitrank[parquet]"`.

## Démarrage rapide

Enregistrez des données (ici les iris fournis avec scikit-learn) :

```bash
python -c "from sklearn.datasets import load_iris; load_iris(as_frame=True).frame.to_csv('iris.csv', index=False)"
```

Décrivez l'expérience dans `fitrank.yaml` :

```yaml
data:
  path: iris.csv
  target: target          # la colonne à prédire
task: classification
models:
  - name: logistic_regression
  - name: decision_tree
  - name: svm
```

Lancez-la :

```bash
fitrank run fitrank.yaml
```

Chaque modèle est réglé sur les mêmes plis de validation croisée, classé, puis
évalué sur un jeu de test mis de côté. Les résultats sont enregistrés dans
`runs/fitrank_<date>_<heure>/`.

## Le classement

L'[exemple Iris](examples/iris.yaml) compare 4 modèles, avec et sans ACP (PCA),
après mise à l'échelle des colonnes. Depuis la racine du dépôt :

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

Les modèles sont classés sur le score de validation croisée (`cv f1_weighted`) :
le score de test est affiché, mais ne sert jamais à choisir. Le dossier de
résultats contient le classement (CSV, JSON, Markdown), la configuration telle
que FitRank l'a lue, les versions utilisées et le meilleur modèle, prêt pour de
nouvelles données :

```bash
# remplacez par le nom de dossier affiché par votre exécution
fitrank predict runs/iris_2026-11-01_15-36-06/best_model.joblib nouvelles_fleurs.csv -o predictions.csv
```

[`examples/iris.yaml`](examples/iris.yaml) explique chaque clé de la
configuration ; [`examples/diabetes.yaml`](examples/diabetes.yaml) est un
exemple de régression.

## Autres commandes

```bash
fitrank validate fitrank.yaml        # vérifie la config et les données, sans rien entraîner
fitrank list models --task regression
fitrank list reductions
fitrank --quiet run fitrank.yaml     # uniquement le classement, pour les scripts
fitrank --verbose run fitrank.yaml   # une ligne par modèle dès qu'il est terminé
```

Codes de sortie : `0` succès, `1` erreur de configuration, `2` erreur dans les
données, `3` tous les modèles ont échoué.

## Utilisation en Python

La même expérience, à partir d'un DataFrame :

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
model = results.best_model()  # un Pipeline scikit-learn déjà entraîné
print(model.predict(df.drop(columns="target").head()))
```

`Benchmark.from_yaml("fitrank.yaml").run()` lance un fichier YAML. Les
arguments nommés reprennent les sections du YAML, et les deux façons donnent
les mêmes résultats.

## Ajouter son propre modèle

```python
from sklearn.ensemble import ExtraTreesClassifier

from fitrank import Benchmark, register_model


@register_model("extra_trees", task="classification")
def extra_trees():
    # Un estimateur neuf, et la grille explorée quand la config ne donne pas de params.
    return ExtraTreesClassifier(), {"n_estimators": [100, 300]}


results = Benchmark(
    data="iris.csv", target="target", task="classification", models=["extra_trees", "svm"]
).run()
```

Un modèle enregistré s'utilise par son nom, comme les modèles fournis. Les
réductions s'ajoutent de la même façon avec `register_reduction`. En v1.0, vos
propres modèles sont disponibles depuis Python ; les charger depuis le fichier
YAML est prévu pour la v1.1.

## Modèles et étapes disponibles

| Tâche | Modèles |
|---|---|
| classification | `logistic_regression`, `svm`, `knn`, `decision_tree`, `random_forest`, `gradient_boosting`, `naive_bayes` |
| régression | `linear_regression`, `ridge`, `lasso`, `polynomial_regression`, `svr`, `knn`, `decision_tree`, `random_forest`, `gradient_boosting` |

| Section | Noms |
|---|---|
| `reduction` | `none`, `pca`, `select_k_best`, `variance_threshold`, `polynomial_features` |
| étapes de `preprocessing` | `imputer` (mean, median, most_frequent, constant), `scaler` (standard, minmax, robust), `encoder` (onehot, ordinal) |

Chaque modèle a une petite grille par défaut ; `params` dans le YAML la
remplace. Les métriques portent les [noms de scorers de scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) :
`f1_weighted` classe la classification par défaut, la RMSE classe la régression.

## Contribuer

Les signalements de bugs et les idées sont les bienvenus dans les
[issues](https://github.com/Adeshinan/fitrank/issues). Pour travailler sur le code :

```bash
git clone https://github.com/Adeshinan/fitrank.git
cd fitrank
pip install -e ".[dev]"
pre-commit install
pytest                       # ajoutez -m "not e2e" pour sauter les tests de bout en bout, plus lents
```

Chaque pull request passe ruff, mypy (strict) et les tests sous Linux, macOS et
Windows, avec Python 3.10 à 3.13. Les conventions sont décrites dans
[CONTRIBUTING.md](CONTRIBUTING.md) (en anglais), et les changements de chaque
version dans le [changelog](CHANGELOG.md).

## Licence

[MIT](LICENSE) © 2026 Faïd Adéshinan ALAGBE
