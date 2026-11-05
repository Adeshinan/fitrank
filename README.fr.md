# FitRank

[![CI](https://github.com/Adeshinan/fitrank/actions/workflows/ci.yml/badge.svg)](https://github.com/Adeshinan/fitrank/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fitrank)](https://pypi.org/project/fitrank/)
[![Python](https://img.shields.io/pypi/pyversions/fitrank)](https://pypi.org/project/fitrank/)
[![Licence : MIT](https://img.shields.io/badge/licence-MIT-blue)](https://github.com/Adeshinan/fitrank/blob/main/LICENSE)

**Comparer des modèles scikit-learn de façon reproductible, à partir d'un fichier de configuration. Aucun prétraitement caché.**

[English](https://github.com/Adeshinan/fitrank/blob/main/README.md) · Français

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

Pour partir plutôt d'un fichier commenté, `fitrank init --data iris.csv --target target`
écrit un `config.yaml` avec vos colonnes, quatre modèles et chaque option expliquée.

Chaque modèle est réglé sur les mêmes plis de validation croisée, classé, puis
évalué sur un jeu de test mis de côté. Les résultats sont enregistrés dans
`runs/fitrank_<date>_<heure>/`.

## Le classement

L'[exemple Iris](https://github.com/Adeshinan/fitrank/blob/main/examples/iris.yaml) compare 4 modèles, avec et sans ACP (PCA),
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

[`examples/iris.yaml`](https://github.com/Adeshinan/fitrank/blob/main/examples/iris.yaml) explique chaque clé de la
configuration ; [`examples/diabetes.yaml`](https://github.com/Adeshinan/fitrank/blob/main/examples/diabetes.yaml) est un
exemple de régression.

## Des données réelles : Titanic

L'[exemple Titanic](https://github.com/Adeshinan/fitrank/blob/main/examples/titanic.yaml) prédit lesquels des 1309 passagers ont
survécu. Contrairement aux iris, les données sont imparfaites : des âges, un
tarif et deux ports d'embarquement manquent, et certaines colonnes sont du
texte. FitRank ne corrige jamais cela de lui-même : sans section
`preprocessing`, `fitrank validate` liste chaque problème et le YAML qui y
répond. L'exemple déclare tout :

```yaml
data:
  path: examples/data/titanic.csv
  target: Survived
  features: all
  exclude: [PassengerId, Name, Ticket, Cabin, HomeDest]   # identifiants et texte libre
preprocessing:
  numeric:
    columns: [Age, Fare, SibSp, Parch]
    steps: [{imputer: {strategy: median}}, scaler]
  categorical:
    columns: [Sex, Pclass, Embarked]
    steps: [{imputer: {strategy: most_frequent}}, {encoder: onehot}]
```

```bash
fitrank run examples/titanic.yaml      # 8 candidats, 15 à 40 s
```

| # | model | reduction | cv f1_weighted | test f1_weighted | test roc_auc |
|---|---|---|---|---|---|
| 1 | svm | pca | 0.8000 | 0.8306 | 0.8427 |
| 2 | random_forest | none | 0.7996 | 0.8273 | 0.8890 |
| 3 | svm | none | 0.7993 | 0.8379 | 0.8616 |
| 4 | decision_tree | none | 0.7889 | 0.8228 | 0.8726 |
| 5 | random_forest | pca | 0.7836 | 0.8183 | 0.8479 |
| 6 | logistic_regression | none | 0.7833 | 0.8039 | 0.8673 |
| 7 | logistic_regression | pca | 0.7696 | 0.8111 | 0.8538 |
| 8 | decision_tree | pca | 0.7594 | 0.7790 | 0.8106 |

Les trois premiers sont à moins de 0,001 d'écart, bien moins que leur cv std
(environ 0,01) : sur ces données ils se valent, et leur ordre pourrait changer
avec un autre `random_state`. Regardez le cv std avant de vous fier à un rang.

Les valeurs manquantes sont remplies dans chaque pli de validation croisée, à
partir des lignes d'entraînement seulement : les scores ne sont pas gonflés par
une fuite de données. Les données viennent d'[OpenML](https://www.openml.org/d/40945)
(licence publique) et sont regénérées par `scripts/make_example_data.py`.

## Autres commandes

```bash
fitrank init --data data.csv --target price --task regression   # écrit un config.yaml commenté
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

## N'importe quel modèle scikit-learn, par sa classe

Un modèle que FitRank ne nomme pas peut être donné par le chemin de sa classe :

```yaml
models:
  - name: random_forest
  - class: sklearn.ensemble.ExtraTreesClassifier
    params: {n_estimators: [100, 300]}   # sans params : ses paramètres par défaut
```

Il apparaît dans le classement sous le nom `ExtraTreesClassifier`. Par sécurité,
seules les classes de `sklearn` sont importées : tout autre chemin (`os.system`,
`builtins.eval`...) est refusé avant le moindre import. Pour utiliser un autre
paquet compatible scikit-learn en qui vous avez confiance, déclarez-le dans le
projet :

```yaml
project:
  allowed_modules: [xgboost, mon_paquet.modeles]   # installé ou importable
models:
  - class: xgboost.XGBClassifier
```

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
réductions s'ajoutent de la même façon avec `register_reduction`. Depuis le
fichier YAML, donnez votre modèle par le chemin de sa classe, avec son paquet
dans `project.allowed_modules`.

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
[CONTRIBUTING.md](https://github.com/Adeshinan/fitrank/blob/main/CONTRIBUTING.md) (en anglais), et les changements de chaque
version dans le [changelog](https://github.com/Adeshinan/fitrank/blob/main/CHANGELOG.md).

## Licence

[MIT](https://github.com/Adeshinan/fitrank/blob/main/LICENSE) © 2026 Faïd Adéshinan ALAGBE
