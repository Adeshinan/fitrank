# Modèles

Il y a trois façons de mettre un modèle dans la comparaison : par son nom
FitRank, par le chemin de sa classe scikit-learn, ou en enregistrant le vôtre.

## Modèles fournis

Chaque modèle a une petite grille par défaut, cherchée quand la configuration
ne donne pas de `params`. `fitrank list models --task classification` affiche
les noms.

### Classification

| Nom | Classe scikit-learn | Grille par défaut |
|---|---|---|
| `decision_tree` | `DecisionTreeClassifier` | `{class_weight: [null, balanced], max_depth: [null, 5, 10]}` |
| `gradient_boosting` | `GradientBoostingClassifier` | `{learning_rate: [0.05, 0.1], n_estimators: [100, 200]}` |
| `knn` | `KNeighborsClassifier` | `{n_neighbors: [3, 5, 11], weights: [uniform, distance]}` |
| `logistic_regression` | `LogisticRegression` | `{C: [0.1, 1, 10], class_weight: [null, balanced]}` |
| `naive_bayes` | `GaussianNB` | `{var_smoothing: [1.0e-09, 1.0e-08, 1.0e-07]}` |
| `random_forest` | `RandomForestClassifier` | `{class_weight: [null, balanced], max_depth: [null, 10], n_estimators: [100, 300]}` |
| `svm` | `SVC` | `{C: [0.1, 1, 10], class_weight: [null, balanced], kernel: [linear, rbf]}` |

### Régression

| Nom | Classe scikit-learn | Grille par défaut |
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

`params` dans le YAML remplace la grille par défaut, il n'est jamais fusionné
avec elle :

```yaml
models:
  - name: random_forest
    params: {n_estimators: [500]}     # seul n_estimators=500 est essayé
```

## N'importe quel modèle scikit-learn, par sa classe

Un modèle que FitRank ne nomme pas peut être donné par le chemin de sa classe :

```yaml
models:
  - name: random_forest
  - class: sklearn.ensemble.ExtraTreesClassifier
    params: {n_estimators: [100, 300]}   # sans params : ses paramètres par défaut
```

Il apparaît dans le classement sous le nom `ExtraTreesClassifier`. FitRank
vérifie que la classe existe, que c'est un estimateur scikit-learn, qu'elle
correspond à la tâche (un régresseur est refusé en classification) et qu'elle
se crée sans arguments.

!!! warning "Seuls les modules de confiance sont importés"
    Un fichier YAML ne doit jamais pouvoir exécuter de code. FitRank n'importe
    que des classes de `sklearn` : tout autre chemin (`os.system`,
    `builtins.eval`...) est refusé **avant le moindre import**. Pour utiliser un
    autre paquet compatible scikit-learn en qui vous avez confiance, déclarez-le
    dans `project.allowed_modules` :

    ```yaml
    project:
      allowed_modules: [xgboost, mon_paquet.modeles]   # installé ou importable
    models:
      - class: xgboost.XGBClassifier
    ```

## Enregistrer votre propre modèle

Depuis Python, `register_model` ajoute un modèle sous un nom, avec sa grille
par défaut. Il s'utilise ensuite par son nom, comme les modèles fournis :

```python
from sklearn.ensemble import ExtraTreesClassifier

from fitrank import Benchmark, register_model


@register_model("extra_trees", task="classification")
def extra_trees():
    # Un estimateur neuf, et la grille cherchée quand la configuration ne donne pas de params.
    return ExtraTreesClassifier(), {"n_estimators": [100, 300]}


results = Benchmark(
    data="iris.csv", target="species", task="classification", models=["extra_trees", "svm"]
).run()
```

La fonction est appelée une fois par candidat : chacun reçoit un estimateur
neuf. Un même nom s'enregistre une fois par tâche. Depuis le fichier YAML,
donnez plutôt votre modèle par le chemin de sa classe, avec son paquet dans
`project.allowed_modules`.

## Enregistrer votre propre réduction

`register_reduction` marche de la même façon. La fonction reçoit la tâche et
renvoie un transformateur et sa grille par défaut :

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
