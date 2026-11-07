# FitRank

**Comparaison de modèles scikit-learn reproductible, pilotée par la configuration. Aucun prétraitement caché.**

FitRank entraîne plusieurs modèles scikit-learn sur vos données tabulaires,
optimise les hyperparamètres de chacun et les classe dans un tableau. Toute
l'expérience tient dans un fichier YAML : relancez-la le mois prochain, ou envoyez-la à un collègue,
vous obtenez le même classement.

```yaml
data:
  path: iris.csv
  target: species         # la colonne à prédire
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

Ou essayez-le dans votre navigateur, sans rien installer :
[ouvrez le notebook dans Colab](https://colab.research.google.com/github/Adeshinan/fitrank/blob/main/examples/try_fitrank.ipynb) et exécutez toutes les cellules
(environ 2 minutes).

## Ce que FitRank garantit

- **Les mêmes conditions pour chaque modèle.** Même découpage
  entraînement/test, mêmes plis de validation croisée, même métrique. Les
  modèles sont classés sur le score de validation croisée ; le score de test
  est affiché, jamais utilisé pour choisir.
- **Aucune transformation n'est appliquée à vos données sans que vous l'ayez
  déclarée.** Les valeurs manquantes et
  les colonnes texte sont signalées avec le YAML qui les corrige ; FitRank ne
  remplit, ne supprime et n'encode jamais rien de lui-même.
- **Aucune fuite de données.** Le prétraitement vit dans le pipeline : il est
  appris sur les plis d'entraînement seulement.
- **Un modèle qui échoue n'arrête jamais l'exécution.** Il est marqué `failed`
  dans le classement, avec son erreur.
- **Des erreurs claires.** Chaque message dit ce qui ne va pas, où (clé YAML ou
  colonne) et comment corriger, sans afficher la pile d'appels (traceback).

## Pourquoi FitRank ?

| | FitRank | [LazyPredict](https://github.com/shankarpandala/lazypredict) | [PyCaret](https://pycaret.org) | AutoML ([auto-sklearn](https://automl.github.io/auto-sklearn/), [FLAML](https://microsoft.github.io/FLAML/), [AutoGluon](https://auto.gluon.ai)…) |
|---|---|---|---|---|
| Classe beaucoup de modèles en une exécution | ✅ | ✅ | ✅ | ✅ (vise un seul modèle final ou un ensemble) |
| Recherche d'hyperparamètres par modèle | ✅ chaque modèle | optionnelle, les meilleurs modèles après le classement | ✅ étape séparée | ✅ |
| Toute l'expérience dans un fichier YAML | ✅ | ❌ | ❌ | ❌ |
| Prétraitement seulement s'il est déclaré | ✅ | ❌ (impute, normalise et encode toujours) | ❌ (impute et encode par défaut) | ❌ |
| Qui décide du pipeline | vous | la bibliothèque | vous ou la bibliothèque | la bibliothèque |

**Utilisez autre chose si** vous voulez le meilleur score possible sans effort
(→ un outil d'AutoML) ou une plateforme low-code complète avec déploiement
(→ PyCaret).

## Installation

```bash
pip install fitrank
```

Python 3.10 à 3.13. Les fichiers Excel demandent `pip install "fitrank[excel]"`,
les fichiers Parquet `pip install "fitrank[parquet]"`.

## Pour continuer

- [Démarrage rapide](quickstart.md) : une première exécution, le classement et le dossier de résultats.
- [Référence de la configuration](configuration.md) : chaque clé YAML, son type et sa valeur par défaut.
- [Modèles](models.md) : les modèles fournis, n'importe quelle classe scikit-learn, vos propres modèles.
- [Ligne de commande](cli.md) et [API Python](api.md).
- Guides : [Classification ou régression ?](guides/task.md) et [Choisir une métrique](guides/metrics.md).
- [FAQ](faq.md).

FitRank est open source (MIT). Le code, les tickets et le journal des
modifications sont sur [GitHub](https://github.com/Adeshinan/fitrank).
