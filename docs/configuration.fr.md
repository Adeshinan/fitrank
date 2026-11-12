# Référence de la configuration

Une configuration FitRank est un fichier YAML fait de sections. Seules `data`,
`task` et `models` sont obligatoires ; toutes les autres clés ont une valeur
par défaut. Une clé inconnue est une erreur qui liste les clés permises : une
faute de frappe ne passe jamais inaperçue.

```yaml
project:        # optionnel : noms, dossiers, reproductibilité
data:           # obligatoire : le fichier, la cible, les variables
task:           # obligatoire : classification ou regression
preprocessing:  # optionnel : quoi faire à quelles colonnes
reduction:      # optionnel : réductions de dimension à essayer
models:         # obligatoire : les modèles à comparer
evaluation:     # optionnel : découpage, plis, recherche, métriques
output:         # optionnel : ce qui est enregistré
```

`fitrank validate config.yaml` vérifie un fichier sans rien entraîner. La
configuration telle que FitRank l'a lue, valeurs par défaut comprises, est
enregistrée dans chaque dossier de résultats sous le nom `config.yaml`.

## project

| Clé | Type | Défaut | Description |
|---|---|---|---|
| `name` | texte | `fitrank` | Nomme le dossier de résultats : `runs/<name>_<date>_<heure>/`. |
| `output_dir` | chemin | `runs` | Où les dossiers de résultats sont créés. |
| `random_state` | nombre entier | `42` | Le même nombre donne le même découpage, les mêmes plis et les mêmes résultats. |
| `n_jobs` | nombre entier | `1` | Cœurs du processeur pour la recherche ; `-1` les utilise tous. `0` et les valeurs inférieures à `-1` sont refusés. |
| `allowed_modules` | liste de noms de modules | `[]` | Paquets que [`models[].class`](models.md#nimporte-quel-modele-scikit-learn-par-sa-classe) peut importer, en plus de `sklearn`. |

```yaml
project:
  name: titanic
  random_state: 42
  n_jobs: -1
```

`n_jobs: -1` donne les mêmes résultats, plus vite sur les grosses recherches
(l'exemple Titanic passe de 26 s à 15 s sur 12 cœurs). Sur de petits jeux de
données, lancer les processus coûte plus que ce qu'ils font gagner : gardez
`1`. Un modèle qui fixe `n_jobs` dans ses `params` alors que `project.n_jobs`
ne vaut pas `1` est refusé : les cœurs ne sont jamais utilisés deux fois.

## data

| Clé | Type | Défaut | Description |
|---|---|---|---|
| `path` | chemin | obligatoire | Un fichier `.csv`, `.xlsx` ou `.parquet`. Un chemin relatif part du dossier où la commande est lancée. |
| `target` | nom de colonne | obligatoire | La colonne à prédire. |
| `features` | `all` ou liste de colonnes | `all` | Les colonnes d'entraînement. `all` veut dire toutes les colonnes sauf la cible. |
| `exclude` | liste de colonnes | `[]` | Avec `features: all`, les colonnes à laisser de côté (identifiants, texte libre...). |

```yaml
data:
  path: data/titanic.csv
  target: Survived
  features: all
  exclude: [PassengerId, Name, Ticket]
```

Utilisez soit une liste de `features`, soit `exclude`, pas les deux. Une
colonne qui n'existe pas est signalée avec les colonnes disponibles et le nom
le plus proche. Les lignes dont la cible manque sont signalées, jamais
supprimées.

## task

`classification` (prédire une catégorie : survivant ou non, une espèce...) ou
`regression` (prédire un nombre : un prix, une durée...). C'est vous qui
décidez : FitRank avertit seulement, sans s'arrêter, quand la cible ne
ressemble pas à la tâche, par exemple une régression sur une cible à deux
valeurs. Vous hésitez ? Voir [Classification ou régression ?](guides/task.md)

## preprocessing

Un ensemble de groupes. Chaque groupe applique ses `steps`, dans l'ordre, à ses
`columns`. Les colonnes hors de tout groupe passent sans changement.

```yaml
preprocessing:
  numeric:                       # n'importe quel nom de groupe
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

| Étape | S'écrit | Ce qu'elle fait |
|---|---|---|
| `imputer` | `imputer` (mean), `imputer: median`, `imputer: {strategy: constant, fill_value: 0}` | Remplit les valeurs manquantes. Stratégies : `mean`, `median`, `most_frequent`, `constant`. Les autres clés sont des options de [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html). |
| `scaler` | `scaler` (standard), `scaler: minmax`, `scaler: robust` | Met les nombres à la même échelle. |
| `encoder` | `encoder` (onehot), `encoder: ordinal` | Transforme les catégories en nombres. `onehot` ignore les catégories jamais vues à l'entraînement ; `ordinal` les code `-1`. |

Règles :

- Une colonne n'appartient qu'à un seul groupe, et une étape n'apparaît qu'une
  fois par groupe.
- Une colonne avec des valeurs manquantes demande un `imputer`, une colonne
  texte un `encoder` : sinon `fitrank validate` les liste, avec un extrait de
  YAML.
- Les étapes sont apprises dans chaque pli de validation croisée, sur les
  lignes d'entraînement seulement : les scores ne sont pas gonflés par une
  fuite de données.

## reduction

Une liste de réductions de dimension ; chacune est combinée avec chaque
modèle. Défaut : `[none]`.

```yaml
reduction:
  - none                              # les colonnes telles quelles
  - pca: {n_components: [3, 5]}       # les valeurs entre [] sont cherchées, comme params
```

| Nom | Grille par défaut | Ce qu'elle fait |
|---|---|---|
| `none` | — | Garde les colonnes telles quelles. |
| `pca` | `{n_components: [0.95]}` | Projette les colonnes sur moins d'axes (0.95 = garder 95 % de la variance). |
| `select_k_best` | `{k: [5, 10]}` | Garde les k colonnes les plus liées à la cible. |
| `variance_threshold` | `{threshold: [0.0, 0.01]}` | Retire les colonnes qui varient à peine. |
| `polynomial_features` | `{degree: [2]}` | Ajoute les produits et les puissances des colonnes. |

Les options écrites après le nom remplacent la grille par défaut. Avec 2
réductions et 4 modèles, FitRank évalue 8 candidats.

## models

Une liste d'au moins un modèle. Chaque modèle a un `name` ou une `class`, et
des `params` optionnels.

```yaml
models:
  - name: logistic_regression                   # grille par défaut
  - name: random_forest
    params: {n_estimators: [100, 300], max_depth: [null, 10]}
  - class: sklearn.ensemble.ExtraTreesClassifier
    params: {n_estimators: [100]}
```

| Clé | Type | Description |
|---|---|---|
| `name` | nom de modèle | Un modèle FitRank : voir [Modèles](models.md). |
| `class` | chemin de classe | N'importe quel estimateur scikit-learn, par exemple `sklearn.svm.LinearSVC`. |
| `params` | ensemble de listes | Les valeurs à essayer pour chaque paramètre. Remplace la grille par défaut ; `null` correspond au `None` de Python. |

Sans `params`, un modèle nommé cherche dans sa petite grille par défaut, et une
classe utilise ses paramètres par défaut. Un paramètre que le modèle n'a pas
est refusé, avec le nom valide le plus proche.

## evaluation

| Clé | Type | Défaut | Description |
|---|---|---|---|
| `test_size` | nombre entre 0 et 1 | `0.2` | Part des lignes mises de côté pour le score de test final. |
| `test_data` | chemin de fichier | aucun | Votre propre jeu de test (CSV, Excel ou Parquet), à la place de `test_size` : voir plus bas. |
| `cv` | nombre entier ≥ 2 | `5` | Nombre de plis de validation croisée sur les autres lignes. |
| `search` | `grid` ou `random` | `grid` | `grid` essaie chaque combinaison de `params` ; `random` en essaie `n_iter`. |
| `n_iter` | nombre entier ≥ 1 | `20` | Avec `search: random`, combinaisons essayées par candidat (au plus la taille de la grille). |
| `primary_metric` | nom de scorer | `f1_weighted` / `neg_root_mean_squared_error` | Classe les modèles (défaut en classification / en régression). |
| `metrics` | liste de noms de scorers | voir plus bas | Aussi calculées sur le jeu de test. |

Les métriques sont les
[noms de scorers de scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
Sans `metrics`, la classification affiche `accuracy`, `precision_weighted` et
`recall_weighted` ; la régression affiche `r2` et `neg_mean_absolute_error`.

- Les erreurs (noms qui commencent par `neg_`) sont affichées positives, et
  plus bas est meilleur : le classement écrit `root_mean_squared_error`.
- En classification, le découpage et les plis gardent la part de chaque classe.
- Une métrique qui demande une méthode absente du modèle (par exemple
  `neg_log_loss`, qui demande `predict_proba`, sur `svm`) est sautée pour ce
  modèle, avec une note.

Vous hésitez sur la métrique ? Voir [Choisir une métrique](guides/metrics.md).

### Votre propre jeu de test

Par défaut, FitRank met de côté `test_size` des lignes. Pour évaluer tous les
modèles sur votre propre jeu de test (le jeu de test officiel d'une
compétition, les derniers mois d'une série temporelle, un jeu de test propre
pendant que les données d'entraînement changent), donnez-le dans `test_data` :

```yaml
data:
  path: data/train.csv
  target: label
evaluation:
  test_data: data/test.csv
  cv: 5
```

- Rien n'est découpé : toutes les lignes de `data.path` servent à la recherche
  et à la validation croisée, et chaque modèle est évalué sur tout le jeu de test.
- Le jeu de test reçoit les mêmes colonnes que l'entraînement (`features` /
  `exclude`) ; ses colonnes en plus sont ignorées.
- `test_size` ne peut pas être écrit avec `test_data` : choisissez l'un des deux.
- Avant l'entraînement, FitRank s'arrête avec une erreur claire si le jeu de
  test n'a pas la cible ou une colonne utilisée, a des valeurs manquantes sans
  imputer, ou (classification) contient une classe absente de l'entraînement.
  Une classe de l'entraînement sans aucune ligne de test donne seulement un
  avertissement : `roc_auc` et les moyennes macro peuvent alors tromper.
- `metadata.json` enregistre l'origine du jeu de test et son empreinte
  SHA-256 : deux exécutions avec la même empreinte ont été évaluées sur
  exactement les mêmes lignes.

En Python, passez un DataFrame ou un chemin : `Benchmark(data=train, test_data=test, ...)`.

## output

| Clé | Type | Défaut | Description |
|---|---|---|---|
| `formats` | liste parmi `csv`, `json`, `markdown` | les trois | Fichiers du classement écrits dans le dossier de résultats. |
| `save_best_model` | true ou false | `true` | Enregistre `best_model.joblib`, pour `fitrank predict`. |

## Un exemple complet

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
