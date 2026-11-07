# Démarrage rapide

## 1. Des données

N'importe quel fichier CSV, Excel (`.xlsx`) ou Parquet avec une ligne par
exemple convient. Ici, les fleurs d'iris fournies avec scikit-learn :

```bash
python -c "from sklearn.datasets import load_iris; d = load_iris(as_frame=True); d.frame.assign(species=d.target_names[d.target]).drop(columns='target').to_csv('iris.csv', index=False)"
```

## 2. Décrire l'expérience

Enregistrez ceci dans `fitrank.yaml` :

```yaml
data:
  path: iris.csv
  target: species         # la colonne à prédire
task: classification      # prédire une catégorie (regression : prédire un nombre)
models:
  - name: logistic_regression
  - name: decision_tree
  - name: svm
```

Ou laissez FitRank écrire un fichier commenté à partir de vos données :

```bash
fitrank init --data iris.csv --target species
```

Il écrit `config.yaml` avec vos colonnes, quatre modèles rapides et chaque
option expliquée en commentaire.

## 3. Vérifier, puis lancer

```bash
fitrank validate fitrank.yaml   # vérifie la configuration et les données, n'entraîne rien
fitrank run fitrank.yaml
```

`validate` est instantané : utilisez-le après chaque modification. Il signale
tous les problèmes d'un coup, par exemple un nom de modèle inconnu avec le nom
valide le plus proche, ou une colonne avec des valeurs manquantes et le YAML
de `preprocessing` qui la traite.

## 4. Lire le classement

Voici l'[exemple Iris](https://github.com/Adeshinan/fitrank/blob/main/examples/iris.yaml),
qui compare 4 modèles avec et sans ACP :

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

- **cv f1_weighted** : le score moyen sur les plis de validation croisée, avec
  les meilleurs paramètres trouvés. Le classement n'utilise que cette colonne.
- **cv std** : à quel point le score varie d'un pli à l'autre. Quand deux
  modèles sont plus proches que leur cv std, ils se valent sur ces données.
- **test f1_weighted** : le score sur des lignes mises de côté dès le départ
  (20 % par défaut). Il indique à quoi s'attendre sur de nouvelles données ; il
  ne décide jamais du rang.
- **best params / error** : les paramètres choisis par la recherche, ou la
  raison de l'échec du modèle.

## 5. Le dossier de résultats

Chaque exécution a son propre dossier : rien n'est jamais écrasé.

```text
runs/iris_2026-11-01_15-36-06/
├── config.yaml          la configuration telle que FitRank l'a lue, valeurs par défaut comprises
├── leaderboard.csv      ┐
├── leaderboard.json     ├ les formats listés dans output.formats
├── leaderboard.md       ┘
├── best_model.joblib    si output.save_best_model vaut true
└── metadata.json        versions, date, durée, meilleur candidat
```

Le meilleur modèle est un `Pipeline` scikit-learn entraîné, prétraitement
compris. Utilisez-le sur de nouvelles données qui ont les mêmes colonnes :

```bash
fitrank predict runs/iris_2026-11-01_15-36-06/best_model.joblib nouvelles_fleurs.csv -o predictions.csv
```

## 6. Des données réelles

Les vraies données sont rarement propres. L'[exemple Titanic](https://github.com/Adeshinan/fitrank/blob/main/examples/titanic.yaml)
a des âges, des tarifs et des ports manquants, et des colonnes texte. FitRank
ne les corrige pas en silence : sans section `preprocessing`,
`fitrank validate` liste chaque problème et le YAML qui y répond. L'exemple
déclare tout :

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

La [référence de la configuration](configuration.md) explique chaque clé.
