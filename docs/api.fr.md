# API Python

Tout ce que fait le fichier YAML est disponible depuis Python, à partir d'un
fichier ou d'un DataFrame pandas. Les arguments nommés de `Benchmark` sont les
sections du YAML, et les deux façons donnent les mêmes résultats.

```python
import pandas as pd

from fitrank import Benchmark

df = pd.read_csv("iris.csv")  # le fichier du démarrage rapide
results = Benchmark(
    data=df,
    target="species",
    task="classification",
    models=["logistic_regression", "svm", {"name": "knn", "params": {"n_neighbors": [3, 5]}}],
    evaluation={"cv": 5, "primary_metric": "f1_macro"},
).run()

print(results.leaderboard()[["rank", "model", "cv_f1_macro", "test_f1_macro"]])
model = results.best_model()  # un Pipeline scikit-learn entraîné
print(model.predict(df.drop(columns="species").head()))
```

Un fichier YAML se lance de la même façon :

```python
results = Benchmark.from_yaml("fitrank.yaml").run()
```

Les erreurs d'utilisation lèvent `ConfigError` ou `DataError`, deux
`FitRankError`, avec les mêmes messages qu'en ligne de commande.

## Comparer deux classements

`compare_rankings` mesure à quel point deux exécutions sont d'accord sur
l'ordre des modèles, par exemple sur des données d'entraînement propres et
bruitées, évaluées sur le même jeu de test :

```python
from fitrank import Benchmark, compare_rankings

clean = Benchmark(data=train_clean, test_data=test, **setup).run(export=False)
noisy = Benchmark(data=train_noisy, test_data=test, **setup).run(export=False)

cmp = compare_rankings(clean, noisy)
cmp.kendall_tau  # 1 = même ordre, 0 = sans rapport, -1 = inversé
cmp.top1_agree  # le même meilleur modèle des deux côtés ?
cmp.regret  # score de test perdu, côté propre, en prenant le gagnant bruité
cmp.table  # model, reduction, reference_rank, other_rank, rank_shift, scores
cmp.notes  # candidats écartés, ex æquo en tête...
```

- Les candidats sont associés par (modèle, réduction). Un candidat en échec,
  ou présent d'un seul côté, est écarté avec une note.
- Les deux côtés doivent avoir la même métrique principale. Ils sont classés
  sur le score de validation croisée, comme dans le leaderboard ;
  `by="test"` les classe sur le score de test.
- `kendall_tau` est le tau-b de Kendall : les scores égaux comptent comme des
  ex æquo.
- Un ex æquo en tête est départagé par l'ordre du leaderboard, avec une
  note : `top1_agree` et `regret` en dépendent.
- Chaque côté peut aussi être un DataFrame de leaderboard, ou le
  `leaderboard.csv` / `leaderboard.json` d'un dossier de résultats.

Avec 5 à 8 modèles, le tau prend peu de valeurs et varie d'un `random_state`
à l'autre. Avant de lire un changement de tau comme un effet des données,
calculez le tau entre deux graines sur les mêmes données : c'est la taille du
bruit.

## Probabilités hors échantillon

`results.oof_proba(model)` donne, pour chaque ligne d'entraînement, les
probabilités des classes prédites par un modèle qui n'a jamais vu cette
ligne : le pipeline du candidat (prétraitement, réduction, modèle) avec ses
meilleurs paramètres, réentraîné sur les autres plis de la validation croisée
du run. Elles servent à la détection d'étiquettes erronées (par exemple
`cleanlab.filter.find_label_issues`), à l'analyse d'erreurs ou à l'empilement
de modèles :

```python
results = Benchmark(data=train, test_data=test, **setup).run(export=False)

proba = results.oof_proba("random_forest")  # lignes d'entraînement x classes
labels = results.y_train.loc[proba.index]  # les étiquettes observées, dans le même ordre
confidence = proba.to_numpy()[range(len(proba)), proba.columns.get_indexer(labels)]
suspects = proba.index[confidence < 0.2]  # les lignes auxquelles le modèle croit à peine
```

- L'index est celui des lignes d'entraînement (`results.X_train`) : avec
  `test_size`, les lignes non mises de côté ; avec `test_data`, tout `data`.
- Classification seulement. Un modèle sans `predict_proba` (`svm`) est
  refusé avec une erreur claire.
- Avec plusieurs réductions, nommez-en une : `oof_proba("knn", reduction="pca")`.
- `run()` ne calcule rien : le premier appel entraîne les plis, les appels
  suivants renvoient une copie du même résultat.

Les lignes ne sont pas parfaitement hors échantillon : chaque ligne est
prédite par un modèle qui ne l'a pas vue, mais les meilleurs paramètres ont
été choisis avec ces mêmes plis, donc avec toutes les lignes. C'est la
pratique courante pour la détection d'erreurs d'étiquettes. `params="default"`
est plus strict : les paramètres qui n'ont qu'une valeur dans `params` (fixés
à l'avance) sont gardés, ceux qui étaient cherchés prennent les valeurs par
défaut de scikit-learn.

## Référence

La référence ci-dessous est générée depuis le code ; elle est en anglais.

::: fitrank.Benchmark

::: fitrank.Results

::: fitrank.compare_rankings

::: fitrank.RankingComparison

::: fitrank.load_model

::: fitrank.register_model

::: fitrank.register_reduction

::: fitrank.FitRankError

::: fitrank.ConfigError

::: fitrank.DataError

::: fitrank.FitRankWarning
