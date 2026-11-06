# FAQ

## Pourquoi FitRank ne remplit-il pas lui-même les valeurs manquantes, et n'encode-t-il pas le texte ?

Parce qu'un choix silencieux change vos résultats sans vous le dire. Remplir
les âges avec la moyenne ou la médiane, supprimer des lignes, transformer des
catégories en nombres : chaque choix est une décision sur vos données. FitRank
signale le problème, avec le YAML qui le corrige, et vous décidez. Ce que vous
avez déclaré est ce qui a tourné, et le `config.yaml` enregistré le montre.

## Pourquoi classer sur le score de validation croisée et pas sur le score de test ?

Le jeu de test doit rester une estimation honnête de ce qui se passe sur de
nouvelles données. S'il choisissait le gagnant, il ferait partie de
l'entraînement, et son score serait trop optimiste. Le score de test est
affiché pour vérifier que le gagnant tient la route, jamais pour le choisir.

## Deux modèles sont presque à égalité. Lequel choisir ?

Regardez `cv std`. Quand deux scores sont plus proches que leur cv std, les
données ne permettent pas de les départager : un autre `random_state` pourrait
les inverser. Prenez le plus simple ou le plus rapide, ou celui dont le score
de test tient le mieux.

## Comment retrouver exactement les mêmes résultats ?

Gardez le même `random_state` (42 par défaut), les mêmes données et les mêmes
versions des bibliothèques. Chaque dossier de résultats enregistre la
configuration telle que FitRank l'a lue (`config.yaml`) et les versions
utilisées (`metadata.json`). `n_jobs` ne change pas les résultats, seulement
le temps.

## `n_jobs: -1` est plus lent sur mes données. C'est normal ?

Oui, sur de petits jeux de données. Lancer les processus coûte plus que le
travail qu'ils se partagent. Ça devient rentable avec beaucoup de combinaisons
ou beaucoup de lignes : l'exemple Titanic passe de 26 s à 15 s sur 12 cœurs.
Sinon, gardez la valeur par défaut `1`.

## Puis-je utiliser XGBoost, LightGBM ou mon propre estimateur ?

Oui, s'il suit l'API de scikit-learn. Donnez le chemin de sa classe et
autorisez son paquet :

```yaml
project:
  allowed_modules: [xgboost]
models:
  - class: xgboost.XGBClassifier
```

Voir [Modèles](models.md#nimporte-quel-modele-scikit-learn-par-sa-classe).

## Un modèle a échoué. Que se passe-t-il ?

Les autres continuent. Le modèle en échec est classé en dernier, marqué
`failed`, avec son erreur dans le classement. L'exécution échoue (code de
sortie `3`) seulement quand tous les candidats ont échoué.

## FitRank avertit que ma cible « looks categorical ». Est-ce une erreur ?

Non, l'exécution continue. La cible contient du texte, des vrai/faux ou deux
valeurs, mais la tâche est `regression` : vous voulez sans doute
`task: classification`. L'avertissement inverse apparaît pour une
classification sur un nombre qui a beaucoup de valeurs différentes. C'est vous
qui décidez : l'avertissement ne bloque jamais.

## Quelle métrique choisir ?

Par défaut, la classification est classée avec `f1_weighted` (robuste quand
les classes sont déséquilibrées) et la régression avec la RMSE (une erreur dans
l'unité de la cible). Choisissez celle qui correspond à l'erreur qui vous coûte
le plus, par exemple `recall_weighted` quand rater un cas positif est pire
qu'une fausse alerte. N'importe quel
[nom de scorer scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
fonctionne.

## FitRank traite-t-il les images ou le texte libre ?

Pas encore : FitRank travaille sur des données tabulaires (CSV, Excel, Parquet,
DataFrame). Laissez les colonnes de texte libre de côté avec `data.exclude`,
ou transformez-les en colonnes numériques avant.
