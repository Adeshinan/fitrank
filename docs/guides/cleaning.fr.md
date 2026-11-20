# Mesurer l'effet d'un nettoyage de données

Vous pensez que certaines étiquettes de vos données d'entraînement sont
fausses, et vous avez un moyen de retirer les lignes suspectes. Deux questions
se posent :

1. **Les étiquettes fausses changent-elles le modèle gagnant ?**
2. **Le nettoyage rend-il le bon classement ?**

FitRank répond aux deux avec la même expérience lancée plusieurs fois, en ne
changeant que les lignes d'entraînement. Ce guide donne le protocole, la
lecture des chiffres et les pièges. Le notebook
[label_noise_demo.ipynb](https://colab.research.google.com/github/Adeshinan/fitrank/blob/main/examples/label_noise_demo.ipynb)
le déroule de bout en bout en une minute environ.

## Le protocole

Chaque exécution ne doit différer **que par les lignes d'entraînement** :
mêmes modèles, mêmes plis, même métrique, même jeu de test.

```python
from fitrank import Benchmark, compare_rankings

setup = {
    "target": "Class",
    "task": "classification",
    "models": [
        "logistic_regression",
        "knn",
        "random_forest",
        {"name": "svm", "calibrate": "sigmoid"},
    ],
    "evaluation": {"cv": 3, "primary_metric": "f1_macro"},
}

clean = Benchmark(data=train, test_data=test, **setup).run(export=False)
noisy = Benchmark(data=train_noisy, test_data=test, **setup).run(export=False)
cleaned = Benchmark(data=train_cleaned, test_data=test, **setup).run(export=False)

before = compare_rankings(clean, noisy)
after = compare_rankings(clean, cleaned)
```

1. **Découpez une seule fois.** Coupez vos données en `train` et `test` une
   fois, avant tout le reste, et gardez les deux fichiers.
2. **Gardez le jeu de test propre et fixe.** Donnez-le avec `test_data` :
   chaque exécution est notée sur exactement les mêmes lignes, et aucune ne
   sert à l'entraînement. Ne nettoyez jamais le jeu de test avec la méthode
   que vous évaluez : elle corrigerait sa propre copie.
3. **Ne changez que les lignes d'entraînement** : `train` (la référence),
   `train_noisy` (avec des étiquettes fausses, réelles ou ajoutées exprès) et
   `train_cleaned` (après votre nettoyage).
4. **Comparez chaque classement à la référence** avec `compare_rankings`.

Avec des données réelles, vous n'avez pas de version propre : comparez le
classement avant et après le nettoyage, `compare_rankings(noisy, cleaned)`.
Vous saurez si le nettoyage change la décision, pas s'il a raison.

## Trouver les lignes suspectes

`oof_proba` donne, pour chaque ligne d'entraînement, les probabilités prédites
par un modèle qui ne l'a jamais vue. La règle la plus simple retire les lignes
dont l'étiquette observée reçoit la plus faible probabilité (*faible
confiance*) :

```python
import numpy as np

proba = noisy.oof_proba("svm (calibrated: sigmoid)")
labels = noisy.y_train.loc[proba.index]
confidence = proba.to_numpy()[range(len(proba)), proba.columns.get_indexer(labels)]

suspects = proba.index[np.argsort(confidence)[: int(0.2 * len(proba))]]  # les 20 % les moins crues
train_cleaned = train_noisy.drop(index=suspects)
```

Les mêmes probabilités alimentent d'autres méthodes, par exemple
`cleanlab.filter.find_label_issues(labels, proba.to_numpy())`. Un modèle sans
probabilités (`svm`) a besoin de `calibrate` : voir
[Modèles calibrés](../configuration.md#modeles-calibres).

## Lire les chiffres

| Chiffre | Sens | Bon signe après nettoyage |
|---|---|---|
| `kendall_tau` | Accord sur tout l'ordre : 1 = le même, 0 = sans rapport, -1 = inversé. | Plus proche de 1 qu'avant. |
| `top1_agree` | Le même meilleur modèle des deux côtés ? | `True` |
| `regret` | Score de test perdu, côté propre, en choisissant le gagnant de l'autre côté. | Plus proche de 0. |
| `table` | Le rang de chaque candidat des deux côtés (`rank_shift`). | De petits décalages. |

Dans le notebook (Dry Bean, 5 000 haricots, 20 % d'étiquettes fausses,
6 modèles) :

| | `kendall_tau` | `top1_agree` | `regret` |
|---|---|---|---|
| propre contre bruité | 0,73 | True | 0,000 |
| propre contre nettoyé | 1,00 | True | 0,000 |

Les étiquettes fausses ont échangé trois modèles au milieu du classement ; les
données nettoyées ont rendu l'ordre propre.

## Les pièges

**La taille du bruit.** Avec 5 à 8 modèles, le tau prend peu de valeurs : un
seul échange entre voisins le déplace déjà de 0,1 à 0,2. Avant de lire un
changement comme un effet du nettoyage, mesurez ce qui change quand rien ne
change : le tau entre deux graines sur les mêmes données.

```python
other_seed = Benchmark(data=train, test_data=test, project={"random_state": 1}, **setup)
baseline = compare_rankings(clean, other_seed.run(export=False))
print(baseline.kendall_tau)  # un écart plus petit que celui-ci, c'est du bruit
```

Répétez ensuite toute l'expérience avec plusieurs graines (le découpage, le
bruit et `random_state`) et donnez la moyenne et la dispersion, pas une seule
exécution.

**Les scores de validation croisée trompent sur des données bruitées.** Ils
sont calculés eux aussi sur les étiquettes bruitées, donc ils baissent pour
tous les modèles. Les scores de test, sur des étiquettes propres, montrent le
vrai effet. `compare_rankings(..., by="test")` classe sur ces scores.

**Retirer des lignes en retire aussi de bonnes.** Comptez combien de lignes
retirées étaient vraiment fausses quand vous le pouvez (bruit ajouté exprès),
essayez plusieurs seuils, et surveillez les classes rares : elles perdent des
lignes en premier.

**Le détecteur a été réglé sur les mêmes plis.** Les meilleurs paramètres
derrière `oof_proba` ont été choisis avec toutes les lignes.
`oof_proba(..., params="default")` est plus strict.

**La calibration.** La règle de faible confiance se fie aux probabilités.
`sigmoid` convient aux petits jeux ; `isotonic` est plus souple mais peut
surapprendre sous environ 1 000 lignes. Ajoutez `neg_brier_score` à
`evaluation.metrics` pour voir lequel donne les meilleures probabilités.
