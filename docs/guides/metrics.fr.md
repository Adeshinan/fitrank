# Choisir une métrique

Ce que mesure chaque métrique, quand l'utiliser, ses pièges, et comment le
choix change une exécution FitRank. Tous les chiffres de cette page ont été
calculés avec scikit-learn.

Il n'existe pas de « meilleure métrique » dans l'absolu. Le bon choix répond à
une seule question :

> **Quelle erreur vous coûte le plus cher ?**

## D'où viennent les noms

- **Les formules** (F1, RMSE, R²…) sont des outils de statistique, plus anciens
  que scikit-learn. Le F1 vient de la « F-measure » de C. J. van Rijsbergen
  (recherche d'information, 1979).
- **Les noms** que vous écrivez dans le YAML (`f1_weighted`,
  `neg_root_mean_squared_error`…) sont les noms de scorers de scikit-learn.
  Pour les lister :
  ```bash
  python -c "from sklearn.metrics import get_scorer_names; print(get_scorer_names())"
  ```
- **FitRank** n'invente aucun nom. Il vérifie que le nom existe (et propose une
  correction en cas de faute de frappe), choisit les métriques par défaut
  (`f1_weighted` en classification, la RMSE en régression) et affiche les
  erreurs en positif.
- Le préfixe **`neg_`** est une convention de scikit-learn, qui cherche
  toujours le plus grand score : les erreurs sont donc stockées en négatif. Une
  RMSE de 55 est stockée −55 ; FitRank affiche 55.

### Une seule métrique principale

```yaml
evaluation:
  primary_metric: f1_macro                        # une seule : classe les modèles
  metrics: [accuracy, recall_macro, roc_auc_ovr]  # plusieurs : affichées pour information
```

La métrique principale sert **deux fois** :

1. à l'intérieur de chaque candidat, pour choisir les meilleurs paramètres (le
   `refit` de `GridSearchCV`) ;
2. entre les candidats, pour choisir le meilleur modèle.

Il n'y en a qu'une parce qu'avec deux critères il n'y a souvent pas de
gagnant. Si A est meilleur en F1 et B en accuracy, vous seul savez ce qui
compte le plus.

## Classification

### L'exemple fil rouge : 100 patients, 10 malades, 90 sains

On compare trois modèles :

- **Notre modèle** : trouve 6 malades, en rate 4, donne 3 fausses alertes.
- **Paresseux** : dit toujours « sain ».
- **Alarmiste** : dit toujours « malade ».

| Métrique | Notre modèle | Paresseux | Alarmiste |
|---|---|---|---|
| `accuracy` | 0.93 | **0.90** ⚠️ | 0.10 |
| `balanced_accuracy` | 0.78 | 0.50 | 0.50 |
| `precision` | 0.67 | 0.00 | 0.10 |
| `recall` | 0.60 | 0.00 | **1.00** ⚠️ |
| `f1` | 0.63 | 0.00 | 0.18 |
| `f1_macro` | 0.80 | 0.47 | 0.09 |
| `f1_weighted` | 0.93 | **0.85** ⚠️ | 0.02 |
| `matthews_corrcoef` | 0.59 | 0.00 | 0.00 |

⚠️ = la métrique se fait tromper par un modèle inutile.

### `accuracy` : la part de bonnes réponses

```text
accuracy = bonnes réponses / total
```

- **Bien pour :** des classes **équilibrées** (iris : 50/50/50, chiffres
  manuscrits) où toutes les erreurs se valent.
- **Piège :** avec des classes déséquilibrées, le paresseux obtient 0.90 sans
  rien faire.
- **Si elle est mal choisie :** un modèle qui ignore la classe rare peut être
  classé premier. Sur un jeu de fraude avec 8 % de fraudes, un modèle qui ne
  signale aucune fraude atteint quand même 0.92.

### `balanced_accuracy` : l'accuracy corrigée

```text
balanced_accuracy = moyenne du rappel de chaque classe
                  = (6/10 malades trouvés + 87/90 sains trouvés) / 2 = 0.78
```

- **Bien pour :** des classes **déséquilibrées**, quand vous voulez un chiffre
  aussi simple à lire que l'accuracy.
- **Repère :** 0.50 = hasard (2 classes). Le paresseux et l'alarmiste tombent
  tous les deux à 0.50.

### `precision` : « quand il dit oui, a-t-il raison ? »

```text
précision = vrais positifs / tout ce qu'il a signalé = 6 / 9 = 0.67
```

- **Bien quand une fausse alerte coûte cher :**
    - un filtre anti-spam : envoyer un vrai mail dans les spams est pire que laisser passer une publicité ;
    - accuser quelqu'un de fraude, recommander un produit cher.
- **Piège :** un modèle qui ne signale presque rien peut avoir une précision
  parfaite. Ne l'utilisez jamais seule comme métrique principale.

### `recall` : « parmi les vrais cas, combien en a-t-il trouvé ? »

```text
rappel = vrais positifs / tous les vrais cas = 6 / 10 = 0.60
```

- **Bien quand rater un cas coûte cher :**
    - le dépistage d'un cancer : un malade non détecté peut mourir, une fausse alerte coûte un examen de plus ;
    - la détection de fraude, une pièce défectueuse sur une chaîne de production.
- **Piège :** l'alarmiste a un rappel de 1.00 en déclarant tout le monde
  malade. Ne l'utilisez jamais seul non plus.

### La bascule précision / rappel

Les deux s'opposent : un modèle prudent gagne en précision et perd en rappel,
un modèle zélé fait l'inverse.

```text
Anti-spam          →  privilégier la précision  (ne jamais bloquer un vrai mail)
Cancer             →  privilégier le rappel     (ne jamais rater un malade)
Les deux comptent  →  F1
```

### La famille F1 : les deux à la fois

```text
F1 = 2 × précision × rappel / (précision + rappel)
```

C'est une moyenne **harmonique**, qui punit le déséquilibre : l'alarmiste
(précision 0.10, rappel 1.00) a une moyenne simple de 0.55, mais un F1 de
0.18.

La formule est toujours la même. Ce qui change, c'est **sur quelle classe** on
la calcule et **comment on combine** les classes :

| Variante | Calcul | Patients |
|---|---|---|
| `f1` | seulement la classe `1` (malade) | 0.63 |
| `f1_macro` | moyenne simple des F1 par classe : (0.63 + 0.96) / 2 | 0.80 |
| `f1_weighted` | moyenne pondérée par le nombre de lignes : (0.63 × 10 + 0.96 × 90) / 100 | 0.93 |
| `f1_micro` | tous les cas mélangés, égal à l'accuracy | 0.93 |

| Variante | Quand | Piège |
|---|---|---|
| `f1` | 2 classes (0 / 1), **une seule** vous intéresse | échoue avec 3 classes ou plus, et avec une cible texte |
| `f1_macro` | classes **déséquilibrées**, **toutes** importantes | une petite classe pèse autant qu'une grosse |
| `f1_weighted` | classes **équilibrées**, note globale | **cache** les classes rares : 0.85 pour le paresseux |
| `f1_micro` | jamais dans FitRank | identique à l'accuracy |

`f1_weighted` est le défaut de FitRank parce qu'il **n'échoue jamais** (2
classes ou plus, n'importe quels noms de classes). Il est **sûr**, pas
**meilleur**.

Avec 3 classes (A : 80 lignes, B : 15, C : 5, mal reconnue) :

```text
F1 par classe : A = 0.96   B = 0.90   C = 0.33

f1_macro    = 0.73   ← voit que C est mal reconnue
f1_weighted = 0.92   ← C ne pèse que 5 %
f1          = erreur (3 classes)
```

### `matthews_corrcoef` (MCC) : le plus honnête

Un coefficient entre −1 et +1 qui utilise les 4 cases de la matrice de
confusion (trouvés, ratés, fausses alertes, bien rejetés).

```text
+1 = parfait     0 = hasard ou modèle inutile     −1 = toujours faux
```

- **Bien pour :** des classes **très déséquilibrées**, quand vous voulez un
  seul chiffre impossible à tromper. Le paresseux et l'alarmiste ont 0.
- **Limite :** moins facile à expliquer.

### `roc_auc` : la qualité du tri, sans seuil

Un modèle donne souvent une probabilité (« malade à 73 % »), puis on coupe à
50 %. L'AUC répond à une autre question :

> Si je prends un malade et un sain au hasard, le modèle donne-t-il une
> probabilité plus haute au malade ?

```text
Vérité :        sain  sain  sain  malade  malade
Probabilités :  0.10  0.20  0.30  0.40    0.45

accuracy (seuil 50 %) = 0.60   → aucun malade détecté
roc_auc               = 1.00   → le tri est parfait, c'est le seuil qui est mal placé
```

- **Bien pour :** juger un modèle avant de choisir le seuil (un score de risque
  de crédit, où la banque choisit elle-même à partir de quel score refuser).
- **Variantes :** `roc_auc` (2 classes, fonctionne avec `predict_proba` ou
  `decision_function`, donc aussi avec `svm`), `roc_auc_ovr` / `roc_auc_ovo`
  (3 classes ou plus, demandent `predict_proba`).

### `neg_log_loss` : punit les erreurs sûres d'elles

Deux modèles avec la même accuracy (0.75) sur 4 cas :

```text
                  cas 1    cas 2    cas 3    cas 4 (vrai = 1)
Sûr de lui :      0.99     0.01     0.99     0.01   ← se trompe à 99 % de certitude
Prudent :         0.80     0.20     0.80     0.40   ← se trompe en hésitant

log_loss : sûr de lui = 1.16     prudent = 0.40   (plus petit = meilleur)
```

- **Bien quand vous utilisez les probabilités elles-mêmes**, pas seulement la
  classe. Par exemple, un assureur qui fixe un prix selon « 12 % de risque
  d'accident ».
- **Demande** `predict_proba` : sur `svm`, FitRank la saute avec une note
  quand elle est secondaire, et le candidat échoue quand elle est principale.

## Régression

### L'exemple fil rouge : le prix de 5 maisons (en k€)

```text
Vrai prix  : 200   250   300   350   400
Modèle A   : 210   240   310   340   410    ← se trompe toujours de 10
Modèle B   : 200   250   300   350   450    ← parfait, sauf UNE grosse erreur de 50
```

| Métrique | Modèle A | Modèle B | Gagnant |
|---|---|---|---|
| MAE (erreur absolue moyenne) | 10.0 | 10.0 | égalité |
| **RMSE** (racine de l'erreur quadratique moyenne) | **10.0** | **22.4** | **A** |
| MedAE (erreur absolue médiane) | 10.0 | **0.0** | **B** |
| max_error (pire erreur) | 10 | 50 | A |
| MAPE (erreur en %) | 3.5 % | 2.5 % | B |
| R² | 0.98 | 0.90 | A |

Mêmes données : selon la métrique, A ou B gagne.

### `neg_mean_absolute_error` (MAE) : l'erreur moyenne, en unités réelles

```text
MAE = moyenne des |vrai − prédit|      → « en moyenne, je me trompe de 10 k€ »
```

- **Bien quand chaque erreur coûte proportionnellement** : une erreur de 50
  coûte 5 fois une erreur de 10, pas plus.
- **Bien avec des valeurs aberrantes** (une maison à 5 M€ dans un quartier à
  300 k€) qui ne doivent pas tout dominer.
- **La plus facile à expliquer** à quelqu'un qui ne fait pas de data.

### `neg_root_mean_squared_error` (RMSE) : le défaut de FitRank

```text
RMSE = racine de la moyenne des erreurs AU CARRÉ
       l'erreur de 50 → 2500 : elle écrase les autres
```

- **Bien quand une grosse erreur est grave** : prévision de consommation
  électrique (une forte sous-estimation provoque une coupure), dosage d'un
  médicament, stock d'un entrepôt.
- Même unité que la cible, donc lisible.
- **Piège :** une seule valeur aberrante peut faire exploser la RMSE et
  influencer tout le classement.

### `neg_median_absolute_error` (MedAE) : ignore les exceptions

```text
Erreurs de B : 0, 0, 0, 0, 50   → médiane = 0
```

- **Bien quand les données contiennent des valeurs aberrantes connues**
  (erreurs de saisie, cas exceptionnels) et qu'on juge le cas typique.
- **Piège :** elle ignore les grosses erreurs. B obtient un score parfait
  alors qu'il s'est trompé de 50 k€.

### `neg_max_error` : la pire erreur

- **Bien quand une seule erreur au-delà d'un seuil est inacceptable** :
  tolérance d'usinage, sécurité.
- **Piège :** elle ne regarde qu'une ligne et ignore toutes les autres.

### `neg_mean_absolute_percentage_error` (MAPE) : l'erreur en %

```text
MAPE = moyenne des |erreur| / vrai     → « je me trompe de 3.5 % en moyenne »
```

- **Bien pour comparer des échelles différentes** : se tromper de 100 € sur
  1 000 € (10 %) est pire que sur 100 000 € (0.1 %), par exemple pour les
  ventes d'un petit magasin et d'un hypermarché.
- **Gros piège : les petites valeurs.**
  ```text
  Vrai   :  1     100    1000
  Prédit :  3     102    1002      ← erreur de 2 partout

  MAE  = 2
  MAPE = 67 %     (le « 1 → 3 » compte pour une erreur de 200 %)
  ```
  Si la cible peut valoir 0, la MAPE devient infinie.

### `r2` (R²) : quelle part de la variation est expliquée

```text
R² = 1     → parfait
R² = 0     → pas mieux que de prédire toujours la moyenne
R² < 0     → pire que de prédire la moyenne
```

- **Bien pour communiquer** (« le modèle explique 90 % de la variation du
  prix ») et pour comparer des problèmes d'échelles différentes.
- **Piège :** il ne dit pas de combien on se trompe en unités réelles. À
  utiliser à côté de la RMSE ou de la MAE.

### `neg_root_mean_squared_log_error` (RMSLE) : erreurs relatives sur des données étalées

```text
Vrai      : 100    1 000    100 000
Modèle 1  : 200    1 000    100 000     ← double un petit prix
Modèle 2  : 100    1 000    100 100     ← +100 sur un gros prix (0.1 %)

RMSE  : 57.7 et 57.7        (identiques : même erreur absolue de 100)
RMSLE : 0.397 et 0.0006     (doubler un prix est grave, 0.1 % ne l'est pas)
```

- **Bien pour des cibles très étalées et positives** : prix immobiliers d'un
  pays entier, salaires, nombre de vues d'une vidéo.
- **Demande** des valeurs ≥ 0.

## Comment choisir

### L'arbre de décision

```text
CLASSIFICATION
│
├─ Les classes sont-elles équilibrées ?
│     OUI → f1_weighted (défaut) ou accuracy
│     NON ↓
│
├─ Une seule classe vous intéresse (cible 0/1) ?
│     OUI → Qu'est-ce qui coûte le plus cher ?
│            • rater un cas         → recall    (+ f1 en secondaire)
│            • une fausse alerte    → precision (+ f1 en secondaire)
│            • les deux             → f1
│     NON → f1_macro, balanced_accuracy ou matthews_corrcoef
│
└─ Allez-vous utiliser les probabilités (scores, seuil choisi plus tard) ?
      → roc_auc (tri) ou neg_log_loss (probabilités justes)


RÉGRESSION
│
├─ Une grosse erreur est-elle grave ?                OUI → RMSE (défaut)
├─ Des valeurs aberrantes à ne pas surpondérer ?     OUI → MAE
├─ Des échelles très différentes ?
│     • valeurs jamais proches de 0 → MAPE
│     • valeurs très étalées, > 0   → RMSLE
└─ Pour communiquer                                  → r2 en secondaire
```

### L'impact du choix, à trois niveaux

1. **Sur le classement.** Un autre modèle peut passer premier : le modèle qui
   ne signale aucune fraude peut être premier en accuracy et dernier en
   `f1_macro`.
2. **Sur les paramètres choisis.** La métrique principale guide aussi
   `GridSearchCV` quand il choisit C, max_depth… Avec `recall`, il tend à
   préférer `class_weight: balanced`, qui pousse le modèle à signaler plus de
   cas rares. Le même modèle sort donc différent.
3. **Sur le comportement en production.** Le modèle gagnant optimise ce qu'on
   a mesuré. Mesurer l'accuracy sur de la fraude, c'est déployer un modèle qui
   ne détecte pas la fraude, et ne s'en rendre compte qu'une fois en service.

### Ce qui compte, et ne compte pas, dans le classement FitRank

Le classement repose sur un seul chiffre : la **moyenne des scores de
validation croisée** de la métrique principale (la colonne `cv_…`), mesurée
dans des conditions identiques pour tous (même découpage, mêmes plis, même
`random_state`).

| Colonne | Rôle |
|---|---|
| `cv_…` | **classe** les modèles |
| `test_…` | **vérifie** le gagnant, ne le choisit jamais |
| `cv_std` | stabilité entre les plis, à interpréter vous-même |
| métriques secondaires | comprendre d'où vient le score |
| `time` | information, n'entre pas dans le classement |

Limites à garder en tête :

- **Les petits écarts ne veulent souvent rien dire.** Sur l'exemple Diabetes,
  lasso (RMSE 55.36) et ridge (55.37) sont séparés de 0.02, alors que les plis
  varient d'environ ± 2.35. Ils se valent : prenez le plus simple.
- **Regardez `cv_std`** : 0.90 ± 0.01 est plus fiable que 0.91 ± 0.08.
- **Comparez `cv` et `test`** : un score de test bien plus bas signale un
  surapprentissage, ou un jeu de test trop petit.
- **Sur peu de données, tout est bruité** : le classement peut changer avec
  `random_state`.

### Exemples de YAML selon le cas

```yaml
# Iris, chiffres manuscrits : classes équilibrées
evaluation:
  primary_metric: f1_weighted
  metrics: [accuracy]

# Dépistage d'une maladie (cible 0/1) : ne rater personne
evaluation:
  primary_metric: recall
  metrics: [precision, f1, balanced_accuracy]

# Filtre anti-spam (cible 0/1) : ne jamais bloquer un vrai mail
evaluation:
  primary_metric: precision
  metrics: [recall, f1]

# Fraude, churn : classes déséquilibrées, toutes importantes
evaluation:
  primary_metric: f1_macro
  metrics: [balanced_accuracy, matthews_corrcoef, roc_auc]

# Prix de maisons dans une ville : une grosse erreur coûte cher
evaluation:
  primary_metric: neg_root_mean_squared_error
  metrics: [neg_mean_absolute_error, r2]

# Ventes de magasins de tailles très différentes
evaluation:
  primary_metric: neg_mean_absolute_percentage_error
  metrics: [neg_mean_absolute_error, r2]
```

### Les trois règles à retenir

1. **Choisir la métrique AVANT de regarder les résultats.** Changer de
   métrique jusqu'à ce que son modèle préféré gagne, c'est se tromper
   soi-même.
2. **Une métrique principale, plusieurs secondaires.** La principale décide,
   les secondaires vérifient qu'il n'y a pas de piège (un rappel de 1.00 avec
   une précision de 0.10, par exemple).
3. **Toujours comparer au modèle paresseux.** Une accuracy de 0.90 ne veut
   rien dire si 90 % des lignes sont dans la même classe.

!!! note "Une limite actuelle"
    FitRank vérifie qu'un nom de métrique **existe**, pas qu'il **va avec la
    tâche** : `task: classification` avec `primary_metric: r2` est accepté
    aujourd'hui, tout comme des scores de clustering comme
    `adjusted_rand_score`. Une vérification de la métrique selon la tâche
    pourra venir dans une version ultérieure.
