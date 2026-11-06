# Classification ou régression ?

FitRank vous demande de déclarer `task:` dans le YAML. Ce guide montre comment
reconnaître la tâche d'un problème, et quels autres types de tâches existent.

## La règle de base

Regardez **la nature de ce que vous prédisez**, pas son format.

| Vous prédisez… | Tâche | Exemples |
|---|---|---|
| **Une catégorie** (une étiquette parmi une liste fixe) | **Classification** | Survivant ou non ? Spam ou pas ? Quelle espèce de fleur ? |
| **Une quantité** (un nombre sur une échelle continue) | **Régression** | Quel prix ? Quelle température demain ? Combien de minutes de trajet ? |

## Trois questions pour trancher

1. **Est-ce qu'une moyenne a du sens ?**
    - La moyenne de 200 000 € et 300 000 € est 250 000 €, ça a un sens → **régression**
    - La moyenne de « chat » et « chien » n'a aucun sens → **classification**

2. **Est-ce que la taille d'une erreur compte ?**
    - Prédire 210 000 € au lieu de 200 000 € est presque juste, 900 000 € est très faux → **régression**
    - Prédire « chien » au lieu de « chat » est juste faux, il n'y a pas de « presque » → **classification**

3. **Est-ce que le modèle peut sortir une valeur jamais vue ?**
    - Un prix de 237 451 € peut ne figurer nulle part dans les données, mais c'est une réponse valable → **régression**
    - Une espèce de fleur inventée n'est pas une réponse valable → **classification**

!!! warning "Attention au piège du format"
    Un nombre n'est **pas forcément** une quantité. Un code postal (75001), une
    classe de billet (1, 2, 3) ou une étiquette 0/1 sont des **catégories**
    écrites avec des chiffres.

    Indice : peu de valeurs possibles, et aucun calcul n'a de sens entre elles
    → c'est de la classification.

## Les cas limites

| Cible | Plutôt… | Pourquoi |
|---|---|---|
| Une note de 1 à 5 étoiles | Ça dépend | Il y a un ordre, mais pas une vraie échelle continue. On parle de **régression ordinale** (voir plus bas). |
| Nombre d'enfants (0, 1, 2…) | Régression | C'est un comptage : la moyenne a un sens. Il existe une variante dédiée, la **régression de Poisson**. |
| Âge en tranches (« 18-25 », « 26-35 ») | Classification | Ce sont des catégories, même si elles viennent d'un nombre. |
| Probabilité de défaut de paiement | Classification | Les classes sont « défaut / pas défaut », et le modèle donne une probabilité pour chaque classe. |

## Les variantes de la classification et de la régression

- **Classification binaire** : deux classes (oui / non).
- **Classification multiclasse** : plus de deux classes, une seule réponse par exemple (iris : setosa, versicolor ou virginica).
- **Classification multilabel** : plusieurs étiquettes à la fois (un film à la fois « comédie » et « romance »).
- **Régression multi-sortie** : plusieurs nombres à la fois (latitude et longitude).

FitRank traite la classification binaire et multiclasse, et la régression à
une seule sortie.

## Les autres types de tâches

La classification et la régression font partie de l'**apprentissage
supervisé** : les exemples viennent avec la bonne réponse (la *target*, ou
cible). Il existe d'autres familles.

### Supervisé, mais différent

| Tâche | Exemple |
|---|---|
| Régression ordinale | Prédire une note de 1 à 5 en respectant l'ordre |
| Prévision de séries temporelles | Les ventes du mois prochain à partir des mois passés |
| Ranking (classement) | Ordonner des résultats de recherche |
| Analyse de survie | Combien de temps avant qu'un client parte ou qu'une machine tombe en panne |

### Non supervisé (pas de cible, on cherche une structure)

| Tâche | Exemple |
|---|---|
| Clustering (regroupement) | Regrouper des clients aux comportements proches |
| Réduction de dimension | Résumer 50 colonnes en 3 (par exemple avec une ACP) |
| Détection d'anomalies | Repérer une transaction bancaire frauduleuse |

### Autres familles

- **Apprentissage par renforcement** : un agent apprend par essais et erreurs, avec des récompenses (jeux, robots).
- **Modèles génératifs** : produire du nouveau contenu (texte, images).
- **Semi-supervisé** : quelques exemples ont une cible, la plupart n'en ont pas.

## Le réflexe à garder

> Avant tout modèle, demandez-vous : **« ma réponse est-elle une étiquette ou une quantité ? »**
>
> - Une étiquette → **classification**
> - Une quantité → **régression**

Sur des données tabulaires, cette question suffit dans la grande majorité des
cas.

### Pourquoi FitRank ne devine pas

La tâche ne se déduit pas de la cible seule. Une note de 1 à 5 peut être une
classification **ou** une régression : c'est un choix lié au problème, pas aux
données. C'est pour ça que FitRank demande de déclarer `task:` explicitement.
Il avertit seulement, sans s'arrêter, quand la cible semble peu probable pour
la tâche (voir la [référence de la configuration](../configuration.md#task)).
