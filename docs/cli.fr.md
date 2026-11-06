# Ligne de commande

```bash
fitrank [--verbose | --quiet] COMMANDE [ARGUMENTS]
fitrank --version
fitrank COMMANDE --help
```

| Commande | Ce qu'elle fait |
|---|---|
| `fitrank run CONFIG` | Évalue chaque candidat, affiche le classement et enregistre le dossier de résultats. |
| `fitrank validate CONFIG` | Vérifie la configuration et les données, n'entraîne rien. |
| `fitrank init` | Écrit une configuration de départ commentée. |
| `fitrank predict MODELE DONNEES` | Prédit de nouvelles données avec un modèle enregistré. |
| `fitrank list models` | Liste les noms de modèles, pour chaque tâche. |
| `fitrank list reductions` | Liste les noms de réductions. |

## run

```bash
fitrank run examples/iris.yaml
```

Affiche une barre de progression, puis le classement sur la sortie standard,
et enregistre les résultats dans `<output_dir>/<name>_<date>_<heure>/`. Tout le
reste (progression, avertissements, erreurs) va sur la sortie d'erreur :
`fitrank --quiet run config.yaml > classement.txt` ne garde que le tableau.

## validate

```bash
fitrank validate examples/titanic.yaml
```

```text
Config OK: examples/titanic.yaml
Data OK: 1309 rows, 7 features, target 'Survived'
8 candidates: 2 reduction(s) x 4 model(s), 5-fold cross-validation
```

Lit les données et fait toutes les vérifications de `run`, sans entraîner.
Tous les problèmes sont listés d'un coup.

## init

```bash
fitrank init --data data.csv --target price --task regression
```

| Option | Défaut | Description |
|---|---|---|
| `-o`, `--output` | `config.yaml` | Le fichier à écrire. |
| `--task` | `classification` | `classification` ou `regression`. |
| `--data` | — | Un fichier de données : remplit `data.path`, la liste des variables et des groupes de prétraitement suggérés. |
| `--target` | — | La colonne à prédire. |
| `--force` | désactivé | Écrase un fichier existant. |

Les groupes de prétraitement suggérés sont en commentaire : rien n'est
appliqué tant que vous ne les décommentez pas.

## predict

```bash
fitrank predict runs/iris_2026-11-01_15-36-06/best_model.joblib nouvelles_fleurs.csv -o predictions.csv
```

Ajoute une colonne `prediction` aux données et les écrit en CSV, sur la sortie
standard ou dans `-o FICHIER`. Les données doivent avoir les mêmes colonnes
qu'à l'entraînement. Un avertissement s'affiche quand le modèle a été
enregistré avec une autre version de scikit-learn.

## list

```bash
fitrank list models --task regression
fitrank list reductions
```

## Options globales

| Option | Description |
|---|---|
| `--quiet`, `-q` | Seulement le classement, pour les scripts. |
| `--verbose`, `-v` | Une ligne par candidat dès qu'il est fini, et la trace d'appels en cas d'erreur. |
| `--version` | Affiche la version de FitRank. |

## Codes de sortie

| Code | Signification |
|---|---|
| `0` | Succès (avertissements compris). |
| `1` | Erreur de configuration : clé, modèle, étape ou métrique inconnus, mauvaise valeur. |
| `2` | Erreur de données : fichier introuvable, colonne absente, valeurs manquantes sans imputer... |
| `3` | Tous les candidats ont échoué. |

Les erreurs s'affichent en rouge, sans trace d'appels : ce qui ne va pas, où,
et comment corriger. Les avertissements s'affichent en jaune et n'arrêtent
jamais la commande.
