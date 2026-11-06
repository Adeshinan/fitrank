# Command line

```bash
fitrank [--verbose | --quiet] COMMAND [ARGS]
fitrank --version
fitrank COMMAND --help
```

| Command | What it does |
|---|---|
| `fitrank run CONFIG` | Evaluates every candidate, shows the leaderboard and saves the results folder. |
| `fitrank validate CONFIG` | Checks the config and the data, trains nothing. |
| `fitrank init` | Writes a commented starter config. |
| `fitrank predict MODEL DATA` | Predicts new data with a saved model. |
| `fitrank list models` | Lists the model names, for each task. |
| `fitrank list reductions` | Lists the reduction names. |

## run

```bash
fitrank run examples/iris.yaml
```

Shows a progress bar, then the leaderboard on the standard output, and saves
the results in `<output_dir>/<name>_<date>_<time>/`. Everything except the
leaderboard (progress, warnings, errors) goes to the error output, so
`fitrank --quiet run config.yaml > leaderboard.txt` keeps only the table.

## validate

```bash
fitrank validate examples/titanic.yaml
```

```text
Config OK: examples/titanic.yaml
Data OK: 1309 rows, 7 features, target 'Survived'
8 candidates: 2 reduction(s) x 4 model(s), 5-fold cross-validation
```

Reads the data and runs every check of `run`, without training. All the
problems are listed at once.

## init

```bash
fitrank init --data data.csv --target price --task regression
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | `config.yaml` | The file to write. |
| `--task` | `classification` | `classification` or `regression`. |
| `--data` | — | A data file: fills `data.path`, the feature list and suggested preprocessing groups. |
| `--target` | — | The column to predict. |
| `--force` | off | Overwrites an existing file. |

The suggested preprocessing groups are commented out: nothing is applied until
you uncomment them.

## predict

```bash
fitrank predict runs/iris_2026-11-01_15-36-06/best_model.joblib new_flowers.csv -o predictions.csv
```

Adds a `prediction` column to the data and writes it as CSV, to the standard
output or to `-o FILE`. The data needs the same feature columns as for
training. A warning is shown when the model was saved with another
scikit-learn version.

## list

```bash
fitrank list models --task regression
fitrank list reductions
```

## Global options

| Option | Description |
|---|---|
| `--quiet`, `-q` | Only the leaderboard, for scripts. |
| `--verbose`, `-v` | One line per candidate as soon as it is done, and tracebacks on errors. |
| `--version` | Shows the FitRank version. |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success (warnings included). |
| `1` | Configuration error: unknown key, model, step or metric, wrong value. |
| `2` | Data error: file not found, missing column, missing values without an imputer... |
| `3` | Every candidate failed. |

Errors are shown in red, without a traceback: what is wrong, where, and how to
fix it. Warnings are shown in yellow and never stop the command.
