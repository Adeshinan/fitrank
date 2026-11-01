# Changelog

All notable changes to FitRank are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-11-06

First public release: train, compare and rank scikit-learn models on tabular
data from a single YAML file.

### Added

- **Configuration**: one YAML file describes the whole experiment (`project`, `data`,
  `task`, `preprocessing`, `reduction`, `models`, `evaluation`, `output`), validated
  before anything runs. Errors name the key, explain the problem and suggest a fix.
- **Data**: CSV, Excel (`fitrank[excel]`) and Parquet (`fitrank[parquet]`) files, or a
  pandas DataFrame from Python. Missing values and text columns are reported, never
  fixed silently.
- **Preprocessing, only when declared**: groups of columns, each with ordered steps:
  `imputer`, `scaler`, `encoder`. Fitted inside each cross-validation fold.
- **Reductions**: `none`, `pca`, `select_k_best`, `variance_threshold`,
  `polynomial_features`; each one is combined with each model, and its options can be a
  grid.
- **Models**: 7 for classification and 9 for regression, each with a small default
  grid that `params` replaces. Add your own with `@register_model` and
  `@register_reduction`.
- **Evaluation**: a test set kept aside (stratified for classification), the same
  cross-validation folds for every candidate, grid or random search, one primary metric
  to rank (`f1_weighted` or RMSE by default) and secondary metrics. A failing model is
  marked `failed` and the others carry on.
- **Results**: a ranked leaderboard; each run writes a timestamped folder with the
  leaderboard (CSV, JSON, Markdown), the configuration used, library versions and the
  best model refitted on the training set. `load_model` warns when the scikit-learn
  version differs.
- **Command line**: `fitrank run`, `validate`, `predict`, `list models`,
  `list reductions`, with `--verbose` / `--quiet` and exit codes 0, 1 (configuration),
  2 (data) and 3 (every model failed).
- **Python API**: `Benchmark` (from a YAML file or from arguments), `Results`,
  `load_model`. The API and the command give the same results.
- **Examples**: `examples/iris.yaml` (classification, commented key by key) and
  `examples/diabetes.yaml` (regression), tested end to end in CI.
- README in English and French; contributing guide, code of conduct, issue templates.

## [0.1.0] - 2026-10-26

### Added

- Package skeleton published to reserve the `fitrank` name on PyPI.

[Unreleased]: https://github.com/Adeshinan/fitrank/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Adeshinan/fitrank/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/Adeshinan/fitrank/releases/tag/v0.1.0
