# FitRank — project context for Claude Code

FitRank is an open source Python library (MIT) that trains, compares and ranks
scikit-learn models from a single YAML config. Owner: Adeshinan (GitHub).

## Key documents
- Specification (cahier des charges, FR): https://claude.ai/artifact/3UrLNjZrcPVz4iVSo1Gs5f
- 2-week plan (FR): https://claude.ai/artifact/4cfTZgVYZwq6N2tMKN2YQg
- Trello board (one card per day, with branch, examples and tests): https://trello.com/b/VgVFOwaX/fitrank-v10

## Scope v1.0 (release Friday 2026-11-06, 4 h/day)
- Tabular data only (CSV, Excel, Parquet, DataFrame). Images/HOG = v2.
- Classification and regression. Task declared by the user.
- Preprocessing is ALWAYS explicit: nothing is applied unless declared.
  FitRank reports data problems, it never fixes them silently.
- Registries for models, reductions and preprocessing steps
  (`@register_model(name, task)`): adding one never touches the engine.
- Same split, same CV folds, same metric for every candidate.
  Preprocessing lives inside the pipeline (no data leakage).
- Defaults: primary metric `f1_weighted` (classification),
  `neg_root_mean_squared_error` (regression, shown positive).
- A failing model never stops the run: it is marked `failed` in the leaderboard.
- Deferred to v1.1: MkDocs site, Titanic example, `fitrank init`,
  EF-04 (exclude), EF-11 (task/target warning), EF-44 (class path), EF-57 (n_jobs).

## Conventions
- Code, identifiers, docstrings, error messages and comments: English.
  User-facing docs: English (README.md) and French (README.fr.md).
- GitHub Flow: `main` is protected; one Trello card = one branch = one PR.
  Branch names: `<type>/<day>-<topic>`, e.g. `feat/j02-config-schema`.
- Conventional Commits: `feat(config): add evaluation section`.
- Error messages say what is wrong, where (YAML key / column) and how to fix it.
  User errors (`ConfigError`, `DataError`) never show a raw traceback.
- Python 3.10–3.13, `src/` layout, version in `src/fitrank/__init__.py` only.

## Commands
```bash
pip install -e ".[dev]"
pytest                      # tests
pytest -m "not e2e"         # skip the end-to-end tests (real fitrank processes, ~1 min)
pytest --cov                # coverage (target >= 85 %)
ruff check . && ruff format --check .
mypy                        # strict
pre-commit run --all-files
```

## Definition of done (every card)
Checklist done, tests pass locally, pre-commit clean, CI green, PR merged.
