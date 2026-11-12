# Contributing to FitRank

Thank you for helping! Bug reports, ideas, documentation fixes and new models are
all welcome. Please follow the [code of conduct](CODE_OF_CONDUCT.md).

## Before you start

- **Found a bug or have an idea?** Open an [issue](https://github.com/Adeshinan/fitrank/issues)
  with one of the templates.
- **Want to add a model?** Use the "New model" template first, so we can agree on
  the default grid before you write code.
- **Planning a bigger change?** Open an issue to discuss it before the pull request.

FitRank has a few rules that shape every change:

- Preprocessing is always explicit: FitRank reports data problems, it never fixes
  them silently.
- Models, reductions and preprocessing steps go through registries: adding one
  never touches the engine.
- Every candidate gets the same split, the same folds and the same metric.
- Error messages say what is wrong, where (YAML key or column) and how to fix it.

## Set up a development environment

Python 3.10 to 3.13.

```bash
git clone https://github.com/Adeshinan/fitrank.git
cd fitrank
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install               # ruff and mypy run before each commit
```

## Run the checks

```bash
pytest                           # all tests
pytest -m "not e2e"              # skip the end-to-end tests (real fitrank processes, ~1 min)
pytest --cov                     # coverage; CI fails below 85 %
ruff check . && ruff format --check .
mypy                             # strict
pre-commit run --all-files
```

New code comes with tests. Code, identifiers, docstrings, error messages and
comments are in English.

## Branches and commits

We follow GitHub Flow: `main` is always releasable and only changes through pull
requests. One topic = one branch = one pull request.

Branch names: `<type>/<topic>`, for example `feat/extra-trees-model`.

| Type | Use |
|---|---|
| `feat/` | a new feature |
| `fix/` | a bug fix |
| `docs/` | documentation |
| `test/` | tests only |
| `chore/` | tooling, CI, packaging |

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/),
in the imperative:

```
feat(models): add extra_trees for classification
fix(data): raise a clear error when the target column is missing
docs(readme): add a regression example
```

## Open a pull request

1. Create your branch from an up-to-date `main`:
   ```bash
   git switch main
   git pull
   git switch -c feat/extra-trees-model
   ```
2. Commit in small steps, then push: `git push -u origin feat/extra-trees-model`.
3. Open the pull request and fill in the template: link the issue, describe what
   changed, list the tests you added.
4. CI must be green: ruff, mypy, the tests on Linux, macOS and Windows with
   Python 3.10 to 3.13, coverage of at least 85 %, and the non-regression job.
   **A pull request that breaks `tests/regression/` is not merged**: see
   [Non-regression tests](#non-regression-tests).
5. Pull requests are merged with **Squash and merge**, so the title becomes the
   commit message: write it as a Conventional Commit.

User-facing changes get a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

## Add a model

Built-in models live in `src/fitrank/models/classification.py` and
`src/fitrank/models/regression.py`:

```python
@register_model("extra_trees", task=TASK)
def extra_trees() -> tuple[ExtraTreesClassifier, Grid]:
    return ExtraTreesClassifier(), {"n_estimators": [100, 300], "max_depth": [None, 10]}
```

Keep the default grid small (a first run must stay fast), add the model to the
tables of `README.md`, `README.fr.md`, `docs/models.en.md` and `docs/models.fr.md`
(a test checks the docs), and add a test.

## Documentation

The site lives in `docs/`: each page exists in English (`page.en.md`) and in
French (`page.fr.md`). The Python API reference is generated from the
docstrings.

```bash
pip install -e ".[docs]"
mkdocs serve                     # http://127.0.0.1:8000, reloads on each change
mkdocs build --strict            # what CI runs: fails on a broken link
```

Every push to `main` publishes the site to https://adeshinan.github.io/fitrank/.

## Non-regression tests

`tests/regression/` checks that what worked in FitRank 1.1 still works:

- **Golden leaderboards**: the Iris, Diabetes and Titanic examples (frozen copies
  in `tests/regression/configs/`) give the same ranking, parameters and notes,
  and the same scores to 1e-6, as the files in `tests/regression/golden/`. They
  run in the "Non-regression" CI job, on Linux with the versions pinned in
  `tests/regression/constraints.txt`. Elsewhere they are skipped.
- **Public API**: names exported by `fitrank`, parameters and their defaults,
  config keys and defaults, registered models, reductions and steps.
- **Command line**: commands, options, exit codes, the files of a run folder and
  the columns of `leaderboard.csv` / `leaderboard.json`.
- **Documented configs**: every YAML example of the READMEs and of `docs/` is
  still valid.

Adding something never breaks these tests; removing, renaming or changing a
default does. If a change of behaviour is really intended, update the test in
the same pull request and say why in its description.

The golden leaderboards are recorded on Linux: PCA goes through OpenBLAS, whose
last digits differ between platforms, and trees can turn them into other
choices of parameters. After an intended change of results, download the
artifact "golden-leaderboards" of the "Non-regression" job (the leaderboards
that job computed) and copy its CSV files into `tests/regression/golden/`.

The notebook `examples/try_fitrank.ipynb` is executed before each release (and
by hand from the Actions tab, workflow "Notebook").

## Releases

Maintainers release by pushing a `vX.Y.Z` tag on `main`, after setting
`__version__` in `src/fitrank/__init__.py` and dating the CHANGELOG section. The
release workflow publishes to PyPI and creates the GitHub release. A `vX.Y.ZrcN`
tag publishes a release candidate to TestPyPI first.
