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
   Python 3.10 to 3.13, coverage of at least 85 %.
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
tables of `README.md` and `README.fr.md`, and add a test.

## Releases

Maintainers release by pushing a `vX.Y.Z` tag on `main`, after setting
`__version__` in `src/fitrank/__init__.py` and dating the CHANGELOG section. The
release workflow publishes to PyPI and creates the GitHub release. A `vX.Y.ZrcN`
tag publishes a release candidate to TestPyPI first.
