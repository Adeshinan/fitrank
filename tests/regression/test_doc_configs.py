"""Every YAML example of the READMEs and the documentation site is still accepted.

A snippet that is only part of a config (``models:``, ``evaluation:``...) is
completed with a small base config, then checked like ``fitrank validate``
checks a file: schema, model, step and reduction names. When the snippet
declares a data file that exists in the repository, the data is checked too,
and no warning is allowed.
"""

import copy
import importlib.util
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml
from sklearn.decomposition import TruncatedSVD

from fitrank import Benchmark
from fitrank.config import FitRankConfig, validate_config
from fitrank.pipeline import build_candidates
from fitrank.reduction import REDUCTIONS, register_reduction

REPO = Path(__file__).resolve().parents[2]
PAGES = [REPO / "README.md", REPO / "README.fr.md", *sorted((REPO / "docs").rglob("*.md"))]
YAML_BLOCK = re.compile(r"^( *)```ya?ml\n(.*?)^\1```", re.S | re.M)

# The data files the snippets name, as they are in this repository.
DATA = {"iris.csv": "examples/data/iris.csv"}
BASES = {
    "classification": {
        "data": {"path": "examples/data/iris.csv", "target": "species"},
        "task": "classification",
        "models": [{"name": "logistic_regression"}],
    },
    "regression": {
        "data": {"path": "examples/data/diabetes.csv", "target": "progression"},
        "task": "regression",
        "models": [{"name": "ridge"}],
    },
}
REGRESSION_METRICS = ("neg_", "r2", "explained_variance", "max_error")


class _UniqueKeyLoader(yaml.SafeLoader):
    """A YAML loader that refuses a key written twice, so such blocks are split."""

    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise yaml.constructor.ConstructorError(None, None, "duplicate key", node.start_mark)
        return super().construct_mapping(node, deep=deep)


def _snippets() -> list[Any]:
    """One pytest param per YAML snippet; a block with repeated sections gives several."""
    params = []
    for page in PAGES:
        if not page.is_file():
            continue
        text = page.read_text(encoding="utf-8")
        for i, match in enumerate(YAML_BLOCK.finditer(text)):
            indent = len(match.group(1))
            block = "\n".join(line[indent:] for line in match.group(2).splitlines())
            try:
                parts = [yaml.load(block, Loader=_UniqueKeyLoader)]
            except yaml.constructor.ConstructorError:  # alternatives, one per paragraph
                parts = [yaml.safe_load(p) for p in re.split(r"\n\s*\n", block) if p.strip()]
            for j, part in enumerate(parts):
                name = f"{page.relative_to(REPO).as_posix()}#{i + 1}" + (
                    f".{j + 1}" if len(parts) > 1 else ""
                )
                params.append(pytest.param(part, id=name))
    return params


@pytest.fixture(autouse=True)
def svd_reduction() -> Any:
    """The svd reduction that docs/models.*.md registers just before using it."""
    if "svd" in REDUCTIONS:
        yield
        return
    register_reduction("svd")(lambda task: (TruncatedSVD(), {"n_components": [2, 5]}))
    yield
    del REDUCTIONS["svd"]


def _complete(snippet: dict[str, Any]) -> dict[str, Any]:
    """The snippet, with the base config for the sections it does not declare."""
    metric = str((snippet.get("evaluation") or {}).get("primary_metric", ""))
    task = snippet.get("task") or (
        "regression" if metric.startswith(REGRESSION_METRICS) else "classification"
    )
    config = copy.deepcopy(BASES[task])
    if "data" in snippet and "path" not in snippet["data"]:
        snippet = {**snippet, "data": {**config["data"], **snippet["data"]}}
    config.update(copy.deepcopy(snippet))
    path = str(config["data"]["path"])
    config["data"]["path"] = DATA.get(path, path)
    return config


def _importable(model: dict[str, Any]) -> bool:
    """False for a class from a package that is not installed here (xgboost...)."""
    path = model.get("class")
    return path is None or importlib.util.find_spec(str(path).split(".")[0]) is not None


@pytest.mark.parametrize("snippet", _snippets())
def test_documented_config_is_valid(snippet: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(snippet, dict), "a YAML example must be a mapping of config sections"
    if all(value is None for value in snippet.values()):  # an outline of the sections
        assert list(snippet) == list(FitRankConfig.model_fields)
        return
    raw = _complete(snippet)
    validate_config(raw, "the snippet")  # the schema, with every model as written

    checked = {**raw, "models": [m for m in raw["models"] if _importable(m)]}
    if not checked["models"]:
        checked["models"] = BASES[raw["task"]]["models"]
    config = validate_config(checked, "the snippet")
    build_candidates(config)  # model, step and reduction names, model params

    declares_data = "path" in snippet.get("data", {})
    if declares_data and (REPO / config.data.path).is_file():
        monkeypatch.chdir(REPO)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no FitRankWarning either
            Benchmark.from_config(config).validate()


@pytest.mark.skipif(not (REPO / "docs").is_dir(), reason="docs/ is not in the source package")
def test_snippets_are_found() -> None:
    """The extraction itself works: the READMEs and docs/ give dozens of snippets."""
    assert len(_snippets()) >= 40
