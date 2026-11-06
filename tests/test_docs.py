"""Tests that the documentation site stays in step with the code."""

from pathlib import Path

import pytest

from fitrank.config import (
    DataConfig,
    EvaluationConfig,
    FitRankConfig,
    ModelConfig,
    OutputConfig,
    ProjectConfig,
)
from fitrank.preprocessing import STEPS
from fitrank.reduction import list_reductions
from fitrank.registry import TASKS, list_models

DOCS = Path(__file__).resolve().parents[1] / "docs"
LANGUAGES = ["en", "fr"]

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="docs/ is not in the source package")


def test_every_page_exists_in_both_languages() -> None:
    pages = {path.name.split(".")[0] for path in DOCS.glob("*.md")}
    for page in pages:
        for language in LANGUAGES:
            assert (DOCS / f"{page}.{language}.md").is_file(), f"{page}.{language}.md is missing"


@pytest.mark.parametrize("language", LANGUAGES)
def test_models_page_lists_every_model(language: str) -> None:
    text = (DOCS / f"models.{language}.md").read_text(encoding="utf-8")
    for task in TASKS:
        for name in list_models(task):
            assert f"| `{name}` |" in text, f"{name} ({task}) is missing from models.{language}.md"


@pytest.mark.parametrize("language", LANGUAGES)
def test_configuration_page_lists_every_reduction_and_step(language: str) -> None:
    text = (DOCS / f"configuration.{language}.md").read_text(encoding="utf-8")
    for name in [*list_reductions(), *STEPS]:
        assert f"| `{name}` |" in text, f"{name} is missing from configuration.{language}.md"


@pytest.mark.parametrize("language", LANGUAGES)
def test_configuration_page_documents_every_key(language: str) -> None:
    text = (DOCS / f"configuration.{language}.md").read_text(encoding="utf-8")
    sections = [ProjectConfig, DataConfig, ModelConfig, EvaluationConfig, OutputConfig]
    for section in sections:
        for name, field in section.model_fields.items():
            key = field.alias or name
            assert f"| `{key}` |" in text, f"{key} is missing from configuration.{language}.md"
    for name in FitRankConfig.model_fields:
        assert f"## {name}" in text, f"section {name} is missing from configuration.{language}.md"
