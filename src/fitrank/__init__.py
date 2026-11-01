"""FitRank: train, compare and rank scikit-learn models from a single YAML config."""

from fitrank.benchmark import Benchmark
from fitrank.exceptions import ConfigError, DataError, FitRankError, FitRankWarning
from fitrank.reduction import register_reduction
from fitrank.registry import register_model
from fitrank.results import Results, load_model

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "ConfigError",
    "DataError",
    "FitRankError",
    "FitRankWarning",
    "Results",
    "__version__",
    "load_model",
    "register_model",
    "register_reduction",
]
