"""FitRank: train, compare and rank scikit-learn models from a single YAML config."""

from fitrank.benchmark import Benchmark
from fitrank.compare import RankingComparison, compare_rankings
from fitrank.exceptions import ConfigError, DataError, FitRankError, FitRankWarning
from fitrank.reduction import register_reduction
from fitrank.registry import register_model
from fitrank.results import Results, load_model

__version__ = "1.1.2"

__all__ = [
    "Benchmark",
    "ConfigError",
    "DataError",
    "FitRankError",
    "FitRankWarning",
    "RankingComparison",
    "Results",
    "__version__",
    "compare_rankings",
    "load_model",
    "register_model",
    "register_reduction",
]
