"""Built-in classification models and their default grids.

Grids are small (4 to 12 combinations) so that a first run stays fast; widen
them with ``params`` in the YAML. Models that support it search
``class_weight`` to cope with imbalanced classes.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fitrank.registry import Grid, Task, register_model

TASK: Task = "classification"
_BALANCE = [None, "balanced"]


@register_model("logistic_regression", task=TASK)
def logistic_regression() -> tuple[LogisticRegression, Grid]:
    return LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10], "class_weight": _BALANCE}


@register_model("svm", task=TASK)
def svm() -> tuple[SVC, Grid]:
    return SVC(), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "class_weight": _BALANCE,
    }


@register_model("knn", task=TASK)
def knn() -> tuple[KNeighborsClassifier, Grid]:
    return KNeighborsClassifier(), {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]}


@register_model("decision_tree", task=TASK)
def decision_tree() -> tuple[DecisionTreeClassifier, Grid]:
    return DecisionTreeClassifier(), {"max_depth": [None, 5, 10], "class_weight": _BALANCE}


@register_model("random_forest", task=TASK)
def random_forest() -> tuple[RandomForestClassifier, Grid]:
    return RandomForestClassifier(), {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "class_weight": _BALANCE,
    }


@register_model("gradient_boosting", task=TASK)
def gradient_boosting() -> tuple[GradientBoostingClassifier, Grid]:
    return GradientBoostingClassifier(), {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}


@register_model("naive_bayes", task=TASK)
def naive_bayes() -> tuple[GaussianNB, Grid]:
    return GaussianNB(), {"var_smoothing": [1e-9, 1e-8, 1e-7]}
