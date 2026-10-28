"""Built-in regression models and their default grids.

Grids are small (2 to 6 combinations) so that a first run stays fast; widen
them with ``params`` in the YAML.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from fitrank.registry import Grid, Task, register_model

TASK: Task = "regression"


@register_model("linear_regression", task=TASK)
def linear_regression() -> tuple[LinearRegression, Grid]:
    return LinearRegression(), {"fit_intercept": [True, False]}


@register_model("ridge", task=TASK)
def ridge() -> tuple[Ridge, Grid]:
    return Ridge(), {"alpha": [0.1, 1, 10]}


@register_model("lasso", task=TASK)
def lasso() -> tuple[Lasso, Grid]:
    return Lasso(max_iter=10_000), {"alpha": [0.01, 0.1, 1]}


@register_model("polynomial_regression", task=TASK)
def polynomial_regression() -> tuple[Pipeline, Grid]:
    pipeline = Pipeline(
        [("poly", PolynomialFeatures(include_bias=False)), ("linear", LinearRegression())]
    )
    return pipeline, {"poly__degree": [2, 3]}


@register_model("svr", task=TASK)
def svr() -> tuple[SVR, Grid]:
    return SVR(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}


@register_model("knn", task=TASK)
def knn() -> tuple[KNeighborsRegressor, Grid]:
    return KNeighborsRegressor(), {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]}


@register_model("decision_tree", task=TASK)
def decision_tree() -> tuple[DecisionTreeRegressor, Grid]:
    return DecisionTreeRegressor(), {"max_depth": [None, 5, 10]}


@register_model("random_forest", task=TASK)
def random_forest() -> tuple[RandomForestRegressor, Grid]:
    return RandomForestRegressor(), {"n_estimators": [100, 300], "max_depth": [None, 10]}


@register_model("gradient_boosting", task=TASK)
def gradient_boosting() -> tuple[GradientBoostingRegressor, Grid]:
    return GradientBoostingRegressor(), {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
