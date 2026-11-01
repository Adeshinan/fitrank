"""Regenerate the example datasets in examples/data from scikit-learn.

Both datasets ship with scikit-learn, so their origin is public and the CSV
files can be rebuilt at any time::

    python scripts/make_example_data.py

- iris.csv: 150 flowers, 4 measurements in cm, the species to predict.
- diabetes.csv: 442 patients, 10 measurements in their original units (not
  the scaled version), and the disease progression one year later. The blood
  serum columns s1 to s6 get the names given in the scikit-learn docs.
"""

from pathlib import Path

from sklearn.datasets import load_diabetes, load_iris

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

IRIS_COLUMNS = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
}

DIABETES_COLUMNS = {
    "s1": "total_cholesterol",
    "s2": "ldl",
    "s3": "hdl",
    "s4": "cholesterol_hdl_ratio",
    "s5": "log_triglycerides",
    "s6": "glucose",
    "target": "progression",
}


def make_iris() -> Path:
    iris = load_iris(as_frame=True)
    frame = iris.frame.rename(columns=IRIS_COLUMNS)
    frame["species"] = iris.target_names[frame.pop("target")]  # 0 -> setosa...
    path = DATA_DIR / "iris.csv"
    frame.to_csv(path, index=False)
    return path


def make_diabetes() -> Path:
    frame = load_diabetes(as_frame=True, scaled=False).frame
    frame = frame.rename(columns=DIABETES_COLUMNS)
    frame["sex"] = frame["sex"].astype(int)  # a category coded 1 or 2, not a measurement
    path = DATA_DIR / "diabetes.csv"
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (make_iris(), make_diabetes()):
        print(f"wrote {path.relative_to(DATA_DIR.parents[1]).as_posix()}")
