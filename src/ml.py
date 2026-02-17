# src/ml.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class MLResult:
    scores: Dict[str, float]
    best_name: str
    best_estimator: Any  # <- pipeline (scaler + model)


def build_models(task: str) -> Dict[str, Any]:
    """
    task: 'classification' o 'regression'
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    if task == "classification":
        return {
            "LogReg": LogisticRegression(max_iter=2000),
            "SVM": SVC(probability=True),
            "RandomForest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "GradBoost": GradientBoostingClassifier(),
        }
    else:
        return {
            "Ridge": Ridge(),
            "SVR": SVR(),
            "RandomForest": RandomForestRegressor(),
            "KNN": KNeighborsRegressor(),
            "GradBoost": GradientBoostingRegressor(),
        }


def evaluate_models(X: np.ndarray, y: np.ndarray, task: str, cv: int = 5) -> MLResult:
    """
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = build_models(task)
    scores = {}
    pipes = {}

    scoring = "accuracy" if task == "classification" else "r2"

    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipes[name] = pipe
        s = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        scores[name] = float(np.mean(s))

    best_name = max(
        scores, key=scores.get
    )  # pyright: ignore[reportArgumentType, reportCallIssue]
    best_estimator = pipes[best_name]  # el pipeline ganador aún sin entrenar "final"
    return MLResult(scores=scores, best_name=best_name, best_estimator=best_estimator)


# src/ml.py
import re
from typing import List


def generate_lr_labels(
    ch_names: List[str],
    mode: str = "binary",
) -> List[str]:
    """
    Genera etiquetas izquierda/derecha (binario)
    o izquierda/centro/derecha (three)
    según convención EEG:
      - impar -> izquierda
      - par -> derecha
      - 'z' -> centro

    mode: 'binary' | 'three'
    """
    labels = []

    for ch in ch_names:
        s = ch.strip().replace(".", "")
        s_low = s.lower()

        # centro
        if s_low.endswith("z"):
            labels.append("0" if mode == "binary" else "2")
            continue

        # número final
        m = re.search(r"(\d+)$", s)
        if not m:
            labels.append("0" if mode == "binary" else "2")
            continue

        n = int(m.group(1))
        is_left = n % 2 == 1

        if mode == "binary":
            labels.append("0" if is_left else "1")
        else:
            labels.append("0" if is_left else "1")

    return labels
