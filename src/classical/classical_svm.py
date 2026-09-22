""" Trains and evaluates the classical support vector machine baseline using scikit-learn. """
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def fit_predict_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    degree: int = 3,
    class_weight: str | dict | None = None,
) -> np.ndarray:
    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, class_weight=class_weight)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
