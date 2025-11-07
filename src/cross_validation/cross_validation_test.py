from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from cross_validation import CrossValidation, CrossValidationResult


@pytest.fixture(scope="module")
def toy_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Crea un dataset de juguete multiclase para usar en los tests.

    Returns
    -------
    X : pd.DataFrame
        Features simulados.
    y : pd.Series
        Etiquetas de clase.
    """
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        n_classes=5,
        n_clusters_per_class=1,
        class_sep=1.2,
        flip_y=0.01,
        random_state=123,
    )

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_sr = pd.Series(y, name="target")

    return X_df, y_sr


@pytest.fixture()
def cv_default(toy_dataset: Tuple[pd.DataFrame, pd.Series]) -> CrossValidation:
    """
    Devuelve una instancia de CrossValidation con parámetros por defecto.
    """
    X, y = toy_dataset
    return CrossValidation(
        X=X,
        y=y,
        n_splits=5,
        shuffle=True,
        random_state=42,
    )


# =============================================================================
# Tests sobre CrossValidationResult
# =============================================================================


def test_cross_validation_result_dataclass_basic() -> None:
    """
    Verifica que CrossValidationResult almacena los campos básicos correctamente.
    """
    result = CrossValidationResult(
        n_splits=5,
        scoring=["accuracy", "f1_macro"],
        raw_scores={
            "test_accuracy": np.array([0.8, 0.85, 0.83, 0.82, 0.81]),
            "test_f1_macro": np.array([0.79, 0.84, 0.82, 0.80, 0.81]),
        },
        summary={
            "accuracy": {"mean": 0.82, "std": 0.02},
            "f1_macro": {"mean": 0.81, "std": 0.02},
        },
    )

    assert result.n_splits == 5
    assert result.scoring == ["accuracy", "f1_macro"]
    assert "test_accuracy" in result.raw_scores
    assert "accuracy" in result.summary
    assert "mean" in result.summary["accuracy"]
    assert "std" in result.summary["accuracy"]


# =============================================================================
# Tests sobre CrossValidation con scoring por defecto
# =============================================================================


def test_cross_validation_default_scoring_metrics_present(
    cv_default: CrossValidation,
) -> None:
    """
    Verifica que con scoring por defecto se calculan las 4 métricas:
    accuracy, precision_macro, recall_macro, f1_macro.
    """
    model = LogisticRegression(max_iter=500)
    result = cv_default.evaluate(model)

    expected_metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    # Están en summary
    for metric in expected_metrics:
        assert metric in result.summary, f"{metric} no está en summary"

        mean = result.summary[metric]["mean"]
        std = result.summary[metric]["std"]

        # Métricas en rango razonable
        assert 0.0 <= mean <= 1.0
        assert std >= 0.0

    # Están en raw_scores con el prefijo test_
    for metric in expected_metrics:
        key = f"test_{metric}"
        assert key in result.raw_scores
        scores = result.raw_scores[key]
        # Debe haber un score por fold
        assert scores.shape[0] == result.n_splits


def test_cross_validation_default_n_splits_length(
    cv_default: CrossValidation,
) -> None:
    """
    Verifica que la cantidad de scores por métrica coincide con n_splits.
    """
    model = LogisticRegression(max_iter=500)
    result = cv_default.evaluate(model)

    for key, values in result.raw_scores.items():
        # Cada entry en raw_scores debe tener tantos valores como folds
        assert len(values) == result.n_splits, (
            f"{key} tiene {len(values)} valores, "
            f"pero n_splits={result.n_splits}"
        )


# =============================================================================
# Tests sobre scoring personalizado
# =============================================================================


def test_cross_validation_custom_scoring_string(
    toy_dataset: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Verifica que si se pasa un string como scoring, la clase
    lo normaliza a lista con un solo elemento.
    """
    X, y = toy_dataset
    cv = CrossValidation(
        X=X,
        y=y,
        n_splits=3,
        shuffle=True,
        random_state=0,
        scoring="accuracy",
    )

    model = LogisticRegression(max_iter=500)
    result = cv.evaluate(model)

    # Solo se espera la métrica accuracy
    assert result.scoring == ["accuracy"]
    assert "accuracy" in result.summary
    assert "test_accuracy" in result.raw_scores
    assert len(result.raw_scores) == 1  # solo test_accuracy


def test_cross_validation_custom_scoring_list(
    toy_dataset: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Verifica que si se pasa una lista de métricas, la clase las usa tal cual.
    """
    X, y = toy_dataset
    metrics = ["accuracy", "f1_macro"]
    cv = CrossValidation(
        X=X,
        y=y,
        n_splits=4,
        shuffle=True,
        random_state=0,
        scoring=metrics,
    )

    model = LogisticRegression(max_iter=500)
    result = cv.evaluate(model)

    assert set(result.scoring) == set(metrics)
    for metric in metrics:
        assert metric in result.summary
        assert f"test_{metric}" in result.raw_scores


# =============================================================================
# Tests de reproducibilidad y varios estimadores
# =============================================================================


def test_cross_validation_reproducible_with_same_random_state(
    toy_dataset: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Verifica que, con el mismo random_state y mismo modelo,
    los resultados de cross validation sean reproducibles.
    """
    X, y = toy_dataset

    cv1 = CrossValidation(
        X=X,
        y=y,
        n_splits=5,
        shuffle=True,
        random_state=123,
    )

    cv2 = CrossValidation(
        X=X,
        y=y,
        n_splits=5,
        shuffle=True,
        random_state=123,
    )

    model = LogisticRegression(max_iter=500)

    result1 = cv1.evaluate(model)
    result2 = cv2.evaluate(model)

    # Comparamos al menos una métrica clave (accuracy)
    acc1 = result1.raw_scores["test_accuracy"]
    acc2 = result2.raw_scores["test_accuracy"]

    assert np.allclose(acc1, acc2), (
        "Los scores de accuracy no son reproducibles con el mismo random_state"
    )


@pytest.mark.parametrize(
    "estimator_cls",
    [
        LogisticRegression,
        RandomForestClassifier,
        SVC,
    ],
)
def test_cross_validation_works_with_multiple_estimators(
    cv_default: CrossValidation,
    estimator_cls,
) -> None:
    """
    Verifica que CrossValidation funciona con varios tipos de estimadores
    que respetan la API de clasificador de scikit-learn.
    """
    if estimator_cls is LogisticRegression:
        model = estimator_cls(max_iter=500)
    elif estimator_cls is RandomForestClassifier:
        model = estimator_cls(n_estimators=20, random_state=0)
    else:  # SVC
        model = estimator_cls(kernel="rbf", gamma="scale")

    result = cv_default.evaluate(model)

    # Debe haber al menos accuracy
    assert "accuracy" in result.summary
    assert "test_accuracy" in result.raw_scores
    assert len(result.raw_scores["test_accuracy"]) == result.n_splits
