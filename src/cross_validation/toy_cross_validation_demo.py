"""
Ejemplo de uso de la clase CrossValidation con varios clasificadores:

- CatBoostClassifier
- XGBClassifier
- LogisticRegression / LogisticRegressionCV
- SVC / LinearSVC
- LinearDiscriminantAnalysis
- GradientBoostingClassifier
- RandomForestClassifier

Usa un dataset de juguete multiclase (7 clases) para que el cómputo sea bajo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC, SVC
from cross_validation import CrossValidation, CrossValidationResult


# =====================================================================
# Dataset de juguete y modelos
# =====================================================================


def build_toy_dataset(
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye un dataset de juguete multiclase (7 clases) para pruebas.

    Parameters
    ----------
    random_state : int, default=42
        Semilla para reproducibilidad.

    Returns
    -------
    X : pd.DataFrame
        Features simulados.
    y : pd.Series
        Etiquetas de clase.
    """
    X, y = make_classification(
        n_samples=350,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=7,
        n_clusters_per_class=1,
        class_sep=1.5,
        flip_y=0.01,
        random_state=random_state,
    )

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_sr = pd.Series(y, name="target")

    return X_df, y_sr


def build_models(random_state: int = 42) -> Dict[str, ClassifierMixin]:
    """
    Construye todos los modelos a evaluar sobre el dataset de juguete.

    Parameters
    ----------
    random_state : int, default=42
        Semilla para los modelos que lo soporten.

    Returns
    -------
    dict[str, ClassifierMixin]
        Diccionario nombre -> modelo.
    """
    models: Dict[str, ClassifierMixin] = {}

    # Modelos lineales
    models["logistic_regression"] = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=random_state,
    )

    models["logistic_regression_cv"] = LogisticRegressionCV(
        Cs=5,
        cv=3,
        solver="lbfgs",
        max_iter=500,
        n_jobs=-1,
        random_state=random_state,
    )

    # SVM
    models["svc_rbf"] = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,
        random_state=random_state,
    )

    models["linear_svc"] = LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=random_state,
    )

    # LDA
    models["lda"] = LinearDiscriminantAnalysis()

    # Tree-based / ensemble
    models["gradient_boosting"] = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state,
    )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

    # XGBoost
    models["xgb_classifier"] = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    # CatBoost
    models["catboost_classifier"] = CatBoostClassifier(
        iterations=50,
        depth=4,
        learning_rate=0.1,
        loss_function="MultiClass",
        random_state=random_state,
        verbose=False,
    )

    return models


# =====================================================================
# Utilidad para imprimir resultados
# =====================================================================


def print_summary(model_name: str, result: CrossValidationResult) -> None:
    """
    Imprime un resumen legible de las métricas de validación cruzada.

    Parameters
    ----------
    model_name : str
        Nombre del modelo.
    result : CrossValidationResult
        Resultado de la validación cruzada.
    """
    print("=" * 70)
    print(f"Modelo: {model_name}")
    print(f"Folds:  {result.n_splits}")
    print("-" * 70)

    # Métricas principales que esperamos tener
    metrics_to_show = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    for metric in metrics_to_show:
        stats = result.summary.get(metric)
        if stats is None:
            continue
        mean = stats["mean"]
        std = stats["std"]
        print(f"{metric:15s}: {mean:.4f} ± {std:.4f}")


# =====================================================================
# main
# =====================================================================


def main() -> None:
    """
    Ejecuta el ejemplo de validación cruzada sobre todos los modelos.
    """
    random_state = 42

    # 1) Dataset de juguete
    X, y = build_toy_dataset(random_state=random_state)

    # 2) Objeto de validación cruzada
    cv = CrossValidation(
        X=X,
        y=y,
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # 3) Modelos a evaluar
    models = build_models(random_state=random_state)

    # 4) Loop de evaluación
    for name, model in models.items():
        result = cv.evaluate(model)
        print_summary(name, result)


if __name__ == "__main__":
    main()
