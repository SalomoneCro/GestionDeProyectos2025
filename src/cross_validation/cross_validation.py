from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_validate


ScoringType = Union[str, Sequence[str]]


@dataclass
class CrossValidationResult:
    """
    Resultado de la validación cruzada.

    Attributes
    ----------
    n_splits : int
        Número de folds utilizados en la validación cruzada.
    scoring : list[str]
        Lista de nombres de métricas utilizadas.
    raw_scores : dict[str, np.ndarray]
        Diccionario con los scores por fold. Las claves siguen el
        formato de `cross_validate`, por ejemplo: "test_accuracy".
    summary : dict[str, dict[str, float]]
        Resumen con media y desviación estándar por métrica.
        Ejemplo:
        {
            "accuracy": {"mean": 0.85, "std": 0.03},
            "f1_macro": {"mean": 0.82, "std": 0.04},
            ...
        }
    """

    n_splits: int
    scoring: List[str]
    raw_scores: Dict[str, np.ndarray]
    summary: Dict[str, Dict[str, float]]


class CrossValidation:
    """
    Clase utilitaria para realizar validación cruzada estratificada
    en tareas de clasificación multiclase.

    Esta clase asume que:
    - El dataset ya está preprocesado.
    - La tarea es de clasificación multiclase (posiblemente con > 5 clases).
    - El clasificador respeta la API de scikit-learn (fit / predict / score).

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Features ya preprocesados.
    y : pd.Series | np.ndarray
        Vector de etiquetas de clase.
    n_splits : int, default=5
        Número de folds para StratifiedKFold.
    shuffle : bool, default=True
        Si barajar las muestras antes de hacer los splits.
    random_state : int | None, default=None
        Semilla para la aleatoriedad de StratifiedKFold cuando shuffle=True.
    scoring : ScoringType | None, default=None
        Métricas de evaluación a usar. Si es None, se usan:
        ("accuracy", "f1_macro", "precision_macro", "recall_macro").
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        scoring: Optional[ScoringType] = None,
    ) -> None:
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        if scoring is None:
            scoring = (
                "accuracy",
                "f1_macro",
                "precision_macro",
                "recall_macro",
            )

        # Normalizamos scoring a lista de strings internamente
        if isinstance(scoring, str):
            self.scoring: List[str] = [scoring]
        else:
            self.scoring = list(scoring)

    # ============================
    # Métodos públicos
    # ============================

    def evaluate(self, model: ClassifierMixin) -> CrossValidationResult:
        """
        Ejecuta la validación cruzada para el modelo dado.

        Parameters
        ----------
        model : ClassifierMixin
            Clasificador a evaluar. Puede ser, por ejemplo:
            - CatBoostClassifier
            - XGBClassifier
            - LogisticRegression / LogisticRegressionCV
            - SVC / LinearSVC
            - LinearDiscriminantAnalysis
            - GradientBoostingClassifier
            - RandomForestClassifier

        Returns
        -------
        CrossValidationResult
            Objeto con los scores por fold y un resumen (media / std)
            de cada métrica.
        """
        cv = self._build_cv_splitter()

        cv_results = cross_validate(
            estimator=model,
            X=self.X,
            y=self.y,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        summary = self._build_summary(cv_results)

        return CrossValidationResult(
            n_splits=self.n_splits,
            scoring=self.scoring,
            raw_scores={
                key: value
                for key, value in cv_results.items()
                if key.startswith("test_")
            },
            summary=summary,
        )

    # ============================
    # Métodos privados
    # ============================

    def _build_cv_splitter(self) -> StratifiedKFold:
        """
        Construye el objeto StratifiedKFold para clasificación multiclase.

        Returns
        -------
        StratifiedKFold
            Objeto configurado con n_splits, shuffle y random_state.
        """
        return StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

    def _build_summary(
        self,
        cv_results: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """
        Construye un resumen (media y desviación estándar) para cada métrica.

        Parameters
        ----------
        cv_results : dict[str, Any]
            Diccionario retornado por `sklearn.model_selection.cross_validate`.

        Returns
        -------
        dict[str, dict[str, float]]
            Diccionario con media y std por métrica.
        """
        summary: Dict[str, Dict[str, float]] = {}

        for metric_name in self.scoring:
            key = f"test_{metric_name}"
            scores = cv_results.get(key)

            if scores is None:
                # Por si alguna métrica no está disponible
                continue

            scores = np.asarray(scores, dtype=float)
            summary[metric_name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std(ddof=1)),
            }

        return summary
