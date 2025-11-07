from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy import sparse as sp
from numpy.linalg import LinAlgError
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from cross_validation.cross_validation import CrossValidation
from tuning.model_specs import MODELS, ModelSpec, build_preprocessor


def _build_cat_native_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Preprocesado ligero para CatBoost: imputación simple sin OHE.

    - Numéricas: median.
    - Categóricas: fillna("missing") manteniendo strings/categorías.
    """
    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    # Numericas
    med = X_tr[num_cols].median()
    X_tr_num = X_tr[num_cols].fillna(med)
    X_val_num = X_val[num_cols].fillna(med)

    # Categóricas
    X_tr_cat = X_tr[cat_cols].copy()
    X_val_cat = X_val[cat_cols].copy()
    for c in cat_cols:
        X_tr_cat[c] = X_tr_cat[c].astype("object").fillna("missing")
        X_val_cat[c] = X_val_cat[c].astype("object").fillna("missing")

    # Reconstruimos respetando el orden original de columnas
    X_tr_proc = pd.concat([X_tr_num, X_tr_cat], axis=1)[X_tr.columns]
    X_val_proc = pd.concat([X_val_num, X_val_cat], axis=1)[X_val.columns]

    return X_tr_proc, y_tr, X_val_proc, y_val


def build_folds_cache(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor_kind: str,
    n_splits: int = 3,
    random_state: int = 42,
    force_dense: bool = False,
) -> Sequence[Tuple[Any, Any, Any, Any]]:
    """Construye y cachea folds preprocesados para un modelo dado.

    Devuelve una lista de tuplas (X_tr_proc, y_tr, X_val_proc, y_val).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: List[Tuple[Any, Any, Any, Any]] = []

    if preprocessor_kind == "cat_native":
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            folds.append(_build_cat_native_fold(X_tr, y_tr, X_val, y_val))
        return folds

    # ColumnTransformer pipelines (ohe_scale / ohe_only)
    pre = build_preprocessor(X, preprocessor_kind)
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pre_fit = pre.fit(X_tr)
        X_tr_proc = pre_fit.transform(X_tr)
        X_val_proc = pre_fit.transform(X_val)

        if force_dense:
            if sp.issparse(X_tr_proc):
                X_tr_proc = X_tr_proc.toarray()
            if sp.issparse(X_val_proc):
                X_val_proc = X_val_proc.toarray()

        folds.append((X_tr_proc, y_tr.values, X_val_proc, y_val.values))
    return folds


def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: str,
    n_splits: int = 3,
    random_state: int = 42,
    scoring: str = "recall_weighted",
):
    """Crea el objetivo de Optuna para un modelo usando CrossValidation con folds cacheados."""
    spec: ModelSpec = MODELS[model_key]
    # Aseguramos y codificada a enteros 0..K-1 para compatibilidad (e.g., XGBoost)
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y.values), index=y.index, name=y.name)

    folds = build_folds_cache(
        X,
        y_enc,
        spec.preprocessor_kind,
        n_splits=n_splits,
        random_state=random_state,
        force_dense=(model_key == "lda"),
    )

    # Para evaluate_precomputed no usamos X/y internos; n_jobs=1 para evitar over-subscription
    cv = CrossValidation(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
        n_jobs=1,
        scoring=[scoring],
    )

    # fit params por fold para modelos con early stopping
    cat_cols_idx: List[int] = []
    if spec.preprocessor_kind == "cat_native":
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols_idx = [X.columns.get_loc(c) for c in cat_cols]

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest_params(trial)
        est = spec.build_estimator(params)

        fit_params_per_fold: List[Dict[str, Any]] | None = None

        if model_key == "xgb":
            fit_params_per_fold = []
            for X_tr_p, y_tr_p, X_val_p, y_val_p in folds:
                fit_params_per_fold.append(
                    {
                        "eval_set": [(X_val_p, y_val_p)],
                    }
                )

        elif model_key == "catboost":
            fit_params_per_fold = []
            for X_tr_p, y_tr_p, X_val_p, y_val_p in folds:
                fit_params_per_fold.append(
                    {
                        "eval_set": (X_val_p, y_val_p),
                        "use_best_model": True,
                        "cat_features": cat_cols_idx,
                    }
                )

        res = cv.evaluate_precomputed(
            model=est,
            folds=folds,
            scoring=scoring,
            fit_params_per_fold=fit_params_per_fold,
        )

        trial.set_user_attr("recall_std", res.summary[scoring]["std"])
        return res.summary[scoring]["mean"]

    return objective
