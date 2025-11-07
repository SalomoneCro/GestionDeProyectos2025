from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import xgboost as xgb
from catboost import CatBoostClassifier

import optuna

xgb.set_config(verbosity=0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    preprocessor_kind: str  # 'ohe_scale' | 'ohe_only' | 'cat_native'
    build_estimator: Callable[[Dict[str, Any]], Any]
    suggest_params: Callable[[optuna.trial.Trial], Dict[str, Any]]


def build_preprocessor(X: pd.DataFrame, kind: str) -> ColumnTransformer:
    """Build a ColumnTransformer for the requested preprocessing kind.

    - ohe_scale: imputer+scaler (num), imputer+OHE (cat)
    - ohe_only:  imputer (num),        imputer+OHE (cat)
    - cat_native: handled outside of sklearn pipeline (CatBoost); this
                  function should not be called with kind='cat_native'.
    """
    if kind == "cat_native":
        raise ValueError(
            "cat_native preprocessing is handled outside of sklearn pipeline"
        )

    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()

    if kind == "ohe_scale":
        num_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
    elif kind == "ohe_only":
        num_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
    else:
        raise ValueError(f"Unknown preprocessor kind: {kind}")

    return ColumnTransformer(
        [
            ("num", num_pipe, num),
            ("cat", cat_pipe, cat),
        ],
        remainder="drop",
    )


def build_pipeline(
    X: pd.DataFrame, spec: ModelSpec, params: Dict[str, Any]
) -> Pipeline:
    pre = build_preprocessor(X, spec.preprocessor_kind)
    est = spec.build_estimator(params)
    return Pipeline([("prep", pre), ("clf", est)])


# ========================
# Search spaces per model
# ========================


def sp_logreg(t: optuna.Trial) -> Dict[str, Any]:
    return {
        "C": t.suggest_float("logreg_C", 1e-3, 1e3, log=True),
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "n_jobs": 1,
        "class_weight": t.suggest_categorical(
            "logreg_class_weight", [None, "balanced"]
        ),
    }


def sp_logregcv(t: optuna.Trial) -> Dict[str, Any]:
    n_cs = t.suggest_int("logregcv_n_cs", 4, 8)
    Cs = np.logspace(-3, 3, n_cs)
    return {
        "Cs": Cs,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "cv": 3,
        "n_jobs": 1,
        "refit": True,
        "class_weight": t.suggest_categorical(
            "logregcv_class_weight", [None, "balanced"]
        ),
    }


def sp_rf(t: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("rf_n_estimators", 300, 900),
        "max_depth": t.suggest_int("rf_max_depth", 6, 24),
        "min_samples_split": t.suggest_int("rf_min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("rf_min_samples_leaf", 1, 10),
        "max_features": t.suggest_categorical(
            "rf_max_features",
            ["sqrt", "log2", 0.3, 0.5, 0.8],
        ),
        "bootstrap": True,
        "n_jobs": 1,
        "criterion": "gini",
        "class_weight": t.suggest_categorical("rf_class_weight", [None, "balanced"]),
    }


def sp_gb(t: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("gb_n_estimators", 150, 600),
        "learning_rate": t.suggest_float("gb_lr", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("gb_max_depth", 2, 5),
        "min_samples_leaf": t.suggest_int("gb_min_samples_leaf", 1, 20),
        "subsample": t.suggest_float("gb_subsample", 0.7, 1.0),
        "max_features": t.suggest_categorical(
            "gb_max_features",
            ["sqrt", "log2", 0.3, 0.5, 1.0],
        ),
    }


def sp_svc(t: optuna.Trial) -> Dict[str, Any]:
    kernel = t.suggest_categorical(
        "svc_kernel",
        ["rbf", "poly", "sigmoid"],
    )
    params = {
        "C": t.suggest_float("svc_C", 0.1, 1e4, log=True),
        "gamma": t.suggest_float("svc_gamma", 1e-5, 10.0, log=True),
        "kernel": kernel,
        "decision_function_shape": "ovr",
        "probability": False,
        "max_iter": -1,
    }
    if kernel == "poly":
        params["degree"] = t.suggest_int("svc_degree", 2, 5)
        params["coef0"] = t.suggest_float("svc_coef0", 0.0, 5.0)
    elif kernel == "sigmoid":
        params["coef0"] = t.suggest_float("svc_coef0", -5.0, 5.0)
    return params


def sp_linsvc(t: optuna.Trial) -> Dict[str, Any]:
    return {
        "C": t.suggest_float("linsvc_C", 1e-3, 1e4, log=True),
        "penalty": "l2",
        "loss": "squared_hinge",
        "dual": True,
        "max_iter": 5000,
        "class_weight": t.suggest_categorical(
            "linsvc_class_weight", [None, "balanced"]
        ),
    }


def sp_lda(t: optuna.Trial) -> Dict[str, Any]:
    # Avoid 'svd' due to instability (SVD did not converge) on high-dim OHE.
    return {
        "solver": "lsqr",
        "shrinkage": t.suggest_float("lda_shrinkage", 0.0, 1.0)
        if t.suggest_categorical("lda_use_shrinkage", [True, False])
        else None,
    }


def sp_xgb(t: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("xgb_n_estimators", 200, 600),
        "learning_rate": t.suggest_float("xgb_lr", 0.02, 0.3, log=True),
        "max_depth": t.suggest_int("xgb_max_depth", 3, 8),
        "min_child_weight": t.suggest_float("xgb_min_child_weight", 1.0, 10.0),
        "gamma": t.suggest_float("xgb_gamma", 0.0, 5.0),
        "subsample": t.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        # Regularization
        "reg_lambda": t.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": t.suggest_float(
            "xgb_reg_alpha", 0.0, 5.0
        ),  # linear; allows exact 0
        # Fixed stuff
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",  # outer CV optimizes recall_weighted
        "tree_method": "hist",
        "n_jobs": 1,
        "verbosity": 0,
        "random_state": 42,
    }


def sp_cat(t: optuna.Trial) -> Dict[str, Any]:
    return {
        "iterations": t.suggest_int("cat_iterations", 300, 900),
        "learning_rate": t.suggest_float("cat_lr", 0.02, 0.3, log=True),
        "depth": t.suggest_int("cat_depth", 4, 10),
        "l2_leaf_reg": t.suggest_float("cat_l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": t.suggest_float("cat_bagging_temp", 0.0, 5.0),
        "random_strength": t.suggest_float("cat_random_strength", 0.5, 5.0),
        "border_count": t.suggest_int("cat_border_count", 32, 255),
        "loss_function": "MultiClass",
        # We'll let outer CV handle recall_weighted; internal metric can be generic
        "eval_metric": "MultiClass",
        # Configurar overfitting detector a nivel de modelo (no en fit)
        "od_type": "Iter",
        "od_wait": 50,
        "thread_count": 1,
        "verbose": False,
        "task_type": "CPU",
        "allow_writing_files": False,
    }


MODELS: Dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        "logreg",
        "ohe_scale",
        lambda p: LogisticRegression(**p),
        sp_logreg,
    ),
    "logreg_cv": ModelSpec(
        "logreg_cv",
        "ohe_scale",
        lambda p: LogisticRegressionCV(**p),
        sp_logregcv,
    ),
    "rf": ModelSpec(
        "rf",
        "ohe_only",
        lambda p: RandomForestClassifier(**p),
        sp_rf,
    ),
    "gb": ModelSpec(
        "gb",
        "ohe_only",
        lambda p: GradientBoostingClassifier(**p),
        sp_gb,
    ),
    "svc": ModelSpec(
        "svc",
        "ohe_scale",
        lambda p: SVC(**p),
        sp_svc,
    ),
    "linear_svc": ModelSpec(
        "linear_svc",
        "ohe_scale",
        lambda p: LinearSVC(**p),
        sp_linsvc,
    ),
    "lda": ModelSpec(
        "lda",
        "ohe_scale",
        lambda p: LinearDiscriminantAnalysis(**p),
        sp_lda,
    ),
    "xgb": ModelSpec(
        "xgb",
        "ohe_only",
        lambda p: xgb.XGBClassifier(**p),
        sp_xgb,
    ),
    "catboost": ModelSpec(
        "catboost",
        "cat_native",
        lambda p: CatBoostClassifier(**p),
        sp_cat,
    ),
}
