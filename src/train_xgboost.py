# train_xgboost.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

# ====================================================================
# CONFIGURACION Y PARAMETROS
# ====================================================================

# Rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sel_all.csv"  # ¡Asegurate de que este archivo exista!
MODEL_OUTPUT_PATH = BASE_DIR / "xgboost_best_pipeline.pkl"

# Constantes de Entrenamiento
TARGET_COLUMN = 'Risk_Level'
RANDOM_STATE = 42

# Hiperparametros Optimizados (del JSON proporcionado)
BEST_PARAMS = {
    "n_estimators": 7976,
    "learning_rate": 0.48132578560291905,
    "max_depth": 1,
    "min_child_weight": 1.1068147985302066,
    "gamma": 0.02637638445351531,
    "subsample": 0.9996125283562703,
    "colsample_bytree": 0.7148937314170194,
    "reg_lambda": 0.007894121197231548,
    "reg_alpha": 2.031530326241857
}

# ====================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ====================================================================

def inferir_features(data, target_col):
    """Identifica las columnas numericas y categoricas."""
    all_features = [col for col in data.columns if col != target_col]
    num_features = data[all_features].select_dtypes(include=np.number).columns.tolist()
    cat_features = data[all_features].select_dtypes(include=['object', 'category']).columns.tolist()
    return num_features, cat_features

def crear_preprocesador(num_features, cat_features):
    """Crea el ColumnTransformer para preprocesar los datos (Normalizacion y OHE)."""
    
    # 1. Pipeline para características numéricas: Imputación por mediana y Normalización (StandardScaler)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # <-- AQUI SE APLICA LA NORMALIZACION
    ])

    # 2. Pipeline para características categóricas: Imputación por moda y One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop'
    )
    return preprocessor

# ====================================================================
# EJECUCION PRINCIPAL
# ====================================================================

def main_train():
    print("Iniciando entrenamiento de XGBoost con Normalización...")
    
    # 1. Carga de Datos y Preparación
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"✅ Datos cargados: {len(data)} filas.")
    except FileNotFoundError:
        print(f"❌ Error: Archivo de datos no encontrado en {DATA_PATH}.")
        return

    # Inferencia de features
    num_features, cat_features = inferir_features(data, TARGET_COLUMN)
    all_features = num_features + cat_features
    
    X = data[all_features]
    y = data[TARGET_COLUMN]
    
    # Codificar la etiqueta (Target)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.dropna()) 
    X_processed = X.loc[y.dropna().index]

    # 2. Creación y Entrenamiento de la Pipeline Completa
    preprocessor = crear_preprocesador(num_features, cat_features)
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('classifier', xgb.XGBClassifier(
            **BEST_PARAMS,
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=RANDOM_STATE
        ))
    ])
    
    print("🚀 Entrenando Pipeline (Preprocesador + Modelo)...")
    # Al entrenar la pipeline, el preprocesador se ajusta a X_processed, y el scaler (normalizador) 
    # aprenderá la media y desviación estándar de los datos.
    full_pipeline.fit(X_processed, y_encoded)
    print("✅ Entrenamiento completado.")
    
    # 3. Guardar la Pipeline (Contiene el preprocesador ajustado y el modelo entrenado)
    try:
        joblib.dump(full_pipeline, MODEL_OUTPUT_PATH)
        print(f"\n✨ Pipeline completa guardada en: {MODEL_OUTPUT_PATH.name}")
    except Exception as e:
        print(f"❌ Error al guardar la pipeline: {e}")

if __name__ == '__main__':
    main_train()