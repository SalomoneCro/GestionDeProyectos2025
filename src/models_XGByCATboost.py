# %%
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# %%
# Ruta al dataset
DATA_PATH = Path("data/processed/project_risk_clean.csv")


# Leer el dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("Risk_Level", axis=1)
y = df["Risk_Level"]


# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# %%
# Preprocesamiento
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

num_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop"
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# %% [markdown]
# # **XGBoost**

# %%
model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X_train_prep, y_train_encoded)

y_pred = model.predict(X_test_prep)
y_pred_labels = encoder.inverse_transform(y_pred)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred_labels))
print("Recall:", recall_score(y_test, y_pred_labels, average="weighted"))
print("Precision:", precision_score(y_test, y_pred_labels, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_labels, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels))


# Matriz de confusión
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_labels), annot=True, fmt="d", cmap="RdPu")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# %%
# Busca de hiperparámetros
param_grid = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "eval_metric": ["logloss", "mlogloss"],
}

grid = RandomizedSearchCV(
    xgb.XGBClassifier(eval_metric="logloss"),
    param_distributions=param_grid,
    n_iter=50,
    scoring="accuracy",
    cv=3,
    verbose=3,
    random_state=42,
)
grid.fit(X_train_prep, y_train_encoded)

print(
    "Mejores parámetros:", grid.best_params_
)  # {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1} recall 0.6416
# {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.1, 'eval_metric': 'mlogloss'} recall 0.6366
#  {'subsample': 1.0, 'n_estimators': 800, 'max_depth': 3, 'learning_rate': 0.1, 'eval_metric': 'logloss'} recall 0.655

# %%
model_xgb_best = xgb.XGBClassifier(
    eval_metric="mlogloss",
    subsample=1,
    n_estimators=800,
    max_depth=3,
    learning_rate=0.3,
)
model_xgb_best.fit(X_train_prep, y_train_encoded)

y_pred = model_xgb_best.predict(X_test_prep)
y_pred_labels = encoder.inverse_transform(y_pred)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred_labels))
print("Recall:", recall_score(y_test, y_pred_labels, average="weighted"))
print("Precision:", precision_score(y_test, y_pred_labels, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_labels, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels))

# Matriz de confusión
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_labels), annot=True, fmt="d", cmap="RdPu")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# %% [markdown]
# # **CatBoost**

# %%
X_train[cat_cols] = X_train[cat_cols].fillna("missing")
X_test[cat_cols] = X_test[cat_cols].fillna("missing")

model_cat = CatBoostClassifier(
    loss_function="MultiClass", eval_metric="MultiClass", random_seed=42, verbose=False
)
model_cat.fit(X_train, y_train, cat_features=cat_cols)

y_pred = model_cat.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))  # recall 0.7283
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="RdPu")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# %%
model_cat_base = CatBoostClassifier(
    loss_function="MultiClass", eval_metric="MultiClass", random_seed=42, verbose=False
)

param_grid = {
    "iterations": [300, 500, 800],
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg": [1, 3, 5],
    "border_count": [32, 64, 128],
}


grid = RandomizedSearchCV(
    model_cat_base,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring="recall_weighted",
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

# Entrenamiento con categóricas y pesos
grid.fit(X_train, y_train, cat_features=cat_cols)

# Evaluación final
best_model = grid.best_estimator_  # {'learning_rate': 0.05, 'l2_leaf_reg': 1, 'iterations': 800, 'depth': 6, 'border_count': 64}
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))  # recall 0.7283
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="RdPu")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# %%
model_cat_best = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    verbose=False,
    learning_rate=0.05,
    l2_leaf_reg=1,
    iterations=800,
    depth=6,
    border_count=64,
)

model_cat_best.fit(X_train, y_train, cat_features=cat_cols)

y_pred = model_cat_best.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))  # recall 0.7283
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="RdPu")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()
