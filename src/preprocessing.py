# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    mutual_info_classif,
    SelectFromModel,
    VarianceThreshold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# %%
plt.style.use("ggplot")

# %% [markdown]
# ## **Data Cleaning**

# %%
file_path = os.path.join(
    kagglehub.dataset_download("ka66ledata/project-management-risk-raw"),
    "project_risk_raw_dataset.csv",
)

# %%
df = pd.read_csv(file_path, index_col=0)
df.head()

# %%
# duplicated
print("duplicados:", df.duplicated().sum())

# missing
missing = pd.concat([df.isnull().sum(), (df.isnull().mean() * 100)], axis=1)
missing.columns = ["missing_count", "missing_%"]
display(missing.sort_values("missing_count", ascending=False).head(30))

# %%
# target
display(
    pd.concat(
        [
            df["Risk_Level"].value_counts(dropna=False),
            df["Risk_Level"].value_counts(normalize=True, dropna=False).round(2),
        ],
        axis=1,
    )
)

df["Risk_Level"] = (
    df["Risk_Level"]
    .replace({"Low": 0, "Medium": 1, "High": 2, "Critical": 3})
    .astype("int")
)

# %%
# eliminar columnas con más de 50% de datos faltantes
thresh_pct = 50
cols_drop = missing[missing["missing_%"] > thresh_pct].index.tolist()
print("columns to DROP (>50% missing):", cols_drop)
df = df.drop(columns=cols_drop)

# agrupar categorías raras (frecuencia < 1%)
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols: num_cols.remove("Risk_Level")

for c in cat_cols:
    freq = df[c].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    if len(rare) > 0:
        df[c] = df[c].replace(list(rare), "OTHERS")

# %% [markdown]
# ## **Data Processing**

# %%
# features, target
X = df.drop(columns=["Risk_Level"])
y = df["Risk_Level"].copy()

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# preprocesamiento
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


def get_feature_names_from_ct(ct):
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
            ohe = trans.named_steps["onehot"]
            ohe_cols = list(ohe.get_feature_names_out(cols))
            names.extend(ohe_cols)
        else:
            names.extend(list(cols))
    return names


feature_names = get_feature_names_from_ct(preprocessor)

print("feature matrix shape (train):", X_train_prep.shape)

# %% [markdown]
# ## **Principal Component Analysis (PCA)**

# %%
# PCA (numerical features)
pca_num = PCA(random_state=42).fit(num_pipeline.fit_transform(X_train[num_cols]))
expl_num = pca_num.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(expl_num) + 1),
    expl_num.cumsum(),
    color="steelblue",
    alpha=0.5,
    marker="o",
)
plt.title("PCA (numerical features)")
plt.xlabel("n components")
plt.ylabel("cumulative explained")
plt.grid(True)
plt.show()

# %%
# PCA (preprocessed data)
pca_full = PCA(random_state=42).fit(X_train_prep)
expl_full = pca_full.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(expl_full) + 1),
    expl_full.cumsum(),
    color="steelblue",
    alpha=0.5,
    marker="o",
)
plt.title("PCA (numerical features)")
plt.xlabel("n components")
plt.ylabel("cumulative explained")
plt.grid(True)
plt.show()

# %% [markdown]
# ## **Feature Selection**

# %%
rf = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train_prep, y_train)

importances = pd.Series(rf.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(6, 10))
sns.barplot(x=importances.head(30).values, y=importances.head(30).index, color="tomato")
plt.title("feature importances")
plt.xlabel("importance")
plt.ylabel("feature")
plt.show()

# %%
N = 30

selected_features = importances.head(N).index.tolist()
selected_idx = [feature_names.index(f) for f in selected_features]

X_train_sel = X_train_prep[:, selected_idx]
X_test_sel = X_test_prep[:, selected_idx]

# %% [markdown]
# ## **Save Data**

# %%
# ===== PROCESSING =====
df_train_prep = pd.DataFrame(X_train_prep, columns=feature_names)
df_train_prep["Risk_Level"] = y_train.reset_index(drop=True)
df_train_prep["dataset_split"] = "train"  # opcional, para identificar luego

df_test_prep = pd.DataFrame(X_test_prep, columns=feature_names)
df_test_prep["Risk_Level"] = y_test.reset_index(drop=True)

# Combinar ambos
df_prep_all = pd.concat([df_train_prep, df_test_prep], ignore_index=True)

# Guardar dataset combinado
df_prep_all.to_csv("prep_all.csv", index=False)


# ===== FEATURE SELECTION =====
df_train_sel = pd.DataFrame(X_train_sel, columns=selected_features)
df_train_sel["Risk_Level"] = y_train.reset_index(drop=True)

df_test_sel = pd.DataFrame(X_test_sel, columns=selected_features)
df_test_sel["Risk_Level"] = y_test.reset_index(drop=True)

# Combinar ambos
df_sel_all = pd.concat([df_train_sel, df_test_sel], ignore_index=True)

# Guardar dataset combinado
df_sel_all.to_csv("sel_all.csv", index=False)
