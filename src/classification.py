# %% [markdown]
# # ***Modelos de Clasificación***

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# %%
from google.colab import drive
drive.mount('/content/drive')
! ls drive/MyDrive/Colab\ Notebooks/Gestión\ de\ Proyectos/proyecto_final

# %%
risk_dataset = pd.read_csv('drive/MyDrive/Colab Notebooks/Gestión de Proyectos/proyecto_final/project_risk_raw_dataset.csv')
risk_dataset.head()

# %% [markdown]
# + Quitamos variable categórica irrelevante (puedo usar los índices del DataFrame)
#

# %%
risk_dataset.drop(columns = ['Project_ID'], inplace = True)

# %% [markdown]
# + Convertimos la variable target a numérica, respetando el orden lógico de las categorías
#

# %%
risk_dataset["Risk_Level"] = risk_dataset["Risk_Level"].replace({
    "Critical": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3
}).astype("int")

risk_dataset.Risk_Level.value_counts()

# %% [markdown]
# + Separación *train-test*

# %%
X = risk_dataset.drop('Risk_Level', axis=1)
y = risk_dataset['Risk_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# + Transformaciones

# %%
cat_cols = risk_dataset.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = risk_dataset.select_dtypes(include=[np.number]).columns.tolist()
num_cols: num_cols.remove("Risk_Level")

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
], remainder="drop")

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)


# # Get feature names from the preprocessor
# num_features = num_cols
# cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)

# all_features = list(num_features) + list(cat_features)

# # Convert back to DataFrame
# X_train_df = pd.DataFrame(X_train_prep, columns=all_features, index=X_train.index)
# X_test_df  = pd.DataFrame(X_test_prep,  columns=all_features, index=X_test.index)


# %% [markdown]
# + Aplicamos PCA conservando las componentes ortogonales que acumulan el 100% de la varianza

# %%
pca = PCA(n_components=95)
X_train_pca = pca.fit_transform(X_train_prep)
X_test_pca = pca.transform(X_test_prep)

# %% [markdown]
# ## *Log-Regression básica*

# %%
# Entrenamiento con Logistic Regression simple
clf_log = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000, random_state=42)
clf_log.fit(X_train_pca, y_train)

# Predicciones con la muestra de test
y_pred_log = clf_log.predict(X_test_pca)

# Accuracy y Recall en el set de test
acc_test_log = clf_log.score(X_test_pca, y_test)
recall_test_log = recall_score(y_test, y_pred_log, average='weighted')

# Cuadro de métricas
label_names = ["Critical", "Low", "Medium", "High"]
print(classification_report(y_test, y_pred_log, target_names=label_names))

# Matriz de confusión
fig = plt.figure(figsize=(20, 20))
cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot()
plt.xticks(rotation=45)
plt.show()

print(f"\nAccuracy en el set de test: {acc_test_log:.6f}")
print(f"\nRecall en el set de test: {recall_test_log:.6f}")

# %% [markdown]
# ## *Log- Regression CV*

# %%
clf_log = LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs')
clf_log.fit(X_train_pca, y_train)

# Predicciones con la muestra de test
y_pred_log = clf_log.predict(X_test_pca)


# Accuracy y Recall en el set de test
acc_test_log = clf_log.score(X_test_pca, y_test)
recall_test_log = recall_score(y_test, y_pred_log, average='weighted')

# Cuadro de métricas
label_names = ["Critical", "Low", "Medium", "High"]
print(classification_report(y_test, y_pred_log, target_names=label_names))

# Matriz de confusión
fig = plt.figure(figsize=(20, 20))
cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot()
plt.xticks(rotation=45)
plt.show()

print(f"\nAccuracy en el set de test: {acc_test_log:.6f}")
print(f"\nRecall en el set de test: {recall_test_log:.6f}")

# %% [markdown]
# ## *SVM (Kernel RBF)*

# %%
clf_rbf = SVC(C=100.0, kernel='rbf', gamma=0.001, random_state=42)
clf_rbf.fit(X_train_pca, y_train)

y_pred_rbf = clf_rbf.predict(X_test_pca)

# Accuracy y Recall en el set de test
acc_test_rbf = clf_rbf.score(X_test_pca, y_test)
recall_test_rbf = recall_score(y_test, y_pred_rbf, average='weighted')

# Cuadro de métricas
label_names = ["Critical", "Low", "Medium", "High"]
print(classification_report(y_test, y_pred_rbf, target_names=label_names))

# Matriz de confusión
plt.figure(figsize=(20, 20))
cm = confusion_matrix(y_test, y_pred_rbf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot()
plt.xticks(rotation=45)
plt.show()


print(f"\nAccuracy en el set de test: {acc_test_rbf:.6f}")
print(f"\nRecall en el set de test: {recall_test_rbf:.6f}")

# %% [markdown]
# ## *LinearSVC*

# %%
# Entrenamiento con LinearSVC
clf_linear = LinearSVC(C=10.0, random_state=42, max_iter=10000)
clf_linear.fit(X_train_pca, y_train)

# Predicciones
y_pred_linear = clf_linear.predict(X_test_pca)

# Accuracy y Recall en el set de test
acc_test_linear = clf_linear.score(X_test_pca, y_test)
recall_test_linear = recall_score(y_test, y_pred_linear, average='weighted')

# Cuadro de métricas
label_names = ["Critical", "Low", "Medium", "High"]
print(classification_report(y_test, y_pred_linear, target_names=label_names))

# Matriz de confusión
plt.figure(figsize=(20, 20))
cm = confusion_matrix(y_test, y_pred_linear)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot()
plt.xticks(rotation=45)
plt.show()

print(f"\nAccuracy en el set de test: {acc_test_linear:.6f}")
print(f"\nRecall en el set de test: {recall_test_linear:.6f}")

# %% [markdown]
# ## *LDA*

# %%
# Entrenamiento con LDA
clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(X_train_pca, y_train)

# Predicciones con la muestra de test
y_pred_lda = clf_lda.predict(X_test_pca)

# Accuracy y Recall en el set de test
acc_test_lda = clf_lda.score(X_test_pca, y_test)
recall_test_lda = recall_score(y_test, y_pred_lda, average='weighted')

# Cuadro de métricas
label_names = ["Critical", "Low", "Medium", "High"]
print(classification_report(y_test, y_pred_lda, target_names=label_names))

# Matriz de confusión
fig = plt.figure(figsize=(20, 20))
cm = confusion_matrix(y_test, y_pred_lda)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot()
plt.xticks(rotation=45)
plt.show()

print(f"\nAccuracy en el set de test: {acc_test_lda:.6f}")
print(f"\nRecall en el set de test: {recall_test_lda:.6f}")

# %% [markdown]
# ## *Gradient Boosting*

# %%
# Entrenamiento con Gradient Boosting
clf_gb = GradientBoostingClassifier(random_state=42)
clf_gb.fit(X_train_pca, y_train)

# Predicciones con la muestra de test
y_pred_gb = clf_gb.predict(X_test_pca)

# Accuracy y Recall en el set de test
acc_test_gb = clf_gb.score(X_test_pca, y_test)
recall_test_gb = recall_score(y_test, y_pred_gb, average='weighted')

# Cuadro de métricas
label_names = ["Critical", "Low", "Medium", "High"]
print(classification_report(y_test, y_pred_gb, target_names=label_names))

# Matriz de confusión
fig = plt.figure(figsize=(20, 20))
cm = confusion_matrix(y_test, y_pred_gb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot()
plt.xticks(rotation=45)
plt.show()

print(f"\nAccuracy en el set de test: {acc_test_gb:.6f}")
print(f"\nRecall en el set de test: {recall_test_gb:.6f}")
