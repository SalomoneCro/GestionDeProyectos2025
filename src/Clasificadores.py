# clasificador_integral.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path

# Modelos y utilidades de Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
# Clasificadores
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from jinja2 import Environment, FileSystemLoader

# ====================================================================
# 1. CONFIGURACION INICIAL
# ====================================================================

# Rutas y Constantes
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sel_all.csv"  # ¡Asegurate de que este archivo exista en la misma carpeta!
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Configuracion de Entrenamiento
RANDOM_STATE = 42
TEST_SIZE = 0.30
TARGET_COLUMN = 'Risk_Level' # Columna de riesgo (Basado en tus scripts)

# ====================================================================
# 2. FUNCIONES DE PREPROCESAMIENTO Y UTILIDAD
# ====================================================================

def inferir_features(data, target_col):
    """Identifica las columnas numericas y categoricas."""
    all_features = [col for col in data.columns if col != target_col]
    
    # Intentar identificar por tipo de dato de Pandas
    num_features = data[all_features].select_dtypes(include=np.number).columns.tolist()
    cat_features = data[all_features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Quitar cualquier columna que se haya colado en ambas listas (ej. IDs numericos que deberian ser categoricos)
    # Por defecto, si una columna es numerica, la tratamos como tal, a menos que se fuerce a 'object' o 'category'.

    print(f"✅ Features numéricas inferidas: {num_features}")
    print(f"✅ Features categóricas inferidas: {cat_features}")
    
    return num_features, cat_features

def crear_preprocesador(num_features, cat_features):
    """Crea el ColumnTransformer para preprocesar los datos (Scaling y OHE)."""
    
    # 1. Pipeline para características numéricas (Imputacion por mediana y Estandarización)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline para características categóricas (Imputacion por moda y One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Combinar ambos transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop' # Ignora cualquier otra columna
    )
    return preprocessor

# ====================================================================
# 3. FUNCIONES DE EVALUACION Y REPORTE
# ====================================================================

def evaluar_modelo(model_name, model_pipeline, X_test, y_test, classes_sorted, fit_time=0):
    """Genera metricas, grafica la matriz de confusion y devuelve el diccionario de resultados."""
    
    # Para modelos Scikit-learn, medimos solo el tiempo de predicción aquí
    if fit_time == 0:
        t0 = time.time()
        y_pred = model_pipeline.predict(X_test)
        tf = time.time()
        fit_time = tf - t0 # Usamos el tiempo de prediccion si no se paso el tiempo de entrenamiento
    else:
        y_pred = model_pipeline.predict(X_test)

    # 1. Probabilidades (necesarias para AUC/AP)
    y_proba = None
    if hasattr(model_pipeline, 'predict_proba'):
        y_proba = model_pipeline.predict_proba(X_test)

    # 2. Metricas
    acc = accuracy_score(y_test, y_pred)
    p_ma = precision_score(y_test, y_pred, average="macro", zero_division=0)
    r_ma = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_ma = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    auc_macro, ap_macro = None, None
    if y_proba is not None:
        try:
            # AUC-ROC OvR (One-vs-Rest)
            auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            # Convertir y_test a formato binario para average_precision_score
            y_test_bin = pd.get_dummies(y_test).reindex(columns=classes_sorted, fill_value=0)
            ap_macro = average_precision_score(y_test_bin, y_proba, average="macro")
        except ValueError:
             print(f"⚠️ No se pudo calcular AUC/AP para {model_name}.")

    # 3. Reporte de Clasificacion
    cls_rep = classification_report(y_test, y_pred, output_dict=True)
    
    # 4. Generar Matriz de Confusión y guardar imagen
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=classes_sorted, yticklabels=classes_sorted)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Riesgo Real')
    plt.xlabel('Riesgo Previsto')
    img_path = REPORTS_DIR / f'cm_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(img_path)
    plt.close()

    return {
        'name': model_name,
        'accuracy': acc,
        'precision_macro': p_ma,
        'recall_macro': r_ma,
        'f1_macro': f1_ma,
        'auc_macro': auc_macro,
        'ap_macro': ap_macro,
        'reporte_cls': cls_rep,
        'fit_time': fit_time,
        'img_path': str(img_path)
    }

def generar_informe_html(resultados, data_path):
    """Genera el informe HTML final con todos los resultados."""
    
    # Plantilla HTML (para no depender de un archivo externo)
    TEMPLATE_HTML = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Informe Comparativo de Clasificadores de Riesgo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }
            .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #3498db; color: white; }
            .metric-value { font-size: 1.2em; font-weight: bold; color: #2980b9; }
            .model-section { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
            .report-pre { background: #f9f9f9; padding: 10px; border: 1px solid #eee; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏆 Informe Comparativo de Clasificadores de Riesgo</h1>
            <p><strong>Fecha:</strong> {{ fecha_generacion }} | <strong>Archivo de Datos:</strong> {{ data_path }}</p>

            <h2>Tabla Resumen de Métricas</h2>
            <table>
                <thead>
                    <tr>
                        <th>Modelo</th>
                        <th>Accuracy</th>
                        <th>F1-Score (Macro)</th>
                        <th>Recall (Macro)</th>
                        <th>ROC-AUC (Macro)</th>
                        <th>Tiempo (s)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for res in resultados %}
                    <tr>
                        <td><strong>{{ res.name }}</strong></td>
                        <td class="metric-value">{{ '%.4f' % res.accuracy }}</td>
                        <td class="metric-value">{{ '%.4f' % res.f1_macro }}</td>
                        <td class="metric-value">{{ '%.4f' % res.recall_macro }}</td>
                        <td>{{ '%.4f' % (res.auc_macro or 0) }}</td>
                        <td>{{ '%.1f' % res.fit_time }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <hr>

            <h2>Detalle de Modelos y Visualizaciones</h2>
            {% for res in resultados %}
            <div class="model-section">
                <h3>{{ res.name }}</h3>
                
                <h4>Matriz de Confusión</h4>
                <p><img src="reports/{{ res.img_path.split('/')[-1] }}" alt="Matriz de Confusión para {{ res.name }}" style="width: 400px;"></p>

                <h4>Reporte de Clasificación Detallado (Clases, Precision, Recall, F1)</h4>
                <pre class="report-pre">{{ res.reporte_cls | tojson(indent=2) | safe }}</pre>

            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    
    # Configuramos Jinja2 para cargar desde string (la plantilla anterior)
    env = Environment(loader=FileSystemLoader('.'))
    template = env.from_string(TEMPLATE_HTML)
    
    html_output = template.render(
        fecha_generacion=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        data_path=str(data_path),
        resultados=resultados
    )
    
    output_file = 'informe_clasificadores_riesgo.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"\n✅ Informe HTML generado: {output_file}")


# ====================================================================
# 4. EJECUCION PRINCIPAL
# ====================================================================

def main():
    print("Iniciando Proceso de Entrenamiento y Reporte Integral...")
    
    # 1. Carga de Datos
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"✅ Datos cargados: {len(data)} filas.")
    except FileNotFoundError:
        print(f"❌ Error: Archivo de datos no encontrado en {DATA_PATH}. Revise la ruta.")
        return
    except KeyError as e:
        print(f"❌ Error: La columna TARGET '{TARGET_COLUMN}' no se encontró en los datos.")
        return
        
    # 2. Inferencia de Features
    num_features, cat_features = inferir_features(data, TARGET_COLUMN)
    all_features = num_features + cat_features
    
    # 3. Separación de Datos y Codificación de Target
    X = data[all_features]
    y = data[TARGET_COLUMN]
    
    le = LabelEncoder()
    # Aseguramos que solo las filas donde el target no es nulo se usen
    mask = y.notna()
    X_masked = X[mask]
    y_masked = y[mask]

    # Codificar la etiqueta si es texto (necesario para la mayoría de los modelos)
    y_encoded = le.fit_transform(y_masked)
    classes_sorted = sorted(le.classes_) # Para las etiquetas en la matriz de confusion
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_masked, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # 4. Definir Preprocesador
    preprocessor = crear_preprocesador(num_features, cat_features)
    
    # 5. Definición de Pipelines/Modelos
    pipelines = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                solver='saga', max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE
            ))
        ]),
        
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
            ))
        ]),

        'MLP Classifier (Tuned)': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(128, 64), activation='relu', solver='adam', 
                max_iter=500, random_state=RANDOM_STATE
            ))
        ]),
        
        # XGBoost (Necesita OHE, por eso se usa el preprocessor completo)
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE
            )) 
        ]),
        
        # CatBoost (Se entrena por separado, ya que prefiere la columna categórica sin OHE)
        'CatBoost': CatBoostClassifier(
            verbose=0, random_state=RANDOM_STATE, learning_rate = 0.05, 
            l2_leaf_reg = 1, iterations=800, depth = 6, border_count = 64
        ),
    }
    
    # 6. Entrenamiento y Evaluación
    resultados = []
    
    for name, model_or_pipeline in pipelines.items():
        print(f"\n⚙️ Entrenando y evaluando: {name}...")
        
        t0 = time.time()
        
        if name == 'CatBoost':
            # CatBoost usa el DF original y las columnas categóricas
            model_or_pipeline.fit(
                X_train, y_train, 
                cat_features=cat_features,
                verbose=0
            )
            tf = time.time()
            fit_time = tf - t0
            
            # Wrapper para usar la función de evaluación general
            class CatBoostWrapper:
                def __init__(self, model): self.model = model
                def predict(self, X): return self.model.predict(X)
                def predict_proba(self, X): return self.model.predict_proba(X)
            
            cb_wrapper = CatBoostWrapper(model_or_pipeline)
            res = evaluar_modelo(name, cb_wrapper, X_test, y_test, classes_sorted, fit_time=fit_time)
            
        else:
            # Modelos basados en Scikit-learn (usan el Pipeline con preprocessor)
            model_or_pipeline.fit(X_train, y_train)
            tf = time.time()
            fit_time = tf - t0
            
            res = evaluar_modelo(name, model_or_pipeline, X_test, y_test, classes_sorted, fit_time=fit_time)
        
        resultados.append(res)
        
    # 7. Generar Informe Final
    generar_informe_html(resultados, DATA_PATH)

if __name__ == '__main__':
    main()