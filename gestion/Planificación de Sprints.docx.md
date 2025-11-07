# SPRINT 1

*Alcance: 2.1 Obtención del dataset · 2.2 Limpieza y preparación · 2.3 EDA*

*Fuera de alcance: introducción/selección de modelos, entrenamiento, métricas y prototipos.*

# **Sprint**

**Objetivo: Completar 2.1, 2.2 y 2.3 dentro del mismo sprint (dos semanas).**

## **Épicas del Sprint**

* E-DS1: Obtención del dataset (2.1)  
* E-DS2: Limpieza y preparación (2.2)  
* E-DS3: EDA (2.3)

### **E-DS1.HIST-1 | Acceso, mapeo de fuentes y extracción inicial**

**Criterios de aceptación**

* Fuentes/rutas definidas (origen → destino) con tabla de mapeo.  
* Extracción reproducible con script/CLI y parámetros.

**Definition of Done (DoD)**

* Script de extracción versionado con README de uso.  
* Ejecución valida que el volumen y periodo esperado fue descargado.

### **E-DS1.HIST-2 | Normalización a formato intermedio**

**Criterios de aceptación**

* Alineación de nombres (snake\_case) y fechas ISO-8601.  
* Conversión de tipos base (datetime/float/int/str) unificada.  
* Salida en /data/interim.

**Definition of Done (DoD)**

* etl\_interim.py con parámetros (rango de fechas, filtros).  
* Resumen de columnas presentes/ausentes en /reports/etl\_interim\_summary.md.

### **E-DS2.HIST-1 | Limpieza básica y estandarización**

**Criterios de aceptación**

* Definición de claves de duplicados y reglas de merge.  
* Rutinas de deduplicación y parsing/normalización de fechas y unidades.  
* Estandarización de tipos (float/int/enums).  
* Tratamiento inicial de nulos (imputación simple o drop justificado).

**Definition of Done (DoD)**

* Artefacto /data/processed/train\_ready.parquet generado.  
* README con estrategias y justificaciones de limpieza.

### **E-DS2.HIST-2 | Dataset final y documentación mínima**

**Criterios de aceptación**

* Diccionario de datos (nombre, descripción, tipo, valores permitidos).  
* Listado de supuestos y decisiones de limpieza.  
* Registro de ruta y checksum del dataset final.

**Definition of Done (DoD)**

* /docs/dataset\_readme.md actualizado y versionado.  
* /data/processed/train\_ready.parquet con fecha y checksum.

### **E-DS3.HIST-1 | EDA típico (lectura, merge, tipos, limpieza y análisis)**

**Criterios de aceptación**

* Lectura de archivos y verificación de esquemas.  
* Merge/union de datos provenientes de archivos diferentes (join keys definidas).  
* Formateo correcto del tipo de dato de cada columna (datetime/numérico/categórico).  
* Corrección de errores de datos (valores atípicos obvios, strings malformados, fechas inválidas).  
* Análisis general de columnas: media, desviación estándar, min, max, conteos y plots básicos (hist, boxplot).  
* Estrategia de tratamiento de NA (drop o imputación simple) aplicada y documentada.  
* Análisis de PCA (preprocesado: escalado/estandarización) para inspección exploratoria.  
* Reporte entregable en /reports/EDA\_sprint.pdf (o .html).

**Definition of Done (DoD)**

* Notebook/script reproducible con pasos claros (lectura→merge→dtypes→limpieza→EDA→PCA).  
* Sección de limitaciones y próximos pasos de datos (no de modelos).

## **(Tentativo) Tareas del Sprint (Jira-ready)**

| ID | Resumen | Epic/Historia | Owner | Est. (h) | Dependencias |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **E-DS1.T-101** | Inventario de fuentes y estructura (tabla de mapeo) | E-DS1.HIST-1 | PM/Analista | 1 | — |
| **E-DS1.T-103** | Script de extracción con parámetros (rango, filtros) | E-DS1.HIST-1 | DE | 5 | T-101 |
| **E-DS1.T-105** | Transformación a formato intermedio (nombres/fechas/tipos) | E-DS1.HIST-2 | DE | 5 | T-103 |
| **E-DS2.T-201** | Definir clave(s) de duplicados y reglas de merge | E-DS2.HIST-1 | DS/DE | 2 | T-105 |
| **E-DS2.T-202** | Deduplicación y parsing/normalización de fechas | E-DS2.HIST-1 | DE | 4 | T-201 |
| **E-DS2.T-203** | Estandarizar unidades y tipos (float/int/enum) | E-DS2.HIST-1 | DS | 4 | T-202 |
| **E-DS2.T-204** | Tratamiento inicial de nulos (drop/imputación) \+ README | E-DS2.HIST-1 | DS | 3 | T-203 |
| **E-DS2.T-205** | Export a /data/processed/train\_ready.parquet | E-DS2.HIST-1 | DE | 2 | T-204 |
| **E-DS2.T-301** | Generar diccionario de datos (auto \+ edición) | E-DS2.HIST-2 | DS/Analista | 4 | T-205 |
| **E-DS2.T-302** | Registrar ruta y checksum del dataset final | E-DS2.HIST-2 | DE | 2 | T-205 |
| **E-DS3.T-401** | Leer archivos y validar esquemas | E-DS3.HIST-1 | DS | 2 | T-302 |
| **E-DS3.T-402** | Merge/union de datasets (join keys) | E-DS3.HIST-1 | DS/DE | 4 | T-401 |
| **E-DS3.T-403** | Formateo de tipos de datos por columna | E-DS3.HIST-1 | DS | 3 | T-402 |
| **E-DS3.T-404** | Corrección de errores de datos (fechas/strings/outliers obvios) | E-DS3.HIST-1 | DS | 3 | T-403 |
| **E-DS3.T-405** | Análisis general (stats básicas y plots) | E-DS3.HIST-1 | DS | 4 | T-404 |
| **E-DS3.T-406** | Borrado/Imputación de NA según criterio documentado | E-DS3.HIST-1 | DS | 2 | T-405 |
| **E-DS3.T-407** | Preprocesado (escalado) y PCA exploratorio | E-DS3.HIST-1 | DS | 4 | T-406 |
| **E-DS3.T-408** | Compilar reporte EDA (PDF/HTML) y revisión final | E-DS3.HIST-1 | DS/PM | 3 | T-407 |

# **Estructura en Jira**

* EPIC: E-DS1 Obtención del dataset (2.1) — HIST-1, HIST-2  
* EPIC: E-DS2 Limpieza y preparación (2.2) — HIST-1, HIST-2  
* EPIC: E-DS3 EDA (2.3) — HIST-1

# **Hitos (Releases/Milestones)**

* H2 — Dataset procesado listo (cierre del sprint).  
* H3 — Informe EDA entregado (cierre del sprint).

# **Tablero y Definiciones**

*Swimlanes: 2.1 Obtención · 2.2 Limpieza · 2.3 EDA · Bloqueados · Revisión*

**Ready (Listo para Sprint):**

* Fuentes identificadas; estructura de carpetas creada (data/raw, data/interim, data/processed, reports/).  
* Plantillas de README y notebooks listas; herramientas instaladas.

**Done (Hecho):**

* Scripts y artefactos versionados; outputs generados en rutas acordadas.  
* Documentación mínima actualizada (dataset\_readme.md, etl\_interim\_summary.md, EDA\_sprint.pdf/html).  
* Revisión por par aplicada.

# Tareas a asignar con puntos de historia

## 

| ID | Resumen | Epic/Historia | Owner | Puntos de historia | Dependencias | Persona |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **E-DS1.T-101** | Inventario de fuentes y estructura (tabla de mapeo) | E-DS1.HIST-1 | PM/Analista | 1 | — | Diego |
| **E-DS1.T-102** | Script de extracción con parámetros (rango, filtros). Transformación a formato intermedio (nombres/fechas/tipos) | E-DS1.HIST-1 E-DS1.HIST-2 | DE | 2 | T-101 |  |
|  |  |  |  |  |  |  |
| **E-DS2.T-201** | Definir clave(s) de duplicados, reglas de merge y de datos nulos. | E-DS2.HIST-1 | DS/DE | 1 | T-102 |  |
| **E-DS2.T-202**  | Deduplicación y parsing/normalización de fechas Estandarizar unidades y tipos (float/int/enum) Tratamiento inicial de nulos (drop/imputación). Merge/union de datasets (join keys) | E-DS2.HIST-1 E-DS2.HIST-1 E-DS2.HIST-1 | DE DS  | 3  | T-201  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
| **E-DS2.T-301** | Generar diccionario de datos (auto \+ edición) | E-DS2.HIST-2 | DS/Analista | 2 | T-202 | Emi |
| **E-DS2.T-302** | Registrar ruta y checksum del dataset final.  Export a /data/processed/train\_ready.parquet | E-DS2.HIST-2 | DE | 0 | T-202 |  |
| **E-DS3.T-401** | Leer archivos y validar esquemas | E-DS3.HIST-1 | DS | 0 | T-302 |  |
| **E-DS3.T-402** | Análisis general (stats básicas y plots) | E-DS3.HIST-1 | DS | 13 | T-401 | Yesica |
| **E-DS3.T-403** | Preprocesado (escalado) , PCA exploratorio y selección de características relevantes. | E-DS3.HIST-1 | DS | 8 | T-402 | Facundo |
| **E-DS3.T-404** | Compilar reporte EDA (PDF/HTML) y revisión final | E-DS3.HIST-1 | DS/PM | 5 | T-403 |  |

README????

**E-DS1: Obtención del dataset (2.1)**

Historia de usuario:

Como científico de datos, quiero obtener y consolidar el dataset necesario desde las fuentes definidas para poder comenzar el análisis y desarrollo del modelo.

**E-DS2: Limpieza y preparación (2.2)**

Historia de usuario:

Como científico de datos, quiero limpiar y preparar el dataset para asegurar que los datos estén listos y en un formato adecuado para el análisis posterior.

**E-DS3: EDA (2.3)**

Historia de usuario:

Como analista de datos, quiero realizar un análisis exploratorio de los datos (EDA) para identificar patrones, tendencias y relaciones que orienten la construcción del modelo o las decisiones del proyecto.

# SPRINT 2

*Alcance: 3.1 Selección de algoritmos de Machine Learning· 3.2 Entrenamiento y validación de modelos· 3.3 Optimización de modelos seleccionados 3.4 Pruebas del sistema*

*Fuera de alcance: Fase de Evaluación y Entrega Final..*

# **Sprint**

**Objetivo: Completar 3.1, 3.2, 3.3 y 3.4 dentro del mismo sprint (dos semanas).**

## **Épicas del Sprint**

* E1: Selección de algoritmos de Machine Learning (3.1)  
* E2: Entrenamiento y validación de modelos (3.2)  
* E3: Optimización de modelos seleccionados (3.3)

### **E1: Selección de algoritmos de Machine Learning**

**Historia de usuario:**

Como científico de datos, quiero poder seleccionar los algoritmos de Machine Learning más adecuados según el tipo de problema y los datos disponibles, para garantizar que el modelo tenga el mejor desempeño posible desde las primeras etapas del desarrollo.

**Criterios de aceptación:**

* Se debe poder comparar diferentes tipos de algoritmos.

* Se documentan las razones de la selección de cada algoritmo.

### **E2: Entrenamiento y validación de modelos**

**Historia de usuario**

Como científico de datos, quiero entrenar y validar los modelos seleccionados utilizando los conjuntos de datos de entrenamiento y validación, para evaluar su desempeño y validar la generalización del modelo a nuevos datos.

**Criterios de aceptación:**

* El proceso de entrenamiento debe ser reproducible y automatizable.

* Se deben aplicar técnicas de validación (por ejemplo, cross-validation o hold-out).

* Los resultados de la validación deben incluir métricas relevantes.

* Los resultados deben registrarse para su comparación posterior.

### **E3: Optimización de modelos seleccionados**

**Historia de usuario**

Como científico de datos, quiero optimizar los hiperparámetros de los modelos seleccionados, para mejorar su rendimiento y obtener la mejor versión posible.

**Criterios de aceptación:**

* Se debe poder realizar búsqueda de hiperparámetros.

* Se deben comparar los resultados antes y después de la optimización.

* El modelo optimizado debe guardar tanto los parámetros finales como las métricas obtenidas.

* La optimización debe integrarse con el flujo de entrenamiento existente.

### 

# Tareas a asignar con puntos de historia

| ID | Resumen | Epic / Historia | Owner | Puntos de historia | Dependencias | Persona |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **3.1.1.1** | Elegir métricas de evaluación (ROC-AUC, F1, Recall, Precision, etc.) | Fase 3.1 — Selección de algoritmos de ML | PO \+ DS | 2 | Ninguna |  |
| **3.1.2** | Crear división de datos (train/test, stratify/time-split) | Fase 3.2 — Entrenamiento y validación | DS | 1 | Datasets listos |  |
| **3.1.3.1** | Implementar modelo baseline (Logistic Regression) | Fase 3.1 — Selección de algoritmos de ML | DS / MLE | 3 | 3.1.2 |  |
| **3.1.3.2** | Implementar modelo Random Forest | Fase 3.1 — Selección de algoritmos de ML | DS / MLE | 3 | 3.1.2 |  |
| **3.1.3.3** | Implementar modelo XGBoost / LightGBM | Fase 3.1 — Selección de algoritmos de ML | DS / MLE | 3 | 3.1.2 |  |
| **3.1.3.4** | Implementar modelo adicional (SVM o NN pequeña) | Fase 3.1 — Selección de algoritmos de ML | DS / MLE | 3 | 3.1.2 |  |
| **3.2.1** | Ejecutar validación cruzada (K-fold / Time-series CV) | Fase 3.2 — Entrenamiento y validación | DS / MLE | 3 | 3.1.3.x, 3.1.2 |  |
| **3.2.2** | Comparar resultados entre modelos (tabla y visualizaciones ROC/PR) | Fase 3.2 — Entrenamiento y validación | DS | 5 | 3.2.1 |  |
| **3.3.1** | Ajuste de hiperparámetros y validación de resultados | Fase 3.3 — Optimización de modelos | MLE \+ DS | 8 | 3.2.2 |  |
| **3.3.3** | Documentar resultados finales, métricas y decisiones | Fase 3.3 — Optimización de modelos | DS (rev. TL/PO) | 1 | 3.3.1 |  |

