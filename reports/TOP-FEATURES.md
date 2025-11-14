# Análisis de las Variables más Influyentes en el Nivel de Riesgo del Proyecto

## 1. Introducción

Con el objetivo de identificar los factores que más influyen en la predicción del **nivel de riesgo del proyecto (Risk_Level)**, se realizó un análisis de las variables incluidas en el dataset, evaluando su impacto mediante diversas técnicas.

El enfoque combinó métricas basadas en coeficientes, importancias nativas de modelos de árboles, *Permutation Importance* para modelos lineales y no lineales, y finalmente un **ranking consolidado** que sintetiza la importancia promedio entre todos los algoritmos. Esto permitió aislar las variables cuya influencia es consistente independientemente de la técnica empleada.

El presente informe detalla los resultados para las **10 variables más influyentes**, analizando su significado, su aporte al riesgo y su rol dentro de la dinámica de un proyecto.

## 2. Metodología

Para asegurar robustez en la selección de variables, se aplicó una estrategia multitécnica que incluyó:

- **Coefficient Importance** para modelos lineales.
- **Feature Importance** para modelos basados en árboles.
- **Permutation Importance** para modelos lineales, no lineales y redes neuronales.
- **Normalización por modelo** y construcción de un **ranking combinado** para obtener una medida agregada de importancia.

Este enfoque garantiza que las variables identificadas como más relevantes lo sean tanto desde perspectivas lineales, no lineales y jerárquicas, evitando sesgos propios de cada algoritmo.

## 3. Resultados

El ranking consolidado determinó que las 10 variables más influyentes en el riesgo del proyecto son:

1. **Complexity_Score**
2. **Technology_Familiarity**
3. **Key_Stakeholder_Availability**
4. **Team_Experience_Level**
5. **Stakeholder_Engagement_Level**
6. **Change_Control_Maturity**
7. **Risk_Management_Maturity**
8. **Org_Process_Maturity**
9. **Project_Manager_Experience**
10. **Requirement_Stability**

Estas variables representan dimensiones críticas del proyecto: técnica, humana, organizacional y de procesos, lo que demuestra que el riesgo es un fenómeno multidimensional.

## 4. Análisis de las Variables Más Influyentes

### 4.1 Complexity_Score  

**Tipo:** Numérica  
**Dimensión:** Técnica

Complexity_Score emerge como el predictor más determinante. Proyectos con alta complejidad presentan:

- más dependencias técnicas y funcionales,
- mayor esfuerzo de integración,
- propagación de riesgos entre componentes,
- múltiples puntos de fallo potenciales,
- mayor probabilidad de desvíos en alcance, cronograma y presupuesto.

La complejidad funciona como un **amplificador estructural del riesgo**, aumentando la sensibilidad del proyecto ante cualquier perturbación.

### 4.2 Technology_Familiarity  

**Tipo:** Categórica  
**Dimensión:** Técnica

- **Expert:** reduce significativamente el riesgo.  
- **New:** incrementa riesgo técnico-operativo.

La familiaridad del equipo con la tecnología define la curva de aprendizaje, la velocidad de desarrollo, la calidad técnica y la probabilidad de re-trabajo. En entornos con tecnología desconocida, se incrementa la incertidumbre y la tasa de errores, aumentando el riesgo general.

### 4.3 Key_Stakeholder_Availability  

**Tipo:** Categórica  
**Dimensión:** Gobernanza

La disponibilidad de stakeholders clave es un impulsor crítico de riesgo social y estratégico.

- **Poor:** ralentiza la toma de decisiones, dificulta aclaraciones, aumenta el re-trabajo y genera desalineaciones.  
- **Excellent:** mitiga riesgos y agiliza la ejecución.

La ausencia de stakeholders en momentos clave suele ser un factor desencadenante de desviaciones importantes.

### 4.4 Team_Experience_Level  

**Tipo:** Categórica  
**Dimensión:** Recursos Humanos

La experiencia del equipo determina su capacidad para resolver problemas, anticipar riesgos y mantener un rendimiento estable.

- **Expert:** menor tasa de defectos, mayor autonomía, mejor estimación y ejecución más predecible.  
- **Junior:** aumentan la variabilidad y el riesgo operativo.

Es un indicador directo de la madurez técnica del proyecto.

### 4.5 Stakeholder_Engagement_Level  

**Tipo:** Categórica  
**Dimensión:** Stakeholder Management

El engagement refleja no sólo presencia, sino compromiso activo. Un bajo nivel de participación genera:

- falta de alineación en expectativas,
- cambios tardíos,
- entregables ambiguos,
- menor soporte para decisiones críticas.

Un alto engagement actúa como mitigador del riesgo organizacional.

### 4.6 Change_Control_Maturity  

**Tipo:** Categórica  
**Dimensión:** Procesos Organizacionales

La madurez del proceso de gestión de cambios es fundamental para evitar distorsiones del alcance.

- **Advanced:** el impacto de los cambios se controla y se documenta adecuadamente.  
- **None:** los cambios se introducen caóticamente, generando incertidumbre y re-trabajo.

La ausencia de un proceso formal de control de cambios es un predictor muy poderoso de riesgo elevado.

### 4.7 Risk_Management_Maturity  

**Tipo:** Categórica  
**Dimensión:** Madurez Organizacional

Organizaciones con alta madurez en gestión de riesgos poseen:

- identificación temprana de amenazas,  
- mitigación efectiva,  
- monitoreo continuo.

La falta de madurez expone al proyecto a riesgos no anticipados y sin tratamiento, elevando significativamente el nivel global de riesgo.

### 4.8 Org_Process_Maturity  

**Tipo:** Categórica  
**Dimensión:** Madurez Organizacional

Esta variable refleja la solidez metodológica de la organización.  
Procesos “Optimizing” permiten una ejecución coordinada y predecible; procesos “Ad-hoc” generan inconsistencias, falta de estandarización y mayor variabilidad operativa.

Es un factor sistemático que condiciona la estabilidad del proyecto.

### 4.9 Project_Manager_Experience  

**Tipo:** Categórica  
**Dimensión:** Recursos Humanos

La experiencia del Project Manager influye directamente en:

- gestión del alcance,  
- conducción de equipos,  
- calidad de la comunicación,  
- manejo de stakeholders,  
- mitigación de riesgos,  
- priorización y toma de decisiones.

PMs senior o certificados gestionan mejor la incertidumbre y reducen las probabilidades de desviaciones críticas.

### 4.10 Requirement_Stability  

**Tipo:** Categórica  
**Dimensión:** Gestión del Alcance

La estabilidad de los requisitos es un factor crítico para la predictibilidad.

- **Stable:** menor probabilidad de re-trabajo y estimaciones más precisas.  
- **Volatile:** mayor incertidumbre, cambios continuos en el alcance y desalineación con los objetivos del proyecto.

Requerimientos volátiles son uno de los componentes más directos de riesgo operativo.

## 5. Conclusiones

El análisis revela que el riesgo de un proyecto se explica por una combinación de factores **técnicos, organizacionales, humanos y de procesos**. Las variables de mayor influencia coinciden con áreas tradicionalmente críticas en la literatura de gestión de proyectos: complejidad, claridad del alcance, madurez institucional, experiencia del equipo y participación activa de stakeholders.

Los factores que tienden a **incrementar el riesgo** incluyen:

- alta complejidad,
- baja familiaridad tecnológica,
- falta de disponibilidad o compromiso de stakeholders,
- volatilidad en los requisitos,
- baja madurez organizacional,
- equipos o PMs poco experimentados.

Los factores que **mitigan el riesgo** incluyen:

- experiencia técnica y de gestión,
- procesos maduros,
- claridad y estabilidad del alcance,
- fuerte apoyo y participación de stakeholders.

## 6. Recomendaciones

1. **Evaluar la complejidad tempranamente** y ajustar la planificación acorde a su nivel.
2. **Capacitar al equipo cuando se trabajen tecnologías nuevas** o incorporar expertos externos.
3. **Establecer acuerdos formales de disponibilidad con stakeholders clave**.
4. **Asignar Project Managers experimentados a proyectos de alta criticidad**.
5. **Fortalecer procesos organizacionales**, en especial:  
   - control de cambios,  
   - gestión de riesgos,  
   - estandarización operativa.
6. **Asegurar estabilidad y claridad en los requisitos** antes de avanzar a fases críticas del proyecto.
