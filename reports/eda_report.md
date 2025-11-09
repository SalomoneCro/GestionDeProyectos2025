# ***Reporte de Análisis Exploratorio del dataset `Project Management Risk Raw`***

<br>

## ***Variables numéricas***

De la matriz de correlación podemos observar que las variables `Team_Size`, `Project_Budget_USD`, `Estimated_Timeline_Months` `Complexity_Score`, `Stakeholder_Count` poseen un coeficiente de correlación $|r| \geq 0.6$, lo cual implica que exhiben información similar del dataset. La existencia de redundancia puede dañar la capacidad de interpretación de modelos de clasificación e inflar la complejidad del análisis innecesariamente. Aplicar PCA sería la opción adecuada en este caso para elegir un representante de este cluster. 

<br>

Por otro lado, las variables `Schedule_Pressure`, `Technical_Debt_Level`, `Change_Request_Frequency`, `Organizational_Change_Frequency` y `Comunication_Frequency` poseen una alta cantidad de valores anómalos en los boxplots realizados. Es recomendable analizar en detalle a qué filas corresponden dichos valores y analizarlos en su contexto. En caso de que representen datos espurios, considerar eliminarlos del dataset, catalogándolos como *outliers*.

<br>
<br>

## ***Variables Categóricas***
La variable target `Risk_Level` se encuentra medianamente balanceada respecto a sus clases. 

Las características `Project_Phase`, `Regulatory_Compliance_Level`, `Stakeholder_Engagement_Level`y `Priority_Level` poseen una varianza casi nula de acuerdo a sus histogramas. Esto implica que su contribución al nivel de riesgo del proyecto no es tan crucial respecto de otras, y pueden considerarse irrelevantes. Lo mismo podemos decir de `Project_ID` que es un mero identificador.


<br>
<br>

## ***Interpretaciones e insights***

1. `Vendor_Reliability_Score` es el principal predictor de riesgo. Los proyectos con menor fiabilidad del proveedor se concentran en clases de `Risk_Level` más altas. Esto sugiere una fuerte influencia de los proveedores externos en el riesgo del proyecto.

2. `Estimated_Timeline_Months` es el segundo predictor más importante: los proyectos más largos son más riesgosos, probablemente debido a una mayor exposición a ampliación del alcance, dependencias y cambios en los recursos.

3. `Resource_Availability` es un factor clave: una menor disponibilidad corresponde a un mayor riesgo, consistente con la idea de que las restricciones de recursos aumentan la probabilidad de fracaso.

4. `Integration_Complexity` y `External_Dependencies` son relevantes: los proyectos que requieren integraciones complejas o múltiples proveedores externos enfrentan riesgos de coordinación e incertidumbre, lo que los empuja hacia niveles de riesgo más altos.

5. `Team_Size` muestra un efecto moderado: equipos muy pequeños (y a veces extremadamente grandes) se asocian con mayor riesgo. Se sospechan efectos no lineales (rendimientos decrecientes y sobrecarga de coordinación).

6. `Communication_Frequency` y la `Organizational_Changes_Frequency` muestran asociaciones más pequeñas pero consistentes con el riesgo.

## ***Predicciones e hipótesis***

- Los proyectos con `Vendor_Reliability_Score` por debajo del percentil 25 tienen una probabilidad significativamente mayor de estar en un nivel de riesgo alto.

- Los proyectos en el cuartil superior de `Estimated_Timeline_Months` son más propensos a ser de riesgo medio/alto.

- Los proyectos con más de 3 `External_Dependencies` presentan mayor riesgo, especialmente cuando el `Vendor_Reliability_Score` es bajo (efecto de interacción).

- Una baja `Resource_Availability` combinada con una alta `Integration_Complexity` constituye un factor de riesgo compuesto (efecto de interacción).
