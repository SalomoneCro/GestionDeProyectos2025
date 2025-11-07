Diccionario de Datos

La idea es describir brevemente las características del dataset. Las características se encuentran divididas en distintos grupos.

Características demográficas del problema:

- **“Project_ID”**: Se le asigna un número a cada proyecto para poder luego identificarlo
- **“Project_Type”**: Describe el tipo de proyecto indicando en qué industria o área va a ser realizado. Dentro de las variables podemos encontrar: “Construcción”, “IT”, “Healthcare” y otras más.
- **“Project_Budget_USD”**: Se pide un estimativo del presupuesto total del proyecto en dólares. Esta es una característica numérica.
- **“Estimated_Timeline_Months”**: Se pide la duración estimada del proyecto dada en meses.
- **“Team_Size”**: Describe el tamaño del equipo, indicando cuántas personas estarán asignadas al proyecto. Estas son variables numéricas.
- **“Complexity_Score”**: Esta variable es una escala numérica del 1 al 10 en donde se describe la complejidad y dificultad inherentes al proyecto.

Métricas operativas

- **“Change_Request_Frequency”**: Tasa a la que se presentan solicitudes formales de cambio durante el proyecto. Esto es una forma de medir cuantitativamente la cantidad de veces que se pide un cambio en el producto/proyecto.
- **“Budget_Utilization_Rate”**: Relación entre el presupuesto real gastado y el presupuesto planificado.
- **“Resource_Availability”**: Porcentaje de los recursos requeridos que están disponibles para el proyecto.
- **“Current_Phase_Duration_Months”**: Duración en meses dedicada a la fase actual del proyecto.

Recursos Humanos

- **“Team_Experience”**: Es un promedio de la experiencia de los miembros del equipo que van a llevar a cabo el proyecto. Dentro de estas categorías tenemos “Senior”, “Junior”, “Expert”, etc.
- **“Project_Manager_Experience”**: Experiencia del Project Manager, toma valores como “Senior PM”, “Mid level PM”, etc.
- **“Stakeholder_Engagement_Level”**: Nivel de participación activa y compromiso de los stakeholder clave.
- **“Team_Turnover_Rate”**: Porcentaje de miembros del equipo que dejan el proyecto en su desarrollo.

<--- Page Split --->
Factores Organizacionales

- **“Org_Process_Maturity”**: Describe el nivel de desarrollo o madurez de los procesos y normas de gestión de proyectos dentro de la organización.
- **“Regulatory_Compliance_Level”**: Nivel de supervisión regulatoria o los requisitos específicos de cumplimiento que el proyecto debe cumplir.
- **“Funding_Source”**: Nos dice de donde provienen los recursos financieros
- **“Risk_Management_Maturity”**: Nivel de madurez de la organización para identificar, evaluar y mitigar los riesgos.

Aspectos Técnicos

- **“Technology_Familiarity”**: Nos describe que tan familiarizados están los miembros del equipo con las tecnologías que se van a utilizar en el proyecto.
- **“Technical_Debt_Level”**: Compromisos o atajos técnicos acumulados en desarrollos anteriores, principalmente relevantes para proyectos de IT
- **“Integration_Complexity”**: Dificultad para integrar los distintos componentes, sistemas o procesos del proyecto.
- **“Tech_Environment_Stability”**: Estabilidad de la infraestructura o plataforma técnica subyacente utilizada en el proyecto.

Factores Externos

- **“Market_Volatility”**: Es un puntaje entre el (0,1) que describe la inestabilidad o la imprevisibilidad del mercado relevante para el proyecto.
- **“Industry_Volatility”**: Describe que tan estable o imprevisible es el sector de la industria en donde opera el proyecto.
- **“External_Dependencies_Count”**: Número de factores o entidades externas de las que depende el proyecto.
- **“Client_Experience_Level”**: Experiencia previa del cliente trabajando con la organización de proyectos.

Otros

- **“Stakeholder_Count”**: Es el número de personas claves o grupos interesadas en el desarrollo del proyecto.
- **“Methodology_Used”**: Se describe la metodología adoptada para llevar a cabo el proyecto. En ellas se pueden encontrar “Ágil”, “Scrum”, “Kanban”, etc.
- **“Past_Similar_Project”**: Número de proyectos similares completados por la organización o el equipo.
- **“Project_Phase”**: Etapa actual del ciclo de vida del proyecto.
- **“Requirement_Stability”**: Describe que tan estable y bien definido está el proyecto.
- **“Vendor_Reliability_Score”**: Un puntaje del (0 al 1) que indica la fiabilidad histórica de los proveedores o socios externos.
- **“Historical_Risk_Incidents”**: Número de riesgos pasados o riesgos encontrados en proyectos similares.

<--- Page Split --->
• **"Communication_Frequency"**: Frecuencia con la que se producen comunicaciones formales dentro del proyecto.

• **"Geographical_Distribution"**: Número de ubicaciones geográficas distintas en las que se encuentran los miembros del equipo.

• **"Schedule_Pressure"**: Es un indicador que refleja cuán ajustado está el cronograma del proyecto respecto a su duración ideal.

• **"Executive_Sponsorship"**: Grado de respaldo e involucramiento por parte de la alta dirección.

• **"Priority_Level"**: Grado de relevancia estratégica o nivel de urgencia que el proyecto tiene para la organización.

• **"Organizational_Change_Frequency"**: Frecuencia con la que la organización en su conjunto atraviesa cambios estructurales o estratégicos significativos.

• **"Cross_Functional_Dependecies"**: Número de dependencias de otros departamentos o funciones internas de la organización.

• **"Previous_Delivery_Success_Rate"**: Tasa histórica de éxito de proyectos similares realizados por la organización.

• **"Data_Security_Requirement"**: El nivel de rigor de las regulaciones de seguridad y privacidad de los datos aplicables al proyecto.

• **"Key_Stakeholder_Availability"**: Grado de disponibilidad y capacidad de respuesta de los stakeholders clave.

• **"Contract_Type"**: Tipo de contrato utilizado para trabajos o servicios externos.

• **"Resource_Contention_Level"**: Nivel de competencia por recursos compartidos entre múltiples proyectos dentro de la organización.

• **"Change_Control_Maturity"**: Grado de madurez del proceso para gestionar cambios en el alcance o los requerimientos del proyecto.

• **"Team_Colocation"**: Describe cómo están distribuidos geográficamente los miembros del equipo.

• **"Documentatio_Quality"**: Describe cómo es la calidad o la completitud de la documentación del proyecto.

• **"Proyect_Start_Month"**: Mes en el que se empezó el proyecto. Es un número del 1 al 12 que se corresponde a cada mes del año.

• **"Seasonal_Risk_Factor"**: Un factor multiplicador que indica el aumento de riesgo debido a influencias estacionales en ciertas industrias.

• **"Risk_Level"**: El riesgo general del problema. Esta es la característica que nos va a servir para controlar la clasificación.

<--- Page Split --->
