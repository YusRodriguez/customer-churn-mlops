# Modelo Predictivo para la Estimación del Comportamiento de Pago en Solicitudes de Crédito

## Descripción general

Este proyecto desarrolla una solución de **Machine Learning** orientada a apoyar la evaluación del riesgo crediticio mediante la predicción del comportamiento de pago de nuevos solicitantes de crédito, utilizando información histórica proveniente de una entidad del sector financiero.

El objetivo principal consiste en construir un modelo de clasificación supervisada capaz de estimar la probabilidad de que un cliente realice el pago oportuno de sus obligaciones financieras. Para ello se emplean variables demográficas, laborales, económicas y de historial crediticio, permitiendo complementar el análisis realizado por los especialistas de riesgo durante el proceso de evaluación de una solicitud de crédito.

El proyecto fue desarrollado siguiendo una arquitectura basada en principios de **MLOps**, organizando cada etapa del ciclo de vida del modelo en componentes independientes que favorecen la reproducibilidad, mantenibilidad y escalabilidad de la solución. Como parte de este enfoque se implementan procesos de:

- Carga y validación del conjunto de datos.
- Análisis Exploratorio de Datos (EDA).
- Ingeniería de características (Feature Engineering).
- Entrenamiento y evaluación del modelo.
- Persistencia de artefactos.
- Despliegue para generación de predicciones.
- Monitoreo básico de la calidad de los datos.

La variable objetivo del proyecto es **`Pago_atiempo`**, la cual representa si un cliente cumplió o no oportunamente con el pago de su obligación crediticia. A partir de esta variable se entrena un modelo de **Regresión Logística**, el cual posteriormente puede reutilizarse para generar predicciones sobre nuevas solicitudes de crédito.

---

# Caso de negocio

Las entidades financieras enfrentan diariamente el desafío de evaluar múltiples solicitudes de crédito, procurando equilibrar el crecimiento de la colocación de cartera con una adecuada gestión del riesgo.

Tradicionalmente, estas decisiones dependen del análisis realizado por especialistas de riesgo, quienes consideran variables como ingresos, historial crediticio, nivel de endeudamiento, comportamiento financiero y capacidad de pago del solicitante. Sin embargo, a medida que aumenta el volumen de solicitudes, estos procesos pueden volverse más lentos, costosos y susceptibles a diferencias de criterio entre analistas.

En este contexto, las técnicas de **Machine Learning** permiten complementar la evaluación tradicional mediante modelos capaces de identificar patrones históricos asociados al comportamiento de pago de los clientes.

La solución desarrollada en este proyecto busca proporcionar una estimación objetiva basada en datos históricos que apoye el proceso de análisis crediticio, contribuyendo a:

- Fortalecer la evaluación del riesgo crediticio.
- Apoyar la priorización de solicitudes de crédito.
- Reducir tiempos asociados al análisis manual.
- Mejorar la consistencia en la toma de decisiones.
- Facilitar la incorporación de herramientas analíticas dentro de los procesos de evaluación.

Es importante resaltar que el modelo **no reemplaza la decisión del analista de riesgo**, sino que constituye una herramienta de apoyo que aporta evidencia cuantitativa para fortalecer el proceso de toma de decisiones.

Asimismo, el proyecto fue estructurado bajo principios de **MLOps**, permitiendo que cada etapa del flujo de trabajo sea reproducible, reutilizable y escalable, facilitando futuras integraciones con procesos automatizados de entrenamiento, despliegue y monitoreo.

---

# Objetivos

## Objetivo general

Desarrollar una solución de Machine Learning que permita estimar el comportamiento de pago de nuevos solicitantes de crédito utilizando información histórica del sector financiero, implementando un flujo de trabajo estructurado bajo principios de MLOps que garantice la reproducibilidad, organización y reutilización del proceso.

## Objetivos específicos

- Cargar y validar el conjunto de datos verificando su estructura, integridad y calidad.
- Realizar un Análisis Exploratorio de Datos (EDA) para comprender el comportamiento de las variables e identificar posibles problemas de calidad.
- Diseñar e implementar un proceso de Ingeniería de Características que prepare la información para el entrenamiento del modelo.
- Construir un pipeline de transformación reutilizable que garantice consistencia entre entrenamiento e inferencia.
- Entrenar y evaluar un modelo supervisado de clasificación para estimar el comportamiento de pago.
- Persistir tanto el pipeline de transformación como el modelo entrenado para facilitar su reutilización.
- Implementar un proceso de despliegue que permita generar predicciones sobre nuevos registros.
- Incorporar un mecanismo básico de monitoreo orientado a verificar la calidad de los datos utilizados por el modelo.
- Documentar cada etapa del proyecto siguiendo buenas prácticas de Ciencia de Datos y MLOps.

---

# Arquitectura del proyecto

El proyecto se desarrolló siguiendo una arquitectura modular que separa las diferentes etapas del flujo de trabajo. Esta organización facilita el mantenimiento del código, la reutilización de componentes y la trazabilidad del proceso completo de construcción del modelo.

```text
customer-churn-mlops/
│
├── data/
│   └── Base_de_datos.xlsx - Hoja1.csv
│
├── models/
│   ├── feature_pipeline.pkl
│   └── logistic_regression.pkl
│
├── src/
│   ├── feature_transformers.py
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
│
├── notebooks/
│   ├── Cargar_datos.ipynb
│   └── comprension_eda.ipynb
│
├── predicciones.csv
├── requirements.txt
├── README.md
└── .gitignore
```

### Componentes principales

| Componente | Descripción |
|------------|-------------|
| **Cargar_datos.ipynb** | Realiza la carga del conjunto de datos y ejecuta validaciones iniciales relacionadas con estructura, tipos de datos, valores nulos, duplicados y calidad de la información. |
| **comprension_eda.ipynb** | Desarrolla el Análisis Exploratorio de Datos, identificando patrones, distribuciones, relaciones entre variables y hallazgos relevantes para el problema de negocio. |
| **feature_transformers.py** | Contiene transformadores personalizados reutilizables utilizados dentro del pipeline de Ingeniería de Características. |
| **ft_engineering.py** | Implementa el proceso de Ingeniería de Características, aplica reglas de negocio, construye el pipeline de transformación y persiste el artefacto para su reutilización. |
| **model_training_evaluation.py** | Entrena el modelo de Regresión Logística, calcula métricas de evaluación y almacena el modelo entrenado. |
| **model_deploy.py** | Carga el pipeline y el modelo entrenado para generar predicciones sobre nuevos registros y exportar los resultados. |
| **model_monitoring.py** | Ejecuta un monitoreo básico de la calidad de los datos mediante estadísticas descriptivas, revisión de valores faltantes, duplicados y distribución de la variable objetivo. |

# Tecnologías utilizadas

El desarrollo del proyecto se realizó utilizando herramientas ampliamente empleadas en proyectos de Ciencia de Datos, Machine Learning y MLOps. Cada tecnología fue seleccionada de acuerdo con su propósito dentro del flujo de trabajo, permitiendo construir una solución reproducible, organizada y escalable.

| Tecnología | Propósito |
|------------|-----------|
| **Python** | Lenguaje principal para el desarrollo de la solución de Machine Learning. |
| **Jupyter Notebook** | Desarrollo del proceso de carga de datos, validación inicial y Análisis Exploratorio de Datos (EDA). |
| **Pandas** | Manipulación, limpieza, transformación y análisis del conjunto de datos. |
| **NumPy** | Operaciones numéricas y soporte para procesamiento eficiente de datos. |
| **Matplotlib** | Construcción de visualizaciones utilizadas durante el análisis exploratorio. |
| **Seaborn** | Elaboración de gráficos estadísticos para identificar patrones y relaciones entre variables. |
| **SciPy** | Aplicación de pruebas estadísticas durante el análisis exploratorio. |
| **Statsmodels** | Evaluación de multicolinealidad mediante el Factor de Inflación de la Varianza (VIF). |
| **Scikit-Learn** | Implementación del pipeline de transformación, entrenamiento, evaluación y despliegue del modelo. |
| **Joblib** | Persistencia del pipeline de transformación y del modelo entrenado para su reutilización. |
| **Logging** | Registro estructurado de eventos durante la ejecución de los scripts. |
| **Pathlib** | Gestión portable de rutas para garantizar compatibilidad entre diferentes sistemas operativos. |
| **Git y GitHub** | Control de versiones, administración del repositorio y seguimiento de cambios del proyecto. |

---

# Flujo metodológico del proyecto

El proyecto fue desarrollado siguiendo un flujo de trabajo estructurado que cubre las principales etapas del ciclo de vida de un modelo de Machine Learning. Cada componente cumple una responsabilidad específica y fue implementado de forma modular para facilitar su mantenimiento, reutilización y evolución futura.

```text
Carga de datos
        │
        ▼
Validación de calidad
        │
        ▼
Análisis Exploratorio de Datos (EDA)
        │
        ▼
Ingeniería de Características
        │
        ▼
División Train/Test
        │
        ▼
Construcción del Pipeline
        │
        ▼
Entrenamiento del Modelo
        │
        ▼
Evaluación del Modelo
        │
        ▼
Persistencia de Artefactos
        │
        ▼
Despliegue
        │
        ▼
Monitoreo de la Calidad de los Datos
```

Cada una de estas etapas fue implementada como un componente independiente, permitiendo que el flujo sea reproducible y facilitando futuras mejoras o sustituciones de modelos sin afectar el resto de la arquitectura.

Esta separación también favorece la trazabilidad de las transformaciones aplicadas y reduce el riesgo de inconsistencias entre entrenamiento e inferencia.

---

# Carga y validación del conjunto de datos

La primera etapa del proyecto consistió en la carga y validación del conjunto de datos con el propósito de verificar que la información cumpliera las condiciones mínimas necesarias para iniciar el proceso analítico.

Durante esta fase se ejecutaron diferentes controles orientados a garantizar la integridad del conjunto de datos, entre ellos:

- Verificación de la existencia del archivo fuente.
- Validación del número de registros y variables.
- Revisión de los tipos de datos.
- Identificación de valores faltantes.
- Detección de registros duplicados.
- Validación de la estructura esperada del dataset.
- Revisión preliminar de la variable objetivo.

Estas verificaciones permiten detectar oportunamente posibles problemas que podrían afectar la calidad del análisis o introducir inconsistencias durante el entrenamiento del modelo.

Asimismo, esta etapa constituye el punto de entrada del flujo de trabajo y garantiza que todas las fases posteriores trabajen sobre información consistente y estructurada.

---

# Análisis Exploratorio de Datos (EDA)

El Análisis Exploratorio de Datos tuvo como propósito comprender la estructura del conjunto de datos, identificar patrones relevantes y detectar posibles problemas de calidad que pudieran afectar el desempeño del modelo predictivo.

Durante esta etapa se analizaron las variables demográficas, laborales, financieras y de historial crediticio, evaluando tanto sus características individuales como las relaciones existentes entre ellas y la variable objetivo **Pago_atiempo**.

Las principales actividades desarrolladas fueron:

- Análisis univariado de variables numéricas y categóricas.
- Evaluación de distribuciones y medidas descriptivas.
- Identificación de valores faltantes.
- Detección de registros duplicados.
- Identificación de valores atípicos.
- Análisis bivariado entre variables predictoras y la variable objetivo.
- Evaluación de correlaciones entre variables numéricas.
- Análisis de multicolinealidad mediante el Factor de Inflación de la Varianza (VIF).
- Identificación de variables con posible riesgo de fuga de información (*data leakage*).

Este análisis permitió comprender el comportamiento general del conjunto de datos y proporcionó evidencia para justificar las decisiones adoptadas posteriormente durante la etapa de Ingeniería de Características.

---

# Principales hallazgos del Análisis Exploratorio

El proceso de exploración permitió identificar información relevante tanto desde la perspectiva estadística como desde el contexto del negocio.

## Calidad del conjunto de datos

El conjunto de datos presentó una estructura consistente y sin registros completamente duplicados, lo que constituye una base adecuada para el desarrollo del modelo predictivo.

No obstante, algunas variables financieras presentaron valores faltantes. En lugar de eliminar dichos registros, se decidió tratarlos mediante estrategias de imputación implementadas dentro del pipeline de transformación, preservando la mayor cantidad posible de información disponible.

---

## Variables financieras

Las variables relacionadas con montos, saldos, obligaciones vigentes e historial financiero evidenciaron una alta variabilidad entre clientes, comportamiento esperado en bases de datos crediticias debido a la diversidad de perfiles financieros presentes en la población analizada.

Desde la perspectiva del negocio, estas variables aportan información relevante para caracterizar la capacidad de pago y el nivel de riesgo asociado a cada solicitante.

---

## Variables demográficas y laborales

Las variables demográficas y laborales proporcionan información complementaria sobre el perfil socioeconómico del cliente.

Aunque de manera individual su capacidad predictiva puede ser limitada, su combinación con variables financieras permite enriquecer la representación del comportamiento histórico de los solicitantes y mejorar la capacidad de generalización del modelo.

---

## Historial crediticio

Las variables relacionadas con el historial crediticio representan uno de los componentes más relevantes para la evaluación del riesgo.

Indicadores como el puntaje crediticio, el número de créditos vigentes y los saldos asociados a obligaciones anteriores permiten resumir el comportamiento financiero histórico de cada cliente y constituyen una fuente importante de información para el proceso de modelado.

Durante el análisis exploratorio también se identificó que la variable **puntaje** presentaba una asociación extremadamente alta con la variable objetivo. Debido al riesgo de que dicha variable introdujera **fuga de información (Data Leakage)**, se decidió excluirla del entrenamiento del modelo durante la etapa de Ingeniería de Características, buscando mejorar la capacidad de generalización sobre nuevos datos.

---

## Variable objetivo

La variable objetivo **Pago_atiempo** representa el comportamiento de pago que el modelo busca estimar.

El análisis de su distribución permitió comprender el balance entre clases antes del entrenamiento y sirvió como referencia para interpretar posteriormente las métricas de desempeño del modelo.

Conocer esta distribución también permitió validar la estrategia de partición del conjunto de datos mediante un **Train/Test Split estratificado**, garantizando que ambas particiones conservaran una representación adecuada de las clases presentes en el conjunto original.

# Ingeniería de Características (Feature Engineering)

Una vez finalizado el Análisis Exploratorio de Datos, se implementó un proceso de Ingeniería de Características cuyo propósito fue preparar el conjunto de datos para el entrenamiento del modelo, garantizando que todas las transformaciones aplicadas durante esta etapa pudieran reutilizarse posteriormente durante el despliegue.

El proceso fue desarrollado siguiendo principios de reproducibilidad propios de MLOps, encapsulando todas las transformaciones dentro de un **Pipeline** persistente construido mediante Scikit-Learn.

Las principales actividades realizadas durante esta etapa fueron:

- Validación estructural del conjunto de datos.
- Aplicación de reglas de negocio derivadas del análisis exploratorio.
- Extracción de variables temporales a partir de la fecha del préstamo.
- Exclusión de variables con riesgo de fuga de información (*Data Leakage*).
- Separación entre variables predictoras y variable objetivo.
- División del conjunto de datos en entrenamiento y prueba mediante un muestreo estratificado.
- Construcción de un Pipeline reutilizable para el preprocesamiento.
- Persistencia del Pipeline mediante Joblib para su utilización durante el entrenamiento y el despliegue.

Las transformaciones implementadas fueron diseñadas para garantizar que cualquier nuevo conjunto de datos sea procesado exactamente bajo las mismas reglas utilizadas durante el entrenamiento del modelo, evitando inconsistencias entre ambientes de desarrollo e inferencia.

---

## Reglas de negocio implementadas

Como resultado del Análisis Exploratorio de Datos se identificaron diferentes condiciones que debían controlarse antes del entrenamiento del modelo.

Entre las principales reglas implementadas se encuentran:

- Conservación únicamente de registros con edades válidas para procesos de evaluación crediticia.
- Eliminación de registros con valores no válidos para variables como capital prestado y plazo del crédito.
- Validación de la estructura mínima esperada del conjunto de datos.
- Eliminación de registros completamente duplicados cuando existieran.
- Tratamiento de valores faltantes mediante estrategias de imputación incorporadas dentro del Pipeline.

Estas reglas permiten reducir la presencia de inconsistencias en la información utilizada para el entrenamiento y contribuyen a mejorar la calidad del proceso de modelado.

---

## Prevención de fuga de información (Data Leakage)

Uno de los hallazgos más relevantes del Análisis Exploratorio fue la identificación de la variable **puntaje** como un posible riesgo de fuga de información.

Durante el análisis de correlaciones se observó que esta variable presentaba una asociación significativamente superior respecto al resto de variables predictoras. Desde la perspectiva del negocio, este comportamiento podría indicar que parte de la información contenida en dicha variable incorpora conocimiento muy cercano al resultado que el modelo intenta predecir.

Con el fin de reducir el riesgo de sobreestimar el desempeño del modelo y favorecer su capacidad de generalización sobre nuevos datos, esta variable fue excluida durante la construcción del Pipeline de Ingeniería de Características.

Esta decisión permite que el modelo aprenda patrones asociados al comportamiento histórico de los clientes utilizando únicamente información que estaría disponible en un escenario real de evaluación crediticia.

---

## Construcción del Pipeline

El proceso de transformación fue implementado utilizando un **Pipeline** de Scikit-Learn, permitiendo encapsular todas las etapas de preprocesamiento en un único objeto reutilizable.

La estructura general del Pipeline comprende:

```text
Dataset original
        │
        ▼
Extracción de variables de fecha
        │
        ▼
Eliminación de variables con riesgo de Data Leakage
        │
        ▼
Separación automática de variables numéricas y categóricas
        │
        ▼
Imputación de valores faltantes
        │
        ▼
Escalamiento de variables numéricas
        │
        ▼
Codificación One-Hot de variables categóricas
        │
        ▼
Pipeline persistido (feature_pipeline.pkl)
```

Este enfoque garantiza que todas las observaciones reciban exactamente el mismo tratamiento durante el entrenamiento y el despliegue del modelo, eliminando diferencias entre ambos procesos y fortaleciendo la reproducibilidad de la solución.

---

# Modelo de Machine Learning

El problema abordado corresponde a un escenario de **clasificación supervisada**, donde el objetivo consiste en estimar el comportamiento de pago de nuevos solicitantes de crédito utilizando información histórica proveniente de operaciones crediticias anteriores.

Para este proyecto se seleccionó un modelo de **Regresión Logística**, ampliamente utilizado como algoritmo base en problemas de clasificación binaria debido a sus ventajas en términos de interpretabilidad, estabilidad y eficiencia computacional.

El entrenamiento del modelo se realizó utilizando la librería **Scikit-Learn**, integrando el Pipeline de Ingeniería de Características previamente construido para garantizar que todas las transformaciones aplicadas a los datos fueran consistentes durante las diferentes etapas del ciclo de vida del modelo.

El modelo entrenado fue persistido mediante **Joblib**, permitiendo reutilizarlo posteriormente durante el proceso de despliegue sin necesidad de repetir el entrenamiento.

---

# Entrenamiento del modelo

Después de completar la Ingeniería de Características, el conjunto de datos fue dividido en subconjuntos de entrenamiento y prueba utilizando una partición estratificada, preservando la distribución de la variable objetivo.

Posteriormente, el Pipeline de transformación fue ajustado exclusivamente sobre el conjunto de entrenamiento, evitando que información del conjunto de prueba influyera durante el proceso de aprendizaje del modelo.

Una vez transformados los datos, se entrenó un modelo de Regresión Logística utilizando las variables procesadas por el Pipeline.

Este flujo garantiza que:

- El entrenamiento y el despliegue utilicen exactamente las mismas transformaciones.
- Se reduzca el riesgo de inconsistencias entre entrenamiento e inferencia.
- El proceso completo sea reproducible.
- Los artefactos generados puedan reutilizarse sin necesidad de repetir todas las etapas del proyecto.

Finalmente, tanto el Pipeline de Ingeniería de Características como el modelo entrenado fueron almacenados mediante **Joblib**, permitiendo su reutilización dentro del proceso de despliegue y favoreciendo la escalabilidad de la solución.

# Evaluación del modelo

Una vez finalizado el proceso de entrenamiento, el modelo fue evaluado utilizando el conjunto de prueba con el propósito de medir su capacidad para generalizar sobre datos no utilizados durante el aprendizaje.

Para ello se emplearon métricas ampliamente utilizadas en problemas de clasificación binaria, permitiendo evaluar diferentes dimensiones del desempeño del modelo y obtener una visión integral de su comportamiento.

## Métricas obtenidas

| Métrica | Resultado |
|----------|----------:|
| Accuracy | **0.9981** |
| Precision | **0.9985** |
| Recall | **0.9995** |
| F1-Score | **0.9990** |
| ROC AUC | **1.0000** |

---

## Reporte de clasificación

| Clase | Precision | Recall | F1-Score | Soporte |
|-------|----------:|-------:|---------:|--------:|
| 0 | 0.99 | 0.97 | 0.98 | 102 |
| 1 | 1.00 | 1.00 | 1.00 | 2051 |

**Exactitud global:** **99.81 %**

---

## Matriz de confusión

| | Predicción: 0 | Predicción: 1 |
|---|---:|---:|
| **Valor real: 0** | 99 | 3 |
| **Valor real: 1** | 1 | 2050 |

---

## Interpretación de resultados

Los resultados obtenidos evidencian un desempeño sobresaliente del modelo sobre el conjunto de prueba.

La métrica **Accuracy (99.81 %)** indica que el modelo clasificó correctamente la gran mayoría de las observaciones evaluadas.

La **Precision (99.85 %)** refleja que prácticamente todas las predicciones positivas realizadas por el modelo corresponden efectivamente a clientes con comportamiento de pago oportuno.

El **Recall (99.95 %)** demuestra una elevada capacidad para identificar correctamente los clientes pertenecientes a la clase positiva, minimizando la presencia de falsos negativos.

El **F1-Score (99.90 %)** confirma un equilibrio adecuado entre Precision y Recall, mostrando un comportamiento consistente durante la clasificación.

Finalmente, el **ROC AUC (1.0000)** evidencia una excelente capacidad de discriminación entre ambas clases dentro del conjunto de evaluación.

Es importante resaltar que, aunque estas métricas muestran un desempeño excepcional, en un entorno productivo sería recomendable complementar la evaluación mediante validación cruzada, pruebas sobre nuevos datos y monitoreo continuo para confirmar la capacidad de generalización del modelo a lo largo del tiempo.

---

# Persistencia de artefactos

Como parte del enfoque MLOps implementado en este proyecto, los principales componentes generados durante el entrenamiento fueron almacenados para permitir su reutilización sin necesidad de repetir todo el proceso de construcción del modelo.

Los artefactos persistidos fueron:

| Artefacto | Descripción |
|------------|-------------|
| **feature_pipeline.pkl** | Pipeline de Ingeniería de Características utilizado para transformar nuevos datos de manera consistente. |
| **logistic_regression.pkl** | Modelo de Regresión Logística entrenado y listo para generar predicciones. |

La persistencia de estos componentes permite garantizar que las mismas transformaciones aplicadas durante el entrenamiento sean utilizadas posteriormente durante el despliegue del modelo.

---

# Despliegue del modelo

El proceso de despliegue fue implementado mediante el script `model_deploy.py`.

Este componente tiene como objetivo reutilizar el pipeline de transformación y el modelo previamente entrenado para generar predicciones sobre nuevos registros.

El flujo general de ejecución es el siguiente:

```text
Carga del dataset
        │
        ▼
Carga del Pipeline
        │
        ▼
Carga del modelo entrenado
        │
        ▼
Transformación de nuevas observaciones
        │
        ▼
Generación de predicciones
        │
        ▼
Exportación de resultados
```

El resultado del proceso corresponde a un archivo denominado:

```text
predicciones.csv
```

que contiene las predicciones generadas para cada uno de los registros procesados.

---

# Monitoreo de la calidad de los datos

Con el propósito de fortalecer la mantenibilidad de la solución, se implementó un componente de monitoreo encargado de generar información descriptiva sobre el conjunto de datos utilizado por el modelo.

El monitoreo desarrollado permite identificar posibles cambios en la calidad de los datos antes de ejecutar nuevas predicciones.

Entre las validaciones implementadas se incluyen:

- Verificación de dimensiones del conjunto de datos.
- Revisión de valores faltantes.
- Identificación de registros duplicados.
- Estadísticas descriptivas de variables numéricas.
- Distribución de la variable objetivo.
- Generación de información útil para el seguimiento de la calidad de los datos.

Este proceso constituye una primera aproximación al monitoreo dentro de una arquitectura MLOps y facilita la detección temprana de posibles inconsistencias que podrían afectar el desempeño del modelo.

---

# Conclusiones

El desarrollo de este proyecto permitió implementar un flujo completo de Machine Learning para apoyar la estimación del comportamiento de pago de nuevos solicitantes de crédito utilizando información histórica del sector financiero.

A lo largo del proyecto se integraron las principales etapas del ciclo de vida de un modelo predictivo, incluyendo la carga y validación de datos, el análisis exploratorio, la ingeniería de características, el entrenamiento del modelo, la evaluación, la persistencia de artefactos, el despliegue y el monitoreo básico de la calidad de los datos.

Desde la perspectiva técnica, la utilización de un Pipeline de Ingeniería de Características garantiza la consistencia entre entrenamiento e inferencia, favoreciendo la reproducibilidad y reduciendo el riesgo de errores derivados de transformaciones manuales.

Asimismo, la persistencia tanto del pipeline como del modelo entrenado facilita su reutilización en futuros procesos de inferencia sin necesidad de repetir todas las etapas del entrenamiento.

Desde la perspectiva del negocio, la solución desarrollada constituye una herramienta de apoyo para los procesos de evaluación del riesgo crediticio, aportando evidencia cuantitativa que complementa el análisis realizado por los especialistas responsables de la toma de decisiones.

---

# Trabajo futuro

Como parte de la evolución natural de este proyecto, podrían incorporarse mejoras orientadas a fortalecer la capacidad predictiva y la escalabilidad de la solución.

Entre ellas se destacan:

- Comparar el desempeño de diferentes algoritmos de clasificación.
- Implementar procesos de validación cruzada para fortalecer la evaluación del modelo.
- Incorporar técnicas de optimización de hiperparámetros.
- Implementar monitoreo avanzado para detectar cambios en la distribución de los datos (Data Drift) y en el desempeño del modelo (Concept Drift).
- Desarrollar una interfaz web para facilitar la generación de predicciones por parte de usuarios de negocio.
- Automatizar completamente el pipeline mediante herramientas de integración y despliegue continuo.

---

# Autor

**Yustin Perico Rodríguez**

Proyecto Integrador – Ciencia de Datos

Bootcamp de Ciencia de Datos

---

# Licencia

Este proyecto fue desarrollado con fines académicos como parte del Proyecto Integrador del Bootcamp de Ciencia de Datos.