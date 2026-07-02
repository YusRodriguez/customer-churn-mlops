# Predicción del Comportamiento de Pago mediante Machine Learning y MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-red)

---

# Proyecto Integrador

## Ciencia de Datos – Machine Learning – MLOps

Autor:
**Yus Rodríguez**

---

# Descripción del Proyecto

Este proyecto desarrolla una solución integral de Machine Learning para predecir el comportamiento de pago de nuevos solicitantes de crédito mediante un pipeline reproducible de Ciencia de Datos y MLOps.

La solución implementa todo el ciclo de vida del modelo, comenzando con la comprensión del problema de negocio, la exploración y preparación de los datos, el diseño del Feature Engineering, el entrenamiento y evaluación del modelo predictivo, hasta llegar a su despliegue mediante una API REST y su contenerización utilizando Docker.

El objetivo principal consiste en transformar un proceso tradicional de análisis en una solución lista para producción, asegurando consistencia entre entrenamiento e inferencia, trazabilidad de los artefactos y facilidad de despliegue en diferentes entornos.

Todo el flujo fue desarrollado bajo principios de Machine Learning reproducible, reutilizando exactamente el mismo pipeline de transformación tanto durante el entrenamiento como durante la generación de predicciones, evitando inconsistencias entre ambientes de desarrollo y producción.

---

# Objetivo General

Desarrollar una solución de Machine Learning capaz de predecir el comportamiento de pago de nuevos clientes del sector financiero mediante un flujo completo de MLOps que permita entrenar, evaluar, desplegar y consumir el modelo de manera reproducible.

---

# Objetivos Específicos

- Comprender el problema de negocio y el contexto financiero.
- Analizar la calidad del conjunto de datos.
- Realizar el proceso de limpieza y preparación de información.
- Construir variables derivadas mediante Feature Engineering.
- Implementar un pipeline reutilizable utilizando Scikit-Learn.
- Entrenar un modelo supervisado de clasificación.
- Evaluar el desempeño mediante métricas especializadas.
- Persistir el modelo y el pipeline entrenado.
- Construir una API REST utilizando FastAPI.
- Automatizar el proceso de inferencia reutilizando exactamente el mismo pipeline.
- Contenerizar toda la solución utilizando Docker.
- Versionar el proyecto mediante Git y GitHub.
- Aplicar principios fundamentales de MLOps para garantizar reproducibilidad y escalabilidad.

---

# Problema de Negocio

Las entidades financieras deben evaluar diariamente cientos de solicitudes de crédito provenientes de nuevos clientes.

Una decisión incorrecta puede incrementar el riesgo de cartera, afectar los indicadores financieros y generar pérdidas económicas derivadas del incumplimiento de pago.

Tradicionalmente este proceso depende del análisis manual realizado por analistas de riesgo, quienes consideran múltiples variables relacionadas con historial crediticio, ingresos, endeudamiento y comportamiento financiero.

Sin embargo, el crecimiento del volumen de solicitudes hace necesario incorporar herramientas analíticas capaces de apoyar la toma de decisiones mediante modelos predictivos.

En este contexto, el presente proyecto propone una solución basada en Machine Learning que permita estimar si un nuevo solicitante realizará oportunamente el pago de su obligación financiera.

El resultado del modelo puede utilizarse como un mecanismo de apoyo para fortalecer la evaluación crediticia, disminuir el riesgo operativo y optimizar el proceso de análisis de solicitudes.

---

# Arquitectura General

El flujo completo desarrollado sigue una arquitectura MLOps compuesta por las siguientes etapas:

Datos
↓

Comprensión del negocio

↓

Análisis Exploratorio (EDA)

↓

Limpieza de datos

↓

Feature Engineering

↓

Pipeline reutilizable

↓

Entrenamiento del modelo

↓

Evaluación

↓

Persistencia de artefactos

↓

API REST (FastAPI)

↓

Docker

↓

Despliegue
# Arquitectura del Proyecto

El proyecto fue diseñado siguiendo una arquitectura modular basada en principios de **Machine Learning Operations (MLOps)**, separando cada etapa del ciclo de vida del modelo en componentes independientes y reutilizables.

Esta organización facilita la mantenibilidad del código, la reproducibilidad de los experimentos, la reutilización de artefactos y la escalabilidad hacia ambientes de producción.

```text
credit-risk-mlops/
│
├── models/
│   ├── feature_pipeline.pkl
│   └── logistic_regression.pkl
│
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── feature_transformers.py
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
│
├── api.py
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── Base_de_datos.xlsx - Hoja1.csv
│
└── predicciones.csv
```

Cada componente cumple una responsabilidad específica dentro del flujo de Machine Learning, permitiendo desacoplar el entrenamiento, la inferencia, el monitoreo y el despliegue del modelo.

---

# Flujo Metodológico

La solución fue desarrollada siguiendo un flujo estructurado que cubre todas las etapas del ciclo de vida de un modelo de Machine Learning.

```text
Comprensión del problema de negocio
                │
                ▼
Carga y validación del conjunto de datos
                │
                ▼
Análisis Exploratorio de Datos (EDA)
                │
                ▼
Ingeniería de Características
                │
                ▼
Construcción del Pipeline
                │
                ▼
Entrenamiento del Modelo
                │
                ▼
Evaluación del Desempeño
                │
                ▼
Persistencia de Artefactos
                │
                ▼
API REST (FastAPI)
                │
                ▼
Contenerización con Docker
                │
                ▼
Despliegue del Servicio
                │
                ▼
Monitoreo Básico
```

Este enfoque garantiza que las transformaciones utilizadas durante el entrenamiento sean exactamente las mismas empleadas durante la inferencia, reduciendo inconsistencias entre ambientes y fortaleciendo la reproducibilidad del proyecto.

---

# Tecnologías Utilizadas

El proyecto integra herramientas ampliamente utilizadas en proyectos profesionales de Ciencia de Datos, Machine Learning y MLOps.

| Tecnología | Función dentro del proyecto |
|------------|-----------------------------|
| **Python 3.11** | Lenguaje principal del desarrollo. |
| **Pandas** | Manipulación, limpieza y transformación de datos. |
| **NumPy** | Operaciones numéricas y procesamiento matricial. |
| **Matplotlib** | Visualización de datos durante el EDA. |
| **Seaborn** | Análisis estadístico y exploración gráfica. |
| **SciPy** | Aplicación de pruebas estadísticas. |
| **Statsmodels** | Evaluación de multicolinealidad mediante VIF. |
| **Scikit-Learn** | Construcción del Pipeline, entrenamiento y evaluación del modelo. |
| **Joblib** | Persistencia del Pipeline y del modelo entrenado. |
| **FastAPI** | Desarrollo de la API REST para inferencia del modelo. |
| **Uvicorn** | Servidor ASGI utilizado para ejecutar la API. |
| **Docker** | Contenerización de toda la solución para facilitar el despliegue. |
| **Jupyter Notebook** | Desarrollo del EDA y análisis exploratorio. |
| **Git** | Control de versiones del proyecto. |
| **GitHub** | Administración del repositorio y colaboración. |
| **Logging** | Registro estructurado de eventos durante la ejecución de los scripts. |
| **Pathlib** | Gestión portable de rutas del proyecto. |

---

# Componentes del Proyecto

Cada módulo fue desarrollado para cumplir una responsabilidad específica dentro de la arquitectura.

| Componente | Descripción |
|------------|-------------|
| **Cargar_datos.ipynb** | Carga del conjunto de datos, validación estructural, análisis inicial de calidad y revisión de integridad. |
| **comprension_eda.ipynb** | Desarrollo del Análisis Exploratorio de Datos (EDA), análisis estadístico y comprensión del problema. |
| **feature_transformers.py** | Implementación de transformadores personalizados reutilizados dentro del Pipeline de Feature Engineering. |
| **ft_engineering.py** | Construcción del Pipeline de transformación, reglas de negocio, tratamiento de variables y persistencia del Feature Pipeline. |
| **model_training_evaluation.py** | Entrenamiento del modelo de Regresión Logística, evaluación mediante métricas de clasificación y almacenamiento del modelo entrenado. |
| **model_deploy.py** | Reutilización del Pipeline y del modelo para generar predicciones sobre nuevos registros. |
| **model_monitoring.py** | Monitoreo básico de calidad de datos y generación de estadísticas descriptivas para validar nuevas entradas. |
| **api.py** | Implementación de la API REST utilizando FastAPI para consumir el modelo mediante solicitudes HTTP. |
| **app.py** | Interfaz desarrollada en Streamlit para facilitar la interacción con el modelo desde una aplicación web. |

---

# Carga y Validación del Conjunto de Datos

El desarrollo del proyecto inició con un proceso de carga y validación del conjunto de datos con el propósito de garantizar la calidad de la información antes de comenzar cualquier proceso analítico.

Durante esta etapa se verificó la existencia del archivo fuente, la estructura del dataset, los tipos de datos, la presencia de valores faltantes, registros duplicados y la consistencia general de las variables.

Las principales validaciones realizadas fueron:

- Verificación del número de registros y variables.
- Revisión de los tipos de datos.
- Identificación de valores nulos.
- Detección de registros duplicados.
- Validación de la variable objetivo.
- Inspección inicial de variables numéricas y categóricas.

Este proceso permitió establecer una línea base de calidad sobre la cual se desarrollaron las siguientes etapas del proyecto, reduciendo el riesgo de inconsistencias durante el entrenamiento del modelo.

---

# Análisis Exploratorio de Datos (EDA)

Posteriormente se desarrolló un proceso de Análisis Exploratorio de Datos (EDA) orientado a comprender el comportamiento de las variables, identificar patrones relevantes y detectar posibles problemas que pudieran afectar el desempeño del modelo predictivo.

El análisis se realizó tanto desde una perspectiva estadística como desde el contexto del negocio, permitiendo comprender cómo interactúan las variables demográficas, laborales, financieras y de historial crediticio con el comportamiento de pago de los clientes.

Las actividades desarrolladas durante esta etapa incluyeron:

- Análisis univariado de variables numéricas y categóricas.
- Estadísticas descriptivas.
- Distribución de la variable objetivo.
- Identificación de valores faltantes.
- Detección de registros duplicados.
- Identificación de valores atípicos.
- Matriz de correlación.
- Evaluación de multicolinealidad mediante el Factor de Inflación de la Varianza (VIF).
- Análisis bivariado entre variables predictoras y la variable objetivo.

Los resultados obtenidos permitieron comprender la estructura del conjunto de datos y fundamentar técnicamente las decisiones adoptadas durante la etapa de Ingeniería de Características.

---

# Principales Hallazgos del EDA

El análisis exploratorio permitió identificar diferentes aspectos relevantes tanto para el negocio como para el proceso de modelado.

## Calidad de los datos

El conjunto de datos presentó una estructura consistente, sin registros completamente duplicados y con una distribución adecuada de la mayoría de las variables.

Las variables que contenían valores faltantes fueron tratadas posteriormente dentro del Pipeline de Feature Engineering mediante estrategias de imputación, evitando eliminar información potencialmente valiosa.

---

## Variables financieras

Las variables financieras mostraron una alta variabilidad, comportamiento esperado dentro de procesos de evaluación crediticia debido a la diversidad de perfiles económicos presentes en la población analizada.

Estas variables aportan información fundamental para caracterizar la capacidad de pago y el nivel de riesgo asociado a cada solicitante.

---

## Variables demográficas y laborales

Las variables relacionadas con edad, ocupación, ingresos y características laborales complementan el perfil financiero del cliente.

Aunque individualmente presentan una capacidad predictiva moderada, en conjunto enriquecen significativamente la representación del comportamiento histórico de los solicitantes.

---

## Variables de historial crediticio

El historial crediticio constituye uno de los principales factores explicativos del comportamiento de pago.

Variables relacionadas con obligaciones vigentes, puntaje crediticio, número de créditos y comportamiento financiero previo aportan información crítica para estimar el riesgo de incumplimiento.

Durante esta etapa también se identificó una posible situación de **Data Leakage** asociada a la variable **puntaje**, razón por la cual fue excluida posteriormente del entrenamiento para favorecer la capacidad de generalización del modelo.

---

# Ingeniería de Características (Feature Engineering)

Una vez finalizado el análisis exploratorio, se implementó un proceso de Ingeniería de Características orientado a preparar la información para el entrenamiento del modelo y garantizar la reutilización de todas las transformaciones durante la inferencia.

A diferencia de un proceso manual de preparación de datos, todas las transformaciones fueron encapsuladas dentro de un **Pipeline de Scikit-Learn**, permitiendo que exactamente las mismas reglas sean aplicadas tanto durante el entrenamiento como durante el despliegue del modelo.

Las principales transformaciones implementadas fueron:

- Validación estructural del conjunto de datos.
- Extracción de variables temporales a partir de la fecha del préstamo.
- Eliminación de variables con riesgo de fuga de información.
- Separación automática entre variables numéricas y categóricas.
- Imputación de valores faltantes.
- Escalamiento de variables numéricas.
- Codificación One-Hot para variables categóricas.
- Persistencia del Pipeline mediante Joblib.

Este enfoque reduce significativamente el riesgo de inconsistencias entre entrenamiento e inferencia y constituye una de las principales prácticas recomendadas dentro de arquitecturas MLOps.

---

# Construcción del Pipeline de Transformación

Todo el proceso de preprocesamiento fue encapsulado dentro de un Pipeline reutilizable implementado con Scikit-Learn.

```text
Datos originales
        │
        ▼
Extracción de variables temporales
        │
        ▼
Eliminación de variables con Data Leakage
        │
        ▼
Separación de variables
        │
        ▼
Imputación de valores faltantes
        │
        ▼
Escalamiento de variables numéricas
        │
        ▼
Codificación de variables categóricas
        │
        ▼
Feature Pipeline Persistido
```

El Pipeline fue almacenado mediante Joblib como **feature_pipeline.pkl**, permitiendo reutilizar exactamente las mismas transformaciones durante el entrenamiento, el despliegue mediante la API REST y la aplicación desarrollada en Streamlit.

Esta estrategia fortalece la reproducibilidad del proyecto y evita diferencias entre ambientes de desarrollo y producción.

---

# Modelo de Machine Learning

El problema abordado corresponde a un escenario de **clasificación binaria supervisada**, cuyo objetivo consiste en estimar si un nuevo solicitante realizará oportunamente el pago de su obligación crediticia utilizando información histórica proveniente del sector financiero.

Después de evaluar la naturaleza del problema y la interpretabilidad requerida para un contexto de riesgo crediticio, se seleccionó un modelo de **Regresión Logística**, ampliamente utilizado en la industria financiera debido a su estabilidad, eficiencia computacional y facilidad para interpretar la influencia de las variables predictoras.

El entrenamiento se realizó utilizando Scikit-Learn e integrando directamente el Pipeline de Ingeniería de Características, garantizando que todas las transformaciones aplicadas durante el entrenamiento sean exactamente las mismas utilizadas posteriormente durante la inferencia.

---

# Entrenamiento del Modelo

Una vez construido el Pipeline de transformación, el conjunto de datos fue dividido en entrenamiento y prueba mediante una partición estratificada, preservando la distribución de la variable objetivo.

Posteriormente se ejecutó el siguiente flujo de entrenamiento:

```text
Dataset original
        │
        ▼
Train / Test Split
        │
        ▼
Ajuste del Feature Pipeline
        │
        ▼
Transformación de datos
        │
        ▼
Entrenamiento del modelo
        │
        ▼
Evaluación
        │
        ▼
Persistencia de artefactos
```

Esta estrategia garantiza que el modelo nunca tenga acceso al conjunto de prueba durante el entrenamiento, reduciendo el riesgo de sobreajuste y favoreciendo la capacidad de generalización sobre nuevos datos.

Una vez finalizado el entrenamiento, tanto el modelo como el Pipeline fueron almacenados mediante Joblib para permitir su reutilización durante el despliegue.

---

# Evaluación del Modelo

El desempeño del modelo fue evaluado utilizando un conjunto de datos independiente, empleando diferentes métricas de clasificación para obtener una visión integral de su comportamiento.

## Métricas obtenidas

| Métrica | Resultado |
|----------|----------:|
| Accuracy | **99.81 %** |
| Precision | **99.85 %** |
| Recall | **99.95 %** |
| F1-Score | **99.90 %** |
| ROC AUC | **1.0000** |

---

## Interpretación de Resultados

Los resultados obtenidos evidencian un excelente desempeño del modelo sobre el conjunto de prueba.

El modelo logró clasificar correctamente la gran mayoría de las observaciones, manteniendo un equilibrio adecuado entre precisión y sensibilidad.

La métrica ROC AUC cercana a 1 refleja una elevada capacidad para discriminar entre ambas clases, mientras que el F1-Score confirma la consistencia del modelo al combinar Precision y Recall.

Aunque los resultados obtenidos son altamente satisfactorios, es importante considerar que, en un entorno productivo, el desempeño debe complementarse mediante procesos de validación continua y monitoreo para garantizar que el modelo mantenga su capacidad predictiva frente a nuevos datos.

---

# Persistencia de Artefactos

Con el propósito de garantizar la reproducibilidad del proyecto y facilitar su despliegue, los principales componentes generados durante el entrenamiento fueron almacenados como artefactos persistentes.

Los artefactos generados fueron:

| Artefacto | Propósito |
|------------|-------------|
| **feature_pipeline.pkl** | Pipeline reutilizable encargado de aplicar todas las transformaciones de Ingeniería de Características. |
| **logistic_regression.pkl** | Modelo entrenado utilizado para generar predicciones sobre nuevos solicitantes. |

La persistencia de estos componentes permite desacoplar el entrenamiento del proceso de inferencia, evitando repetir el entrenamiento cada vez que se requieran nuevas predicciones.

---

# API REST con FastAPI

Como parte del enfoque MLOps implementado en este proyecto, se desarrolló una **API REST** utilizando **FastAPI**, permitiendo consumir el modelo mediante solicitudes HTTP.

Durante el inicio de la aplicación, la API carga automáticamente el modelo entrenado y el Pipeline de Ingeniería de Características previamente persistidos.

A partir de ese momento, cualquier solicitud enviada al servicio sigue exactamente el mismo flujo utilizado durante el entrenamiento del modelo.

La API incorpora validación de datos de entrada mediante **Pydantic**, manejo estructurado de excepciones y registro de eventos utilizando **Logging**, fortaleciendo la robustez y mantenibilidad del servicio.

---

## Endpoints Implementados

| Endpoint | Método | Descripción |
|-----------|--------|-------------|
| `/` | GET | Información general del servicio. |
| `/health` | GET | Verifica el estado de la API y la disponibilidad del modelo y del Pipeline. |
| `/predict` | POST | Recibe nuevos registros y genera las predicciones del comportamiento de pago. |

La documentación interactiva de la API se encuentra disponible mediante **Swagger UI**, facilitando las pruebas y el consumo del servicio desde cualquier navegador.

---

# Aplicación Web con Streamlit

Además de la API REST, se desarrolló una interfaz gráfica utilizando **Streamlit**, permitiendo consumir el modelo desde una aplicación web de manera sencilla.

La aplicación reutiliza exactamente el mismo Pipeline y el mismo modelo entrenado utilizados por la API, garantizando consistencia entre ambos mecanismos de inferencia.

Esta interfaz facilita la interacción con el modelo para usuarios no técnicos, permitiendo realizar predicciones sin necesidad de ejecutar código ni consumir directamente los endpoints de la API.

---

# Contenerización con Docker

Con el propósito de garantizar la portabilidad y reproducibilidad del entorno de ejecución, toda la solución fue contenerizada mediante **Docker**.

Se construyó una imagen que incorpora:

- El código fuente del proyecto.
- El modelo entrenado.
- El Pipeline de Ingeniería de Características.
- La API REST desarrollada con FastAPI.
- Todas las dependencias necesarias para ejecutar la solución.

Posteriormente se creó un contenedor encargado de exponer la API mediante el puerto **8000**, permitiendo consumir el servicio desde cualquier equipo con Docker instalado.

Esta estrategia elimina diferencias entre ambientes de desarrollo y producción, simplifica el proceso de despliegue y facilita futuras migraciones hacia servicios Cloud.

El proceso general de despliegue se resume en el siguiente flujo:

```text
Entrenamiento del modelo
        │
        ▼
Persistencia de artefactos
        │
        ▼
Construcción de la API REST
        │
        ▼
Construcción de la imagen Docker
        │
        ▼
Ejecución del contenedor
        │
        ▼
Consumo del servicio mediante HTTP
```

La contenerización constituye uno de los principales componentes de la estrategia MLOps implementada en este proyecto, permitiendo distribuir la solución de manera consistente y reproducible.

# Monitoreo del Modelo

Como complemento al proceso de entrenamiento y despliegue, se implementó un componente de monitoreo orientado a evaluar continuamente la calidad de los datos utilizados por el modelo.

El objetivo principal consiste en identificar posibles inconsistencias antes de ejecutar nuevas predicciones, permitiendo detectar oportunamente problemas que puedan afectar el desempeño del modelo.

Entre las validaciones implementadas se encuentran:

- Verificación de la estructura del conjunto de datos.
- Revisión de dimensiones del dataset.
- Identificación de valores faltantes.
- Detección de registros duplicados.
- Estadísticas descriptivas de variables numéricas.
- Distribución de la variable objetivo.
- Registro de información relevante mediante Logging.

Aunque este componente representa un monitoreo básico, constituye el punto de partida para evolucionar hacia arquitecturas MLOps con monitoreo continuo del desempeño del modelo y de la calidad de los datos.

---

# Flujo General de la Solución

El proyecto integra todas las etapas del ciclo de vida de un modelo de Machine Learning dentro de una arquitectura organizada y reproducible.

```text
Comprensión del problema de negocio
                │
                ▼
Carga y validación del dataset
                │
                ▼
Análisis Exploratorio de Datos
                │
                ▼
Ingeniería de Características
                │
                ▼
Construcción del Feature Pipeline
                │
                ▼
Entrenamiento del Modelo
                │
                ▼
Evaluación
                │
                ▼
Persistencia del Modelo
                │
                ▼
API REST (FastAPI)
                │
                ▼
Aplicación Web (Streamlit)
                │
                ▼
Docker
                │
                ▼
Despliegue
                │
                ▼
Monitoreo
```

Este flujo garantiza que todas las etapas del proyecto permanezcan desacopladas, facilitando el mantenimiento, la reutilización de componentes y la evolución futura de la solución.

---

# Resultados Alcanzados

Durante el desarrollo del proyecto se logró construir una solución completa de Machine Learning siguiendo principios de MLOps.

Entre los principales resultados obtenidos se destacan:

- Desarrollo de un proceso estructurado para la carga y validación de datos.
- Implementación de un Análisis Exploratorio de Datos (EDA) para comprender el comportamiento de las variables y apoyar la toma de decisiones durante el modelado.
- Construcción de un Pipeline reutilizable de Ingeniería de Características utilizando Scikit-Learn.
- Desarrollo de transformadores personalizados para encapsular reglas de negocio dentro del proceso de preprocesamiento.
- Entrenamiento y evaluación de un modelo de Regresión Logística para estimar el comportamiento de pago.
- Persistencia del modelo entrenado y del Feature Pipeline mediante Joblib.
- Implementación de una API REST utilizando FastAPI para consumir el modelo mediante solicitudes HTTP.
- Desarrollo de una aplicación web con Streamlit para facilitar la interacción con el modelo.
- Contenerización de la solución mediante Docker para garantizar portabilidad y reproducibilidad.
- Organización del proyecto bajo una arquitectura modular orientada a buenas prácticas de MLOps.

---

# Conclusiones

El presente proyecto permitió desarrollar una solución integral de Machine Learning para apoyar la estimación del comportamiento de pago de nuevos solicitantes de crédito mediante una arquitectura basada en principios de MLOps.

Desde la perspectiva analítica, el desarrollo del Análisis Exploratorio de Datos permitió comprender la estructura del conjunto de datos, identificar variables relevantes para el problema y establecer criterios técnicos para el diseño del proceso de Ingeniería de Características.

Desde la perspectiva del modelado, la construcción de un Pipeline reutilizable garantizó que las mismas transformaciones aplicadas durante el entrenamiento fueran utilizadas posteriormente durante la inferencia, fortaleciendo la reproducibilidad y reduciendo el riesgo de inconsistencias entre ambientes.

Desde la perspectiva de ingeniería, la implementación de una API REST mediante FastAPI y la contenerización de la aplicación utilizando Docker permitieron desacoplar el modelo del entorno de desarrollo, facilitando su despliegue, reutilización e integración con otros sistemas.

En conjunto, el proyecto demuestra la importancia de combinar técnicas de Ciencia de Datos, Machine Learning e Ingeniería de Software para construir soluciones analíticas escalables, mantenibles y preparadas para evolucionar hacia ambientes productivos.

---

# Trabajo Futuro

Aunque la solución implementada cubre el ciclo completo de desarrollo y despliegue de un modelo de Machine Learning, existen diferentes oportunidades para continuar fortaleciendo la arquitectura.

Las principales líneas de evolución identificadas son:

- Implementar pipelines de Integración y Despliegue Continuo (CI/CD) para automatizar el entrenamiento, validación y publicación de nuevas versiones del modelo.
- Incorporar herramientas especializadas para el versionamiento de modelos, datos y experimentos, como MLflow o DVC, fortaleciendo la trazabilidad de los artefactos generados.
- Implementar monitoreo avanzado mediante indicadores de Data Drift, Concept Drift y seguimiento continuo del desempeño del modelo en producción.
- Incorporar mecanismos de autenticación, autorización y control de acceso para fortalecer la seguridad de la API.
- Desplegar la solución sobre plataformas Cloud utilizando servicios administrados que permitan escalar horizontalmente la aplicación.
- Implementar pruebas automatizadas, monitoreo de disponibilidad y observabilidad para garantizar la confiabilidad del servicio en ambientes productivos.
- Evaluar algoritmos adicionales y estrategias de optimización de hiperparámetros para comparar el desempeño predictivo frente al modelo base desarrollado en este proyecto.

Estas mejoras permitirían evolucionar la solución hacia una arquitectura MLOps más robusta, preparada para escenarios reales de operación y mantenimiento continuo.

---

# Autor

**Yus Rodríguez**

Proyecto Integrador – Bootcamp de Ciencia de Datos

---

# Licencia

Este proyecto fue desarrollado con fines académicos como parte del Proyecto Integrador del Bootcamp de Ciencia de Datos.

Su propósito es demostrar la implementación de un flujo completo de Machine Learning y MLOps, integrando análisis de datos, entrenamiento de modelos, despliegue mediante API REST y contenerización utilizando Docker.