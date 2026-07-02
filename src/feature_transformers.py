"""
==============================================================
Proyecto Integrador - Ciencia de Datos
Predicción del Comportamiento de Pago
Transformadores Personalizados para Feature Engineering
==============================================================

Este módulo implementa los transformadores personalizados utilizados durante la etapa de ingeniería de características del modelo de
Machine Learning para la predicción del comportamiento de pago de nuevos solicitantes de crédito.

Las clases definidas en este archivo forman parte del pipeline de preprocesamiento persistido mediante Joblib. Mantener estos
transformadores en un módulo independiente garantiza su correcta serialización y deserialización desde cualquier componente del
proyecto, evitando dependencias del contexto de ejecución.

Esta arquitectura permite reutilizar exactamente las mismas transformaciones durante el entrenamiento, la evaluación, el
despliegue, la API y la aplicación desarrollada en Streamlit, garantizando consistencia, reproducibilidad y trazabilidad en todo
el flujo MLOps.

Este módulo es reutilizado por los siguientes componentes del proyecto:

- ft_engineering.py
- model_training_evaluation.py
- model_deploy.py
- api.py
- app.py

sin necesidad de redefinir clases ni modificar el pipeline
entrenado.

Autor: Yus Rodriguez
==============================================================
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================
# CONFIGURACIÓN DEL LOGGER
# ==============================================================

logger = logging.getLogger(__name__)

# ==============================================================
# CONSTANTES
# ==============================================================

FECHA_PRESTAMO_FORMATO: str = "%m/%d/%y %H:%M"


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Genera variables temporales a partir de la fecha del préstamo.

    A partir de la columna ``fecha_prestamo`` se generan las variables:

    - anio_prestamo
    - mes_prestamo

    Posteriormente la columna original es eliminada para evitar
    redundancia dentro del conjunto de datos.

    Al formar parte del pipeline persistido, esta transformación se
    ejecuta exactamente de la misma forma durante el entrenamiento y
    la inferencia, garantizando consistencia en todo el proceso de
    Machine Learning.
    """

    def __init__(self, source_column: str = "fecha_prestamo") -> None:
        """
        Inicializa el transformador.

        Parameters
        ----------
        source_column : str, default="fecha_prestamo"
            Nombre de la columna que contiene la fecha del préstamo.
        """
        self.source_column = source_column

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "DateFeatureExtractor":
        """
        Ajusta el transformador.

        Este transformador no aprende parámetros durante el
        entrenamiento, por lo que simplemente retorna la instancia.

        Parameters
        ----------
        X : pd.DataFrame
            Datos de entrenamiento.

        y : pd.Series, optional
            Variable objetivo.

        Returns
        -------
        DateFeatureExtractor
            Instancia del transformador.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Genera las variables temporales derivadas de la fecha del
        préstamo.

        Parameters
        ----------
        X : pd.DataFrame
            Conjunto de datos de entrada.

        Returns
        -------
        pd.DataFrame
            DataFrame transformado.
        """

        X_transformed = X.copy()

        if self.source_column not in X_transformed.columns:
            logger.warning(
                "La columna '%s' no está presente en el DataFrame.",
                self.source_column,
            )
            return X_transformed

        fechas = pd.to_datetime(
            X_transformed[self.source_column],
            format=FECHA_PRESTAMO_FORMATO,
            errors="coerce",
        )

        registros_invalidos = int(fechas.isna().sum())

        if registros_invalidos > 0:
            logger.warning(
                "Se encontraron %d registros con fechas inválidas en '%s'.",
                registros_invalidos,
                self.source_column,
            )

        X_transformed["anio_prestamo"] = fechas.dt.year
        X_transformed["mes_prestamo"] = fechas.dt.month

        X_transformed.drop(
            columns=[self.source_column],
            inplace=True,
        )

        return X_transformed
class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Elimina columnas especificadas del DataFrame cuando están presentes.
    """
  

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()

        columnas_presentes = [
            columna
            for columna in self.columns
            if columna in X_transformed.columns
        ]

        if columnas_presentes:
            logger.info(
                "Columnas eliminadas: %s",
                columnas_presentes,
            )

            X_transformed.drop(
                columns=columnas_presentes,
                inplace=True,
            )

        return X_transformed