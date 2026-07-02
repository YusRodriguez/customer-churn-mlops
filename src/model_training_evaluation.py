"""
==============================================================
Proyecto Integrador - Predicción del Comportamiento de Pago
==============================================================

Proyecto Integrador - Ciencia de Datos

Este script entrena un modelo de Regresion Logistica para la
prediccion de pago a tiempo, utilizando el pipeline de Feature
Engineering previamente generado por ft_engineering.py, y evalua
su desempeno sobre un conjunto de prueba independiente.

Autor: Yus Rodriguez
==============================================================
"""

# ==========================================================
# IMPORTACION DE LIBRERIAS
# ==========================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# CONFIGURACION DEL LOGGER
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# RUTAS DEL PROYECTO
# ==========================================================

BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_PATH: Path = BASE_DIR / "Base_de_datos.xlsx - Hoja1.csv"

MODELS_DIR: Path = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

PIPELINE_PATH: Path = MODELS_DIR / "feature_pipeline.pkl"

MODEL_PATH: Path = MODELS_DIR / "logistic_regression.pkl"


# ==========================================================
# VARIABLE OBJETIVO
# ==========================================================

TARGET: str = "Pago_atiempo"

RANDOM_STATE: int = 42


# ==========================================================
# CARGA DEL DATASET
# ==========================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """Carga el dataset oficial desde un archivo CSV.

    Args:
        path: Ruta del archivo CSV a cargar.

    Returns:
        DataFrame con el dataset cargado.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset oficial en: {path}"
        )

    df = pd.read_csv(path)

    logger.info(
        "Dataset cargado correctamente (%s filas, %s columnas).",
        df.shape[0],
        df.shape[1],
    )

    return df


# ==========================================================
# SEPARACION X / y
# ==========================================================

def split_features_target(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa el dataset en variables predictoras y variable objetivo.

    Args:
        df: Dataset completo.
        target: Nombre de la columna objetivo.

    Returns:
        Tupla (X, y) con las variables predictoras y la variable objetivo.

    Raises:
        ValueError: Si la columna objetivo no existe en el dataset.
    """
    if target not in df.columns:
        raise ValueError(f"No existe la columna objetivo '{target}'.")

    X = df.drop(columns=[target])
    y = df[target]

    logger.info("Variables predictoras: %s | Registros: %s", X.shape[1], X.shape[0])

    return X, y


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

def train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Realiza la particion de los datos en entrenamiento y prueba.

    Args:
        X: Variables predictoras.
        y: Variable objetivo.
        test_size: Proporcion del conjunto de prueba.
        random_state: Semilla de aleatoriedad para reproducibilidad.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Entrenamiento: %s | Prueba: %s", X_train.shape, X_test.shape)

    return X_train, X_test, y_train, y_test


# ==========================================================
# CARGA DEL FEATURE PIPELINE
# ==========================================================

def load_pipeline(path: Path) -> BaseEstimator:
    """Carga el pipeline de Feature Engineering previamente entrenado.

    Args:
        path: Ruta del archivo .pkl del pipeline.

    Returns:
        Objeto transformador ajustado (ColumnTransformer).

    Raises:
        FileNotFoundError: Si el archivo del pipeline no existe.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el Feature Pipeline en: {path}"
        )

    pipeline = joblib.load(path)

    logger.info("Feature Pipeline cargado correctamente.")

    return pipeline


# ==========================================================
# ENTRENAMIENTO DEL MODELO
# ==========================================================

def train_model(
    X_train, y_train, random_state: int = RANDOM_STATE
) -> LogisticRegression:
    """Entrena un modelo de Regresion Logistica.

    Args:
        X_train: Variables predictoras de entrenamiento ya transformadas.
        y_train: Variable objetivo de entrenamiento.
        random_state: Semilla de aleatoriedad para reproducibilidad.

    Returns:
        Modelo de Regresion Logistica entrenado.
    """
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)

    logger.info("Modelo entrenado correctamente.")

    return model


# ==========================================================
# EVALUACION DEL MODELO
# ==========================================================

def evaluate_model(
    model: LogisticRegression, X_test, y_test: pd.Series
) -> dict:
    """Evalua el modelo entrenado sobre el conjunto de prueba.

    Calcula e imprime Accuracy, Precision, Recall, F1 Score, ROC AUC,
    matriz de confusion y reporte de clasificacion.

    Args:
        model: Modelo entrenado.
        X_test: Variables predictoras de prueba ya transformadas.
        y_test: Variable objetivo de prueba.

    Returns:
        Diccionario con las metricas calculadas.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall: %.4f", metrics["recall"])
    logger.info("F1 Score: %.4f", metrics["f1_score"])
    logger.info("ROC AUC: %.4f", metrics["roc_auc"])

    logger.info(
        "Classification Report:\n%s",
        classification_report(y_test, y_pred, zero_division=0),
    )
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    return metrics


# ==========================================================
# GUARDAR MODELO
# ==========================================================

def save_model(model: LogisticRegression, path: Path) -> None:
    """Persiste el modelo entrenado en disco mediante Joblib.

    Args:
        model: Modelo entrenado a persistir.
        path: Ruta de destino del archivo .pkl.
    """
    joblib.dump(model, path)

    logger.info("Modelo almacenado en %s", path)


# ==========================================================
# EJECUCION PRINCIPAL
# ==========================================================

def main() -> None:
    """Ejecuta el proceso completo de entrenamiento y evaluacion."""
    try:
        logger.info("=" * 50)
        logger.info("INICIANDO ENTRENAMIENTO")
        logger.info("=" * 50)

        df = load_dataset(DATA_PATH)

        logger.info("Separando variables")
        X, y = split_features_target(df, TARGET)

        logger.info("Realizando Train/Test Split")
        X_train, X_test, y_train, y_test = train_test(X, y)

        logger.info("Cargando Feature Pipeline")
        pipeline = load_pipeline(PIPELINE_PATH)

        logger.info("Transformando datos")
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        logger.info("Entrenando modelo")
        model = train_model(X_train_transformed, y_train)

        logger.info("Evaluando modelo")
        evaluate_model(model, X_test_transformed, y_test)

        logger.info("Guardando modelo...")
        save_model(model, MODEL_PATH)

        logger.info("=" * 50)
        logger.info("ENTRENAMIENTO FINALIZADO")
        logger.info("=" * 50)

    except FileNotFoundError as error:
        logger.error("Archivo no encontrado: %s", error)
        raise
    except ValueError as error:
        logger.error("Error de validacion: %s", error)
        raise
    except Exception as error:
        logger.error("Error inesperado durante el entrenamiento: %s", error)
        raise


# ==========================================================
# PUNTO DE ENTRADA
# ==========================================================

if __name__ == "__main__":
    main()