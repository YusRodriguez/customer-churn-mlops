"""
Proyecto Integrador - Predicción del Comportamiento de Pago

Carga el pipeline de Feature Engineering y el modelo entrenado
para realizar predicciones sobre nuevos datos.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"

MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"

DATA_PATH = BASE_DIR / "Base_de_datos.xlsx - Hoja1.csv"

# ============================================================
# CARGAR DATOS
# ============================================================

def load_data():

    logger.info("Cargando datos para predicción...")

    df = pd.read_csv(DATA_PATH)

    logger.info(
        f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas"
    )

    return df


# ============================================================
# CARGAR PIPELINE
# ============================================================

def load_pipeline():

    logger.info("Cargando Feature Pipeline...")

    pipeline = joblib.load(PIPELINE_PATH)

    logger.info("Pipeline cargado correctamente.")

    return pipeline


# ============================================================
# CARGAR MODELO
# ============================================================

def load_model():

    logger.info("Cargando modelo entrenado...")

    model = joblib.load(MODEL_PATH)

    logger.info("Modelo cargado correctamente.")

    return model


# ============================================================
# PREPARAR VARIABLES
# ============================================================
def split_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Separa las variables predictoras.

    Si la variable objetivo existe, la elimina.
    Si no existe (caso de la API), devuelve el DataFrame
    sin modificar.
    """

    if "Pago_atiempo" in df.columns:
        return df.drop(columns=["Pago_atiempo"])

    return df.copy()

# ============================================================
# REALIZAR PREDICCIONES
# ============================================================

def predict(model, pipeline, X):

    logger.info("Aplicando Feature Engineering...")

    X_processed = pipeline.transform(X)

    logger.info("Realizando predicciones...")

    predictions = model.predict(X_processed)

    return predictions


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

def save_predictions(df, predictions):

    output_path = BASE_DIR / "predicciones.csv"

    df_result = df.copy()

    df_result["Prediccion"] = predictions

    df_result.to_csv(output_path, index=False)

    logger.info(f"Predicciones almacenadas en:\n{output_path}")


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    logger.info("=" * 60)
    logger.info("INICIANDO DESPLIEGUE DEL MODELO")
    logger.info("=" * 60)

    df = load_data()

    X = split_features(df)

    pipeline = load_pipeline()

    model = load_model()

    predictions = predict(
        model,
        pipeline,
        X,
    )

    save_predictions(
        df,
        predictions,
    )

    logger.info("=" * 60)
    logger.info("DESPLIEGUE FINALIZADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()