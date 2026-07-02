"""
==============================================================
Proyecto Integrador - Predicción del Comportamiento de Pago
==============================================================

API REST desarrollada con FastAPI para consumir el modelo de
Machine Learning encargado de predecir el comportamiento de pago
de nuevos solicitantes de crédito.

La API reutiliza el mismo pipeline de Feature Engineering y el
modelo entrenado utilizados durante el entrenamiento y el
despliegue, garantizando consistencia entre todos los componentes
del proyecto.

Autor: Yus Rodriguez
==============================================================
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ==========================================================
# CONFIGURACIÓN DEL LOGGER
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# RUTAS DEL PROYECTO
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model_deploy

# ==========================================================
# APLICACIÓN FASTAPI
# ==========================================================

app = FastAPI(
    title="Predicción del Comportamiento de Pago",
    description=(
        "API REST para apoyar la evaluación del riesgo crediticio "
        "mediante un modelo de Machine Learning."
    ),
    version="1.0.0",
)

# ==========================================================
# CARGA DE ARTEFACTOS
# ==========================================================

try:

    pipeline = model_deploy.load_pipeline()

    model = model_deploy.load_model()

    logger.info("Modelo y pipeline cargados correctamente.")

except Exception as error:

    logger.exception(error)

    pipeline = None

    model = None

# ==========================================================
# MODELO DE ENTRADA
# ==========================================================

class PredictionRequest(BaseModel):
    data: list[dict[str, Any]]

# ==========================================================
# ENDPOINT DE ESTADO
# ==========================================================

@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """
    Verifica el estado de la API y la disponibilidad
    de los artefactos del modelo.
    """

    if pipeline is None or model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo o el pipeline no pudieron cargarse."
            ),
        )

    return {
        "status": "ok",
        "message": (
            "API disponible. Modelo y Feature Pipeline cargados correctamente."
        ),
    }


# ==========================================================
# ENDPOINT DE PREDICCIÓN
# ==========================================================

@app.post("/predict", tags=["Predicción"])
def predict(request: PredictionRequest) -> dict[str, Any]:
    """
    Genera predicciones para nuevos solicitantes de crédito
    utilizando el mismo pipeline empleado durante el
    entrenamiento del modelo.
    """

    if pipeline is None or model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo o el pipeline no están disponibles."
            ),
        )

    if len(request.data) == 0:
        raise HTTPException(
            status_code=400,
            detail="No se recibieron registros para procesar.",
        )

    try:

        df = pd.DataFrame(request.data)

        X = model_deploy.split_features(df)

        predictions = model_deploy.predict(
            model=model,
            pipeline=pipeline,
            X=X,
        )

    except KeyError as error:

        raise HTTPException(
            status_code=400,
            detail=f"Columna faltante: {error}",
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        logger.exception(error)

        raise HTTPException(
            status_code=500,
            detail="Error durante la generación de predicciones.",
        ) from error
            # Validación del esquema esperado utilizando el dataset oficial
        dataset_referencia = pd.read_csv(model_deploy.DATA_PATH)

        columnas_esperadas = [
            c for c in dataset_referencia.columns
            if c != "Pago_atiempo"
        ]

        columnas_faltantes = sorted(
            set(columnas_esperadas) - set(df.columns)
        )

        if columnas_faltantes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Faltan columnas requeridas: "
                    f"{columnas_faltantes}"
                ),
            )

        columnas_extra = sorted(
            set(df.columns) - set(columnas_esperadas)
        )

        return {
            "success": True,
            "message": (
                "Predicciones generadas correctamente."
            ),
            "total_registros": int(len(predictions)),
            "columnas_adicionales": columnas_extra,
            "predicciones": predictions.tolist(),
        }


# ==========================================================
# ENDPOINT RAÍZ
# ==========================================================

@app.get("/", tags=["Inicio"])
def root() -> dict[str, str]:
    """
    Endpoint informativo de la API.
    """

    return {
        "proyecto": (
            "Predicción del Comportamiento de Pago"
        ),
        "descripcion": (
            "API REST para apoyar la evaluación del riesgo "
            "crediticio mediante Machine Learning."
        ),
        "variable_objetivo": "Pago_atiempo",
        "estado": "Activa",
    }


# ==========================================================
# PUNTO DE ENTRADA
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )