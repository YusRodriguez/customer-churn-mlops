"""Predicción del Comportamiento de Pago - Entrenamiento y Evaluación del Modelo

Monitorea la calidad de los datos utilizados por el modelo
y genera un reporte básico de monitoreo.
"""

# ============================================================
# IMPORTACIONES
# ============================================================

import logging
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "Base_de_datos.xlsx - Hoja1.csv"

# ============================================================
# CARGAR DATASET
# ============================================================

def load_data():

    logger.info("=" * 60)
    logger.info("CARGANDO DATASET")
    logger.info("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    logger.info(
        f"Dataset cargado correctamente ({df.shape[0]} filas, {df.shape[1]} columnas)"
    )

    return df


# ============================================================
# REVISIÓN GENERAL
# ============================================================

def dataset_overview(df):

    logger.info("=" * 60)
    logger.info("RESUMEN DEL DATASET")
    logger.info("=" * 60)

    logger.info(f"Filas    : {df.shape[0]}")
    logger.info(f"Columnas : {df.shape[1]}")

    logger.info("Tipos de datos:")

    print(df.dtypes)


# ============================================================
# REVISIÓN DE NULOS
# ============================================================

def check_missing_values(df):

    logger.info("=" * 60)
    logger.info("VALORES NULOS")
    logger.info("=" * 60)

    missing = df.isnull().sum()

    print(missing)

    logger.info(f"Total de valores nulos: {missing.sum()}")


# ============================================================
# REVISIÓN DE DUPLICADOS
# ============================================================

def check_duplicates(df):

    logger.info("=" * 60)
    logger.info("REGISTROS DUPLICADOS")
    logger.info("=" * 60)

    duplicates = df.duplicated().sum()

    logger.info(f"Duplicados encontrados: {duplicates}")

    # ============================================================
# DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# ============================================================

def check_target_distribution(df):

    logger.info("=" * 60)
    logger.info("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
    logger.info("=" * 60)

    target = "Pago_atiempo"

    if target in df.columns:

        distribution = df[target].value_counts()

        print(distribution)

        logger.info("Distribución calculada correctamente.")

    else:

        logger.warning(
            f"La columna '{target}' no existe en el dataset."
        )


# ============================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

def descriptive_statistics(df):

    logger.info("=" * 60)
    logger.info("ESTADÍSTICAS DESCRIPTIVAS")
    logger.info("=" * 60)

    print(df.describe(include="all"))


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    logger.info("=" * 60)
    logger.info("INICIANDO MONITOREO DEL MODELO")
    logger.info("=" * 60)

    df = load_data()

    dataset_overview(df)

    check_missing_values(df)

    check_duplicates(df)

    check_target_distribution(df)

    descriptive_statistics(df)

    logger.info("=" * 60)
    logger.info("MONITOREO FINALIZADO")
    logger.info("=" * 60)


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()