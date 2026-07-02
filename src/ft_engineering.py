"""
==============================================================
Prediccion de Comportamiento de Pago - Feature Engineering
==============================================================

Proyecto Integrador - Ciencia de Datos (Sector Financiero)

Este modulo construye y persiste el pipeline de ingenieria de caracteristicas utilizado para preparar los datos de creditos
antes del entrenamiento y la inferencia del modelo de prediccion de pago a tiempo (variable objetivo: Pago_atiempo).

El pipeline resultante es autocontenido: recibe el dataset crudo (tal como lo entrega la fuente oficial) y ejecuta internamente la
extraccion de variables temporales y la exclusion de columnas con riesgo de fuga de informacion, de modo que sea compatible tanto
con el flujo de entrenamiento como con el de despliegue sin requerir preprocesamiento adicional externo.

Autor: Yus Rodriguez
==============================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_transformers import ColumnDropper, DateFeatureExtractor


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


# ==========================================================
# CONFIGURACION DEL PROBLEMA
# ==========================================================

TARGET: str = "Pago_atiempo"

RANDOM_STATE: int = 42

TEST_SIZE: float = 0.20

# Esquema minimo esperado del dataset oficial. Se utiliza para
# validar que el archivo recibido contenga las columnas requeridas
# antes de procesar cualquier informacion.
EXPECTED_COLUMNS: List[str] = [
    "tipo_credito",
    "fecha_prestamo",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
    TARGET,
]

# Columnas excluidas del modelado por riesgo de fuga de informacion.
# 'puntaje' presenta una correlacion de 0.92 con la variable objetivo
# segun el analisis exploratorio (comprension_eda.ipynb), muy por
# encima del resto de variables, lo que sugiere que podria calcularse
# a partir de informacion posterior al evento que se busca predecir.
LEAKAGE_COLUMNS: List[str] = ["puntaje"]

# Reglas de negocio derivadas del EDA, aplicadas unicamente sobre el
# conjunto utilizado para entrenar el pipeline (no se aplican dentro
# del pipeline persistido, para no alterar el numero de registros
# recibidos durante la inferencia).
EDAD_CLIENTE_RANGO: Tuple[int, int] = (18, 100)


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
        raise FileNotFoundError(f"No se encontro el dataset oficial en: {path}")

    df = pd.read_csv(path)

    logger.info(
        "Dataset cargado correctamente (%s filas, %s columnas).",
        df.shape[0],
        df.shape[1],
    )

    return df


# ==========================================================
# VALIDACION DEL DATASET
# ==========================================================

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Valida integridad estructural y de calidad del dataset.

    Verifica columnas obligatorias, dimensiones, tipos de datos,
    duplicados y valores nulos. Elimina duplicados exactos si
    existieran.

    Args:
        df: Dataset a validar.

    Returns:
        DataFrame validado, sin registros duplicados.

    Raises:
        ValueError: Si faltan columnas obligatorias del esquema esperado.
    """
    columnas_faltantes = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias en el dataset: {columnas_faltantes}"
        )

    logger.info("Filas: %s | Columnas: %s", df.shape[0], df.shape[1])
    logger.info("Tipos de datos:\n%s", df.dtypes)

    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        logger.info("Valores nulos por columna:\n%s", nulos)
    else:
        logger.info("No se encontraron valores nulos.")

    duplicados = int(df.duplicated().sum())
    logger.info("Registros duplicados encontrados: %s", duplicados)

    if duplicados > 0:
        df = df.drop_duplicates()
        logger.info("Duplicados eliminados. Filas restantes: %s", df.shape[0])

    return df


# ==========================================================
# REGLAS DE NEGOCIO
# ==========================================================

def apply_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas de negocio derivadas del EDA sobre el dataset.

    Estas reglas se aplican unicamente al conjunto utilizado para
    entrenar el pipeline y el modelo; no se incorporan al pipeline
    persistido, ya que alteran el numero de registros y romperian
    la alineacion entre entradas y predicciones durante la
    inferencia.

    Reglas aplicadas (justificadas en comprension_eda.ipynb):
        - edad_cliente debe estar en el rango [18, 100].
        - capital_prestado debe ser mayor que 0.
        - plazo_meses debe ser mayor que 0.

    Args:
        df: Dataset sobre el cual aplicar las reglas.

    Returns:
        DataFrame filtrado segun las reglas de negocio.
    """
    filas_iniciales = df.shape[0]

    if "edad_cliente" in df.columns:
        edad_min, edad_max = EDAD_CLIENTE_RANGO
        df = df[df["edad_cliente"].between(edad_min, edad_max)]

    if "capital_prestado" in df.columns:
        df = df[df["capital_prestado"] > 0]

    if "plazo_meses" in df.columns:
        df = df[df["plazo_meses"] > 0]

    filas_eliminadas = filas_iniciales - df.shape[0]

    logger.info(
        "Reglas de negocio aplicadas. Registros eliminados: %s (de %s).",
        filas_eliminadas,
        filas_iniciales,
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

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
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
# CONSTRUCCION DEL PIPELINE
# ==========================================================

def build_feature_pipeline() -> Pipeline:
    """Construye el pipeline completo de ingenieria de caracteristicas.

    El pipeline es autocontenido: acepta el dataset crudo (con
    'fecha_prestamo' y todas las columnas originales) y ejecuta
    internamente, en orden:
        1. Extraccion de variables temporales (anio/mes de prestamo).
        2. Exclusion de columnas con riesgo de fuga de informacion.
        3. Imputacion, escalado y codificacion de variables numericas
           y categoricas, seleccionadas dinamicamente por tipo de dato.

    Returns:
        Pipeline de scikit-learn sin ajustar.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=object),
            ),
        ]
    )

    feature_pipeline = Pipeline(
        steps=[
            ("date_features", DateFeatureExtractor()),
            ("drop_leakage_columns", ColumnDropper(LEAKAGE_COLUMNS)),
            ("preprocessor", preprocessor),
        ]
    )

    logger.info("Pipeline de Feature Engineering construido correctamente.")

    return feature_pipeline


# ==========================================================
# ENTRENAMIENTO DEL PIPELINE
# ==========================================================

def train_pipeline(pipeline: Pipeline, X_train: pd.DataFrame) -> Pipeline:
    """Ajusta el pipeline unicamente sobre el conjunto de entrenamiento.

    Args:
        pipeline: Pipeline de Feature Engineering sin ajustar.
        X_train: Variables predictoras de entrenamiento.

    Returns:
        Pipeline ajustado.
    """
    pipeline.fit(X_train)

    logger.info("Pipeline entrenado correctamente sobre el conjunto de entrenamiento.")

    return pipeline


# ==========================================================
# GUARDAR PIPELINE
# ==========================================================

def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Persiste el pipeline ajustado en disco mediante Joblib.

    Args:
        pipeline: Pipeline ajustado a persistir.
        path: Ruta de destino del archivo .pkl.
    """
    joblib.dump(pipeline, path)

    logger.info("Pipeline almacenado en: %s", path)


# ==========================================================
# EJECUCION PRINCIPAL
# ==========================================================

def main() -> None:
    """Ejecuta el proceso completo de Feature Engineering."""
    try:
        logger.info("=" * 50)
        logger.info("INICIANDO FEATURE ENGINEERING")
        logger.info("=" * 50)

        df = load_dataset(DATA_PATH)

        df = validate_dataset(df)
        logger.info("Validacion completada")

        df = apply_business_rules(df)

        logger.info("Aplicando Feature Engineering")

        logger.info("Separando variables")
        X, y = split_features_target(df, TARGET)

        logger.info("Train/Test Split")
        X_train, _, _, _ = split_dataset(X, y)

        logger.info("Construyendo pipeline")
        pipeline = build_feature_pipeline()

        logger.info("Entrenando pipeline")
        pipeline = train_pipeline(pipeline, X_train)

        save_pipeline(pipeline, PIPELINE_PATH)
        logger.info("Pipeline almacenado")

        logger.info("=" * 50)
        logger.info("FEATURE ENGINEERING FINALIZADO")
        logger.info("=" * 50)

    except FileNotFoundError as error:
        logger.error("Archivo no encontrado: %s", error)
        raise
    except ValueError as error:
        logger.error("Error de validacion: %s", error)
        raise
    except Exception as error:
        logger.error("Error inesperado durante el Feature Engineering: %s", error)
        raise


# ==========================================================
# PUNTO DE ENTRADA
# ==========================================================

if __name__ == "__main__":
    main()