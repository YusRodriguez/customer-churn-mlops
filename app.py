"""
Aplicación Streamlit para predicción del comportamiento de pago.

La aplicación consume los artefactos existentes del proyecto sin modificar
notebooks, scripts, modelos, datasets ni archivos de configuración.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"

TARGET_COLUMN = "Pago_atiempo"
PREDICTION_COLUMN = "Prediccion"

REQUIRED_COLUMNS = [
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
]


def configure_page() -> None:
    """Configura la pagina principal de Streamlit."""
    st.set_page_config(
        page_title="Prediccion del Comportamiento de Pago",
        page_icon="bar_chart",
        layout="wide",
    )


def import_deployment_module() -> Any:
    """Importa el modulo de despliegue existente del proyecto.

    Returns:
        Modulo ``model_deploy`` ya importado.

    Raises:
        ImportError: Si el modulo no puede importarse desde ``src``.
    """
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import model_deploy  # pylint: disable=import-error,import-outside-toplevel

    return model_deploy


def show_header() -> None:
    """Muestra encabezado, descripcion, objetivo y caso de negocio."""
    st.title("Prediccion del Comportamiento de Pago")
    st.subheader(
        "Sistema de apoyo para evaluacion de riesgo crediticio mediante "
        "Machine Learning."
    )
    st.write(
        "Aplicacion empresarial para evaluar solicitudes de credito mediante "
        "un modelo supervisado entrenado sobre informacion crediticia, "
        "financiera y laboral de solicitantes."
    )
    st.info(
        "Objetivo: generar predicciones de comportamiento de pago para nuevos "
        "registros, reutilizando los artefactos existentes del pipeline MLOps."
    )


def show_sidebar() -> None:
    """Muestra informacion ejecutiva del proyecto en la barra lateral."""
    st.sidebar.title("Proyecto")
    st.sidebar.write("Prediccion del Comportamiento de Pago")
    st.sidebar.write("Version: 1.0.0")
    st.sidebar.write("Autor: Yus Rodriguez")
    st.sidebar.write("Modelo: Regresion Logistica")
    st.sidebar.write("Pipeline: Feature Engineering")
    st.sidebar.write("MLOps: inferencia con artefactos versionados")


def get_file_metadata(path: Path) -> Optional[str]:
    """Obtiene fecha de modificacion de un archivo.

    Args:
        path: Ruta del archivo a inspeccionar.

    Returns:
        Fecha en formato legible o ``None`` si el archivo no existe.
    """
    if not path.exists():
        return None

    modified_at = datetime.fromtimestamp(path.stat().st_mtime)
    return modified_at.strftime("%Y-%m-%d %H:%M:%S")


def check_artifacts() -> Dict[str, bool]:
    """Verifica existencia del modelo y del pipeline."""
    return {
        "modelo": MODEL_PATH.exists(),
        "pipeline": PIPELINE_PATH.exists(),
    }


def show_model_information() -> None:
    """Presenta informacion del modelo y artefactos utilizados."""
    st.header("Informacion del modelo")

    model_date = get_file_metadata(MODEL_PATH)
    pipeline_date = get_file_metadata(PIPELINE_PATH)

    col_model, col_pipeline, col_target = st.columns(3)
    col_model.metric("Modelo utilizado", "Regresion Logistica")
    col_pipeline.metric("Pipeline utilizado", "Feature Engineering")
    col_target.metric("Variable objetivo", TARGET_COLUMN)

    st.write(
        {
            "ruta_modelo": str(MODEL_PATH),
            "ruta_pipeline": str(PIPELINE_PATH),
            "fecha_modelo": model_date or "No disponible",
            "fecha_pipeline": pipeline_date or "No disponible",
        }
    )


def show_artifact_status() -> bool:
    """Muestra estado de artefactos y retorna si estan disponibles."""
    st.header("Verificacion automatica")
    artifacts = check_artifacts()

    if artifacts["modelo"]:
        st.success("Modelo encontrado: models/logistic_regression.pkl")
    else:
        st.error("No se encontro el modelo: models/logistic_regression.pkl")

    if artifacts["pipeline"]:
        st.success("Pipeline encontrado: models/feature_pipeline.pkl")
    else:
        st.error("No se encontro el pipeline: models/feature_pipeline.pkl")

    return all(artifacts.values())


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """Carga el modelo usando la funcion existente de model_deploy.py."""
    model_deploy = import_deployment_module()
    return model_deploy.load_model()


@st.cache_resource(show_spinner=False)
def load_pipeline() -> Any:
    """Carga el pipeline usando la funcion existente de model_deploy.py."""
    model_deploy = import_deployment_module()
    return model_deploy.load_pipeline()


def load_data(uploaded_file: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Carga un CSV suministrado por el usuario.

    Args:
        uploaded_file: Archivo recibido desde ``st.file_uploader``.

    Returns:
        Tupla con DataFrame y mensaje de error opcional.
    """
    if uploaded_file is None:
        return None, "Carga un archivo CSV para iniciar la evaluacion."

    try:
        data = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        return None, "El archivo CSV esta vacio."
    except UnicodeDecodeError:
        return None, "No fue posible leer el CSV por problemas de codificacion."
    except Exception as error:  # pragma: no cover - mensaje para UI
        return None, f"No fue posible leer el archivo CSV: {error}"

    if data.empty:
        return None, "El archivo CSV no contiene registros."

    return data, None


def get_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Obtiene variables predictoras respetando la logica existente.

    Si el archivo incluye la variable objetivo, se utiliza ``split_features``
    de ``model_deploy.py``. Si no la incluye, se asume que son nuevos
    solicitantes y se predice sobre las columnas recibidas.
    """
    if TARGET_COLUMN in data.columns:
        model_deploy = import_deployment_module()
        return model_deploy.split_features(data)

    return data.copy()


def validate_dataset(data: pd.DataFrame) -> Dict[str, Any]:
    """Valida esquema de entrada para prediccion.

    Args:
        data: DataFrame cargado por el usuario.

    Returns:
        Diccionario con columnas faltantes, adicionales y estado del esquema.
    """
    expected = set(REQUIRED_COLUMNS)
    observed = set(data.columns) - {TARGET_COLUMN}

    missing_columns = sorted(expected - observed)
    additional_columns = sorted(observed - expected)

    return {
        "is_valid": not missing_columns,
        "missing_columns": missing_columns,
        "additional_columns": additional_columns,
        "required_columns": REQUIRED_COLUMNS,
    }


def show_data_preview(data: pd.DataFrame) -> None:
    """Muestra dimensiones, vista previa y tipos de datos."""
    st.header("Vista previa del archivo")

    rows, columns = data.shape
    col_rows, col_columns = st.columns(2)
    col_rows.metric("Filas", rows)
    col_columns.metric("Columnas", columns)

    st.subheader("Primeras observaciones")
    st.dataframe(data.head(), use_container_width=True)

    st.subheader("Tipos de datos")
    dtypes = (
        data.dtypes.astype(str)
        .rename("tipo_dato")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    st.dataframe(dtypes, use_container_width=True)


def show_validation(validation: Dict[str, Any]) -> None:
    """Muestra el resultado de validacion del esquema."""
    st.header("Validaciones")

    if validation["is_valid"]:
        st.success("El archivo contiene las columnas obligatorias.")
    else:
        st.error("El archivo no cumple el esquema minimo para prediccion.")

    col_missing, col_extra = st.columns(2)
    with col_missing:
        st.write("Columnas faltantes")
        if validation["missing_columns"]:
            st.dataframe(pd.DataFrame({"columna": validation["missing_columns"]}))
        else:
            st.write("No se detectaron columnas faltantes.")

    with col_extra:
        st.write("Columnas adicionales")
        if validation["additional_columns"]:
            st.dataframe(pd.DataFrame({"columna": validation["additional_columns"]}))
        else:
            st.write("No se detectaron columnas adicionales.")


def predict(
    model: Any,
    pipeline: Any,
    data: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Genera predicciones reutilizando ``model_deploy.predict``.

    Args:
        model: Modelo entrenado.
        pipeline: Pipeline de transformacion.
        data: Dataset validado.

    Returns:
        Tupla con resultados y mensaje de error opcional.
    """
    try:
        model_deploy = import_deployment_module()
        feature_data = get_feature_frame(data)
        predictions = model_deploy.predict(model, pipeline, feature_data)
    except Exception as error:  # pragma: no cover - mensaje para UI
        return None, f"No fue posible generar predicciones: {error}"

    results = data.copy()
    results[PREDICTION_COLUMN] = predictions

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(pipeline.transform(feature_data))[:, 1]
            results["Probabilidad_pago_a_tiempo"] = probabilities
        except Exception:
            pass

    return results, None


def summarize_predictions(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula conteos y porcentajes de predicciones."""
    summary = (
        results[PREDICTION_COLUMN]
        .value_counts(dropna=False)
        .rename_axis(PREDICTION_COLUMN)
        .reset_index(name="conteo")
    )
    summary["porcentaje"] = summary["conteo"] / len(results) * 100
    return summary.sort_values(PREDICTION_COLUMN)


def plot_prediction_distribution(summary: pd.DataFrame) -> None:
    """Grafica conteo y porcentaje de clases predichas."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(summary[PREDICTION_COLUMN].astype(str), summary["conteo"])
    axes[0].set_title("Conteo de clases predichas")
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Registros")

    axes[1].bar(summary[PREDICTION_COLUMN].astype(str), summary["porcentaje"])
    axes[1].set_title("Distribucion porcentual")
    axes[1].set_xlabel("Prediccion")
    axes[1].set_ylabel("Porcentaje")

    plt.tight_layout()
    st.pyplot(fig)


def show_results(results: pd.DataFrame) -> None:
    """Muestra resultados, graficas y descarga de predicciones."""
    st.header("Resultados")

    summary = summarize_predictions(results)
    positive = int((results[PREDICTION_COLUMN] == 1).sum())
    negative = int((results[PREDICTION_COLUMN] == 0).sum())

    col_total, col_positive, col_negative = st.columns(3)
    col_total.metric("Registros evaluados", len(results))
    col_positive.metric("Predicciones positivas", positive)
    col_negative.metric("Predicciones negativas", negative)

    st.subheader("Distribucion de predicciones")
    st.dataframe(summary, use_container_width=True)
    plot_prediction_distribution(summary)

    st.subheader("Tabla de resultados")
    st.dataframe(results, use_container_width=True)

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar predicciones.csv",
        data=csv,
        file_name="predicciones.csv",
        mime="text/csv",
    )


def main() -> None:
    """Ejecuta la aplicacion Streamlit."""
    configure_page()
    show_sidebar()
    show_header()
    show_model_information()

    artifacts_ready = show_artifact_status()
    if not artifacts_ready:
        st.warning(
            "La aplicacion requiere el modelo y el pipeline entrenados para "
            "ejecutar predicciones."
        )
        return

    uploaded_file = st.file_uploader(
        "Carga un archivo CSV con nuevas solicitudes de credito",
        type=["csv"],
    )

    data, load_error = load_data(uploaded_file)
    if load_error:
        st.info(load_error)
        return

    assert data is not None
    show_data_preview(data)

    validation = validate_dataset(data)
    show_validation(validation)

    if not validation["is_valid"]:
        st.warning(
            "Corrige el esquema del archivo antes de generar predicciones."
        )
        return

    if st.button("Generar predicciones", type="primary"):
        with st.spinner("Cargando artefactos y generando predicciones..."):
            try:
                pipeline = load_pipeline()
                model = load_model()
            except Exception as error:  # pragma: no cover - mensaje para UI
                st.error(f"No fue posible cargar los artefactos: {error}")
                return

            results, prediction_error = predict(model, pipeline, data)

        if prediction_error:
            st.error(prediction_error)
            return

        assert results is not None
        show_results(results)


if __name__ == "__main__":
    main()
