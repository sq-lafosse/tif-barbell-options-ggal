"""download_ccl.py — Construye el Contado con Liquidación (CCL) diario a partir de Yahoo Finance.

El CCL (Contado con Liquidación) es el tipo de cambio implícito argentino. Se calcula
a partir de la doble cotización de un mismo activo en dos plazas (ARS local y USD exterior).
No existe un ticker directo del CCL en Yahoo Finance — hay que construirlo.

Fórmula adoptada:
    CCL_t = (GGAL.BA_close_t × ADR_RATIO) / GGAL_ADR_close_t

Donde:
    - GGAL.BA  : acción de Grupo Financiero Galicia en BYMA (ARS), ticker Yahoo "GGAL.BA"
    - GGAL     : ADR de GGAL en NYSE (USD),                        ticker Yahoo "GGAL"
    - ADR_RATIO: 10 (constante histórica — 1 ADR de GGAL = 10 acciones B locales,
                 relación vigente desde la emisión del ADR y sin variación hasta hoy)

Por qué GGAL y no bonos (GD30 / AL30):
    Los bonos emitidos en el canje 2020 no cubren el periodo 2015-2020 del backtest.
    GGAL tiene doble cotización continua desde antes de 2015, lo que permite cubrir el
    rango completo sin interpolaciones ni empalmes.

Cross-check disponible:
    Los archivos Historial de opciones incluyen una columna CCL (disponible desde
    oct-2023). Una vez que data_loader.py esté implementado, se puede contrastar esta
    serie con esa columna para validar la metodología.

Rol en el proyecto (ver CLAUDE.md §5.4b): insumo para convertir el Merval de ARS a USD
(download_merval.py) y para calcular el spot histórico de GGAL en USD (fx.py).

Uso:
    python scripts/download_ccl.py
    python scripts/download_ccl.py --start 2015-01-01 --end 2026-06-16
    python scripts/download_ccl.py --force
    python scripts/download_ccl.py --output data/raw/ccl/otro_nombre.parquet
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

TICKER_LOCAL = "GGAL.BA"  # acción Galicia en BYMA, cotización en ARS
TICKER_ADR = "GGAL"       # ADR Galicia en NYSE, cotización en USD

# Ratio de conversión: 1 ADR de GGAL equivale a 10 acciones B locales.
# Esta relación es una constante histórica desde la emisión del ADR, sin cambios.
ADR_RATIO = 10

DEFAULT_START = "2015-01-01"
DEFAULT_OUTPUT_PATH = Path("data/raw/ccl/CCL_daily.parquet")

OUTPUT_COLUMNS = ["fecha", "ggal_ba_ars", "ggal_adr_usd", "ccl"]

# Límites de sanity check: rangos históricamente imposibles para el CCL argentino.
CCL_MIN_WARN = 1.0
CCL_MAX_WARN = 10_000.0


def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """Carga config.yaml si existe en la raíz del repo.

    Args:
        config_path: ruta al archivo de configuración.

    Returns:
        Diccionario con la configuración cargada, o vacío si el archivo no existe.
    """
    if not config_path.exists():
        logger.warning("No se encontró %s — se usarán valores default", config_path)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_output_path(config: dict, cli_output: str | None) -> Path:
    """Resuelve el path de salida del Parquet con prioridad CLI > config.yaml > default.

    Args:
        config: configuración cargada desde config.yaml (puede estar vacía).
        cli_output: valor del flag --output, si fue provisto por el usuario.

    Returns:
        Path donde se va a guardar el archivo Parquet.
    """
    if cli_output:
        return Path(cli_output)
    ccl_path = config.get("data", {}).get("ccl_path")
    if ccl_path:
        return Path(ccl_path)
    return DEFAULT_OUTPUT_PATH


def _download_close(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Descarga el precio de cierre (Close) de un ticker y lo devuelve como DataFrame tidy.

    Encapsula la lógica de descarga, manejo de MultiIndex y normalización de la columna
    fecha. Usado internamente para descargar GGAL.BA y GGAL por separado.

    Args:
        ticker: ticker de Yahoo Finance a descargar.
        start:  fecha de inicio en formato "YYYY-MM-DD".
        end:    fecha de fin en formato "YYYY-MM-DD".

    Returns:
        DataFrame de dos columnas: "fecha" (datetime sin timezone) y "close" (float).

    Raises:
        ValueError: si la descarga falla o Yahoo Finance no devuelve datos.
    """
    logger.info("Descargando %s desde Yahoo Finance: %s → %s", ticker, start, end)

    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    except Exception as exc:
        raise ValueError(
            f"Error al descargar el ticker '{ticker}' desde Yahoo Finance: {exc}. "
            "Verificar conexión a internet."
        ) from exc

    if raw.empty:
        raise ValueError(
            f"Yahoo Finance no devolvió datos para el ticker '{ticker}' "
            f"en el rango {start} → {end}. Verificar conexión a internet o el ticker."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()[["Date", "Close"]].rename(
        columns={"Date": "fecha", "Close": "close"}
    )

    # Normalizar a datetime sin timezone para que el merge posterior sea limpio.
    df["fecha"] = pd.to_datetime(df["fecha"])
    if df["fecha"].dt.tz is not None:
        df["fecha"] = df["fecha"].dt.tz_convert(None)

    logger.info("  → %d filas descargadas para %s", len(df), ticker)

    return df


def download_ccl(start: str, end: str) -> pd.DataFrame:
    """Construye la serie diaria del CCL a partir de GGAL.BA y el ADR de GGAL.

    Descarga ambas series desde Yahoo Finance, hace inner join por fecha y calcula
    CCL = (GGAL.BA_close × ADR_RATIO) / GGAL_ADR_close. Solo se conservan los días
    con cotización simultánea en ambas plazas.

    Args:
        start: fecha de inicio en formato "YYYY-MM-DD".
        end:   fecha de fin en formato "YYYY-MM-DD".

    Returns:
        DataFrame tidy con columnas fecha, ggal_ba_ars, ggal_adr_usd, ccl.

    Raises:
        ValueError: si cualquiera de las dos descargas falla.
    """
    local_df = _download_close(TICKER_LOCAL, start, end)
    adr_df = _download_close(TICKER_ADR, start, end)

    # Inner join: descarta feriados locales sin cotización en NY y viceversa.
    df = pd.merge(
        local_df.rename(columns={"close": "ggal_ba_ars"}),
        adr_df.rename(columns={"close": "ggal_adr_usd"}),
        on="fecha",
        how="inner",
    )

    df["ccl"] = (df["ggal_ba_ars"] * ADR_RATIO) / df["ggal_adr_usd"]

    logger.info("Filas después del inner join: %d", len(df))

    n_nan = int(df["ccl"].isna().sum())
    logger.info("NaN en CCL: %d (esperado 0 tras inner join)", n_nan)

    logger.info(
        "CCL — min: %.2f | max: %.2f | media: %.2f | mediana: %.2f",
        df["ccl"].min(),
        df["ccl"].max(),
        df["ccl"].mean(),
        df["ccl"].median(),
    )

    if df["ccl"].min() < CCL_MIN_WARN or df["ccl"].max() > CCL_MAX_WARN:
        logger.warning(
            "ATENCIÓN: CCL fuera del rango esperado [%.0f, %.0f]. "
            "min=%.4f, max=%.4f — posible error en los datos de origen.",
            CCL_MIN_WARN,
            CCL_MAX_WARN,
            df["ccl"].min(),
            df["ccl"].max(),
        )

    return df[OUTPUT_COLUMNS]


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el DataFrame en formato Parquet, creando las carpetas necesarias.

    Args:
        df: DataFrame ya normalizado a guardar.
        output_path: ruta de destino del archivo .parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los argumentos start, end, force y output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Construye el CCL diario (ARS/USD) a partir de GGAL.BA y el ADR de GGAL (NYSE)."
        )
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Fecha de inicio en formato YYYY-MM-DD (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Fecha de fin en formato YYYY-MM-DD (default: hoy).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribir el archivo de salida si ya existe.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path de salida del Parquet. Default: data.ccl_path en config.yaml "
            f"si existe, sino {DEFAULT_OUTPUT_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Orquesta la construcción del CCL: descarga, join, cálculo y guardado.

    Returns:
        Código de salida: 0 si terminó OK (incluye el caso idempotente de "ya existe"),
        1 si la descarga falló.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

    args = parse_args()
    end = args.end or date.today().isoformat()

    config = load_config()
    output_path = resolve_output_path(config, args.output)

    if output_path.exists() and not args.force:
        logger.warning(
            "El archivo %s ya existe — no se sobrescribe. Usar --force para forzar la descarga.",
            output_path,
        )
        return 0

    try:
        df = download_ccl(args.start, end)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    save_parquet(df, output_path)

    logger.info("Rango solicitado: %s → %s", args.start, end)
    logger.info("Primera fecha efectiva: %s", df["fecha"].min().date())
    logger.info("Última fecha efectiva: %s", df["fecha"].max().date())
    logger.info("Archivo guardado en: %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
