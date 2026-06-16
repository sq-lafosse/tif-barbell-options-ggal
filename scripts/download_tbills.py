"""download_tbills.py — Descarga el histórico diario de T-Bills 3M desde Yahoo Finance.

Fuente: Yahoo Finance vía la librería `yfinance`, ticker `^IRX` ("13 Week
Treasury Bill"). Yahoo publica este ticker como yield anualizado expresado en
puntos porcentuales (ej. 4.25 significa 4.25%), misma convención que la serie
DGS3MO de FRED.
Rol en el proyecto (ver CLAUDE.md §5.4c): tasa libre de riesgo del polo seguro
del Barbell y cross-check vs la columna TLR de las planillas de opciones.

Nota técnica — pivote de fuente: este script usó originalmente la API CSV
pública de FRED (`fredgraph.csv`), pero FRED estuvo caído/severamente lento
(504 Gateway Timeout, timeouts en `requests` incluso con timeout=60s). Se pivotó
a Yahoo Finance (`^IRX`) porque ya se usa exitosamente para el ADR de GGAL en
`download_adr.py` y resulta más estable.

Uso:
    python scripts/download_tbills.py
    python scripts/download_tbills.py --start 2015-01-01 --end 2026-06-16
    python scripts/download_tbills.py --force
    python scripts/download_tbills.py --output data/raw/tbills/otro_nombre.parquet
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

TICKER = "^IRX"
DEFAULT_START = "2015-01-01"
DEFAULT_OUTPUT_PATH = Path("data/raw/tbills/TBILLS_3M_daily.parquet")

# Columnas finales del Parquet, en el orden en que se guardan.
OUTPUT_COLUMNS = ["fecha", "tasa_pct", "tasa_decimal"]


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
    tbills_path = config.get("data", {}).get("tbills_path")
    if tbills_path:
        return Path(tbills_path)
    return DEFAULT_OUTPUT_PATH


def download_tbills(start: str, end: str) -> pd.DataFrame:
    """Descarga el histórico diario de T-Bills 3M (^IRX) desde Yahoo Finance.

    Args:
        start: fecha de inicio en formato "YYYY-MM-DD".
        end: fecha de fin en formato "YYYY-MM-DD".

    Returns:
        DataFrame tidy con columnas fecha, tasa_pct, tasa_decimal. tasa_pct es
        el yield anualizado en puntos porcentuales tal como lo publica Yahoo
        Finance (convención estándar para tickers de tasa, ej. 4.25 = 4.25%).
        Puede haber NaN en días sin publicación (feriados que no caen en fin
        de semana) — no se eliminan, el consumidor aguas abajo decide cómo
        manejarlos.

    Raises:
        ValueError: si la descarga falla (timeout, sin conexión) o si Yahoo
            Finance no devuelve datos para el ticker/rango pedido.
    """
    logger.info("Descargando %s desde Yahoo Finance: %s → %s", TICKER, start, end)

    try:
        raw = yf.download(TICKER, start=start, end=end, auto_adjust=False, progress=False)
    except Exception as exc:
        raise ValueError(
            f"Error al descargar el ticker '{TICKER}' desde Yahoo Finance: {exc}. "
            "Verificar conexión a internet."
        ) from exc

    if raw.empty:
        raise ValueError(
            f"Yahoo Finance no devolvió datos para el ticker '{TICKER}' "
            f"en el rango {start} → {end}. Verificar conexión a internet o el ticker."
        )

    # yfinance devuelve columnas con MultiIndex cuando se pasa más de un ticker.
    # Acá pedimos uno solo, pero normalizamos por robustez ante cambios de la librería.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Solo nos interesan fecha y Close (el yield) — Open/High/Low/Volume son
    # ruido para una serie de tasa.
    df = raw.reset_index()[["Date", "Close"]].rename(columns={
        "Date": "fecha",
        "Close": "tasa_pct",
    })
    df["tasa_decimal"] = df["tasa_pct"] / 100

    n_nan = df["tasa_pct"].isna().sum()
    logger.info(
        "Valores NaN en tasa_pct: %d de %d filas (feriados sin publicación en Yahoo Finance)",
        n_nan,
        len(df),
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
        description="Descarga el histórico diario de T-Bills 3M (^IRX) desde Yahoo Finance."
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
            "Path de salida del Parquet. Default: data.tbills_path en config.yaml "
            f"si existe, sino {DEFAULT_OUTPUT_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Orquesta la descarga de T-Bills 3M: configuración, descarga, validación y guardado.

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
        df = download_tbills(args.start, end)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    save_parquet(df, output_path)

    logger.info("Rango solicitado: %s → %s", args.start, end)
    logger.info("Filas descargadas: %d", len(df))
    logger.info("Primera fecha efectiva: %s", df["fecha"].min().date())
    logger.info("Última fecha efectiva: %s", df["fecha"].max().date())
    logger.info("Archivo guardado en: %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
