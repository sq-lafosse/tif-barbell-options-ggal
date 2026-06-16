"""download_adr.py — Descarga el histórico diario del ADR de GGAL (NYSE) desde Yahoo Finance.

Fuente: Yahoo Finance vía la librería `yfinance`, ticker "GGAL".
Rol en el proyecto (ver CLAUDE.md §5.4a): benchmark idiosincrático del backtest
("ADR GGAL buy-and-hold") y cross-check de la columna PRECIO GGAL de las planillas.

Uso:
    python scripts/download_adr.py
    python scripts/download_adr.py --start 2019-01-01 --end 2026-06-16
    python scripts/download_adr.py --force
    python scripts/download_adr.py --output data/raw/adr/otro_nombre.parquet
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

TICKER = "GGAL"
DEFAULT_START = "2019-01-01"
DEFAULT_OUTPUT_PATH = Path("data/raw/adr/GGAL_ADR_daily.parquet")

# Columnas finales del Parquet, en el orden en que se guardan.
OUTPUT_COLUMNS = ["fecha", "open", "high", "low", "close", "adj_close", "volume"]


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
    adr_path = config.get("data", {}).get("adr_path")
    if adr_path:
        return Path(adr_path)
    return DEFAULT_OUTPUT_PATH


def download_adr(start: str, end: str) -> pd.DataFrame:
    """Descarga el histórico diario del ADR GGAL desde Yahoo Finance.

    Args:
        start: fecha de inicio en formato "YYYY-MM-DD".
        end: fecha de fin en formato "YYYY-MM-DD".

    Returns:
        DataFrame tidy con una fila por día de trading y columnas
        fecha, open, high, low, close, adj_close, volume.

    Raises:
        ValueError: si Yahoo Finance no devuelve datos para el ticker/rango pedido
            (ticker inexistente, sin conexión, rango sin ruedas, etc.).
    """
    logger.info("Descargando %s desde Yahoo Finance: %s → %s", TICKER, start, end)
    raw = yf.download(TICKER, start=start, end=end, auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError(
            f"Yahoo Finance no devolvió datos para el ticker '{TICKER}' "
            f"en el rango {start} → {end}. Verificar conexión a internet o el ticker."
        )

    # yfinance devuelve columnas con MultiIndex cuando se pasa más de un ticker.
    # Acá pedimos uno solo, pero normalizamos por robustez ante cambios de la librería.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index().rename(columns={
        "Date": "fecha",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

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
        description="Descarga el histórico diario del ADR GGAL (NYSE) desde Yahoo Finance."
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
            "Path de salida del Parquet. Default: data.adr_path en config.yaml "
            f"si existe, sino {DEFAULT_OUTPUT_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Orquesta la descarga del ADR GGAL: configuración, descarga, validación y guardado.

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
        df = download_adr(args.start, end)
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
