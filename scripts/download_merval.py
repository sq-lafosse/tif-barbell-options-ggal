"""download_merval.py — Descarga el Merval en ARS y lo convierte a USD via CCL.

El índice Merval (^MERV) cotiza en pesos argentinos. Para usarlo como benchmark en USD
(ver CLAUDE.md §4, decisión 11), hay que dividirlo por el CCL de cada día.

Dependencia: requiere que `data/raw/ccl/CCL_daily.parquet` exista (producido por
`scripts/download_ccl.py`). Si no existe, el script aborta con exit 1 y un mensaje claro.

Rol en el proyecto (ver CLAUDE.md §5.4b): benchmark de mercado del backtest — representa
al equity argentino en su totalidad, en USD, con el mismo tipo de cambio implícito que
usan los inversores institucionales.

Uso:
    python scripts/download_merval.py
    python scripts/download_merval.py --start 2015-01-01 --end 2026-06-16
    python scripts/download_merval.py --force
    python scripts/download_merval.py --output data/raw/merval/otro_nombre.parquet
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

TICKER = "^MERV"
DEFAULT_START = "2015-01-01"
DEFAULT_OUTPUT_PATH = Path("data/raw/merval/MERVAL_daily.parquet")
DEFAULT_CCL_PATH = Path("data/raw/ccl/CCL_daily.parquet")

OUTPUT_COLUMNS = ["fecha", "merval_ars", "ccl", "merval_usd"]


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
    merval_path = config.get("data", {}).get("merval_path")
    if merval_path:
        return Path(merval_path)
    return DEFAULT_OUTPUT_PATH


def resolve_ccl_path(config: dict) -> Path:
    """Resuelve el path del Parquet del CCL con prioridad config.yaml > default.

    Args:
        config: configuración cargada desde config.yaml (puede estar vacía).

    Returns:
        Path donde se espera encontrar el archivo CCL_daily.parquet.
    """
    ccl_path = config.get("data", {}).get("ccl_path")
    if ccl_path:
        return Path(ccl_path)
    return DEFAULT_CCL_PATH


def download_merval(start: str, end: str) -> pd.DataFrame:
    """Descarga el histórico diario del Merval (^MERV) en ARS desde Yahoo Finance.

    Args:
        start: fecha de inicio en formato "YYYY-MM-DD".
        end:   fecha de fin en formato "YYYY-MM-DD".

    Returns:
        DataFrame con columnas "fecha" (datetime sin timezone) y "merval_ars" (float).

    Raises:
        ValueError: si la descarga falla o Yahoo Finance no devuelve datos.
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
            f"en el rango {start} → {end}. "
            "Nota: ^MERV tiene gaps históricos en Yahoo Finance — si el error persiste, "
            "verificar la disponibilidad del ticker manualmente antes de buscar alternativas."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()[["Date", "Close"]].rename(
        columns={"Date": "fecha", "Close": "merval_ars"}
    )

    # Normalizar a datetime sin timezone para que el merge con el CCL sea limpio.
    df["fecha"] = pd.to_datetime(df["fecha"])
    if df["fecha"].dt.tz is not None:
        df["fecha"] = df["fecha"].dt.tz_convert(None)

    return df


def merge_with_ccl(merval_df: pd.DataFrame, ccl_path: Path) -> pd.DataFrame:
    """Hace inner join del Merval con el CCL y calcula el Merval en USD.

    Args:
        merval_df: DataFrame con columnas "fecha" y "merval_ars".
        ccl_path:  Path al Parquet del CCL generado por download_ccl.py.

    Returns:
        DataFrame con columnas fecha, merval_ars, ccl, merval_usd.
    """
    ccl = pd.read_parquet(ccl_path)[["fecha", "ccl"]]

    df = pd.merge(merval_df, ccl, on="fecha", how="inner")
    df["merval_usd"] = df["merval_ars"] / df["ccl"]

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
        description="Descarga el Merval (^MERV) en ARS y lo convierte a USD via CCL."
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
            "Path de salida del Parquet. Default: data.merval_path en config.yaml "
            f"si existe, sino {DEFAULT_OUTPUT_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Orquesta la descarga y conversión del Merval a USD.

    Returns:
        Código de salida: 0 si terminó OK (incluye el caso idempotente de "ya existe"),
        1 si la descarga falló o si falta el CCL.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

    args = parse_args()
    end = args.end or date.today().isoformat()

    config = load_config()
    output_path = resolve_output_path(config, args.output)
    ccl_path = resolve_ccl_path(config)

    # Dependencia explícita: el CCL debe existir antes de continuar.
    if not ccl_path.exists():
        logger.error(
            "Falta el CCL en %s — correr primero `python scripts/download_ccl.py`.",
            ccl_path,
        )
        return 1

    if output_path.exists() and not args.force:
        logger.warning(
            "El archivo %s ya existe — no se sobrescribe. Usar --force para forzar la descarga.",
            output_path,
        )
        return 0

    try:
        merval_df = download_merval(args.start, end)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    logger.info("Filas descargadas: %d", len(merval_df))

    df = merge_with_ccl(merval_df, ccl_path)

    logger.info("Filas después del inner join con CCL: %d", len(df))

    logger.info(
        "Merval USD — min: %.2f | max: %.2f | media: %.2f",
        df["merval_usd"].min(),
        df["merval_usd"].max(),
        df["merval_usd"].mean(),
    )

    save_parquet(df, output_path)

    logger.info("Rango solicitado: %s → %s", args.start, end)
    logger.info("Primera fecha efectiva: %s", df["fecha"].min().date())
    logger.info("Última fecha efectiva: %s", df["fecha"].max().date())
    logger.info("Archivo guardado en: %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
