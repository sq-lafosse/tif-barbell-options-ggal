"""data_loader.py — Parser unificado de los 17 archivos Historial de opciones GGAL.

Lee los CSV crudos de ``data/raw/options/`` (formato ``GGAL_HIST_YYYY-MM.csv``) y
produce un DataFrame tidy según el esquema de CLAUDE.md §7. Maneja dos esquemas de
columnas que conviven en la serie histórica:

  - Esquema A (20 columnas, 2023-10 → 2025-04): sin griegas. Por defecto, las griegas
    se recalculan con ``src.greeks`` usando la VI del propio archivo (ver CLAUDE.md §5.3).
  - Esquema B (24 columnas, 2025-06 → 2026-06): incluye DELTA/GAMMA/VEGA/THETA
    pre-calculadas por la fuente, que se toman tal como vienen.

Decisiones de diseño:
  1. Las griegas del esquema A se recalculan internamente via ``src.greeks`` cuando
     ``recompute_greeks_scheme_a=True`` (default). Los inputs son el spot local (ARS),
     el strike (ARS), el plazo en años, la TLR y la VI del propio archivo.
  2. La calibración de ``src.greeks`` contra el esquema B (protocolo §5.3) es
     responsabilidad de un test de validación posterior — este módulo solo aplica
     las funciones, no las valida.
  3. Los CSV crudos nunca se modifican. Toda transformación se hace en memoria.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.greeks import (
    black_scholes_price as _bs_price,  # noqa: F401 — importado para verificar disponibilidad
    delta as _bs_delta,
    gamma as _bs_gamma,
    theta as _bs_theta,
    vega as _bs_vega,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SCHEMA_A_NCOLS = 20
SCHEMA_B_NCOLS = 24

SCHEMA_A_COLS = [
    "FECHA", "ESPECIE", "BASE", "TIPO", "ÚLTIMO", "%", "MONTO", "HORA", "APE.",
    "MAX.", "MIN.", "C. ANT.", "NOMINAL", "PRECIO GGAL", "VAR. % GGAL", "TLR",
    "VI %", "VE %", "DÍAS AL VTO.", "PLAZO (años)",
]
SCHEMA_B_EXTRA_COLS = ["DELTA", "GAMMA", "VEGA", "THETA"]

TIDY_COLUMNS = [
    "fecha", "opex", "especie", "tipo", "strike", "prima", "monto", "nominal",
    "ggal_local", "var_ggal", "tlr", "vi_implicita", "valor_extrinseco",
    "dias_vto", "plazo_anios", "delta", "gamma", "vega", "theta",
    "esquema", "fuente_archivo",
]

DEFAULT_RAW_DIR = Path("data/raw/options")
DEFAULT_OUTPUT_PATH = Path("data/processed/options_tidy.parquet")
FILE_PATTERN = "GGAL_HIST_*.csv"

# Extrae "2023-10" de "GGAL_HIST_2023-10.csv"
_OPEX_RE = re.compile(r"GGAL_HIST_(\d{4}-\d{2})\.csv$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Helpers de parseo
# ---------------------------------------------------------------------------

def _parse_arg_decimal(s: str) -> float:
    """Convierte un string decimal en formato argentino a float.

    Args:
        s: string con separador de miles ``.`` y decimal ``,`` (ej. ``"1.234,56"``).
           También acepta valores sin separador de miles (ej. ``"40,03"``).
           Retorna NaN para string vacío, ``"0,000"`` o que no se pueda parsear.

    Returns:
        Float equivalente, o ``np.nan`` si el valor es nulo/vacío.
    """
    if not isinstance(s, str):
        return float(s) if s is not None else np.nan
    s = s.strip()
    if s == "" or s == "0,000":
        return np.nan
    # Formato argentino: puntos como separador de miles, coma como decimal
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_arg_percent(s: str) -> float:
    """Convierte un porcentaje en formato argentino a decimal.

    Args:
        s: string como ``"40,03%"`` o ``"40,03"``.
           Retorna NaN para string vacío o no parseable.

    Returns:
        Decimal equivalente (ej. ``0.4003``), o ``np.nan``.
    """
    if not isinstance(s, str):
        return np.nan
    s = s.strip().rstrip("%")
    return _parse_arg_decimal(s) / 100.0 if s else np.nan


# ---------------------------------------------------------------------------
# Detección de esquema
# ---------------------------------------------------------------------------

def _detect_schema(ncols: int, filepath: str) -> str:
    """Detecta si el archivo pertenece al esquema A (20 cols) o B (24 cols).

    Args:
        ncols:    número de columnas del CSV leído.
        filepath: nombre del archivo (para mensajes de error).

    Returns:
        ``"A"`` o ``"B"``.

    Raises:
        ValueError: si el número de columnas no corresponde a ningún esquema conocido.
    """
    if ncols == SCHEMA_A_NCOLS:
        return "A"
    if ncols == SCHEMA_B_NCOLS:
        return "B"
    raise ValueError(
        f"Archivo '{filepath}' tiene {ncols} columnas. "
        f"Se esperan {SCHEMA_A_NCOLS} (esquema A) o {SCHEMA_B_NCOLS} (esquema B)."
    )


# ---------------------------------------------------------------------------
# Parser de un archivo
# ---------------------------------------------------------------------------

def _parse_file(filepath: Path) -> pd.DataFrame:
    """Parsea un único archivo Historial a formato tidy.

    Args:
        filepath: ruta al CSV original.

    Returns:
        DataFrame con el esquema tidy (sin griegas calculadas para esquema A — NaN).

    Raises:
        ValueError: si el número de columnas es inesperado.
    """
    fname = filepath.name
    match = _OPEX_RE.search(fname)
    opex = match.group(1) if match else fname

    raw = pd.read_csv(
        filepath,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,          # todo como string; parseo manual debajo
        low_memory=False,
    )

    # Limpiar espacios en nombres de columnas (BOM residual, espacios)
    raw.columns = [c.strip() for c in raw.columns]

    schema = _detect_schema(len(raw.columns), fname)
    logger.info("  %s — esquema %s — %d filas", fname, schema, len(raw))

    # Columnas de interés por esquema
    fecha_str = raw["FECHA"].str.strip()
    fecha = pd.to_datetime(fecha_str, format="%d/%m/%Y", errors="coerce")

    especie   = raw["ESPECIE"].str.strip()
    tipo_raw  = raw["TIPO"].str.strip().str.capitalize()
    strike    = raw["BASE"].apply(_parse_arg_decimal)
    prima     = raw["ÚLTIMO"].apply(_parse_arg_decimal)
    monto     = raw["MONTO"].apply(_parse_arg_decimal)
    nominal_s = raw["NOMINAL"].apply(_parse_arg_decimal)
    ggal_local= raw["PRECIO GGAL"].apply(_parse_arg_decimal)
    var_ggal  = raw["VAR. % GGAL"].apply(_parse_arg_percent)
    tlr       = raw["TLR"].apply(_parse_arg_percent)
    vi        = raw["VI %"].apply(_parse_arg_percent)
    ve        = raw["VE %"].apply(_parse_arg_percent)
    dias_vto_s= raw["DÍAS AL VTO."].apply(_parse_arg_decimal)
    plazo     = raw["PLAZO (años)"].apply(_parse_arg_decimal)

    # Griegas: del archivo si esquema B, NaN si esquema A
    if schema == "B":
        g_delta = raw["DELTA"].apply(_parse_arg_decimal)
        g_gamma = raw["GAMMA"].apply(_parse_arg_decimal)
        g_vega  = raw["VEGA"].apply(_parse_arg_decimal)
        g_theta = raw["THETA"].apply(_parse_arg_decimal)
    else:
        nan_col = pd.Series(np.nan, index=raw.index)
        g_delta = g_gamma = g_vega = g_theta = nan_col

    df = pd.DataFrame({
        "fecha":            fecha,
        "opex":             opex,
        "especie":          especie,
        "tipo":             pd.Categorical(tipo_raw, categories=["Call", "Put"]),
        "strike":           strike,
        "prima":            prima,
        "monto":            monto,
        "nominal":          nominal_s,
        "ggal_local":       ggal_local,
        "var_ggal":         var_ggal,
        "tlr":              tlr,
        "vi_implicita":     vi,
        "valor_extrinseco": ve,
        "dias_vto":         dias_vto_s,
        "plazo_anios":      plazo,
        "delta":            g_delta,
        "gamma":            g_gamma,
        "vega":             g_vega,
        "theta":            g_theta,
        "esquema":          pd.Categorical([schema] * len(raw), categories=["A", "B", "SIN"]),
        "fuente_archivo":   fname,
    })

    # Tipos finales
    df["nominal"] = pd.to_numeric(df["nominal"], errors="coerce").astype("Int64")
    df["dias_vto"] = pd.to_numeric(df["dias_vto"], errors="coerce").astype("Int64")

    # Descartar filas-fantasma: líneas con solo separadores (;;;;...) generadas
    # por Excel/Sheets al exportar celdas pre-formateadas pero vacías. Se detectan
    # porque producen fecha=NaT tras el parseo.
    n_antes = len(df)
    df = df.dropna(subset=["fecha"]).reset_index(drop=True)
    n_descartadas = n_antes - len(df)
    if n_descartadas > 0:
        logger.info("  → %d filas-fantasma descartadas (fecha NaT)", n_descartadas)

    return df


# ---------------------------------------------------------------------------
# Recálculo de griegas (esquema A)
# ---------------------------------------------------------------------------

def _recompute_greeks_for_scheme_a(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena delta/gamma/vega/theta para filas del esquema A usando src.greeks.

    Solo procesa filas donde todos los inputs necesarios son válidos (no NaN).
    Las filas con algún input NaN (ej. VI sin cotización) quedan con griega NaN.

    Args:
        df: DataFrame tidy concatenado. Modifica en lugar.

    Returns:
        El mismo DataFrame con las columnas de griegas actualizadas para esquema A.
    """
    mask_a = df["esquema"] == "A"
    if not mask_a.any():
        logger.info("No hay filas del esquema A — sin griegas que recalcular.")
        return df

    sub = df.loc[mask_a].copy()

    S     = sub["ggal_local"].to_numpy(dtype=float)
    K     = sub["strike"].to_numpy(dtype=float)
    T     = sub["plazo_anios"].to_numpy(dtype=float)
    r     = sub["tlr"].to_numpy(dtype=float)
    sigma = sub["vi_implicita"].to_numpy(dtype=float)
    tipo  = sub["tipo"].astype(str).str.lower().to_numpy()

    # Máscara de filas con todos los inputs válidos y S, K > 0
    valid = (
        ~np.isnan(S) & ~np.isnan(K) & ~np.isnan(T) &
        ~np.isnan(r) & ~np.isnan(sigma) &
        (S > 0) & (K > 0) & (T >= 0) & (sigma >= 0)
    )

    n_valid = int(valid.sum())
    n_nan   = int((~valid).sum())
    logger.info(
        "Esquema A — recalculando griegas: %d filas con VI válida, %d quedan NaN.",
        n_valid, n_nan,
    )

    if n_valid == 0:
        return df

    # Separar por tipo para pasar un único option_type a cada llamada
    idx_all  = sub.index[valid]
    tipo_val = tipo[valid]
    S_v, K_v, T_v, r_v, sigma_v = S[valid], K[valid], T[valid], r[valid], sigma[valid]

    is_call = tipo_val == "call"
    is_put  = ~is_call

    delta_arr = np.full(n_valid, np.nan)
    theta_arr = np.full(n_valid, np.nan)

    if is_call.any():
        delta_arr[is_call] = _bs_delta(S_v[is_call], K_v[is_call], T_v[is_call], r_v[is_call], sigma_v[is_call], "call")
        theta_arr[is_call] = _bs_theta(S_v[is_call], K_v[is_call], T_v[is_call], r_v[is_call], sigma_v[is_call], "call")
    if is_put.any():
        delta_arr[is_put]  = _bs_delta(S_v[is_put],  K_v[is_put],  T_v[is_put],  r_v[is_put],  sigma_v[is_put],  "put")
        theta_arr[is_put]  = _bs_theta(S_v[is_put],  K_v[is_put],  T_v[is_put],  r_v[is_put],  sigma_v[is_put],  "put")

    gamma_arr = _bs_gamma(S_v, K_v, T_v, r_v, sigma_v)
    vega_arr  = _bs_vega( S_v, K_v, T_v, r_v, sigma_v)

    df.loc[idx_all, "delta"] = delta_arr
    df.loc[idx_all, "gamma"] = gamma_arr
    df.loc[idx_all, "vega"]  = vega_arr
    df.loc[idx_all, "theta"] = theta_arr

    return df


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def load_historical_options(
    raw_dir: Path = DEFAULT_RAW_DIR,
    pattern: str = FILE_PATTERN,
    recompute_greeks_scheme_a: bool = True,
) -> pd.DataFrame:
    """Lee todos los archivos Historial y devuelve un DataFrame tidy unificado.

    Args:
        raw_dir: directorio con los CSV crudos.
        pattern: glob pattern para encontrar los archivos dentro de ``raw_dir``.
        recompute_greeks_scheme_a: si True, recalcula DELTA/GAMMA/VEGA/THETA
            para los archivos del esquema A usando ``src.greeks`` con la VI del
            propio archivo. Si False, esas columnas quedan NaN.

    Returns:
        DataFrame con el esquema tidy de CLAUDE.md §7. Columnas en el orden
        definido por ``TIDY_COLUMNS``.

    Raises:
        FileNotFoundError: si ``raw_dir`` no existe o no se encuentran archivos.
        ValueError: si algún archivo tiene un número de columnas inesperado.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {raw_dir}")

    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos con patrón '{pattern}' en {raw_dir}"
        )

    logger.info("Procesando %d archivos desde %s", len(files), raw_dir)

    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {"A": 0, "B": 0}

    for fp in files:
        df_file = _parse_file(fp)
        schema_val = df_file["esquema"].iloc[0]
        counts[str(schema_val)] = counts.get(str(schema_val), 0) + 1
        frames.append(df_file)

    df = pd.concat(frames, ignore_index=True)

    logger.info(
        "Archivos procesados — esquema A: %d, esquema B: %d",
        counts.get("A", 0), counts.get("B", 0),
    )
    logger.info("Total de filas concatenadas: %d", len(df))

    if recompute_greeks_scheme_a:
        df = _recompute_greeks_for_scheme_a(df)
    else:
        logger.info("recompute_greeks_scheme_a=False — griegas del esquema A quedan NaN.")

    # Sanity checks (warnings)
    _log_sanity_checks(df)

    return df[TIDY_COLUMNS]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _log_sanity_checks(df: pd.DataFrame) -> None:
    """Loggea estadísticas y advertencias sobre el DataFrame tidy."""
    num_cols = ["prima", "vi_implicita", "delta", "gamma", "vega", "theta"]
    nan_counts = {c: int(df[c].isna().sum()) for c in num_cols if c in df.columns}
    nan_str = ", ".join(f"{c}={n}" for c, n in nan_counts.items())
    logger.info("NaN por columna clave — %s", nan_str)

    logger.info(
        "Rango de fechas: %s → %s",
        df["fecha"].min().date(), df["fecha"].max().date(),
    )
    logger.info("Fechas únicas: %d", df["fecha"].nunique())
    logger.info("Strikes únicos: %d", df["strike"].nunique())
    logger.info("Opex únicos: %d — %s", df["opex"].nunique(), sorted(df["opex"].unique()))


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------

def save_tidy(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Guarda el DataFrame tidy como Parquet, creando la carpeta si no existe.

    Args:
        df:          DataFrame producido por ``load_historical_options``.
        output_path: ruta de destino del ``.parquet``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Archivo guardado en: %s", output_path)


# ---------------------------------------------------------------------------
# CLI standalone (python -m src.data_loader)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parsea los archivos Historial de GGAL y guarda un Parquet tidy."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        help=f"Directorio con los CSV crudos (default: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path de salida del Parquet (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--no-recompute-greeks",
        action="store_true",
        help="No recalcular griegas para archivos del esquema A (quedan NaN).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribir el Parquet de salida si ya existe.",
    )
    return parser.parse_args()


def main() -> int:
    """Punto de entrada CLI: parsea opciones y guarda el Parquet tidy.

    Returns:
        0 si todo OK (incluye caso idempotente), 1 si hubo error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    args = _parse_args()
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        logger.warning(
            "El archivo %s ya existe — no se sobrescribe. Usar --force para forzar.",
            output_path,
        )
        return 0

    try:
        df = load_historical_options(
            raw_dir=Path(args.raw_dir),
            recompute_greeks_scheme_a=not args.no_recompute_greeks,
        )
        save_tidy(df, output_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
