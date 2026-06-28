"""fx.py — Conversión ARS↔USD vía CCL para el dataset tidy de opciones.

Convierte las columnas monetarias del dataset de opciones GGAL (nominadas en ARS)
a dólares usando el tipo de cambio implícito (CCL) diario. El CCL proviene de
``data/raw/ccl/CCL_daily.parquet`` (producido por ``scripts/download_ccl.py``).

Decisiones de diseño:
  1. Forward-fill del CCL: cuando hay precio en ARS pero no hay CCL observado
     (feriado en NYSE pero BYMA abierto), se usa el último CCL disponible. Esta
     convención replica el comportamiento del trader real, que habría valorizado
     su posición con el CCL del último día hábil.
  2. El dataset sintético no se convierte: ya está construido en USD (spot=ADR USD,
     strikes y primas calculados en USD por Black-Scholes). Solo se agregan columnas
     de trazabilidad para que el schema sea homogéneo con el dataset real.
  3. El overlap temporal entre sintético (hasta 2023-10-17) y real (desde 2023-08-18)
     no se filtra acá — la decisión de qué dataset usar en ese período le corresponde
     al motor del backtest (``strategy.py``).
  4. El filtro de liquidez (CLAUDE.md §4, decisión 10) no se aplica acá; queda para
     ``strategy.py``, que opera sobre el dataset ya convertido.
  5. ``pct_otm`` del dataset real se calcula contra ``ggal_local_usd`` (el subyacente
     local convertido vía CCL), no contra el ADR — coherente con la Decisión
     metodológica #1 (CLAUDE.md §4): el subyacente de la estrategia es la opción
     local de GGAL, no un derivado sintético sobre el ADR. El ADR se agrega como
     columna ``ggal_adr_usd`` aparte, solo para cross-check/benchmark (CLAUDE.md §5.4a).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_CCL_PATH       = Path("data/raw/ccl/CCL_daily.parquet")
DEFAULT_ADR_PATH       = Path("data/raw/adr/GGAL_ADR_daily.parquet")
DEFAULT_TIDY_PATH      = Path("data/processed/options_tidy.parquet")
DEFAULT_SYNTHETIC_PATH = Path("data/raw/options/SYNTHETIC_2019_2023.parquet")
DEFAULT_OUTPUT_PATH    = Path("data/processed/options_full_usd.parquet")

STRIKE_USD_MAX  = 100.0   # por encima de esto → warning
STRIKE_USD_MIN  = 0.01    # por debajo de esto → warning
FFILL_WARN_DAYS = 7       # run de ffill más largo que esto → warning


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _max_consecutive_true(series: pd.Series) -> int:
    """Retorna la longitud del run más largo de True en una Series booleana."""
    max_run = 0
    current = 0
    for val in series:
        if val:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


# ---------------------------------------------------------------------------
# Carga del CCL
# ---------------------------------------------------------------------------

def load_ccl(ccl_path: Path = DEFAULT_CCL_PATH) -> pd.DataFrame:
    """Lee el Parquet del CCL y devuelve fecha y ccl, ordenado por fecha.

    Args:
        ccl_path: path al Parquet producido por ``scripts/download_ccl.py``.
            Se esperan al menos las columnas ``fecha`` y ``ccl``.

    Returns:
        DataFrame con columnas ``fecha`` (datetime sin timezone) y ``ccl`` (float),
        ordenado por fecha ascendente con índice reseteado.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    ccl_path = Path(ccl_path)
    if not ccl_path.exists():
        raise FileNotFoundError(
            f"No se encontró el CCL en '{ccl_path}'. "
            "Correr primero `python scripts/download_ccl.py`."
        )

    raw = pd.read_parquet(ccl_path)[["fecha", "ccl"]]
    raw["fecha"] = pd.to_datetime(raw["fecha"])
    if raw["fecha"].dt.tz is not None:
        raw["fecha"] = raw["fecha"].dt.tz_convert(None)

    raw = raw.sort_values("fecha").reset_index(drop=True)
    logger.info(
        "CCL cargado: %d filas — %s → %s",
        len(raw), raw["fecha"].min().date(), raw["fecha"].max().date(),
    )
    return raw


# ---------------------------------------------------------------------------
# Carga del ADR (solo para cross-check / benchmark, no para pricing)
# ---------------------------------------------------------------------------

def load_adr(adr_path: Path = DEFAULT_ADR_PATH) -> pd.DataFrame:
    """Lee el Parquet del ADR y devuelve fecha y cierre en USD, ordenado por fecha.

    Args:
        adr_path: path al Parquet producido por ``scripts/download_adr.py``.
            Se esperan al menos las columnas ``fecha`` y ``close``.

    Returns:
        DataFrame con columnas ``fecha`` (datetime sin timezone) y ``ggal_adr_usd``
        (float), ordenado por fecha ascendente con índice reseteado.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    adr_path = Path(adr_path)
    if not adr_path.exists():
        raise FileNotFoundError(
            f"No se encontró el ADR en '{adr_path}'. "
            "Correr primero `python scripts/download_adr.py`."
        )

    raw = pd.read_parquet(adr_path)[["fecha", "close"]].rename(
        columns={"close": "ggal_adr_usd"}
    )
    raw["fecha"] = pd.to_datetime(raw["fecha"])
    if raw["fecha"].dt.tz is not None:
        raw["fecha"] = raw["fecha"].dt.tz_convert(None)

    raw = raw.sort_values("fecha").reset_index(drop=True)
    logger.info(
        "ADR cargado: %d filas — %s → %s",
        len(raw), raw["fecha"].min().date(), raw["fecha"].max().date(),
    )
    return raw


# ---------------------------------------------------------------------------
# Conversión del dataset real (ARS → USD)
# ---------------------------------------------------------------------------

def convert_options_to_usd(
    df_tidy: pd.DataFrame,
    ccl_path: Path = DEFAULT_CCL_PATH,
    adr_path: Path = DEFAULT_ADR_PATH,
    ffill_limit: int | None = None,
) -> pd.DataFrame:
    """Convierte un DataFrame tidy de opciones (ARS) a USD vía CCL.

    Aplica forward-fill al CCL para fechas donde el dataset tiene cotización pero
    no hay CCL observado (ej. feriado en NYSE con BYMA abierto). También calcula
    ``pct_otm`` contra el subyacente local (Decisión metodológica #1) y agrega el
    ADR como columna de cross-check/benchmark.

    Args:
        df_tidy:     output de ``data_loader.load_historical_options``.
        ccl_path:    path al Parquet del CCL.
        adr_path:    path al Parquet del ADR (solo para cross-check, no pricing).
        ffill_limit: máximo de días consecutivos de forward-fill. None = sin límite.
            Loggea warning si algún run supera ``FFILL_WARN_DAYS`` días.

    Returns:
        DataFrame con las columnas originales más:
            - ``ccl_aplicado``:   CCL usado para esa fila (NaN si sin cobertura).
            - ``strike_usd``:     strike / ccl_aplicado.
            - ``prima_usd``:      prima / ccl_aplicado.
            - ``ggal_local_usd``: ggal_local / ccl_aplicado.
            - ``ccl_ffilled``:    True si el CCL fue rellenado por forward-fill.
            - ``pct_otm``:        distancia porcentual strike vs. ggal_local_usd,
                positiva si la opción está OTM (Call: strike > spot; Put: strike < spot).
            - ``ggal_adr_usd``:   cierre del ADR en USD esa fecha (forward-filled),
                solo de referencia — no se usa para pricing ni moneyness.
    """
    ccl_df = load_ccl(ccl_path)

    # Extender el rango hasta el mínimo del CCL para que el ffill tenga de dónde
    # arrastrar valores cuando el tidy empieza con fechas sin CCL observado
    # (ej. fin de semana inmediatamente posterior al último CCL disponible).
    date_min = min(df_tidy["fecha"].min(), ccl_df["fecha"].min())
    date_max = max(df_tidy["fecha"].max(), ccl_df["fecha"].max())
    all_dates = pd.date_range(date_min, date_max, freq="D")

    ccl_series  = ccl_df.set_index("fecha")["ccl"]
    ccl_on_range = ccl_series.reindex(all_dates)                      # NaN donde no hay dato
    ccl_ffilled  = ccl_on_range.ffill(limit=ffill_limit)              # forward-fill

    was_observed  = ccl_on_range.notna()
    is_valid_post = ccl_ffilled.notna()
    ffill_flag    = (~was_observed) & is_valid_post                   # True donde se aplicó ffill

    max_run = _max_consecutive_true(ffill_flag)
    if max_run > FFILL_WARN_DAYS:
        logger.warning(
            "El CCL tiene un forward-fill de %d días consecutivos — verificar gaps en CCL_daily.parquet.",
            max_run,
        )
    elif max_run > 0:
        logger.info("Forward-fill máximo aplicado: %d días consecutivos.", max_run)

    ccl_lookup = pd.DataFrame({
        "fecha":        all_dates,
        "ccl_aplicado": ccl_ffilled.values,
        "ccl_ffilled":  ffill_flag.values,
    })

    df_out = df_tidy.merge(ccl_lookup, on="fecha", how="left")

    # El left-merge puede producir NaN en ccl_ffilled si alguna fecha del tidy no
    # está representada en ccl_lookup (no debería ocurrir dado el rango extendido,
    # pero se fuerza bool estricto para que ~ y .sum() funcionen correctamente).
    df_out["ccl_ffilled"] = df_out["ccl_ffilled"].fillna(False).astype(bool)

    df_out["strike_usd"]     = df_out["strike"]     / df_out["ccl_aplicado"]
    df_out["prima_usd"]      = df_out["prima"]       / df_out["ccl_aplicado"]
    df_out["ggal_local_usd"] = df_out["ggal_local"]  / df_out["ccl_aplicado"]

    is_call = df_out["tipo"] == "Call"
    moneyness_ratio = df_out["strike_usd"] / df_out["ggal_local_usd"]
    df_out["pct_otm"] = np.where(is_call, moneyness_ratio - 1.0, 1.0 - moneyness_ratio)

    adr_df = load_adr(adr_path)
    adr_lookup = adr_df.set_index("fecha")["ggal_adr_usd"].reindex(all_dates).ffill(limit=ffill_limit)
    df_out = df_out.merge(
        pd.DataFrame({"fecha": all_dates, "ggal_adr_usd": adr_lookup.values}),
        on="fecha", how="left",
    )

    n_ffilled  = int(df_out["ccl_ffilled"].sum())
    n_observed = int((~df_out["ccl_ffilled"]).sum())
    n_nan_ccl  = int(df_out["ccl_aplicado"].isna().sum())
    logger.info(
        "CCL aplicado — observado: %d filas, forward-fill: %d filas, sin CCL: %d filas.",
        n_observed, n_ffilled, n_nan_ccl,
    )
    if n_nan_ccl > 0:
        logger.warning(
            "%d filas sin CCL disponible (fechas anteriores al primer CCL o gap sin datos).",
            n_nan_ccl,
        )

    suspicious = df_out["strike_usd"].notna() & (
        (df_out["strike_usd"] > STRIKE_USD_MAX) | (df_out["strike_usd"] < STRIKE_USD_MIN)
    )
    if suspicious.any():
        sample = df_out.loc[suspicious, ["fecha", "strike", "ccl_aplicado", "strike_usd"]].head(5)
        logger.warning(
            "%d filas con strike_usd fuera del rango esperado [%.2f, %.2f]:\n%s",
            int(suspicious.sum()), STRIKE_USD_MIN, STRIKE_USD_MAX,
            sample.to_string(index=False),
        )

    return df_out


# ---------------------------------------------------------------------------
# Conversión del dataset sintético (ya en USD — solo agrega trazabilidad)
# ---------------------------------------------------------------------------

def convert_synthetic_to_usd(df_synthetic: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de trazabilidad USD al dataset sintético sin convertir.

    El sintético ya está en USD (spot=ADR USD, strikes y primas calculados en USD
    por Black-Scholes), por lo que no se aplica CCL. Se agregan las mismas columnas
    de trazabilidad que el dataset real para que el merge posterior sea homogéneo.

    Args:
        df_synthetic: output de ``scripts/generate_synthetic_options.py``.

    Returns:
        DataFrame con las mismas filas más:
            - ``ccl_aplicado``:   NaN (no se aplicó conversión).
            - ``strike_usd``:     igual a ``strike`` (ya en USD).
            - ``prima_usd``:      igual a ``prima`` (ya en USD).
            - ``ggal_local_usd``: NaN (spot fue ADR USD, no precio local ARS).
            - ``ccl_ffilled``:    False.
    """
    df_out = df_synthetic.copy()
    df_out["ccl_aplicado"]   = np.nan
    df_out["strike_usd"]     = df_out["strike"]
    df_out["prima_usd"]      = df_out["prima"]
    df_out["ggal_local_usd"] = np.nan
    df_out["ccl_ffilled"]    = False

    logger.info(
        "Dataset sintético: %d filas — columnas USD de trazabilidad agregadas (sin conversión CCL).",
        len(df_out),
    )
    return df_out


# ---------------------------------------------------------------------------
# Merge final: real + sintético
# ---------------------------------------------------------------------------

def merge_real_and_synthetic(
    df_real_usd: pd.DataFrame,
    df_synthetic_usd: pd.DataFrame,
) -> pd.DataFrame:
    """Concatena el dataset real (post-CCL) y el sintético en un DataFrame unificado.

    Alinea columnas antes de concatenar: si un DataFrame tiene columnas que el otro
    no tiene, se agregan con NaN donde corresponda. El overlap temporal entre los
    dos datasets no se filtra — esa decisión le corresponde al motor del backtest.

    Args:
        df_real_usd:      output de ``convert_options_to_usd``.
        df_synthetic_usd: output de ``convert_synthetic_to_usd``.

    Returns:
        DataFrame concatenado y ordenado por fecha ascendente.
    """
    cols_real = set(df_real_usd.columns)
    cols_syn  = set(df_synthetic_usd.columns)

    only_real = cols_real - cols_syn
    only_syn  = cols_syn  - cols_real

    df_real_out = df_real_usd.copy()
    df_syn_out  = df_synthetic_usd.copy()

    if only_real:
        logger.info(
            "Columnas solo en real (NaN en sintético): %s", sorted(only_real)
        )
        for c in only_real:
            df_syn_out[c] = np.nan

    if only_syn:
        logger.info(
            "Columnas solo en sintético (NaN en real): %s", sorted(only_syn)
        )
        for c in only_syn:
            df_real_out[c] = np.nan

    real_dates = set(df_real_out["fecha"].dt.normalize().unique())
    syn_dates  = set(df_syn_out["fecha"].dt.normalize().unique())
    overlap    = real_dates & syn_dates
    if overlap:
        n_overlap_real = int(df_real_out["fecha"].dt.normalize().isin(overlap).sum())
        n_overlap_syn  = int(df_syn_out["fecha"].dt.normalize().isin(overlap).sum())
        logger.info(
            "Overlap temporal entre datasets: %d fechas — real: %d filas, sintético: %d filas. "
            "No se filtra (decisión del backtest).",
            len(overlap), n_overlap_real, n_overlap_syn,
        )

    df_out = pd.concat([df_real_out, df_syn_out], ignore_index=True)
    df_out = df_out.sort_values("fecha").reset_index(drop=True)

    logger.info(
        "Merge final: %d filas — %s → %s",
        len(df_out),
        df_out["fecha"].min().date(),
        df_out["fecha"].max().date(),
    )
    return df_out


# ---------------------------------------------------------------------------
# CLI standalone (python -m src.fx)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte el dataset tidy de opciones a USD y lo fusiona con el sintético."
    )
    parser.add_argument(
        "--tidy-path",
        default=str(DEFAULT_TIDY_PATH),
        help=f"Path al Parquet tidy de opciones reales (default: {DEFAULT_TIDY_PATH}).",
    )
    parser.add_argument(
        "--synthetic-path",
        default=str(DEFAULT_SYNTHETIC_PATH),
        help=f"Path al Parquet sintético (default: {DEFAULT_SYNTHETIC_PATH}).",
    )
    parser.add_argument(
        "--ccl-path",
        default=str(DEFAULT_CCL_PATH),
        help=f"Path al Parquet del CCL (default: {DEFAULT_CCL_PATH}).",
    )
    parser.add_argument(
        "--adr-path",
        default=str(DEFAULT_ADR_PATH),
        help=f"Path al Parquet del ADR (default: {DEFAULT_ADR_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path de salida del Parquet final (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribir el archivo de salida si ya existe.",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Procesar solo el dataset real (omitir el sintético).",
    )
    return parser.parse_args()


def main() -> int:
    """Orquesta la conversión ARS→USD y el merge con el dataset sintético.

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
        tidy_path = Path(args.tidy_path)
        if not tidy_path.exists():
            logger.error(
                "No se encontró el tidy en '%s'. Correr primero `python -m src.data_loader`.",
                tidy_path,
            )
            return 1

        df_tidy = pd.read_parquet(tidy_path)
        logger.info("Tidy cargado: %d filas.", len(df_tidy))

        df_real_usd = convert_options_to_usd(
            df_tidy, ccl_path=Path(args.ccl_path), adr_path=Path(args.adr_path)
        )

        if args.no_synthetic:
            df_final = df_real_usd
            logger.info("--no-synthetic activo: dataset sintético omitido.")
        else:
            syn_path = Path(args.synthetic_path)
            if not syn_path.exists():
                logger.error(
                    "No se encontró el sintético en '%s'. "
                    "Correr primero `python scripts/generate_synthetic_options.py` "
                    "o usar --no-synthetic.",
                    syn_path,
                )
                return 1

            df_synthetic = pd.read_parquet(syn_path)
            logger.info("Sintético cargado: %d filas.", len(df_synthetic))

            df_synthetic_usd = convert_synthetic_to_usd(df_synthetic)
            df_final = merge_real_and_synthetic(df_real_usd, df_synthetic_usd)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_path, index=False)
        logger.info("Archivo guardado en: %s (%d filas).", output_path, len(df_final))

    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
