"""generate_synthetic_options.py — Reconstrucción sintética de opciones OTM sobre el ADR de GGAL.

VERSIÓN PRELIMINAR — VI PLANA:
    Este script usa volatilidad realizada (ventana móvil de 30 días) como proxy de la
    volatilidad implícita. Es una simplificación; la versión definitiva requiere calibración
    del skew real contra los archivos Historial 2023-2026 (ver CLAUDE.md §5.5). Una vez que
    data_loader.py esté disponible, reemplazar la VI plana por una superficie de skew calibrada
    (función de moneyness y días al vencimiento).

SUPUESTO DE SPOT EN USD:
    El subyacente es el ADR de GGAL en NYSE (cotización en USD), no la acción local de BYMA
    (ARS). Implica que el inversor modelado operaría opciones sobre el ADR o sus equivalentes
    en USD durante 2019-2023, antes de que los datos de opciones locales estuvieran disponibles.

STRIKES Y PRECIOS EN USD:
    Al usar el ADR como subyacente, strikes y primas quedan en USD. Esto es coherente con el
    objetivo de medir la estrategia Barbell en dólares (ver CLAUDE.md §3 y §4, decisión 11).

COMPATIBILIDAD CON EL FORMATO TIDY (CLAUDE.md §7):
    El output replica las columnas del formato tidy con la adaptación de que `ggal_adr_usd`
    reemplaza a `ggal_local` (spot en USD, no ARS). El campo `esquema = "SIN"` distingue
    estas filas de los archivos Historial reales (esquema A o B). Cuando data_loader.py esté
    implementado, este dataset debe poder concatenarse con el tidy de los archivos Historial.

Uso:
    python scripts/generate_synthetic_options.py
    python scripts/generate_synthetic_options.py --start 2019-01-01 --end 2023-10-17
    python scripts/generate_synthetic_options.py --force
    python scripts/generate_synthetic_options.py --output data/raw/options/otro.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml

from src.greeks import (
    black_scholes_price,
    delta as bs_delta,
    gamma as bs_gamma,
    theta as bs_theta,
    vega as bs_vega,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_START = "2019-01-01"
DEFAULT_END = "2023-10-17"
DEFAULT_OUTPUT_PATH = Path("data/raw/options/SYNTHETIC_2019_2023.parquet")
DEFAULT_ADR_PATH = Path("data/raw/adr/GGAL_ADR_daily.parquet")
DEFAULT_TBILLS_PATH = Path("data/raw/tbills/TBILLS_3M_daily.parquet")

EVEN_MONTHS = [2, 4, 6, 8, 10, 12]
VOL_WINDOW = 30       # días para la ventana de volatilidad realizada
MAX_DIAS_VTO = 90     # máximo días al vencimiento para considerar un opex vigente
PCT_OTM_VALS = [0.10, 0.15, 0.20]
TIPOS = ["Call", "Put"]

FUENTE_ARCHIVO = "SYNTHETIC_2019_2023.parquet"
ESQUEMA = "SIN"

DELTA_CALL_15_MIN, DELTA_CALL_15_MAX = 0.10, 0.35
DELTA_PUT_15_MIN, DELTA_PUT_15_MAX = -0.35, -0.10

OUTPUT_COLUMNS = [
    "fecha", "opex", "especie", "tipo", "strike", "prima",
    "ggal_adr_usd", "tlr", "vi_implicita", "dias_vto", "plazo_anios",
    "delta", "gamma", "vega", "theta", "pct_otm", "esquema", "fuente_archivo",
]


# ---------------------------------------------------------------------------
# Configuración y paths
# ---------------------------------------------------------------------------

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
    """Resuelve el path de salida con prioridad CLI > config.yaml > default.

    Args:
        config:     configuración cargada desde config.yaml (puede estar vacía).
        cli_output: valor del flag --output, si fue provisto por el usuario.

    Returns:
        Path donde se va a guardar el archivo Parquet.
    """
    if cli_output:
        return Path(cli_output)
    out_path = config.get("data", {}).get("synthetic_path")
    if out_path:
        return Path(out_path)
    return DEFAULT_OUTPUT_PATH


def resolve_adr_path(config: dict) -> Path:
    """Resuelve el path del Parquet del ADR con prioridad config.yaml > default.

    Args:
        config: configuración cargada desde config.yaml.

    Returns:
        Path donde se espera el Parquet GGAL_ADR_daily.parquet.
    """
    adr_path = config.get("data", {}).get("adr_path")
    return Path(adr_path) if adr_path else DEFAULT_ADR_PATH


def resolve_tbills_path(config: dict) -> Path:
    """Resuelve el path del Parquet de T-Bills con prioridad config.yaml > default.

    Args:
        config: configuración cargada desde config.yaml.

    Returns:
        Path donde se espera el Parquet TBILLS_3M_daily.parquet.
    """
    tbills_path = config.get("data", {}).get("tbills_path")
    return Path(tbills_path) if tbills_path else DEFAULT_TBILLS_PATH


# ---------------------------------------------------------------------------
# Generación de vencimientos
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    """Calcula el tercer viernes de un mes dado (estándar de vencimientos de opciones).

    Args:
        year:  año (int).
        month: mes (int, 1-12).

    Returns:
        Fecha del tercer viernes del mes.
    """
    first = date(year, month, 1)
    days_to_first_friday = (4 - first.weekday()) % 7  # Friday == 4
    return first + timedelta(days=days_to_first_friday + 14)


def generate_expiries(start: str, end: str) -> list[date]:
    """Genera la grilla de vencimientos (tercer viernes de meses pares) para el periodo.

    Cubre desde el año de inicio hasta el año siguiente al fin, para garantizar que
    las últimas fechas de observación tengan vencimientos vigentes en el futuro.
    La ventana efectiva se controla en `build_active_pairs` con MAX_DIAS_VTO.

    Args:
        start: fecha de inicio "YYYY-MM-DD".
        end:   fecha de fin "YYYY-MM-DD".

    Returns:
        Lista de fechas de vencimiento ordenadas ascendentemente.
    """
    start_year = date.fromisoformat(start).year
    end_year = date.fromisoformat(end).year

    expiries = []
    for year in range(start_year, end_year + 2):
        for month in EVEN_MONTHS:
            expiries.append(_third_friday(year, month))
    return sorted(expiries)


# ---------------------------------------------------------------------------
# Carga y preparación de datos
# ---------------------------------------------------------------------------

def load_market_data(
    adr_path: Path,
    tbills_path: Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Carga y alinea el ADR de GGAL con los T-Bills 3M para el periodo indicado.

    Hace inner join por fecha; forward-fill en T-Bills para cubrir días sin publicación
    oficial (feriados de mercado con spot disponible pero sin yield publicado).
    Filtra al rango [start, end].

    Args:
        adr_path:    path al Parquet del ADR (columnas: fecha, close, ...).
        tbills_path: path al Parquet de T-Bills (columnas: fecha, tasa_decimal, ...).
        start:       fecha de inicio "YYYY-MM-DD".
        end:         fecha de fin "YYYY-MM-DD".

    Returns:
        DataFrame ordenado por fecha con columnas: fecha, ggal_adr_usd, tasa_decimal.

    Raises:
        ValueError: si el dataset resultante está vacío después de filtrar el periodo.
    """
    adr = pd.read_parquet(adr_path)[["fecha", "close"]].rename(
        columns={"close": "ggal_adr_usd"}
    )
    tbills = (
        pd.read_parquet(tbills_path)[["fecha", "tasa_decimal"]]
        .sort_values("fecha")
        .copy()
    )

    logger.info("Filas leídas del ADR: %d", len(adr))
    logger.info("Filas leídas de T-Bills: %d", len(tbills))

    # Forward-fill la tasa: días sin publicación oficial heredan la última tasa conocida.
    tbills["tasa_decimal"] = tbills["tasa_decimal"].ffill()

    df = pd.merge(adr, tbills, on="fecha", how="inner")
    logger.info("Filas tras inner join (ADR ∩ T-Bills): %d", len(df))

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df["fecha"] >= start_ts) & (df["fecha"] <= end_ts)]
    df = df.sort_values("fecha").reset_index(drop=True)

    logger.info("Filas en el periodo %s → %s: %d", start, end, len(df))

    if df.empty:
        raise ValueError(
            f"Sin datos de mercado en el periodo {start} → {end}. "
            "Verificar que los Parquet del ADR y T-Bills cubren este rango."
        )
    return df


def compute_realized_vol(market_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Calcula la VI provisional como volatilidad realizada en ventana de VOL_WINDOW días.

    Los primeros VOL_WINDOW - 1 días no tienen historia suficiente para el rolling std;
    se rellenan hacia atrás (bfill) con el primer valor válido disponible.

    Args:
        market_df: DataFrame con columnas fecha y ggal_adr_usd, ordenado por fecha.

    Returns:
        Tupla (DataFrame con columna vi_implicita agregada, número de fechas rellenadas).
    """
    df = market_df.copy()
    log_ret = np.log(df["ggal_adr_usd"] / df["ggal_adr_usd"].shift(1))
    rv = log_ret.rolling(window=VOL_WINDOW).std() * np.sqrt(252)

    n_backfill = int(rv.isna().sum())
    rv = rv.bfill()  # propaga el primer valor válido hacia los NaN del inicio

    df["vi_implicita"] = rv
    return df, n_backfill


# ---------------------------------------------------------------------------
# Construcción de la grilla de contratos
# ---------------------------------------------------------------------------

def build_active_pairs(market_df: pd.DataFrame, expiries: list[date]) -> pd.DataFrame:
    """Cross join de fechas de observación × vencimientos; filtra los vigentes.

    Un vencimiento es "vigente" en una fecha de observación si:
        0 < (expiry - fecha).days <= MAX_DIAS_VTO

    Esto replica la lógica observada en los archivos Historial, donde cada fecha
    cotiza 1-2 vencimientos simultáneos dentro de un horizonte de ~90 días.

    Args:
        market_df: DataFrame con columna fecha (datetime).
        expiries:  lista de fechas de vencimiento (date).

    Returns:
        DataFrame con columnas: fecha, expiry, opex, dias_vto.
    """
    dates_df = market_df[["fecha"]].assign(_key=1)
    exp_df = pd.DataFrame({
        "expiry": pd.to_datetime(expiries),
        "opex": [e.strftime("%Y-%m") for e in expiries],
        "_key": 1,
    })

    cross = pd.merge(dates_df, exp_df, on="_key").drop(columns="_key")
    cross["dias_vto"] = (cross["expiry"] - cross["fecha"]).dt.days

    active = cross[(cross["dias_vto"] > 0) & (cross["dias_vto"] <= MAX_DIAS_VTO)]
    return active.reset_index(drop=True)


def expand_to_contract_grid(active_pairs: pd.DataFrame) -> pd.DataFrame:
    """Expande los pares (fecha, expiry) a la grilla completa de contratos sintéticos.

    Cada par genera len(PCT_OTM_VALS) × len(TIPOS) = 6 filas:
    3 niveles de OTM (10%, 15%, 20%) × 2 tipos (Call, Put).

    Args:
        active_pairs: DataFrame con columnas fecha, expiry, opex, dias_vto.

    Returns:
        DataFrame con columnas adicionales pct_otm y tipo.
    """
    grid_df = pd.DataFrame([
        {"pct_otm": p, "tipo": t}
        for p in PCT_OTM_VALS
        for t in TIPOS
    ]).assign(_key=1)

    expanded = active_pairs.assign(_key=1).merge(grid_df, on="_key").drop(columns="_key")
    return expanded.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pricing y griegas
# ---------------------------------------------------------------------------

def compute_prices_and_greeks(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega datos de mercado, calcula strikes y aplica Black-Scholes vectorizado.

    Calls y puts se procesan en bloques separados para respetar la firma de src.greeks,
    que recibe un único option_type por llamada. Gamma y vega son independientes del
    tipo y se calculan una sola vez sobre el array completo.

    Args:
        df:        DataFrame con columnas fecha, expiry, opex, dias_vto, pct_otm, tipo.
        market_df: DataFrame con columnas fecha, ggal_adr_usd, tasa_decimal, vi_implicita.

    Returns:
        DataFrame con todas las columnas de output calculadas.
    """
    market_cols = ["fecha", "ggal_adr_usd", "tasa_decimal", "vi_implicita"]
    df = pd.merge(df, market_df[market_cols], on="fecha", how="left")

    df["plazo_anios"] = df["dias_vto"] / 365.0
    df["tlr"] = df["tasa_decimal"]

    S = df["ggal_adr_usd"].values
    r = df["tasa_decimal"].values
    sigma = df["vi_implicita"].values
    T = df["plazo_anios"].values
    pct = df["pct_otm"].values
    is_call = (df["tipo"] == "Call").values

    K = np.where(is_call, S * (1.0 + pct), S * (1.0 - pct))

    gamma_arr = bs_gamma(S, K, T, r, sigma)
    vega_arr = bs_vega(S, K, T, r, sigma)

    prima_arr = np.empty(len(df), dtype=float)
    delta_arr = np.empty(len(df), dtype=float)
    theta_arr = np.empty(len(df), dtype=float)

    for mask, ot in [(is_call, "call"), (~is_call, "put")]:
        if not mask.any():
            continue
        S_m, K_m, T_m, r_m, s_m = S[mask], K[mask], T[mask], r[mask], sigma[mask]
        prima_arr[mask] = black_scholes_price(S_m, K_m, T_m, r_m, s_m, ot)
        delta_arr[mask] = bs_delta(S_m, K_m, T_m, r_m, s_m, ot)
        theta_arr[mask] = bs_theta(S_m, K_m, T_m, r_m, s_m, ot)

    df["strike"] = K
    df["prima"] = prima_arr
    df["delta"] = delta_arr
    df["gamma"] = gamma_arr
    df["vega"] = vega_arr
    df["theta"] = theta_arr

    tipo_code = df["tipo"].map({"Call": "C", "Put": "P"})
    expiry_str = df["expiry"].dt.strftime("%Y-%m-%d")
    strike_str = df["strike"].map("{:.2f}".format)
    df["especie"] = "SYN_GGAL_" + tipo_code + "_" + expiry_str + "_" + strike_str

    df["esquema"] = ESQUEMA
    df["fuente_archivo"] = FUENTE_ARCHIVO

    return df


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(df: pd.DataFrame) -> None:
    """Emite warnings si el dataset presenta anomalías metodológicas.

    Checks:
      1. Más del 5% de calls o puts con prima = 0 → posible problema de VI.
      2. Mediana del delta de calls 15% OTM fuera de [0.10, 0.35] → calibración.
      3. Mediana del delta de puts  15% OTM fuera de [-0.35, -0.10] → calibración.

    Args:
        df: DataFrame de opciones sintéticas con columnas prima, tipo, pct_otm, delta.
    """
    for tipo in TIPOS:
        subset = df[df["tipo"] == tipo]
        n_total = len(subset)
        if n_total == 0:
            continue
        n_zero = int((subset["prima"] == 0.0).sum())
        if n_zero > n_total * 0.05:
            logger.warning(
                "ATENCIÓN: %d filas de %s tienen prima = 0 (%.1f%% del total de %s). "
                "Posible problema con la volatilidad implícita.",
                n_zero, tipo, 100.0 * n_zero / n_total, tipo,
            )

    calls_15 = df[(df["tipo"] == "Call") & (df["pct_otm"] == 0.15)]
    if not calls_15.empty:
        med = float(calls_15["delta"].median())
        if not (DELTA_CALL_15_MIN <= med <= DELTA_CALL_15_MAX):
            logger.warning(
                "ATENCIÓN: mediana delta de calls 15%% OTM = %.4f, "
                "fuera del rango esperado [%.2f, %.2f]. Posible error de calibración.",
                med, DELTA_CALL_15_MIN, DELTA_CALL_15_MAX,
            )

    puts_15 = df[(df["tipo"] == "Put") & (df["pct_otm"] == 0.15)]
    if not puts_15.empty:
        med = float(puts_15["delta"].median())
        if not (DELTA_PUT_15_MIN <= med <= DELTA_PUT_15_MAX):
            logger.warning(
                "ATENCIÓN: mediana delta de puts 15%% OTM = %.4f, "
                "fuera del rango esperado [%.2f, %.2f]. Posible error de calibración.",
                med, DELTA_PUT_15_MIN, DELTA_PUT_15_MAX,
            )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def generate_synthetic_options(
    adr_path: Path,
    tbills_path: Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Orquesta la pipeline completa de generación de opciones sintéticas.

    Pasos:
      1. Carga ADR + T-Bills y hace inner join.
      2. Calcula VI realizada 30d (con backfill al inicio).
      3. Genera grilla de vencimientos (tercer viernes de meses pares).
      4. Cross join fechas × vencimientos vigentes (≤ MAX_DIAS_VTO días).
      5. Expande a 6 contratos por par (3 OTM × 2 tipos).
      6. Aplica Black-Scholes vectorizado para precio y griegas.

    Args:
        adr_path:    path al Parquet del ADR.
        tbills_path: path al Parquet de T-Bills.
        start:       fecha de inicio "YYYY-MM-DD".
        end:         fecha de fin "YYYY-MM-DD".

    Returns:
        DataFrame con el esquema tidy definido en OUTPUT_COLUMNS (CLAUDE.md §7).

    Raises:
        ValueError: si no hay datos de mercado en el periodo indicado.
    """
    market_df = load_market_data(adr_path, tbills_path, start, end)
    market_df, n_backfill = compute_realized_vol(market_df)

    logger.info(
        "Fechas con VI rellenada por backfill (primeros %d días sin historia): %d",
        VOL_WINDOW - 1, n_backfill,
    )

    expiries = generate_expiries(start, end)
    logger.info("Vencimientos candidatos generados: %d", len(expiries))
    logger.info(
        "Rango de vencimientos candidatos: %s → %s",
        expiries[0].isoformat(), expiries[-1].isoformat(),
    )

    active_pairs = build_active_pairs(market_df, expiries)
    logger.info(
        "Pares activos (fecha × vencimiento ≤ %d días): %d",
        MAX_DIAS_VTO, len(active_pairs),
    )

    expanded = expand_to_contract_grid(active_pairs)
    logger.info(
        "Contratos en la grilla (%d niveles OTM × %d tipos): %d",
        len(PCT_OTM_VALS), len(TIPOS), len(expanded),
    )

    df = compute_prices_and_greeks(expanded, market_df)
    return df[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el DataFrame en formato Parquet, creando las carpetas necesarias.

    Args:
        df:          DataFrame normalizado a guardar.
        output_path: ruta de destino del archivo .parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los argumentos start, end, force y output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Genera opciones OTM sintéticas sobre el ADR de GGAL para el tramo 2019-2023, "
            "usando Black-Scholes con volatilidad realizada 30d como VI provisional."
        )
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Fecha de inicio en formato YYYY-MM-DD (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"Fecha de fin en formato YYYY-MM-DD (default: {DEFAULT_END}).",
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
            "Path de salida del Parquet. Default: data.synthetic_path en config.yaml "
            f"si existe, sino {DEFAULT_OUTPUT_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Orquesta la generación de opciones sintéticas: validación, cómputo y guardado.

    Returns:
        Código de salida: 0 si terminó OK (incluye el caso idempotente), 1 si hubo error.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

    args = parse_args()
    config = load_config()
    output_path = resolve_output_path(config, args.output)
    adr_path = resolve_adr_path(config)
    tbills_path = resolve_tbills_path(config)

    # Validación de dependencias antes de cualquier cómputo.
    missing = []
    if not adr_path.exists():
        missing.append((adr_path, "python scripts/download_adr.py"))
    if not tbills_path.exists():
        missing.append((tbills_path, "python scripts/download_tbills.py"))
    if missing:
        for path, cmd in missing:
            logger.error("Falta %s — correr primero `%s`.", path, cmd)
        return 1

    if output_path.exists() and not args.force:
        logger.warning(
            "El archivo %s ya existe — no se sobrescribe. Usar --force para forzar.",
            output_path,
        )
        return 0

    try:
        df = generate_synthetic_options(adr_path, tbills_path, args.start, args.end)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    logger.info("Filas totales generadas: %d", len(df))
    logger.info("Fechas de observación únicas: %d", df["fecha"].nunique())
    logger.info("Vencimientos cubiertos (opex únicos): %d", df["opex"].nunique())

    for tipo in TIPOS:
        subset = df[df["tipo"] == tipo]
        logger.info(
            "Prima %s — min: %.4f | max: %.4f | media: %.4f",
            tipo,
            float(subset["prima"].min()),
            float(subset["prima"].max()),
            float(subset["prima"].mean()),
        )

    run_sanity_checks(df)

    save_parquet(df, output_path)

    logger.info("Rango solicitado: %s → %s", args.start, args.end)
    logger.info("Primera fecha efectiva: %s", df["fecha"].min().date())
    logger.info("Última fecha efectiva:  %s", df["fecha"].max().date())
    logger.info("Archivo guardado en: %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
