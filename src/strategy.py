"""strategy.py — Lógica de la Barbell: compra sistemática de Puts OTM sobre GGAL.

Construye el ledger de trades del polo agresivo (10% del capital, Decisión metodológica
del tutor: T-Bills + Puts OTM, no Calls — la Barbell apuesta a la magnitud de una caída,
no a la suba). El polo seguro (T-Bills) y la composición de ambos polos en una curva de
equity diaria son responsabilidad de ``src/backtest.py`` (no implementado todavía).

Decisiones de diseño (ver plan de implementación / CLAUDE.md):
  1. Rebalanceo event-driven: se compra un Put nuevo apenas vence el anterior. La fuente
     de verdad es ``dias_vto`` de cada fila, no la etiqueta ``opex`` del archivo (que
     identifica el ciclo del archivo, no el vencimiento exacto de cada contrato).
  2. Overlap sintético/real (2023-08-18 → 2023-10-17, CLAUDE.md §5.5): se prefieren datos
     reales (esquema A o B) sobre el sintético (esquema SIN) cuando ambos están disponibles
     la misma fecha — ``esquema_priority`` en config.yaml.
  3. Selección de contrato: entre los Puts con ``dias_vto >= min_dias_vto``, se toma el
     vencimiento más próximo y, dentro de ese vencimiento, el strike con ``pct_otm`` más
     cercano al target de ``config.yaml: moneyness.otm_pct``.
  4. Costos de transacción (Decisión #10): se paga "ask" = prima_usd × (1 + spread/2) al
     entrar; el spread se clasifica por ``monto`` (ARS) vs. el umbral de liquidez. El
     dataset sintético no tiene ``monto``/``nominal``, así que esas filas caen siempre en
     el spread de bajo volumen (supuesto conservador a documentar en la tesis). Al
     vencimiento no se aplica spread: la liquidación es por valor intrínseco, no es una
     operación de mercado.
  5. Spot unificado: ``ggal_local_usd`` en el tramo real, ``ggal_adr_usd`` en el sintético
     (cada uno es el subyacente con el que fue construido ese tramo — Decisión #1).
  6. Unidad económica: el ledger se expresa en USD por contrato/unidad de prima, no en
     lotes discretos — simplificación estándar dado que el dataset no tiene profundidad
     de book para simular tamaños de orden reales.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS_PATH = Path("data/processed/options_full_usd.parquet")
DEFAULT_CONFIG_PATH  = Path("config.yaml")
DEFAULT_OUTPUT_PATH  = Path("data/processed/barbell_trades.parquet")


# ---------------------------------------------------------------------------
# Spot unificado por régimen (Decisión #1 / regla #5)
# ---------------------------------------------------------------------------

def _daily_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Colapsa una columna de precio a una serie de un valor por fecha.

    Args:
        df:     DataFrame con columnas ``fecha`` y ``column``.
        column: nombre de la columna de precio a colapsar.

    Returns:
        Series indexada por fecha (``DatetimeIndex``), ordenada ascendentemente,
        apta para ``.asof()``.
    """
    daily = pd.Series(df[column].values, index=pd.to_datetime(df["fecha"].values))
    daily = daily.groupby(level=0).first().sort_index()
    daily.index.name = "fecha"
    return daily


def get_daily_spot(df: pd.DataFrame) -> pd.Series:
    """Devuelve el spot en USD por fecha, unificando el subyacente según el régimen.

    En el tramo real, el subyacente es ``ggal_local_usd``; en el sintético (donde
    ``ggal_local_usd`` es NaN por construcción) se usa ``ggal_adr_usd``. Todas las filas
    de una misma fecha comparten el mismo spot, así que se colapsa a una serie por fecha.

    ``ggal_local_usd`` (precio de 1 acción local) y ``ggal_adr_usd`` (precio del ADR,
    que representa 10 acciones locales — CLAUDE.md §5.4.b) están en bases distintas.
    Esta serie combinada es apta para mostrar el spot de un día puntual, pero NO debe
    usarse para comparar dos fechas de un mismo trade (entrada vs. vencimiento) si el
    trade pudo haber cruzado de régimen — para eso, ver ``_daily_series`` por separado
    en ``build_barbell_trades``.

    Args:
        df: DataFrame con columnas ``fecha``, ``ggal_local_usd``, ``ggal_adr_usd``.

    Returns:
        Series de spot en USD indexada por fecha (``DatetimeIndex``), ordenada
        ascendentemente, apta para ``.asof()``.
    """
    spot = df["ggal_local_usd"].combine_first(df["ggal_adr_usd"])
    daily = pd.Series(spot.values, index=pd.to_datetime(df["fecha"].values))
    daily = daily.groupby(level=0).first().sort_index()
    daily.index.name = "fecha"
    return daily


# ---------------------------------------------------------------------------
# Costos de transacción por liquidez (Decisión #10 / regla #4)
# ---------------------------------------------------------------------------

def classify_spread(monto: pd.Series, config: dict) -> pd.Series:
    """Clasifica el spread bid-ask de cada fila según el volumen operado (``monto``).

    ``monto`` faltante (NaN) — el caso del dataset sintético, que no tiene esa columna —
    cae en el spread de bajo volumen por construcción: una comparación ``NaN >= umbral``
    es ``False``, así que nunca clasifica como alto volumen.

    Args:
        monto:  volumen operado en ARS nominales (columna ``monto`` del tidy).
        config: dict cargado de ``config.yaml``; usa la sección ``transaction_costs``.

    Returns:
        Series de spread (decimal, ej. 0.015 = 1.5%) alineada al índice de ``monto``.
    """
    tc = config["transaction_costs"]
    is_high_volume = monto >= tc["volume_threshold"]
    return pd.Series(
        np.where(is_high_volume, tc["high_volume_spread"], tc["low_volume_spread"]),
        index=monto.index,
    )


# ---------------------------------------------------------------------------
# Selección de contrato (regla #3)
# ---------------------------------------------------------------------------

def select_put_contract(
    df_date: pd.DataFrame,
    target_otm_pct: float,
    min_dias_vto: int,
) -> pd.Series | None:
    """Elige el Put a comprar entre los candidatos de una fecha de entrada.

    Filtra por ``dias_vto >= min_dias_vto`` (evita comprar algo a punto de vencer) y por
    ``prima_usd`` no nulo (contratos sin operaciones ese día no tienen precio observable
    y no son comprables), toma el vencimiento más próximo entre los que pasan el filtro,
    y dentro de ese vencimiento el strike con ``pct_otm`` más cercano al target.

    Args:
        df_date:         filas de Puts de una sola fecha (ya filtradas a un esquema).
        target_otm_pct:  moneyness objetivo (``config.yaml: moneyness.otm_pct``).
        min_dias_vto:    piso de días al vencimiento para considerar un contrato.

    Returns:
        La fila elegida (``pd.Series``), o ``None`` si no hay candidatos válidos.
    """
    candidates = df_date[
        (df_date["dias_vto"] >= min_dias_vto) & df_date["prima_usd"].notna()
    ]
    if candidates.empty:
        return None

    nearest_expiry = candidates["dias_vto"].min()
    nearest_group = candidates[candidates["dias_vto"] == nearest_expiry]
    idx = (nearest_group["pct_otm"] - target_otm_pct).abs().idxmin()
    return nearest_group.loc[idx]


# ---------------------------------------------------------------------------
# Resolución de filas de entrada (overlap sintético/real, regla #2)
# ---------------------------------------------------------------------------

def resolve_entry_rows(
    df: pd.DataFrame,
    date: pd.Timestamp,
    esquema_priority: list[str],
    max_gap_dias: int = 5,
) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """Busca filas de Puts utilizables para entrar en una fecha, con prioridad de esquema.

    Si no hay Puts exactamente en ``date``, busca hacia adelante hasta ``max_gap_dias``
    días (gap de datos — fin de semana, feriado, archivo faltante). Dentro de la primera
    fecha con datos, prioriza el esquema según ``esquema_priority`` (reales antes que
    sintético, ver CLAUDE.md §5.5).

    Args:
        df:               dataset completo de opciones (``options_full_usd.parquet``).
        date:             fecha de entrada deseada.
        esquema_priority: orden de preferencia de esquemas, ej. ``["A", "B", "SIN"]``.
        max_gap_dias:     máximo de días hacia adelante a buscar si falta la fecha exacta.

    Returns:
        Tupla ``(fecha_resuelta, filas)``. Si no se encontró nada dentro del rango,
        devuelve ``(None, df_vacío)``.
    """
    puts = df[df["tipo"] == "Put"]
    for offset in range(max_gap_dias + 1):
        probe_date = date + pd.Timedelta(days=offset)
        day_rows = puts[puts["fecha"] == probe_date]
        if day_rows.empty:
            continue
        for esquema in esquema_priority:
            subset = day_rows[day_rows["esquema"] == esquema]
            if not subset.empty:
                if offset > 0:
                    logger.warning(
                        "Sin Puts en %s — se usa %s (gap de %d días, esquema %s).",
                        date.date(), probe_date.date(), offset, esquema,
                    )
                return probe_date, subset

    logger.warning(
        "Sin Puts disponibles entre %s y %s (gap > %d días) — ciclo omitido.",
        date.date(), (date + pd.Timedelta(days=max_gap_dias)).date(), max_gap_dias,
    )
    return None, df.iloc[0:0]


# ---------------------------------------------------------------------------
# Ledger de trades (motor event-driven, regla #1)
# ---------------------------------------------------------------------------

def build_barbell_trades(
    df: pd.DataFrame,
    config: dict,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Construye el ledger completo de trades del polo agresivo (compra de Puts OTM).

    Loop event-driven: en cada ciclo se entra en la fecha actual, se mantiene el
    contrato hasta su vencimiento, se liquida por valor intrínseco, y el día siguiente
    al vencimiento arranca el próximo ciclo. No hay rebalanceo de capital entre el polo
    seguro y el agresivo acá — eso es ``backtest.py``; este ledger está expresado en
    USD por unidad de prima (Decisión de diseño #6).

    Args:
        df:         dataset completo de opciones (``options_full_usd.parquet``).
        config:     dict cargado de ``config.yaml``.
        start_date: fecha de inicio del backtest.
        end_date:   fecha de fin del backtest.

    Returns:
        DataFrame con una fila por ciclo de Put comprado, columnas:
        ``entry_date, expiry_date, esquema, especie, strike_usd, spot_entry_usd,
        pct_otm_real, prima_usd, spread_pct, entry_cost_usd, spot_expiry_usd,
        payoff_usd, retorno_premium``. Vacío si no se encontró ningún contrato válido.
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    target_otm_pct = config["moneyness"]["otm_pct"]
    rebalance_cfg = config["rebalance"]
    min_dias_vto = rebalance_cfg["min_dias_vto"]
    max_gap_dias = rebalance_cfg["max_gap_dias"]
    esquema_priority = rebalance_cfg["esquema_priority"]

    daily_spot = get_daily_spot(df)
    local_spot = _daily_series(df, "ggal_local_usd")
    adr_spot = _daily_series(df, "ggal_adr_usd")
    df["spread_pct"] = classify_spread(df["monto"], config)

    trades = []
    current_date = start_date

    while current_date <= end_date:
        entry_date, day_rows = resolve_entry_rows(
            df, current_date, esquema_priority, max_gap_dias
        )
        if entry_date is None:
            current_date = current_date + pd.Timedelta(days=max_gap_dias + 1)
            continue

        contract = select_put_contract(day_rows, target_otm_pct, min_dias_vto)
        if contract is None:
            logger.warning(
                "Sin contrato Put válido en %s (min_dias_vto=%d) — ciclo omitido.",
                entry_date.date(), min_dias_vto,
            )
            current_date = entry_date + pd.Timedelta(days=1)
            continue

        expiry_date = entry_date + pd.Timedelta(days=int(contract["dias_vto"]))
        if expiry_date > end_date:
            break

        spread = contract["spread_pct"]
        entry_cost_usd = contract["prima_usd"] * (1.0 + spread / 2.0)

        # `strike_usd` está en la base del régimen de ENTRADA (1 acción local para
        # esquema A/B, ADR para esquema SIN — bases distintas, ratio ~10x entre sí,
        # CLAUDE.md §5.4.b). El spot al vencimiento debe leerse de la MISMA base,
        # sin importar qué régimen tenga datos disponibles ese día — de lo contrario
        # un trade que entra en sintético y vence ya en el tramo real comparа un
        # strike "ADR" contra un spot "1 acción", inflando el payoff ~10x.
        regime_spot = adr_spot if contract["esquema"] == "SIN" else local_spot
        spot_expiry = regime_spot.asof(expiry_date)
        if pd.isna(spot_expiry):
            spot_expiry = daily_spot.asof(expiry_date)
        payoff_usd = max(contract["strike_usd"] - spot_expiry, 0.0)
        retorno_premium = (
            (payoff_usd - entry_cost_usd) / entry_cost_usd
            if entry_cost_usd > 0 else np.nan
        )

        trades.append({
            "entry_date":      entry_date,
            "expiry_date":     expiry_date,
            "esquema":         contract["esquema"],
            "especie":         contract["especie"],
            "strike_usd":      contract["strike_usd"],
            "spot_entry_usd":  daily_spot.asof(entry_date),
            "pct_otm_real":    contract["pct_otm"],
            "prima_usd":       contract["prima_usd"],
            "spread_pct":      spread,
            "entry_cost_usd":  entry_cost_usd,
            "spot_expiry_usd": spot_expiry,
            "payoff_usd":      payoff_usd,
            "retorno_premium": retorno_premium,
        })

        current_date = expiry_date + pd.Timedelta(days=1)

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# CLI standalone (python -m src.strategy)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el ledger de trades de la Barbell (Puts OTM sobre GGAL)."
    )
    parser.add_argument(
        "--options-path",
        default=str(DEFAULT_OPTIONS_PATH),
        help=f"Path al Parquet de opciones en USD (default: {DEFAULT_OPTIONS_PATH}).",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path a config.yaml (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path de salida del ledger (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribir el archivo de salida si ya existe.",
    )
    return parser.parse_args()


def main() -> int:
    """Orquesta la construcción del ledger de trades desde la línea de comandos.

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

    options_path = Path(args.options_path)
    if not options_path.exists():
        logger.error(
            "No se encontró el dataset de opciones en '%s'. "
            "Correr primero `python -m src.fx`.",
            options_path,
        )
        return 1

    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.error("No se encontró config.yaml en '%s'.", config_path)
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = pd.read_parquet(options_path)
    logger.info("Opciones cargadas: %d filas.", len(df))

    trades = build_barbell_trades(
        df, config,
        start_date=config["backtest"]["start_date"],
        end_date=config["backtest"]["end_date"],
    )
    logger.info("Ledger construido: %d trades.", len(trades))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(output_path, index=False)
    logger.info("Archivo guardado en: %s (%d filas).", output_path, len(trades))

    return 0


if __name__ == "__main__":
    sys.exit(main())
