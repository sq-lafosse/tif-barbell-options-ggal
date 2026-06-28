"""benchmark.py — Curvas buy-and-hold de los 2 benchmarks (Decisión metodológica #11).

`backtest.py` mide la Barbell; este módulo mide los dos puntos de comparación que el
tutor cerró en CLAUDE.md §4 decisión 11:
  - ADR GGAL buy-and-hold:  efecto activo (Barbell vs. el subyacente que cubre).
  - Merval en USD:          efecto de mercado (Barbell vs. el equity argentino completo).

Ambos ya están en USD antes de llegar acá — el ADR cotiza en USD nativamente en NYSE
(`data/raw/adr/GGAL_ADR_daily.parquet`, columna `close`), y el Merval ya viene dividido
por el CCL diario (`data/raw/merval/MERVAL_daily.parquet`, columna `merval_usd`,
generado por `scripts/download_merval.py`). Por eso no hace falta ningún paso de
conversión adicional acá: comparar contra la curva de equity de la Barbell (también en
USD-CCL de punta a punta, ver `src/fx.py`) ya es una comparación directa.

Las métricas se calculan sobre retornos DIARIOS (a diferencia de `metrics.py`, que
opera por ciclo de ~2 meses porque así está estructurado el ledger de la Barbell) —
un buy-and-hold tiene un precio observable todos los días de mercado, así que no hay
razón para degradar la resolución. La anualización usa `periods_per_year=252`
(convención estándar de mercado, no 365: los índices/ADRs no cotizan fines de semana).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.metrics import calmar_ratio, expected_shortfall_core, sharpe_core, sortino_core

logger = logging.getLogger(__name__)

DEFAULT_ADR_PATH = Path("data/raw/adr/GGAL_ADR_daily.parquet")
DEFAULT_MERVAL_PATH = Path("data/raw/merval/MERVAL_daily.parquet")
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_OUTPUT_DIR = Path("data/processed")

TRADING_DAYS_PER_YEAR = 252.0
DEFAULT_ES_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Carga de precios
# ---------------------------------------------------------------------------

def load_adr_price(adr_path: Path = DEFAULT_ADR_PATH) -> pd.Series:
    """Lee el cierre diario del ADR GGAL en USD.

    Args:
        adr_path: path al Parquet de `scripts/download_adr.py`.

    Returns:
        Series de precio en USD, indexada por fecha ascendente, sin NaN (días sin
        cierre observado — ej. el último día descargado a mitad de rueda — se eliminan).
    """
    df = pd.read_parquet(adr_path)[["fecha", "close"]].dropna()
    df["fecha"] = pd.to_datetime(df["fecha"])
    series = pd.Series(df["close"].values, index=df["fecha"].values).sort_index()
    series.index.name = "fecha"
    return series


def load_merval_usd_price(merval_path: Path = DEFAULT_MERVAL_PATH) -> pd.Series:
    """Lee el cierre diario del Merval ya convertido a USD vía CCL.

    Args:
        merval_path: path al Parquet de `scripts/download_merval.py`.

    Returns:
        Series de precio en USD, indexada por fecha ascendente.
    """
    df = pd.read_parquet(merval_path)[["fecha", "merval_usd"]].dropna()
    df["fecha"] = pd.to_datetime(df["fecha"])
    series = pd.Series(df["merval_usd"].values, index=df["fecha"].values).sort_index()
    series.index.name = "fecha"
    return series


# ---------------------------------------------------------------------------
# Curva de equity buy-and-hold
# ---------------------------------------------------------------------------

def buy_and_hold_equity(
    price: pd.Series,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    initial_capital: float,
) -> pd.Series:
    """Escala una serie de precio a una curva de equity buy-and-hold en la ventana dada.

    Args:
        price:            Series de precio (cualquier moneda, ya resuelta en USD acá).
        start_date:        fecha de inicio del backtest (`config.yaml: backtest.start_date`).
        end_date:           fecha de fin del backtest.
        initial_capital:   capital invertido en `start_date`.

    Returns:
        Series de equity (mismas unidades que `initial_capital`), indexada por fecha,
        partiendo del primer precio disponible en la ventana.

    Raises:
        ValueError: si no hay ningún precio dentro de la ventana solicitada.
    """
    window = price.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    if window.empty:
        raise ValueError(
            f"Sin precios entre {start_date} y {end_date} — no se puede construir la curva."
        )
    return window / window.iloc[0] * initial_capital


# ---------------------------------------------------------------------------
# Resumen (mismo formato que backtest.summarize + metrics.summarize_risk)
# ---------------------------------------------------------------------------

def summarize_benchmark(
    equity: pd.Series,
    initial_capital: float,
    es_alpha: float = DEFAULT_ES_ALPHA,
) -> dict:
    """Resumen de retorno y riesgo de una curva de equity buy-and-hold.

    Args:
        equity:           output de `buy_and_hold_equity`.
        initial_capital:  capital inicial (denominador del retorno total y del CAGR).
        es_alpha:         nivel de cola para Expected Shortfall.

    Returns:
        Dict con `final_capital`, `total_return_pct`, `cagr`, `max_drawdown_pct`, `calmar`,
        `sharpe`, `sortino`, `expected_shortfall`, `periods_per_year`.
    """
    returns = equity.pct_change().dropna()

    final_capital = float(equity.iloc[-1])
    total_return_pct = (final_capital / initial_capital - 1.0) * 100.0

    n_years = (equity.index[-1] - equity.index[0]).days / 365.0
    cagr = (
        ((final_capital / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0
        if n_years > 0 else np.nan
    )

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown_pct = float(drawdown.min() * 100.0)

    return {
        "final_capital": final_capital,
        "total_return_pct": total_return_pct,
        "cagr": cagr,
        "max_drawdown_pct": max_drawdown_pct,
        "calmar": calmar_ratio(cagr, max_drawdown_pct),
        "sharpe": sharpe_core(returns, TRADING_DAYS_PER_YEAR),
        "sortino": sortino_core(returns, TRADING_DAYS_PER_YEAR),
        "expected_shortfall": expected_shortfall_core(returns, es_alpha),
        "periods_per_year": TRADING_DAYS_PER_YEAR,
    }


# ---------------------------------------------------------------------------
# CLI standalone (python -m src.benchmark)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula las curvas buy-and-hold de los benchmarks (ADR GGAL, Merval USD)."
    )
    parser.add_argument("--adr-path", default=str(DEFAULT_ADR_PATH))
    parser.add_argument("--merval-path", default=str(DEFAULT_MERVAL_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--es-alpha", type=float, default=DEFAULT_ES_ALPHA)
    return parser.parse_args()


def main() -> int:
    """Calcula y guarda el resumen de los 2 benchmarks sobre la ventana del backtest.

    Returns:
        0 si todo OK, 1 si falta algún archivo de entrada.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    args = _parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.error("No se encontró config.yaml en '%s'.", config_path)
        return 1
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]
    initial_capital = config["capital_models"]["initial_capital"]

    adr_path = Path(args.adr_path)
    merval_path = Path(args.merval_path)
    if not adr_path.exists():
        logger.error("No se encontró %s. Correr `python scripts/download_adr.py`.", adr_path)
        return 1
    if not merval_path.exists():
        logger.error("No se encontró %s. Correr `python scripts/download_merval.py`.", merval_path)
        return 1

    benchmarks = {
        config["benchmarks"]["adr"]["name"]: load_adr_price(adr_path),
        config["benchmarks"]["merval"]["name"]: load_merval_usd_price(merval_path),
    }

    rows = []
    for name, price in benchmarks.items():
        equity = buy_and_hold_equity(price, start_date, end_date, initial_capital)
        stats = summarize_benchmark(equity, initial_capital, args.es_alpha)
        stats["model"] = name
        rows.append(stats)
        logger.info("Benchmark '%s' calculado sobre %d días.", name, len(equity))

    summary_df = pd.DataFrame(rows).set_index("model")
    summary_df = summary_df[[
        "final_capital", "total_return_pct", "cagr", "max_drawdown_pct", "calmar",
        "sharpe", "sortino", "expected_shortfall", "periods_per_year",
    ]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_comparison.csv"
    summary_df.to_csv(summary_path)

    logger.info("Comparación de benchmarks:\n%s", summary_df.to_string())
    logger.info("Resumen guardado en: %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
