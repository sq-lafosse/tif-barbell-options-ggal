"""backtest.py — Motor de simulación: compone los ciclos de strategy.py en una curva de equity.

`strategy.py` resuelve la economía de CADA ciclo de Put (cuánto cuesta entrar, cuánto
paga al vencimiento, expresado como retorno por $1 invertido — ver `retorno_premium`
en `barbell_trades.parquet`). Lo que falta es decidir CUÁNTO capital del polo seguro se
destina a cada ciclo, y eso es lo que resuelve este módulo.

Surgió una pregunta metodológica real al diseñar la Barbell: si durante varios ciclos
seguidos los Puts vencen sin valor (sin evento de cola), ¿se sigue invirtiendo un % fijo
del capital corriente (erosión geométrica del principal) o solo los intereses generados
por el T-Bill (principal intacto, posición más chica)? Se comparan 3 reglas alternativas
sobre el dataset real antes de fijar una para la tesis.

Las 3 reglas comparten el mismo motor de ciclo (``run_cycle_engine``) y solo difieren en
la función de presupuesto:
  - ``interest_only``:  presupuesto = interés devengado del T-Bill durante el ciclo.
                         El principal nunca se toca; en períodos de tasa ~0 el
                         presupuesto puede ser ~0 (no se compra protección ese ciclo).
  - ``fixed_ratio``:     presupuesto = % fijo (``portfolio.aggressive_weight``) del
                         capital TOTAL actual, recalculado cada ciclo — erosión
                         geométrica si no hay eventos, pero proporcional siempre.
  - ``hybrid``:          presupuesto = max(interés, piso de % del capital total) —
                         usa el interés si alcanza un mínimo viable, completa con
                         principal solo si no alcanza.

En los 3 casos, el "reset tras evento" (si un ciclo paga fuerte, el próximo presupuesto
se calcula sobre una base mayor) es automático: el motor de ciclo siempre parte del
capital total actualizado, no hace falta una regla aparte para eso.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from src.metrics import calmar_ratio

logger = logging.getLogger(__name__)

DEFAULT_TRADES_PATH = Path("data/processed/barbell_trades.parquet")
DEFAULT_TBILLS_PATH = Path("data/raw/tbills/TBILLS_3M_daily.parquet")
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_OUTPUT_DIR  = Path("data/processed")

BudgetFn = Callable[[float, float], float]


# ---------------------------------------------------------------------------
# Carga de la tasa libre de riesgo (T-Bills 3M, FRED)
# ---------------------------------------------------------------------------

def load_tbills_rate(tbills_path: Path = DEFAULT_TBILLS_PATH) -> pd.Series:
    """Lee el Parquet de T-Bills 3M y devuelve la tasa anualizada por fecha.

    Args:
        tbills_path: path al Parquet con columnas ``fecha``, ``tasa_decimal``.

    Returns:
        Series de tasa anualizada (decimal) indexada por fecha, ordenada
        ascendentemente, apta para ``.asof()``.

    Raises:
        FileNotFoundError: si ``tbills_path`` no existe.
    """
    tbills_path = Path(tbills_path)
    if not tbills_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de T-Bills: {tbills_path}")

    raw = pd.read_parquet(tbills_path)[["fecha", "tasa_decimal"]]
    raw["fecha"] = pd.to_datetime(raw["fecha"])
    raw = raw.sort_values("fecha").reset_index(drop=True)

    rate = pd.Series(raw["tasa_decimal"].values, index=raw["fecha"].values)
    rate.index.name = "fecha"
    return rate


# ---------------------------------------------------------------------------
# Motor de ciclo genérico
# ---------------------------------------------------------------------------

def run_cycle_engine(
    trades: pd.DataFrame,
    tbills_rate: pd.Series,
    initial_capital: float,
    budget_fn: BudgetFn,
) -> pd.DataFrame:
    """Compone los ciclos de ``trades`` en una curva de equity, según ``budget_fn``.

    En cada ciclo, el capital del polo agresivo es transitorio: se retira del polo
    seguro al entrar, se invierte en el Put del ciclo (escalado por ``retorno_premium``,
    que ya viene de ``strategy.py`` expresado por $1), y vuelve íntegro al polo seguro
    al vencimiento (la liquidación es a valor intrínseco, no hay posición que arrastrar
    entre ciclos).

    Args:
        trades:          ledger de ``build_barbell_trades`` (``strategy.py``), con
                          columnas ``entry_date``, ``expiry_date``, ``retorno_premium``.
        tbills_rate:      Series de ``load_tbills_rate``, tasa anualizada por fecha.
        initial_capital:  capital total al inicio del primer ciclo.
        budget_fn:        función ``(total_capital, accrued_interest) -> presupuesto``
                          que decide cuánto invertir ese ciclo. El presupuesto se
                          recorta a ``[0, total_capital]``.

    Returns:
        DataFrame con una fila por ciclo: ``cycle, entry_date, expiry_date, days_held,
        tbills_rate, accrued_interest, total_capital_start, budget, principal_touched,
        aggressive_ending_value, safe_capital_end``.
    """
    trades = trades.sort_values("entry_date").reset_index(drop=True)

    rows = []
    safe_capital = float(initial_capital)

    for cycle, trade in trades.iterrows():
        days_held = (trade["expiry_date"] - trade["entry_date"]).days
        rate = tbills_rate.asof(trade["entry_date"])
        accrued_interest = safe_capital * rate * days_held / 365.0
        total_capital = safe_capital + accrued_interest

        budget = budget_fn(total_capital, accrued_interest)
        budget = min(max(budget, 0.0), total_capital)

        aggressive_ending_value = budget * (1.0 + trade["retorno_premium"])
        safe_capital = (total_capital - budget) + aggressive_ending_value

        rows.append({
            "cycle":                   cycle,
            "entry_date":              trade["entry_date"],
            "expiry_date":             trade["expiry_date"],
            "days_held":               days_held,
            "tbills_rate":             rate,
            "accrued_interest":        accrued_interest,
            "total_capital_start":     total_capital,
            "budget":                  budget,
            "principal_touched":       budget > accrued_interest + 1e-9,
            "aggressive_ending_value": aggressive_ending_value,
            "safe_capital_end":        safe_capital,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Las 3 reglas de presupuesto
# ---------------------------------------------------------------------------

def run_interest_only(
    trades: pd.DataFrame,
    tbills_rate: pd.Series,
    initial_capital: float,
) -> pd.DataFrame:
    """Modelo 1: el presupuesto de cada ciclo es exactamente el interés devengado.

    El principal nunca se toca. En ciclos de tasa ~0, el presupuesto es ~0 y no se
    compra protección ese ciclo (el costo de la convexidad es ceder el rendimiento
    libre de riesgo, no erosionar el capital base).
    """
    return run_cycle_engine(
        trades, tbills_rate, initial_capital,
        budget_fn=lambda total_capital, accrued_interest: accrued_interest,
    )


def run_fixed_ratio(
    trades: pd.DataFrame,
    tbills_rate: pd.Series,
    initial_capital: float,
    aggressive_weight: float,
) -> pd.DataFrame:
    """Modelo 2: el presupuesto es un % fijo del capital total, recalculado cada ciclo.

    Si no hay eventos de cola, el principal se erosiona geométricamente
    (``initial_capital * (1 - aggressive_weight) ** n_ciclos``), pero el % invertido
    respecto al capital corriente es siempre el mismo.
    """
    return run_cycle_engine(
        trades, tbills_rate, initial_capital,
        budget_fn=lambda total_capital, accrued_interest: total_capital * aggressive_weight,
    )


def run_hybrid(
    trades: pd.DataFrame,
    tbills_rate: pd.Series,
    initial_capital: float,
    min_position_pct: float,
) -> pd.DataFrame:
    """Modelo 3: usa el interés devengado si alcanza un piso mínimo; si no, completa con principal.

    El piso es ``min_position_pct`` del capital total de ese ciclo (escala con el NAV,
    igual que el modelo `fixed_ratio`, pero solo se activa cuando el interés no alcanza).
    """
    return run_cycle_engine(
        trades, tbills_rate, initial_capital,
        budget_fn=lambda total_capital, accrued_interest: max(
            accrued_interest, total_capital * min_position_pct
        ),
    )


# ---------------------------------------------------------------------------
# Resumen comparativo
# ---------------------------------------------------------------------------

def summarize(ledger: pd.DataFrame, initial_capital: float) -> dict:
    """Calcula métricas mínimas para comparar modelos de capitalización.

    No reemplaza a ``metrics.py`` (Sharpe, Sortino, Expected Shortfall — pendiente de
    otra iteración); son solo las métricas necesarias para decidir entre los 3 modelos.

    Args:
        ledger:           output de ``run_cycle_engine`` (o sus wrappers).
        initial_capital:  capital inicial, para calcular retorno total y CAGR.

    Returns:
        Dict con ``final_capital``, ``total_return_pct``, ``cagr``, ``max_drawdown_pct``,
        ``calmar``, ``n_cycles``, ``n_principal_touches``, ``pct_cycles_itm``.
    """
    if ledger.empty:
        return {
            "final_capital": initial_capital, "total_return_pct": 0.0, "cagr": 0.0,
            "max_drawdown_pct": 0.0, "calmar": np.nan, "n_cycles": 0,
            "n_principal_touches": 0, "pct_cycles_itm": 0.0,
        }

    equity = pd.concat([
        pd.Series([initial_capital]), ledger["safe_capital_end"],
    ], ignore_index=True)

    final_capital = equity.iloc[-1]
    total_return_pct = (final_capital / initial_capital - 1.0) * 100.0

    n_years = (ledger["expiry_date"].iloc[-1] - ledger["entry_date"].iloc[0]).days / 365.0
    cagr = (
        (final_capital / initial_capital) ** (1.0 / n_years) - 1.0
        if n_years > 0 else np.nan
    )

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100.0

    n_cycles_itm = (ledger["aggressive_ending_value"] > ledger["budget"]).sum()
    cagr_pct = cagr * 100.0

    return {
        "final_capital":       final_capital,
        "total_return_pct":    total_return_pct,
        "cagr":                cagr_pct,
        "max_drawdown_pct":    max_drawdown_pct,
        "calmar":              calmar_ratio(cagr_pct, max_drawdown_pct),
        "n_cycles":            len(ledger),
        "n_principal_touches": int(ledger["principal_touched"].sum()),
        "pct_cycles_itm":      n_cycles_itm / len(ledger) * 100.0,
    }


# ---------------------------------------------------------------------------
# CLI standalone (python -m src.backtest)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara 3 modelos de capitalización del polo agresivo de la Barbell."
    )
    parser.add_argument("--trades-path", default=str(DEFAULT_TRADES_PATH))
    parser.add_argument("--tbills-path", default=str(DEFAULT_TBILLS_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    """Corre los 3 modelos sobre el ledger real y muestra una tabla comparativa.

    Returns:
        0 si todo OK, 1 si hubo error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    args = _parse_args()

    trades_path = Path(args.trades_path)
    if not trades_path.exists():
        logger.error(
            "No se encontró el ledger de trades en '%s'. Correr primero `python -m src.strategy`.",
            trades_path,
        )
        return 1

    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.error("No se encontró config.yaml en '%s'.", config_path)
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trades = pd.read_parquet(trades_path)
    tbills_rate = load_tbills_rate(Path(args.tbills_path))
    logger.info("Trades cargados: %d ciclos.", len(trades))

    initial_capital = config["capital_models"]["initial_capital"]
    aggressive_weight = config["portfolio"]["aggressive_weight"]
    min_position_pct = config["capital_models"]["min_position_pct"]

    models = {
        "interest_only": run_interest_only(trades, tbills_rate, initial_capital),
        "fixed_ratio": run_fixed_ratio(trades, tbills_rate, initial_capital, aggressive_weight),
        "hybrid": run_hybrid(trades, tbills_rate, initial_capital, min_position_pct),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, ledger in models.items():
        ledger.to_parquet(output_dir / f"backtest_{name}.parquet", index=False)
        stats = summarize(ledger, initial_capital)
        stats["model"] = name
        summary_rows.append(stats)
        logger.info("Modelo '%s' guardado: %d ciclos.", name, len(ledger))

    summary_df = pd.DataFrame(summary_rows).set_index("model")
    summary_df = summary_df[[
        "final_capital", "total_return_pct", "cagr", "max_drawdown_pct", "calmar",
        "n_cycles", "n_principal_touches", "pct_cycles_itm",
    ]]
    summary_path = output_dir / "backtest_comparison.csv"
    summary_df.to_csv(summary_path)

    logger.info("Comparación de modelos:\n%s", summary_df.to_string())
    logger.info("Resumen guardado en: %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
