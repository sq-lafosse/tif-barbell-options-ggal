"""metrics.py — Métricas de riesgo/retorno sobre los ledgers de backtest.py.

`backtest.py` ya calcula lo mínimo para decidir entre los 3 modelos de capitalización
(capital final, CAGR, max drawdown). Este módulo agrega las métricas de riesgo ajustado
por volatilidad que CLAUDE.md §6 asigna a `metrics.py`: Sharpe, Sortino y Expected
Shortfall, todas calculadas sobre los retornos POR CICLO del capital total (no diarios:
el ledger de `run_cycle_engine` solo tiene un punto por ciclo de ~2 meses).

Definiciones (por ciclo, ver `cycle_returns`):
  - retorno del ciclo:       safe_capital_end / total_capital_start - 1.
  - retorno libre de riesgo: tbills_rate × days_held / 365 (mismo período que el ciclo).
  - retorno en exceso:       retorno del ciclo − retorno libre de riesgo.

Sharpe y Sortino se anualizan escalando por sqrt(ciclos_por_año), con
ciclos_por_año = 365 / días_promedio_por_ciclo (la cadencia real es bimestral, no
anual — ver discusión en backtest.py). Expected Shortfall se reporta sin anualizar:
es el retorno promedio del ciclo en el percentil de cola, una medida de severidad de
evento, no una tasa.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/processed")
MODELS = ["interest_only", "fixed_ratio", "hybrid"]
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_ES_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Retornos por ciclo
# ---------------------------------------------------------------------------

def cycle_returns(ledger: pd.DataFrame) -> pd.DataFrame:
    """Calcula el retorno total y libre de riesgo de cada ciclo del ledger.

    Args:
        ledger: output de `run_cycle_engine` (o sus wrappers en backtest.py), con
            columnas `total_capital_start`, `safe_capital_end`, `tbills_rate`,
            `days_held`.

    Returns:
        DataFrame con columnas `total_return`, `rf_return`, `excess_return`, alineado
        al índice de `ledger`.
    """
    total_return = ledger["safe_capital_end"] / ledger["total_capital_start"] - 1.0
    rf_return = ledger["tbills_rate"] * ledger["days_held"] / 365.0
    return pd.DataFrame({
        "total_return": total_return,
        "rf_return": rf_return,
        "excess_return": total_return - rf_return,
    })


def periods_per_year(ledger: pd.DataFrame) -> float:
    """Cadencia anualizada de ciclos, a partir del promedio de `days_held`.

    Args:
        ledger: output de `run_cycle_engine`.

    Returns:
        Número (no necesariamente entero) de ciclos por año.
    """
    return 365.0 / ledger["days_held"].mean()


# ---------------------------------------------------------------------------
# Núcleos genéricos (operan sobre cualquier serie de retornos, no solo ciclos —
# reutilizados por src/benchmark.py para los retornos diarios de ADR/Merval).
# ---------------------------------------------------------------------------

def sharpe_core(excess: pd.Series, periods_per_year_: float) -> float:
    """Sharpe anualizado de una serie de retornos en exceso ya calculada.

    Args:
        excess:            retornos en exceso (cualquier frecuencia).
        periods_per_year_: cadencia anual de esa frecuencia (ej. 252 para diario).

    Returns:
        Sharpe anualizado, o NaN si no hay variabilidad.
    """
    if excess.std(ddof=1) == 0:
        return np.nan
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year_))


def sortino_core(excess: pd.Series, periods_per_year_: float) -> float:
    """Sortino anualizado de una serie de retornos en exceso ya calculada.

    Args:
        excess:            retornos en exceso (cualquier frecuencia).
        periods_per_year_: cadencia anual de esa frecuencia.

    Returns:
        Sortino anualizado, o NaN si no hay observaciones con retorno negativo.
    """
    downside = excess[excess < 0.0]
    if downside.empty:
        return np.nan
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return np.nan
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year_))


def expected_shortfall_core(returns: pd.Series, alpha: float) -> float:
    """Expected Shortfall (CVaR) de una serie de retornos, al nivel `alpha`.

    Args:
        returns: retornos (cualquier frecuencia).
        alpha:   proporción de la cola izquierda a promediar.

    Returns:
        Expected Shortfall, como retorno decimal (negativo = pérdida).
    """
    cutoff = returns.quantile(alpha)
    tail = returns[returns <= cutoff]
    if tail.empty:
        tail = returns.nsmallest(1)
    return float(tail.mean())


def calmar_ratio(cagr_pct: float, max_drawdown_pct: float) -> float:
    """Calmar ratio: CAGR sobre el valor absoluto del max drawdown, ambos en %.

    A diferencia de Sharpe/Sortino (que penalizan la volatilidad de los retornos),
    el Calmar penaliza específicamente la peor caída histórica — la métrica estándar
    para argumentar "retorno por unidad de riesgo de cola", el eje central de la
    tesis (Barbell vs. benchmarks lineales). Reutilizable con el `cagr`/`max_drawdown_pct`
    que ya calculan tanto `backtest.summarize` (por ciclo) como
    `benchmark.summarize_benchmark` (diario) — no depende de la frecuencia de origen.

    Args:
        cagr_pct:          CAGR en % (ej. 11.2, no 0.112).
        max_drawdown_pct:  max drawdown en %, valor negativo (ej. -10.2).

    Returns:
        Calmar ratio, o NaN si el drawdown es 0 (curva sin caídas, división indefinida).
    """
    if max_drawdown_pct == 0.0:
        return np.nan
    return float(cagr_pct / abs(max_drawdown_pct))


# ---------------------------------------------------------------------------
# Sharpe / Sortino (por ciclo)
# ---------------------------------------------------------------------------

def sharpe_ratio(ledger: pd.DataFrame) -> float:
    """Sharpe ratio anualizado sobre los retornos en exceso por ciclo.

    Args:
        ledger: output de `run_cycle_engine`.

    Returns:
        Sharpe ratio anualizado, o NaN si no hay variabilidad en los retornos.
    """
    excess = cycle_returns(ledger)["excess_return"]
    return sharpe_core(excess, periods_per_year(ledger))


def sortino_ratio(ledger: pd.DataFrame) -> float:
    """Sortino ratio anualizado: igual que Sharpe pero con desvío solo de la cola negativa.

    Args:
        ledger: output de `run_cycle_engine`.

    Returns:
        Sortino ratio anualizado. NaN si no hay ciclos con retorno en exceso negativo
        (no hay downside que medir).
    """
    excess = cycle_returns(ledger)["excess_return"]
    return sortino_core(excess, periods_per_year(ledger))


# ---------------------------------------------------------------------------
# Expected Shortfall (CVaR, por ciclo)
# ---------------------------------------------------------------------------

def expected_shortfall(ledger: pd.DataFrame, alpha: float = DEFAULT_ES_ALPHA) -> float:
    """Expected Shortfall (CVaR) del retorno total por ciclo, al nivel `alpha`.

    Promedio del retorno en el peor `alpha` de los ciclos (ej. alpha=0.05 -> peor 5%).
    Con pocos ciclos (44 en el dataset real), el percentil cae sobre un puñado de
    observaciones — interpretar como una medida de severidad de cola, no una
    estimación estadísticamente precisa.

    Args:
        ledger: output de `run_cycle_engine`.
        alpha:  proporción de la cola izquierda a promediar (default 5%).

    Returns:
        Expected Shortfall, como retorno decimal por ciclo (negativo = pérdida).
    """
    total_return = cycle_returns(ledger)["total_return"]
    return expected_shortfall_core(total_return, alpha)


# ---------------------------------------------------------------------------
# Max drawdown (consistente con backtest.summarize, repetido acá por completitud
# de CLAUDE.md §6: metrics.py es el dueño de MDD/ES/Sortino/Sharpe)
# ---------------------------------------------------------------------------

def max_drawdown_pct(ledger: pd.DataFrame, initial_capital: float) -> float:
    """Drawdown máximo peak-to-trough sobre la curva de `safe_capital_end` por ciclo.

    Args:
        ledger:          output de `run_cycle_engine`.
        initial_capital: capital inicial, primer punto de la curva de equity.

    Returns:
        Drawdown máximo en %, valor negativo (o 0.0 si la curva nunca cae).
    """
    if ledger.empty:
        return 0.0
    equity = pd.concat(
        [pd.Series([initial_capital]), ledger["safe_capital_end"]], ignore_index=True
    )
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min() * 100.0)


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def summarize_risk(
    ledger: pd.DataFrame,
    initial_capital: float,
    es_alpha: float = DEFAULT_ES_ALPHA,
) -> dict:
    """Resumen de métricas de riesgo/retorno ajustado de un ledger.

    Args:
        ledger:          output de `run_cycle_engine`.
        initial_capital: capital inicial del backtest.
        es_alpha:        nivel de cola para Expected Shortfall.

    Returns:
        Dict con `sharpe`, `sortino`, `expected_shortfall`, `max_drawdown_pct`,
        `periods_per_year`.
    """
    if ledger.empty:
        return {
            "sharpe": np.nan, "sortino": np.nan, "expected_shortfall": np.nan,
            "max_drawdown_pct": 0.0, "periods_per_year": np.nan,
        }
    return {
        "sharpe": sharpe_ratio(ledger),
        "sortino": sortino_ratio(ledger),
        "expected_shortfall": expected_shortfall(ledger, es_alpha),
        "max_drawdown_pct": max_drawdown_pct(ledger, initial_capital),
        "periods_per_year": periods_per_year(ledger),
    }


# ---------------------------------------------------------------------------
# CLI standalone (python -m src.metrics)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula Sharpe, Sortino, Expected Shortfall y MDD de los 3 modelos de backtest.py."
    )
    parser.add_argument("--ledgers-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--es-alpha", type=float, default=DEFAULT_ES_ALPHA)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    """Corre las métricas de riesgo sobre los 3 ledgers ya generados por backtest.py.

    Returns:
        0 si todo OK, 1 si falta algún ledger.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    args = _parse_args()
    ledgers_dir = Path(args.ledgers_dir)

    comparison_path = ledgers_dir / "backtest_comparison.csv"
    if not comparison_path.exists():
        logger.error(
            "No se encontró %s. Correr primero `python -m src.backtest`.", comparison_path
        )
        return 1
    cagr_by_model = pd.read_csv(comparison_path, index_col="model")["cagr"]

    rows = []
    for model in MODELS:
        ledger_path = ledgers_dir / f"backtest_{model}.parquet"
        if not ledger_path.exists():
            logger.error(
                "No se encontró %s. Correr primero `python -m src.backtest`.", ledger_path
            )
            return 1
        ledger = pd.read_parquet(ledger_path)
        stats = summarize_risk(ledger, args.initial_capital, args.es_alpha)
        stats["calmar"] = calmar_ratio(cagr_by_model.loc[model], stats["max_drawdown_pct"])
        stats["model"] = model
        rows.append(stats)
        logger.info("Métricas de '%s' calculadas sobre %d ciclos.", model, len(ledger))

    summary_df = pd.DataFrame(rows).set_index("model")
    summary_df = summary_df[
        ["sharpe", "sortino", "expected_shortfall", "max_drawdown_pct", "calmar", "periods_per_year"]
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "risk_metrics_comparison.csv"
    summary_df.to_csv(summary_path)

    logger.info("Comparación de métricas de riesgo:\n%s", summary_df.to_string())
    logger.info("Resumen guardado en: %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
