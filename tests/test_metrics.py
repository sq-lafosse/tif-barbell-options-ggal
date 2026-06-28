"""tests/test_metrics.py — Tests de src/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    cycle_returns,
    expected_shortfall,
    max_drawdown_pct,
    periods_per_year,
    sharpe_ratio,
    sortino_ratio,
    summarize_risk,
)


def _ledger_row(
    total_capital_start: float,
    safe_capital_end: float,
    tbills_rate: float = 0.0,
    days_held: int = 60,
) -> dict:
    return {
        "total_capital_start": total_capital_start,
        "safe_capital_end": safe_capital_end,
        "tbills_rate": tbills_rate,
        "days_held": days_held,
    }


# ---------------------------------------------------------------------------
# 1. cycle_returns / periods_per_year
# ---------------------------------------------------------------------------

def test_cycle_returns_calcula_total_y_exceso():
    ledger = pd.DataFrame([
        _ledger_row(100_000.0, 110_000.0, tbills_rate=0.05, days_held=73),  # 73/365 = 0.2
    ])
    returns = cycle_returns(ledger)

    assert returns.iloc[0]["total_return"] == pytest.approx(0.10)
    assert returns.iloc[0]["rf_return"] == pytest.approx(0.01)  # 0.05 * 0.2
    assert returns.iloc[0]["excess_return"] == pytest.approx(0.09)


def test_periods_per_year_usa_dias_promedio():
    ledger = pd.DataFrame([
        _ledger_row(100.0, 100.0, days_held=60),
        _ledger_row(100.0, 100.0, days_held=70),
    ])
    assert periods_per_year(ledger) == pytest.approx(365.0 / 65.0)


# ---------------------------------------------------------------------------
# 2. sharpe_ratio
# ---------------------------------------------------------------------------

def test_sharpe_positivo_con_retornos_consistentes_sobre_rf():
    ledger = pd.DataFrame([
        _ledger_row(100_000.0, 102_000.0, tbills_rate=0.0, days_held=60),
        _ledger_row(100_000.0, 103_000.0, tbills_rate=0.0, days_held=60),
        _ledger_row(100_000.0, 101_000.0, tbills_rate=0.0, days_held=60),
    ])
    assert sharpe_ratio(ledger) > 0.0


def test_sharpe_nan_sin_variabilidad():
    """Retorno idéntico en todos los ciclos -> std=0 -> Sharpe indefinido."""
    ledger = pd.DataFrame([
        _ledger_row(100_000.0, 101_000.0, days_held=60),
        _ledger_row(100_000.0, 101_000.0, days_held=60),
    ])
    assert np.isnan(sharpe_ratio(ledger))


# ---------------------------------------------------------------------------
# 3. sortino_ratio
# ---------------------------------------------------------------------------

def test_sortino_ignora_volatilidad_al_alza():
    """Un downside chico (-0.01) junto a upside grande (+0.05, +0.10): el desvío
    completo (Sharpe) queda inflado por la dispersión positiva, pero el downside
    deviation (Sortino) solo mide la cola negativa, mucho más chica -> Sortino > Sharpe."""
    ledger = pd.DataFrame([
        _ledger_row(100_000.0, 99_000.0, days_held=60),    # retorno -0.01 (downside chico)
        _ledger_row(100_000.0, 105_000.0, days_held=60),   # retorno +0.05 (upside)
        _ledger_row(100_000.0, 110_000.0, days_held=60),   # retorno +0.10 (upside)
    ])
    assert sortino_ratio(ledger) > sharpe_ratio(ledger)


def test_sortino_nan_sin_downside():
    ledger = pd.DataFrame([
        _ledger_row(100_000.0, 101_000.0, days_held=60),
        _ledger_row(100_000.0, 102_000.0, days_held=60),
    ])
    assert np.isnan(sortino_ratio(ledger))


# ---------------------------------------------------------------------------
# 4. expected_shortfall
# ---------------------------------------------------------------------------

def test_expected_shortfall_promedia_la_cola_negativa():
    """10 ciclos: 1 con retorno -0.50 (cola), 9 con retorno +0.01. Al 10% de cola,
    ES debe ser exactamente el peor retorno."""
    rows = [_ledger_row(100_000.0, 100_000.0 * 1.01, days_held=60) for _ in range(9)]
    rows.append(_ledger_row(100_000.0, 100_000.0 * 0.50, days_held=60))
    ledger = pd.DataFrame(rows)

    es = expected_shortfall(ledger, alpha=0.10)
    assert es == pytest.approx(-0.50)


def test_expected_shortfall_con_un_solo_ciclo_usa_ese_retorno():
    ledger = pd.DataFrame([_ledger_row(100_000.0, 90_000.0, days_held=60)])
    assert expected_shortfall(ledger, alpha=0.05) == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# 5. max_drawdown_pct
# ---------------------------------------------------------------------------

def test_max_drawdown_desde_el_pico():
    ledger = pd.DataFrame([
        _ledger_row(100_000.0, 200_000.0, days_held=60),  # sube a 200k
        _ledger_row(200_000.0, 100_000.0, days_held=60),  # cae a 100k (-50% desde el pico)
    ])
    assert max_drawdown_pct(ledger, initial_capital=100_000.0) == pytest.approx(-50.0)


def test_max_drawdown_cero_si_nunca_cae():
    ledger = pd.DataFrame([
        _ledger_row(100_000.0, 110_000.0, days_held=60),
        _ledger_row(110_000.0, 120_000.0, days_held=60),
    ])
    assert max_drawdown_pct(ledger, initial_capital=100_000.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. summarize_risk
# ---------------------------------------------------------------------------

def test_summarize_risk_devuelve_todas_las_claves():
    ledger = pd.DataFrame([
        _ledger_row(100_000.0, 102_000.0, tbills_rate=0.03, days_held=60),
        _ledger_row(100_000.0, 98_000.0, tbills_rate=0.03, days_held=60),
    ])
    stats = summarize_risk(ledger, initial_capital=100_000.0)

    assert set(stats.keys()) == {
        "sharpe", "sortino", "expected_shortfall", "max_drawdown_pct", "periods_per_year",
    }
    assert stats["periods_per_year"] == pytest.approx(365.0 / 60.0)


def test_summarize_risk_ledger_vacio():
    stats = summarize_risk(pd.DataFrame(), initial_capital=100_000.0)
    assert stats["max_drawdown_pct"] == 0.0
    assert np.isnan(stats["sharpe"])
