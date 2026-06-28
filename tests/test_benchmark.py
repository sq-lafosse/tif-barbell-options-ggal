"""tests/test_benchmark.py — Tests de src/benchmark.py."""

import numpy as np
import pandas as pd
import pytest

from src.benchmark import buy_and_hold_equity, summarize_benchmark


def _price_series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=dates)


# ---------------------------------------------------------------------------
# 1. buy_and_hold_equity
# ---------------------------------------------------------------------------

def test_equity_arranca_en_initial_capital():
    price = _price_series([100.0, 110.0, 120.0])
    equity = buy_and_hold_equity(price, "2024-01-01", "2024-01-03", initial_capital=1000.0)
    assert equity.iloc[0] == pytest.approx(1000.0)
    assert equity.iloc[-1] == pytest.approx(1200.0)  # +20% en precio -> +20% en equity


def test_equity_recorta_a_la_ventana_solicitada():
    price = _price_series([100.0, 110.0, 120.0, 130.0])
    equity = buy_and_hold_equity(price, "2024-01-02", "2024-01-03", initial_capital=1000.0)
    assert len(equity) == 2
    assert equity.iloc[0] == pytest.approx(1000.0)


def test_ventana_sin_datos_lanza_error():
    price = _price_series([100.0, 110.0])
    with pytest.raises(ValueError):
        buy_and_hold_equity(price, "2025-01-01", "2025-01-05", initial_capital=1000.0)


# ---------------------------------------------------------------------------
# 2. summarize_benchmark
# ---------------------------------------------------------------------------

def test_summarize_devuelve_todas_las_claves():
    price = _price_series([100.0, 105.0, 95.0, 110.0])
    equity = buy_and_hold_equity(price, "2024-01-01", "2024-01-04", initial_capital=1000.0)
    stats = summarize_benchmark(equity, initial_capital=1000.0)

    assert set(stats.keys()) == {
        "final_capital", "total_return_pct", "cagr", "max_drawdown_pct", "calmar",
        "sharpe", "sortino", "expected_shortfall", "periods_per_year",
    }
    assert stats["periods_per_year"] == pytest.approx(252.0)


def test_total_return_consistente_con_el_precio():
    price = _price_series([100.0, 150.0])  # +50%
    equity = buy_and_hold_equity(price, "2024-01-01", "2024-01-02", initial_capital=1000.0)
    stats = summarize_benchmark(equity, initial_capital=1000.0)
    assert stats["total_return_pct"] == pytest.approx(50.0)
    assert stats["final_capital"] == pytest.approx(1500.0)


def test_max_drawdown_desde_el_pico():
    price = _price_series([100.0, 200.0, 100.0])  # sube a 2x, vuelve a 1x -> dd -50% desde el pico
    equity = buy_and_hold_equity(price, "2024-01-01", "2024-01-03", initial_capital=1000.0)
    stats = summarize_benchmark(equity, initial_capital=1000.0)
    assert stats["max_drawdown_pct"] == pytest.approx(-50.0)


def test_sortino_nan_sin_retornos_negativos():
    price = _price_series([100.0, 110.0, 120.0])  # siempre sube
    equity = buy_and_hold_equity(price, "2024-01-01", "2024-01-03", initial_capital=1000.0)
    stats = summarize_benchmark(equity, initial_capital=1000.0)
    assert np.isnan(stats["sortino"])
