"""tests/test_backtest.py — Tests de src/backtest.py."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    run_cycle_engine,
    run_fixed_ratio,
    run_hybrid,
    run_interest_only,
    summarize,
)


def _trade(entry_date: str, expiry_date: str, retorno_premium: float) -> dict:
    return {
        "entry_date": pd.Timestamp(entry_date),
        "expiry_date": pd.Timestamp(expiry_date),
        "retorno_premium": retorno_premium,
    }


def _rate_series(value: float, dates: list[str]) -> pd.Series:
    """Tasa constante en todas las fechas dadas, indexada para .asof()."""
    idx = pd.to_datetime(sorted(dates))
    return pd.Series([value] * len(idx), index=idx)


# ---------------------------------------------------------------------------
# 1. run_cycle_engine — encadenamiento básico
# ---------------------------------------------------------------------------

def test_encadena_capital_sin_huecos():
    """safe_capital_end de un ciclo es la base del total_capital_start del siguiente."""
    trades = pd.DataFrame([
        _trade("2019-01-01", "2019-03-01", retorno_premium=-1.0),  # put vence sin valor
        _trade("2019-03-02", "2019-05-01", retorno_premium=-1.0),
    ])
    rate = _rate_series(0.0, ["2019-01-01", "2019-03-02"])

    ledger = run_cycle_engine(
        trades, rate, initial_capital=100_000.0,
        budget_fn=lambda total, interest: 10_000.0,
    )

    assert len(ledger) == 2
    # tasa 0 -> accrued_interest=0 -> total_capital_start ciclo 2 == safe_capital_end ciclo 1
    assert ledger.iloc[1]["total_capital_start"] == pytest.approx(ledger.iloc[0]["safe_capital_end"])


def test_presupuesto_se_recorta_a_total_capital():
    """Un budget_fn que pide más de lo disponible se recorta a total_capital."""
    trades = pd.DataFrame([_trade("2019-01-01", "2019-03-01", retorno_premium=0.0)])
    rate = _rate_series(0.0, ["2019-01-01"])

    ledger = run_cycle_engine(
        trades, rate, initial_capital=100.0,
        budget_fn=lambda total, interest: total * 5.0,  # pide 5x el capital
    )

    assert ledger.iloc[0]["budget"] == pytest.approx(100.0)


def test_payoff_fuerte_aumenta_capital_para_el_siguiente_ciclo():
    """'Reset tras evento': un ciclo con retorno alto sube la base del próximo ciclo."""
    trades = pd.DataFrame([
        _trade("2019-01-01", "2019-03-01", retorno_premium=50.0),  # x50 sobre el budget
        _trade("2019-03-02", "2019-05-01", retorno_premium=0.0),
    ])
    rate = _rate_series(0.0, ["2019-01-01", "2019-03-02"])

    ledger = run_cycle_engine(
        trades, rate, initial_capital=100_000.0,
        budget_fn=lambda total, interest: 10_000.0,
    )

    # ciclo 1: 90k se queda + 10k*51 = 510k vuelve -> safe_capital_end = 600k
    assert ledger.iloc[0]["safe_capital_end"] == pytest.approx(600_000.0)
    assert ledger.iloc[1]["total_capital_start"] == pytest.approx(600_000.0)


# ---------------------------------------------------------------------------
# 2. run_interest_only
# ---------------------------------------------------------------------------

def test_interest_only_tasa_cero_no_toca_principal():
    """Con tasa 0, el presupuesto es 0 en todos los ciclos -> principal intacto."""
    trades = pd.DataFrame([
        _trade("2019-01-01", "2019-03-01", retorno_premium=-1.0),
        _trade("2019-03-02", "2019-05-01", retorno_premium=-1.0),
    ])
    rate = _rate_series(0.0, ["2019-01-01", "2019-03-02"])

    ledger = run_interest_only(trades, rate, initial_capital=100_000.0)

    assert (ledger["budget"] == 0.0).all()
    assert ledger.iloc[-1]["safe_capital_end"] == pytest.approx(100_000.0)
    assert not ledger["principal_touched"].any()


def test_interest_only_usa_exactamente_el_interes_devengado():
    trades = pd.DataFrame([_trade("2019-01-01", "2019-03-02", retorno_premium=-1.0)])  # 60 días
    rate = _rate_series(0.05, ["2019-01-01"])

    ledger = run_interest_only(trades, rate, initial_capital=100_000.0)

    expected_interest = 100_000.0 * 0.05 * 60 / 365.0
    assert ledger.iloc[0]["budget"] == pytest.approx(expected_interest)
    assert ledger.iloc[0]["accrued_interest"] == pytest.approx(expected_interest)


# ---------------------------------------------------------------------------
# 3. run_fixed_ratio
# ---------------------------------------------------------------------------

def test_fixed_ratio_erosion_geometrica_sin_eventos():
    """3 ciclos sin evento (retorno -1.0): capital cae como initial * (1-weight)^3."""
    trades = pd.DataFrame([
        _trade("2019-01-01", "2019-02-01", retorno_premium=-1.0),
        _trade("2019-02-02", "2019-03-02", retorno_premium=-1.0),
        _trade("2019-03-03", "2019-04-02", retorno_premium=-1.0),
    ])
    rate = _rate_series(0.0, ["2019-01-01", "2019-02-02", "2019-03-03"])

    ledger = run_fixed_ratio(trades, rate, initial_capital=100_000.0, aggressive_weight=0.10)

    expected_final = 100_000.0 * (0.90 ** 3)
    assert ledger.iloc[-1]["safe_capital_end"] == pytest.approx(expected_final)


def test_fixed_ratio_recalcula_sobre_capital_corriente():
    """El presupuesto del segundo ciclo se calcula sobre el capital YA reducido."""
    trades = pd.DataFrame([
        _trade("2019-01-01", "2019-02-01", retorno_premium=-1.0),
        _trade("2019-02-02", "2019-03-02", retorno_premium=-1.0),
    ])
    rate = _rate_series(0.0, ["2019-01-01", "2019-02-02"])

    ledger = run_fixed_ratio(trades, rate, initial_capital=100_000.0, aggressive_weight=0.10)

    assert ledger.iloc[0]["budget"] == pytest.approx(10_000.0)
    assert ledger.iloc[1]["budget"] == pytest.approx(9_000.0)  # 10% de 90k, no de 100k


# ---------------------------------------------------------------------------
# 4. run_hybrid
# ---------------------------------------------------------------------------

def test_hybrid_usa_interes_si_supera_el_piso():
    """Tasa alta -> el interés devengado supera el piso -> no se toca principal."""
    trades = pd.DataFrame([_trade("2019-01-01", "2019-03-02", retorno_premium=-1.0)])  # 60 días
    rate = _rate_series(0.20, ["2019-01-01"])  # tasa alta

    ledger = run_hybrid(trades, rate, initial_capital=100_000.0, min_position_pct=0.012)

    expected_interest = 100_000.0 * 0.20 * 60 / 365.0  # ~3287, > piso de 1200
    assert ledger.iloc[0]["budget"] == pytest.approx(expected_interest)
    assert not ledger.iloc[0]["principal_touched"]


def test_hybrid_usa_piso_si_interes_no_alcanza():
    """Tasa ~0 -> el interés no alcanza el piso -> se completa con principal."""
    trades = pd.DataFrame([_trade("2019-01-01", "2019-03-02", retorno_premium=-1.0)])
    rate = _rate_series(0.0, ["2019-01-01"])

    ledger = run_hybrid(trades, rate, initial_capital=100_000.0, min_position_pct=0.012)

    assert ledger.iloc[0]["budget"] == pytest.approx(100_000.0 * 0.012)
    assert ledger.iloc[0]["principal_touched"]


# ---------------------------------------------------------------------------
# 5. summarize
# ---------------------------------------------------------------------------

def test_summarize_capital_final_y_retorno():
    trades = pd.DataFrame([_trade("2019-01-01", "2020-01-01", retorno_premium=-1.0)])
    rate = _rate_series(0.0, ["2019-01-01"])
    ledger = run_fixed_ratio(trades, rate, initial_capital=100_000.0, aggressive_weight=0.10)

    stats = summarize(ledger, initial_capital=100_000.0)

    assert stats["final_capital"] == pytest.approx(90_000.0)
    assert stats["total_return_pct"] == pytest.approx(-10.0)
    assert stats["n_cycles"] == 1


def test_summarize_max_drawdown_sobre_curva_conocida():
    """Capital sube y luego cae: drawdown debe medirse desde el pico, no desde el inicio."""
    trades = pd.DataFrame([
        _trade("2019-01-01", "2019-02-01", retorno_premium=1.0),   # x2 -> sube
        _trade("2019-02-02", "2019-03-02", retorno_premium=-1.0),  # vence sin valor -> baja
    ])
    rate = _rate_series(0.0, ["2019-01-01", "2019-02-02"])
    ledger = run_cycle_engine(
        trades, rate, initial_capital=100_000.0,
        budget_fn=lambda total, interest: total,  # todo el capital al put cada ciclo
    )
    # ciclo 1: 100k * (1+1.0) = 200k. ciclo 2: 200k * (1+(-1.0)) = 0
    stats = summarize(ledger, initial_capital=100_000.0)

    assert stats["max_drawdown_pct"] == pytest.approx(-100.0)


def test_summarize_principal_touches_y_pct_itm():
    trades = pd.DataFrame([
        _trade("2019-01-01", "2019-03-02", retorno_premium=5.0),   # ITM
        _trade("2019-03-03", "2019-05-02", retorno_premium=-1.0),  # OTM
    ])
    rate = _rate_series(0.0, ["2019-01-01", "2019-03-03"])
    ledger = run_hybrid(trades, rate, initial_capital=100_000.0, min_position_pct=0.012)

    stats = summarize(ledger, initial_capital=100_000.0)

    assert stats["n_principal_touches"] == 2  # tasa 0 -> siempre se usa el piso
    assert stats["pct_cycles_itm"] == pytest.approx(50.0)
