"""tests/test_report.py — Tests de src/report.py."""

import pandas as pd
import pytest

from src.report import load_barbell_equity


def test_equity_arranca_en_initial_capital_y_sigue_los_ciclos(tmp_path):
    ledger = pd.DataFrame([
        {"entry_date": pd.Timestamp("2024-01-01"), "expiry_date": pd.Timestamp("2024-03-01"),
         "safe_capital_end": 110_000.0},
        {"entry_date": pd.Timestamp("2024-03-02"), "expiry_date": pd.Timestamp("2024-05-01"),
         "safe_capital_end": 95_000.0},
    ])
    ledger_path = tmp_path / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)

    equity = load_barbell_equity(ledger_path, initial_capital=100_000.0)

    assert len(equity) == 3
    assert equity.iloc[0] == pytest.approx(100_000.0)
    assert equity.index[0] == pd.Timestamp("2024-01-01")
    assert equity.iloc[1] == pytest.approx(110_000.0)
    assert equity.index[1] == pd.Timestamp("2024-03-01")
    assert equity.iloc[2] == pytest.approx(95_000.0)
    assert equity.index[2] == pd.Timestamp("2024-05-01")
    assert equity.index.is_monotonic_increasing
