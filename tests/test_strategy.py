"""tests/test_strategy.py — Tests de src/strategy.py."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import (
    build_barbell_trades,
    classify_spread,
    get_daily_spot,
    resolve_entry_rows,
    select_put_contract,
)

CONFIG = {
    "moneyness": {"otm_pct": 0.15},
    "transaction_costs": {
        "high_volume_spread": 0.015,
        "low_volume_spread": 0.09,
        "volume_threshold": 1000.0,
    },
    "rebalance": {
        "min_dias_vto": 15,
        "max_gap_dias": 5,
        "esquema_priority": ["A", "B", "SIN"],
    },
}


def _row(
    fecha: str,
    tipo: str = "Put",
    strike_usd: float = 5.0,
    prima_usd: float = 0.10,
    ggal_local_usd: float = 6.0,
    ggal_adr_usd: float = 60.0,
    monto: float = 5000.0,
    dias_vto: int = 30,
    pct_otm: float = 0.15,
    esquema: str = "A",
    especie: str = "GFGV50000O",
) -> dict:
    """Dict con el mínimo de columnas de options_full_usd.parquet usadas por strategy.py."""
    return {
        "fecha":          pd.Timestamp(fecha),
        "tipo":           tipo,
        "especie":        especie,
        "strike_usd":     strike_usd,
        "prima_usd":      prima_usd,
        "monto":          monto,
        "ggal_local_usd": ggal_local_usd,
        "ggal_adr_usd":   ggal_adr_usd,
        "pct_otm":        pct_otm,
        "dias_vto":       dias_vto,
        "esquema":        esquema,
    }


# ---------------------------------------------------------------------------
# 1. get_daily_spot
# ---------------------------------------------------------------------------

def test_spot_usa_local_cuando_disponible():
    """Régimen real: spot = ggal_local_usd."""
    df = pd.DataFrame([_row("2024-01-15", ggal_local_usd=6.0, ggal_adr_usd=60.0)])
    spot = get_daily_spot(df)
    assert spot.loc[pd.Timestamp("2024-01-15")] == pytest.approx(6.0)


def test_spot_usa_adr_cuando_local_es_nan():
    """Régimen sintético: ggal_local_usd es NaN → usa ggal_adr_usd."""
    df = pd.DataFrame([_row("2022-01-15", ggal_local_usd=np.nan, ggal_adr_usd=40.0)])
    spot = get_daily_spot(df)
    assert spot.loc[pd.Timestamp("2022-01-15")] == pytest.approx(40.0)


def test_spot_una_fila_por_fecha():
    """Múltiples contratos en la misma fecha colapsan a un único spot por fecha."""
    df = pd.DataFrame([
        _row("2024-01-15", strike_usd=5.0, ggal_local_usd=6.0),
        _row("2024-01-15", strike_usd=5.5, ggal_local_usd=6.0),
    ])
    spot = get_daily_spot(df)
    assert len(spot) == 1
    assert spot.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# 2. classify_spread
# ---------------------------------------------------------------------------

def test_spread_alto_volumen():
    monto = pd.Series([5000.0])
    result = classify_spread(monto, CONFIG)
    assert result.iloc[0] == pytest.approx(0.015)


def test_spread_bajo_volumen():
    monto = pd.Series([500.0])
    result = classify_spread(monto, CONFIG)
    assert result.iloc[0] == pytest.approx(0.09)


def test_spread_nan_cae_en_bajo_volumen():
    """Dataset sintético sin `monto` (NaN) → spread conservador de bajo volumen."""
    monto = pd.Series([np.nan])
    result = classify_spread(monto, CONFIG)
    assert result.iloc[0] == pytest.approx(0.09)


def test_spread_umbral_exacto_es_alto_volumen():
    """monto == volume_threshold clasifica como alto volumen (>=)."""
    monto = pd.Series([1000.0])
    result = classify_spread(monto, CONFIG)
    assert result.iloc[0] == pytest.approx(0.015)


# ---------------------------------------------------------------------------
# 3. select_put_contract
# ---------------------------------------------------------------------------

def test_selecciona_strike_mas_cercano_al_target():
    """Entre varios strikes del mismo vencimiento, elige el pct_otm más cercano al target."""
    df_date = pd.DataFrame([
        _row("2024-01-15", strike_usd=5.0, pct_otm=0.10, dias_vto=30, especie="A"),
        _row("2024-01-15", strike_usd=4.5, pct_otm=0.15, dias_vto=30, especie="B"),
        _row("2024-01-15", strike_usd=4.0, pct_otm=0.25, dias_vto=30, especie="C"),
    ])
    result = select_put_contract(df_date, target_otm_pct=0.15, min_dias_vto=15)
    assert result["especie"] == "B"


def test_selecciona_vencimiento_mas_proximo():
    """Entre dos vencimientos válidos, elige el más próximo aunque su pct_otm no sea el mejor."""
    df_date = pd.DataFrame([
        _row("2024-01-15", pct_otm=0.15, dias_vto=20, especie="CERCANO"),
        _row("2024-01-15", pct_otm=0.15, dias_vto=50, especie="LEJANO"),
    ])
    result = select_put_contract(df_date, target_otm_pct=0.15, min_dias_vto=15)
    assert result["especie"] == "CERCANO"


def test_filtra_por_min_dias_vto():
    """Contratos a punto de vencer (dias_vto < min_dias_vto) quedan excluidos."""
    df_date = pd.DataFrame([
        _row("2024-01-15", dias_vto=5, especie="MUY_CERCA"),
        _row("2024-01-15", dias_vto=40, especie="VALIDO"),
    ])
    result = select_put_contract(df_date, target_otm_pct=0.15, min_dias_vto=15)
    assert result["especie"] == "VALIDO"


def test_filtra_contratos_sin_prima_observable():
    """Contratos sin operaciones ese día (prima_usd NaN) no son comprables."""
    df_date = pd.DataFrame([
        _row("2024-01-15", dias_vto=30, prima_usd=np.nan, especie="SIN_PRIMA"),
        _row("2024-01-15", dias_vto=30, prima_usd=0.15, especie="CON_PRIMA"),
    ])
    result = select_put_contract(df_date, target_otm_pct=0.15, min_dias_vto=15)
    assert result["especie"] == "CON_PRIMA"


def test_sin_candidatos_devuelve_none():
    df_date = pd.DataFrame([_row("2024-01-15", dias_vto=3)])
    result = select_put_contract(df_date, target_otm_pct=0.15, min_dias_vto=15)
    assert result is None


# ---------------------------------------------------------------------------
# 4. resolve_entry_rows
# ---------------------------------------------------------------------------

def test_prioriza_esquema_real_sobre_sintetico():
    """En el overlap, si hay Puts reales (A/B) y sintéticos (SIN) el mismo día, gana el real."""
    df = pd.DataFrame([
        _row("2023-09-01", esquema="A", especie="REAL"),
        _row("2023-09-01", esquema="SIN", especie="SINTETICO"),
    ])
    fecha, filas = resolve_entry_rows(df, pd.Timestamp("2023-09-01"), ["A", "B", "SIN"])
    assert fecha == pd.Timestamp("2023-09-01")
    assert set(filas["especie"]) == {"REAL"}


def test_usa_sintetico_si_no_hay_real():
    df = pd.DataFrame([_row("2022-01-15", esquema="SIN", especie="SINTETICO")])
    fecha, filas = resolve_entry_rows(df, pd.Timestamp("2022-01-15"), ["A", "B", "SIN"])
    assert fecha == pd.Timestamp("2022-01-15")
    assert set(filas["especie"]) == {"SINTETICO"}


def test_busca_hacia_adelante_si_falta_la_fecha_exacta():
    """Si no hay datos el día exacto (fin de semana), busca hasta max_gap_dias adelante."""
    df = pd.DataFrame([_row("2024-01-17", especie="DISPONIBLE")])  # miércoles
    fecha, filas = resolve_entry_rows(
        df, pd.Timestamp("2024-01-15"), ["A", "B", "SIN"], max_gap_dias=5  # lunes
    )
    assert fecha == pd.Timestamp("2024-01-17")
    assert set(filas["especie"]) == {"DISPONIBLE"}


def test_sin_datos_dentro_del_gap_devuelve_none():
    df = pd.DataFrame([_row("2024-02-01", especie="MUY_LEJOS")])
    fecha, filas = resolve_entry_rows(
        df, pd.Timestamp("2024-01-15"), ["A", "B", "SIN"], max_gap_dias=5
    )
    assert fecha is None
    assert filas.empty


def test_solo_considera_puts():
    """Filas de Calls no deben confundirse con Puts disponibles."""
    df = pd.DataFrame([_row("2024-01-15", tipo="Call", especie="CALL")])
    fecha, filas = resolve_entry_rows(df, pd.Timestamp("2024-01-15"), ["A", "B", "SIN"])
    assert fecha is None
    assert filas.empty


# ---------------------------------------------------------------------------
# 5. build_barbell_trades — end-to-end chico
# ---------------------------------------------------------------------------

def _ciclo_sintetico(fecha_entrada: str, dias_vto: int, strike_usd: float, spot: float) -> list[dict]:
    """Genera filas para un ciclo: entrada + (opcional) una fila más cerca del vencimiento."""
    return [_row(
        fecha_entrada, esquema="SIN", dias_vto=dias_vto,
        strike_usd=strike_usd, prima_usd=0.10, pct_otm=0.15,
        ggal_local_usd=np.nan, ggal_adr_usd=spot, monto=np.nan,
        especie=f"SYN_{fecha_entrada}",
    )]


def test_ledger_encadena_ciclos_sin_huecos():
    """Dos ciclos consecutivos: el segundo entry_date es expiry_date + 1 día del primero."""
    rows = []
    rows += _ciclo_sintetico("2019-01-01", dias_vto=30, strike_usd=42.0, spot=50.0)
    # Spot al vencimiento del primer ciclo (2019-01-31): necesitamos una fila ese día
    # para que get_daily_spot tenga con qué resolver el payoff.
    rows.append(_row(
        "2019-01-31", esquema="SIN", dias_vto=0, strike_usd=42.0, prima_usd=0.0,
        pct_otm=0.15, ggal_local_usd=np.nan, ggal_adr_usd=45.0, monto=np.nan,
        especie="SYN_MARKET_31",
    ))
    rows += _ciclo_sintetico("2019-02-01", dias_vto=28, strike_usd=40.0, spot=45.0)
    rows.append(_row(
        "2019-03-01", esquema="SIN", dias_vto=0, strike_usd=40.0, prima_usd=0.0,
        pct_otm=0.15, ggal_local_usd=np.nan, ggal_adr_usd=38.0, monto=np.nan,
        especie="SYN_MARKET_28",
    ))
    df = pd.DataFrame(rows)

    trades = build_barbell_trades(df, CONFIG, start_date="2019-01-01", end_date="2019-04-01")

    assert len(trades) == 2
    assert trades.iloc[0]["expiry_date"] == pd.Timestamp("2019-01-31")
    assert trades.iloc[1]["entry_date"] == pd.Timestamp("2019-02-01")


def test_payoff_intrinseco_correcto():
    """Put ITM al vencimiento: payoff = strike - spot. Put OTM: payoff = 0."""
    rows = []
    rows += _ciclo_sintetico("2019-01-01", dias_vto=30, strike_usd=42.0, spot=50.0)
    # Al vencimiento, spot cae por debajo del strike -> Put ITM
    rows.append(_row(
        "2019-01-31", esquema="SIN", dias_vto=0, strike_usd=42.0, prima_usd=0.0,
        pct_otm=0.15, ggal_local_usd=np.nan, ggal_adr_usd=35.0, monto=np.nan,
        especie="SYN_MARKET",
    ))
    df = pd.DataFrame(rows)

    trades = build_barbell_trades(df, CONFIG, start_date="2019-01-01", end_date="2019-02-01")

    assert len(trades) == 1
    assert trades.iloc[0]["payoff_usd"] == pytest.approx(42.0 - 35.0)


def test_ledger_vacio_sin_contratos():
    df = pd.DataFrame([_row("2024-01-15", tipo="Call")])  # solo Calls, sin Puts
    trades = build_barbell_trades(df, CONFIG, start_date="2024-01-01", end_date="2024-02-01")
    assert trades.empty


def test_payoff_usa_base_adr_si_entrada_fue_sintetica_aunque_vto_caiga_en_real():
    """Regresión: trade que entra en régimen sintético (ADR) y vence ya con datos
    reales (local) disponibles ese día no debe mezclar bases de precio (ratio ADR
    ~10x vs. 1 acción local, CLAUDE.md §5.4.b). El payoff debe leerse contra el
    spot ADR, no contra el spot de 1 acción local, aunque este último exista esa
    fecha."""
    rows = []
    rows += _ciclo_sintetico("2019-01-01", dias_vto=30, strike_usd=42.0, spot=50.0)
    # Al vencimiento, YA hay datos reales (esquema A) ese día además del sintético —
    # el spot "local" (1 acción) es ~10x menor que el ADR, pero no debe usarse para
    # este trade porque entró en régimen sintético.
    rows.append(_row(
        "2019-01-31", esquema="A", dias_vto=0, strike_usd=4.2, prima_usd=0.0,
        pct_otm=0.15, ggal_local_usd=3.5, ggal_adr_usd=35.0, monto=5000.0,
        especie="REAL_MARKET",
    ))
    df = pd.DataFrame(rows)

    trades = build_barbell_trades(df, CONFIG, start_date="2019-01-01", end_date="2019-02-01")

    assert len(trades) == 1
    # Correcto: strike(42) - spot_ADR(35) = 7. Bug: strike(42) - spot_local(3.5) = 38.5.
    assert trades.iloc[0]["payoff_usd"] == pytest.approx(42.0 - 35.0)
