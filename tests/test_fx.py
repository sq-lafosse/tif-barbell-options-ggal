"""tests/test_fx.py — Tests de src/fx.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx import (
    convert_options_to_usd,
    convert_synthetic_to_usd,
    load_ccl,
    merge_real_and_synthetic,
)

# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------

def _make_ccl_parquet(tmp_path: Path, rows: list[tuple]) -> Path:
    """Crea un Parquet de CCL mínimo con columnas fecha, ggal_ba_ars, ggal_adr_usd, ccl."""
    p = tmp_path / "ccl" / "CCL_daily.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["fecha", "ggal_ba_ars", "ggal_adr_usd", "ccl"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    df.to_parquet(p, index=False)
    return p


def _tidy_row(fecha: str, strike: float = 5000.0, prima: float = 100.0, ggal_local: float = 4800.0) -> dict:
    """Dict con el mínimo de columnas del formato tidy necesarias para los tests."""
    return {
        "fecha":            pd.Timestamp(fecha),
        "opex":             "2024-02",
        "especie":          "GFGC50000CO",
        "tipo":             "Call",
        "strike":           strike,
        "prima":            prima,
        "monto":            10000.0,
        "nominal":          10,
        "ggal_local":       ggal_local,
        "var_ggal":         0.005,
        "tlr":              0.05,
        "vi_implicita":     0.40,
        "valor_extrinseco": 0.10,
        "dias_vto":         30,
        "plazo_anios":      0.082,
        "delta":            0.25,
        "gamma":            0.01,
        "vega":             0.80,
        "theta":            -0.02,
        "esquema":          "A",
        "fuente_archivo":   "GGAL_HIST_2024-02.csv",
    }


def _synthetic_row(fecha: str, strike: float = 4.5, prima: float = 0.1) -> dict:
    """Dict mínimo con columnas del dataset sintético."""
    return {
        "fecha":          pd.Timestamp(fecha),
        "opex":           "2022-02",
        "especie":        "SYN_GGAL_C_2022-02-18_4.50",
        "tipo":           "Call",
        "strike":         strike,
        "prima":          prima,
        "ggal_adr_usd":   40.0,
        "tlr":            0.04,
        "vi_implicita":   0.45,
        "dias_vto":       30,
        "plazo_anios":    0.082,
        "delta":          0.25,
        "gamma":          0.01,
        "vega":           0.80,
        "theta":          -0.02,
        "pct_otm":        0.10,
        "esquema":        "SIN",
        "fuente_archivo": "SYNTHETIC_2019_2023.parquet",
    }


def _make_real_usd_df(fechas: list[str]) -> pd.DataFrame:
    """Mini dataset real con las columnas post-conversión."""
    rows = []
    for f in fechas:
        r = _tidy_row(f)
        r.update({
            "ccl_aplicado":   1000.0,
            "strike_usd":     5.0,
            "prima_usd":      0.1,
            "ggal_local_usd": 4.8,
            "ccl_ffilled":    False,
        })
        rows.append(r)
    return pd.DataFrame(rows)


def _make_synthetic_usd_df(fechas: list[str]) -> pd.DataFrame:
    """Mini dataset sintético con columnas post-convert_synthetic_to_usd."""
    rows = []
    for f in fechas:
        r = _synthetic_row(f)
        r.update({
            "ccl_aplicado":   np.nan,
            "strike_usd":     r["strike"],
            "prima_usd":      r["prima"],
            "ggal_local_usd": np.nan,
            "ccl_ffilled":    False,
        })
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. load_ccl
# ---------------------------------------------------------------------------

def test_load_ccl_columnas(tmp_path):
    """load_ccl devuelve solo fecha y ccl, ordenado por fecha ascendente."""
    ccl_path = _make_ccl_parquet(tmp_path, [
        ("2024-01-03", 1100.0, 10.0, 1100.0),
        ("2024-01-01", 1050.0,  9.8, 1050.0),   # desordenado
        ("2024-01-02", 1080.0,  9.9, 1080.0),
    ])
    df = load_ccl(ccl_path)

    assert list(df.columns) == ["fecha", "ccl"]
    assert df["fecha"].is_monotonic_increasing
    assert df["ccl"].tolist() == [1050.0, 1080.0, 1100.0]


def test_load_ccl_no_existe():
    """FileNotFoundError si el archivo no existe."""
    with pytest.raises(FileNotFoundError):
        load_ccl(Path("data/raw/ccl/no_existe_xyz.parquet"))


# ---------------------------------------------------------------------------
# 2. convert_options_to_usd — caso básico
# ---------------------------------------------------------------------------

def test_convert_basico_prima(tmp_path):
    """Prima 100 ARS con CCL 1000 → prima_usd = 0.1."""
    ccl_path = _make_ccl_parquet(tmp_path, [("2024-01-15", 1000.0, 10.0, 1000.0)])
    df_tidy  = pd.DataFrame([_tidy_row("2024-01-15", prima=100.0)])

    result = convert_options_to_usd(df_tidy, ccl_path=ccl_path)

    assert result["prima_usd"].iloc[0] == pytest.approx(0.1)
    assert result["ccl_aplicado"].iloc[0] == pytest.approx(1000.0)


def test_convert_basico_strike(tmp_path):
    """Strike 5000 ARS con CCL 1000 → strike_usd = 5.0."""
    ccl_path = _make_ccl_parquet(tmp_path, [("2024-01-15", 1000.0, 10.0, 1000.0)])
    df_tidy  = pd.DataFrame([_tidy_row("2024-01-15", strike=5000.0)])

    result = convert_options_to_usd(df_tidy, ccl_path=ccl_path)

    assert result["strike_usd"].iloc[0] == pytest.approx(5.0)


def test_convert_basico_ggal_local(tmp_path):
    """ggal_local 4800 ARS con CCL 1000 → ggal_local_usd = 4.8."""
    ccl_path = _make_ccl_parquet(tmp_path, [("2024-01-15", 1000.0, 10.0, 1000.0)])
    df_tidy  = pd.DataFrame([_tidy_row("2024-01-15", ggal_local=4800.0)])

    result = convert_options_to_usd(df_tidy, ccl_path=ccl_path)

    assert result["ggal_local_usd"].iloc[0] == pytest.approx(4.8)


def test_convert_preserva_columnas_originales(tmp_path):
    """El resultado conserva todas las columnas del tidy original."""
    ccl_path = _make_ccl_parquet(tmp_path, [("2024-01-15", 1000.0, 10.0, 1000.0)])
    df_tidy  = pd.DataFrame([_tidy_row("2024-01-15")])

    result = convert_options_to_usd(df_tidy, ccl_path=ccl_path)

    for col in df_tidy.columns:
        assert col in result.columns, f"Columna original '{col}' perdida tras conversión"


# ---------------------------------------------------------------------------
# 3. Forward-fill
# ---------------------------------------------------------------------------

def test_ffill_aplicado(tmp_path):
    """Gap en CCL (martes sin dato): la fila del martes usa el CCL del lunes."""
    ccl_path = _make_ccl_parquet(tmp_path, [
        ("2024-01-15", 1000.0, 10.0, 1000.0),   # lunes
        # martes 2024-01-16 no tiene CCL (feriado US)
        ("2024-01-17", 1020.0, 10.2, 1020.0),   # miércoles
    ])
    df_tidy = pd.DataFrame([
        _tidy_row("2024-01-15"),
        _tidy_row("2024-01-16"),   # martes — gap en CCL
        _tidy_row("2024-01-17"),
    ])

    result = convert_options_to_usd(df_tidy, ccl_path=ccl_path)

    fila_martes = result[result["fecha"] == pd.Timestamp("2024-01-16")]
    assert len(fila_martes) == 1
    assert fila_martes["ccl_aplicado"].iloc[0] == pytest.approx(1000.0)
    assert fila_martes["ccl_ffilled"].iloc[0] is True or fila_martes["ccl_ffilled"].iloc[0] == True


def test_ffill_marca_correctamente(tmp_path):
    """Fechas con CCL observado tienen ccl_ffilled=False."""
    ccl_path = _make_ccl_parquet(tmp_path, [
        ("2024-01-15", 1000.0, 10.00, 1000.0),
        ("2024-01-16", 1005.0, 10.05, 1005.0),
    ])
    df_tidy = pd.DataFrame([
        _tidy_row("2024-01-15"),
        _tidy_row("2024-01-16"),
    ])

    result = convert_options_to_usd(df_tidy, ccl_path=ccl_path)

    assert not result["ccl_ffilled"].any(), (
        f"Se esperaba ccl_ffilled=False en todas las filas observadas:\n{result['ccl_ffilled']}"
    )


def test_ffill_usa_ultimo_disponible(tmp_path):
    """Dos días consecutivos sin CCL usan el mismo valor anterior."""
    ccl_path = _make_ccl_parquet(tmp_path, [
        ("2024-01-12", 900.0, 9.0, 900.0),   # viernes
        ("2024-01-15", 950.0, 9.5, 950.0),   # lunes (sábado y domingo sin CCL)
    ])
    df_tidy = pd.DataFrame([
        _tidy_row("2024-01-13"),  # sábado — no hay CCL
        _tidy_row("2024-01-14"),  # domingo — no hay CCL
    ])

    result = convert_options_to_usd(df_tidy, ccl_path=ccl_path)

    # Ambas deben usar el CCL del viernes (900)
    assert result["ccl_aplicado"].eq(900.0).all()
    assert result["ccl_ffilled"].all()


# ---------------------------------------------------------------------------
# 4. Dataset sintético
# ---------------------------------------------------------------------------

def test_sintetico_no_aplica_ccl():
    """convert_synthetic_to_usd deja ccl_aplicado=NaN y strike_usd=strike."""
    df_syn = pd.DataFrame([_synthetic_row("2022-01-15", strike=4.5, prima=0.1)])

    result = convert_synthetic_to_usd(df_syn)

    assert np.isnan(result["ccl_aplicado"].iloc[0])
    assert result["strike_usd"].iloc[0] == pytest.approx(4.5)
    assert result["prima_usd"].iloc[0] == pytest.approx(0.1)
    assert np.isnan(result["ggal_local_usd"].iloc[0])
    assert result["ccl_ffilled"].iloc[0] == False  # noqa: E712


def test_sintetico_columnas_de_trazabilidad_agregadas():
    """convert_synthetic_to_usd agrega exactamente las 5 columnas de trazabilidad."""
    df_syn = pd.DataFrame([_synthetic_row("2022-01-15")])
    cols_antes = set(df_syn.columns)

    result = convert_synthetic_to_usd(df_syn)

    cols_nuevas = set(result.columns) - cols_antes
    assert cols_nuevas == {
        "ccl_aplicado", "strike_usd", "prima_usd", "ggal_local_usd", "ccl_ffilled"
    }


def test_sintetico_no_modifica_original():
    """convert_synthetic_to_usd no modifica el DataFrame de entrada."""
    df_syn = pd.DataFrame([_synthetic_row("2022-01-15")])
    cols_originales = list(df_syn.columns)

    convert_synthetic_to_usd(df_syn)

    assert list(df_syn.columns) == cols_originales


# ---------------------------------------------------------------------------
# 5. merge_real_and_synthetic
# ---------------------------------------------------------------------------

def test_merge_columnas_homogeneas():
    """El merge alinea columnas: columnas extra de uno se agregan con NaN en el otro."""
    df_real = _make_real_usd_df(["2024-01-15"])
    df_syn  = _make_synthetic_usd_df(["2022-01-15"])

    result = merge_real_and_synthetic(df_real, df_syn)

    for col in df_real.columns:
        assert col in result.columns, f"Columna de real '{col}' faltante en el merge"
    for col in df_syn.columns:
        assert col in result.columns, f"Columna de sintético '{col}' faltante en el merge"


def test_merge_orden_fecha():
    """El output del merge está ordenado por fecha ascendente."""
    df_real = _make_real_usd_df(["2024-01-15", "2024-06-01"])
    df_syn  = _make_synthetic_usd_df(["2022-01-15", "2019-03-10"])

    result = merge_real_and_synthetic(df_real, df_syn)

    assert result["fecha"].is_monotonic_increasing


def test_merge_periodos_disjuntos_sin_filtro():
    """El overlap entre sintético y real NO se elimina — ambas filas quedan en el output."""
    # 2023-09-01 aparece en ambos datasets (overlap esperado según CLAUDE.md §5.5)
    df_real = _make_real_usd_df(["2023-09-01", "2024-01-15"])
    df_syn  = _make_synthetic_usd_df(["2023-09-01", "2022-01-15"])

    result = merge_real_and_synthetic(df_real, df_syn)

    overlap_rows = result[result["fecha"] == pd.Timestamp("2023-09-01")]
    assert len(overlap_rows) == 2, (
        f"El overlap debe conservarse (2 filas esperadas), se obtuvieron {len(overlap_rows)}"
    )


def test_merge_total_filas():
    """El total de filas del merge es la suma de real + sintético."""
    df_real = _make_real_usd_df(["2024-01-15", "2024-01-16", "2024-01-17"])
    df_syn  = _make_synthetic_usd_df(["2022-01-15", "2022-01-16"])

    result = merge_real_and_synthetic(df_real, df_syn)

    assert len(result) == len(df_real) + len(df_syn)


def test_merge_no_modifica_inputs():
    """merge_real_and_synthetic no modifica los DataFrames de entrada."""
    df_real = _make_real_usd_df(["2024-01-15"])
    df_syn  = _make_synthetic_usd_df(["2022-01-15"])

    cols_real_antes = list(df_real.columns)
    cols_syn_antes  = list(df_syn.columns)

    merge_real_and_synthetic(df_real, df_syn)

    assert list(df_real.columns) == cols_real_antes
    assert list(df_syn.columns) == cols_syn_antes
