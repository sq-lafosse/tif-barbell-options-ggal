"""tests/test_data_loader.py — Tests de src/data_loader.py."""

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    TIDY_COLUMNS,
    _detect_schema,
    _parse_arg_decimal,
    _parse_arg_percent,
    load_historical_options,
)

# ---------------------------------------------------------------------------
# 1. _parse_arg_decimal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entrada, esperado", [
    ("1.234,56",   1234.56),
    ("0,003",      0.003),
    ("40,03",      40.03),
    ("",           np.nan),
    ("0,000",      np.nan),
    ("100",        100.0),
    ("  25,50 ",   25.50),
    ("-1.000,00",  -1000.0),
])
def test_parse_arg_decimal(entrada, esperado):
    resultado = _parse_arg_decimal(entrada)
    if np.isnan(esperado):
        assert np.isnan(resultado), f"Se esperaba NaN para '{entrada}', obtuvo {resultado}"
    else:
        assert resultado == pytest.approx(esperado), (
            f"Para '{entrada}': esperado {esperado}, obtuvo {resultado}"
        )


# ---------------------------------------------------------------------------
# 2. _parse_arg_percent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entrada, esperado", [
    ("40,03%",   0.4003),
    ("40,03",    0.4003),
    ("5%",       0.05),
    ("100%",     1.0),
    ("",         np.nan),
    ("  0,00%",  0.0),
    ("0,000%",   np.nan),   # "0,000" → NaN antes de dividir
])
def test_parse_arg_percent(entrada, esperado):
    resultado = _parse_arg_percent(entrada)
    if np.isnan(esperado):
        assert np.isnan(resultado), f"Se esperaba NaN para '{entrada}', obtuvo {resultado}"
    else:
        assert resultado == pytest.approx(esperado, abs=1e-8), (
            f"Para '{entrada}': esperado {esperado}, obtuvo {resultado}"
        )


# ---------------------------------------------------------------------------
# 3. _detect_schema
# ---------------------------------------------------------------------------

def test_detect_schema_a():
    assert _detect_schema(20, "GGAL_HIST_2023-10.csv") == "A"


def test_detect_schema_b():
    assert _detect_schema(24, "GGAL_HIST_2025-06.csv") == "B"


def test_detect_schema_invalido():
    with pytest.raises(ValueError, match="columnas"):
        _detect_schema(18, "GGAL_HIST_2099-01.csv")


def test_detect_schema_invalido_22_cols():
    with pytest.raises(ValueError, match="22"):
        _detect_schema(22, "GGAL_HIST_2024-01.csv")


# ---------------------------------------------------------------------------
# 4. Carga de archivo real — skip si no están los CSV
# ---------------------------------------------------------------------------

RAW_OPTIONS_DIR = Path("data/raw/options")
REAL_FILES = sorted(RAW_OPTIONS_DIR.glob("GGAL_HIST_*.csv")) if RAW_OPTIONS_DIR.exists() else []

_skip_if_no_data = pytest.mark.skipif(
    not REAL_FILES,
    reason="No hay archivos GGAL_HIST_*.csv en data/raw/options/ — test omitido.",
)


@_skip_if_no_data
def test_load_single_file_columnas():
    """El archivo más chico carga sin errores y el DataFrame tiene las columnas esperadas."""
    smallest = min(REAL_FILES, key=lambda p: p.stat().st_size)
    df = load_historical_options(
        raw_dir=RAW_OPTIONS_DIR,
        pattern=smallest.name,
        recompute_greeks_scheme_a=False,
    )
    for col in TIDY_COLUMNS:
        assert col in df.columns, f"Columna '{col}' faltante"


@_skip_if_no_data
def test_load_single_file_tipos():
    """Columnas de tipo, esquema y fuente_archivo tienen los valores esperados."""
    smallest = min(REAL_FILES, key=lambda p: p.stat().st_size)
    df = load_historical_options(
        raw_dir=RAW_OPTIONS_DIR,
        pattern=smallest.name,
        recompute_greeks_scheme_a=False,
    )
    assert df["tipo"].isin(["Call", "Put"]).all(), "Valores inesperados en columna 'tipo'"
    assert df["esquema"].isin(["A", "B"]).all(), "Valores inesperados en columna 'esquema'"
    assert df["fuente_archivo"].str.match(r"GGAL_HIST_\d{4}-\d{2}\.csv").all()


@_skip_if_no_data
def test_load_todos_los_archivos_shape():
    """Carga todos los archivos y verifica que la concatenación no esté vacía."""
    df = load_historical_options(
        raw_dir=RAW_OPTIONS_DIR,
        recompute_greeks_scheme_a=False,
    )
    assert len(df) > 0
    assert set(TIDY_COLUMNS).issubset(df.columns)


# ---------------------------------------------------------------------------
# 5. Recálculo de griegas para esquema A (usando mini-CSV sintético)
# ---------------------------------------------------------------------------

def _make_mini_csv_schema_a() -> str:
    """CSV mínimo de esquema A (20 columnas) con una fila de Call y una de Put."""
    header = (
        "FECHA;ESPECIE;BASE;TIPO;ÚLTIMO;%;MONTO;HORA;APE.;MAX.;MIN.;"
        "C. ANT.;NOMINAL;PRECIO GGAL;VAR. % GGAL;TLR;VI %;VE %;DÍAS AL VTO.;PLAZO (años)"
    )
    # Call OTM: S=100, K=110, T=30d (~0.082 años), r=5%, VI=40%
    row_call = (
        "15/02/2024;GFGC11000CO;110,00;Call;1,50;-5,00%;12000,00;17:00;1,60;1,80;1,20;"
        "1,60;100;100,00;0,50%;5,00%;40,00%;10,00%;30;0,082192"
    )
    # Put OTM: S=100, K=90, T=30d
    row_put = (
        "15/02/2024;GFGV9000CO;90,00;Put;1,20;-3,00%;8000,00;17:00;1,30;1,40;1,00;"
        "1,25;80;100,00;0,50%;5,00%;40,00%;8,00%;30;0,082192"
    )
    return "\n".join([header, row_call, row_put])


def _write_mini_csv(tmp_path: Path, content: str, filename: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8-sig")
    return p


def test_greeks_recomputation_delta_not_nan(tmp_path):
    """Después del recálculo, filas del esquema A con VI válida tienen delta no-NaN."""
    csv_content = _make_mini_csv_schema_a()
    _write_mini_csv(tmp_path, csv_content, "GGAL_HIST_2024-02.csv")

    df = load_historical_options(
        raw_dir=tmp_path,
        pattern="GGAL_HIST_2024-02.csv",
        recompute_greeks_scheme_a=True,
    )

    assert df["esquema"].unique().tolist() == ["A"]
    # Ambas filas tienen VI válida → delta debe estar calculado
    assert df["delta"].notna().all(), f"Delta tiene NaN inesperado:\n{df[['tipo','delta']]}"
    assert df["gamma"].notna().all()
    assert df["vega"].notna().all()
    assert df["theta"].notna().all()


def test_greeks_recomputation_call_delta_positivo(tmp_path):
    """Delta de call OTM debe ser positivo (entre 0 y 0.5)."""
    _write_mini_csv(tmp_path, _make_mini_csv_schema_a(), "GGAL_HIST_2024-02.csv")
    df = load_historical_options(tmp_path, "GGAL_HIST_2024-02.csv", recompute_greeks_scheme_a=True)

    call_delta = df.loc[df["tipo"] == "Call", "delta"].iloc[0]
    assert 0 < call_delta < 0.5, f"Delta call OTM inesperado: {call_delta:.4f}"


def test_greeks_recomputation_put_delta_negativo(tmp_path):
    """Delta de put OTM debe ser negativo (entre -0.5 y 0)."""
    _write_mini_csv(tmp_path, _make_mini_csv_schema_a(), "GGAL_HIST_2024-02.csv")
    df = load_historical_options(tmp_path, "GGAL_HIST_2024-02.csv", recompute_greeks_scheme_a=True)

    put_delta = df.loc[df["tipo"] == "Put", "delta"].iloc[0]
    assert -0.5 < put_delta < 0, f"Delta put OTM inesperado: {put_delta:.4f}"


def test_greeks_no_recompute_quedan_nan(tmp_path):
    """Con recompute_greeks_scheme_a=False, griegas del esquema A quedan NaN."""
    _write_mini_csv(tmp_path, _make_mini_csv_schema_a(), "GGAL_HIST_2024-02.csv")
    df = load_historical_options(tmp_path, "GGAL_HIST_2024-02.csv", recompute_greeks_scheme_a=False)

    assert df["delta"].isna().all(), "Se esperaban delta NaN con recompute=False"
    assert df["gamma"].isna().all()


def test_greeks_vi_nan_queda_nan(tmp_path):
    """Fila con VI NaN (strike sin cotización) debe tener griega NaN incluso con recompute=True."""
    header = (
        "FECHA;ESPECIE;BASE;TIPO;ÚLTIMO;%;MONTO;HORA;APE.;MAX.;MIN.;"
        "C. ANT.;NOMINAL;PRECIO GGAL;VAR. % GGAL;TLR;VI %;VE %;DÍAS AL VTO.;PLAZO (años)"
    )
    # VI vacía → NaN
    row_no_vi = (
        "15/02/2024;GFGC11000CO;110,00;Call;0,000;-%;0,000;17:00;0,000;0,000;0,000;"
        "0,000;0;100,00;0,50%;5,00%;;0,00%;30;0,082192"
    )
    content = "\n".join([header, row_no_vi])
    _write_mini_csv(tmp_path, content, "GGAL_HIST_2024-02.csv")

    df = load_historical_options(tmp_path, "GGAL_HIST_2024-02.csv", recompute_greeks_scheme_a=True)

    assert np.isnan(df["delta"].iloc[0]), "Delta debe ser NaN cuando VI es NaN"


# ---------------------------------------------------------------------------
# 6. Esquema B — griegas vienen del archivo
# ---------------------------------------------------------------------------

def _make_mini_csv_schema_b() -> str:
    """CSV mínimo de esquema B (24 columnas) con griegas del archivo."""
    header = (
        "FECHA;ESPECIE;BASE;TIPO;ÚLTIMO;%;MONTO;HORA;APE.;MAX.;MIN.;"
        "C. ANT.;NOMINAL;PRECIO GGAL;VAR. % GGAL;TLR;VI %;VE %;DÍAS AL VTO.;PLAZO (años);"
        "DELTA;GAMMA;VEGA;THETA"
    )
    row = (
        "15/08/2025;GFGC22000CO;220,00;Call;5,50;2,00%;50000,00;17:00;5,20;5,80;5,00;"
        "5,20;200;210,00;1,00%;5,00%;45,00%;12,00%;45;0,123288;"
        "0,3500;0,0120;0,8500;-0,0250"
    )
    return "\n".join([header, row])


def test_schema_b_griegas_del_archivo(tmp_path):
    """Esquema B: las griegas se toman del archivo sin recalcular."""
    _write_mini_csv(tmp_path, _make_mini_csv_schema_b(), "GGAL_HIST_2025-08.csv")

    df = load_historical_options(tmp_path, "GGAL_HIST_2025-08.csv", recompute_greeks_scheme_a=True)

    assert df["esquema"].iloc[0] == "B"
    assert df["delta"].iloc[0] == pytest.approx(0.35)
    assert df["gamma"].iloc[0] == pytest.approx(0.012)
    assert df["vega"].iloc[0] == pytest.approx(0.85)
    assert df["theta"].iloc[0] == pytest.approx(-0.025)


# ---------------------------------------------------------------------------
# 7. Columnas de salida y orden
# ---------------------------------------------------------------------------

def test_columnas_salida_orden(tmp_path):
    """El DataFrame retornado tiene exactamente las columnas en el orden de TIDY_COLUMNS."""
    _write_mini_csv(tmp_path, _make_mini_csv_schema_a(), "GGAL_HIST_2024-02.csv")
    df = load_historical_options(tmp_path, "GGAL_HIST_2024-02.csv")

    assert list(df.columns) == TIDY_COLUMNS, (
        f"Columnas esperadas:\n{TIDY_COLUMNS}\nObtenidas:\n{list(df.columns)}"
    )


def test_opex_extraido_del_nombre(tmp_path):
    """El campo opex se extrae correctamente del nombre del archivo."""
    _write_mini_csv(tmp_path, _make_mini_csv_schema_a(), "GGAL_HIST_2024-06.csv")
    df = load_historical_options(tmp_path, "GGAL_HIST_2024-06.csv")

    assert (df["opex"] == "2024-06").all()


def test_fecha_parseada_correctamente(tmp_path):
    """FECHA dd/mm/yyyy se parsea como pd.Timestamp."""
    _write_mini_csv(tmp_path, _make_mini_csv_schema_a(), "GGAL_HIST_2024-02.csv")
    df = load_historical_options(tmp_path, "GGAL_HIST_2024-02.csv")

    assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
    assert df["fecha"].iloc[0] == pd.Timestamp("2024-02-15")


# ---------------------------------------------------------------------------
# 8. Manejo de directorio vacío
# ---------------------------------------------------------------------------

def test_load_directorio_sin_archivos(tmp_path):
    """FileNotFoundError si no hay archivos con el patrón."""
    with pytest.raises(FileNotFoundError):
        load_historical_options(raw_dir=tmp_path)


def test_load_directorio_inexistente():
    """FileNotFoundError si el directorio no existe."""
    with pytest.raises(FileNotFoundError):
        load_historical_options(raw_dir=Path("data/raw/options/no_existe_xyz"))


# ---------------------------------------------------------------------------
# 9. Filtrado de filas-fantasma (fecha NaT)
# ---------------------------------------------------------------------------

def _make_csv_con_filas_fantasma() -> str:
    """CSV esquema A con una fila de datos válida y dos filas con solo separadores."""
    header = (
        "FECHA;ESPECIE;BASE;TIPO;ÚLTIMO;%;MONTO;HORA;APE.;MAX.;MIN.;"
        "C. ANT.;NOMINAL;PRECIO GGAL;VAR. % GGAL;TLR;VI %;VE %;DÍAS AL VTO.;PLAZO (años)"
    )
    fila_valida = (
        "15/02/2024;GFGC11000CO;110,00;Call;1,50;-5,00%;12000,00;17:00;1,60;1,80;1,20;"
        "1,60;100;100,00;0,50%;5,00%;40,00%;10,00%;30;0,082192"
    )
    # Líneas con solo separadores — exactamente como las genera Excel al exportar
    fila_fantasma_1 = ";" * 19   # 20 campos vacíos separados por 19 puntos y coma
    fila_fantasma_2 = ";" * 19
    return "\n".join([header, fila_valida, fila_fantasma_1, fila_fantasma_2])


def test_filas_fantasma_descartadas(tmp_path):
    """Líneas con solo separadores (fecha NaT) se descartan en el parseo."""
    _write_mini_csv(tmp_path, _make_csv_con_filas_fantasma(), "GGAL_HIST_2024-02.csv")
    df = load_historical_options(tmp_path, "GGAL_HIST_2024-02.csv", recompute_greeks_scheme_a=False)

    # Solo debe quedar la fila válida
    assert len(df) == 1, f"Se esperaba 1 fila, se obtuvieron {len(df)}"
    assert df["fecha"].notna().all(), "Todas las fechas deben ser válidas tras el filtrado"


def test_filas_fantasma_fecha_valida_conservada(tmp_path):
    """La fila de datos válida conserva sus valores correctamente tras el filtrado."""
    _write_mini_csv(tmp_path, _make_csv_con_filas_fantasma(), "GGAL_HIST_2024-02.csv")
    df = load_historical_options(tmp_path, "GGAL_HIST_2024-02.csv", recompute_greeks_scheme_a=False)

    assert df["fecha"].iloc[0] == pd.Timestamp("2024-02-15")
    assert df["especie"].iloc[0] == "GFGC11000CO"
    assert df["strike"].iloc[0] == pytest.approx(110.0)
