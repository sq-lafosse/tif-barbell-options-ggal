# data/ — Guía de datos del proyecto

## Datos versionados en el repositorio

Los archivos de datos **públicos** se incluyen en el repositorio para facilitar la reproducibilidad.
Al clonar, ya tenés todo lo necesario para correr el pipeline sin descargas manuales.

### Datos crudos de opciones (`data/raw/options/`)

Los 17 archivos `GGAL_HIST_YYYY-MM.csv` están versionados. Cubren el período
**2023-08-18 → 2026-06-12** con cobertura continua (un archivo por ciclo Opex).

> **Fuente:** archivos `Historial` provenientes de fuente pública.

**Formato:** decimales argentinos (coma como separador decimal, punto como separador de miles),
separador de campos `;`. El parser `src/data_loader.py` los convierte automáticamente.

**Dos esquemas de columnas:**
- **Esquema A** (2023-10 → 2025-04, 10 archivos): 20 columnas, sin griegas.
- **Esquema B** (2025-06 → 2026-06, 7 archivos): 24 columnas, incluye DELTA, GAMMA, VEGA, THETA.

Las griegas del esquema A se recalculan con `src/greeks.py` calibrado contra los archivos B.
Ver protocolo completo en `CLAUDE.md §5.3`.

### Datos externos descargados (`data/raw/adr/`, `data/raw/ccl/`, `data/raw/merval/`, `data/raw/tbills/`)

Estos parquets también están versionados. Se generan con los scripts de descarga y **no es
necesario volver a correrlos** salvo que quieras actualizar la serie hasta hoy:

| Archivo | Script | Fuente |
|---|---|---|
| `data/raw/adr/GGAL_ADR_daily.parquet` | `scripts/download_adr.py` | Yahoo Finance (`GGAL`) |
| `data/raw/ccl/CCL_daily.parquet` | `scripts/download_ccl.py` | Ratio GGAL local / ADR |
| `data/raw/merval/MERVAL_daily.parquet` | `scripts/download_merval.py` | Yahoo Finance (`M.BA`) |
| `data/raw/tbills/TBILLS_3M_daily.parquet` | `scripts/download_tbills.py` | FRED (`DTB3`) |
| `data/raw/options/SYNTHETIC_2019_2023.parquet` | `scripts/generate_synthetic_options.py` | Black-Scholes calibrado |

---

## Outputs procesados (`data/processed/`)

También versionados. Se regeneran corriendo el pipeline completo:

| Archivo | Genera | Descripción |
|---|---|---|
| `options_tidy.parquet` | `python -m src.data_loader` | 17 CSV → formato tidy unificado (40 624 filas) |
| `options_full_usd.parquet` | `python -m src.fx` | Tidy + sintético, todo en USD (53 610 filas) |

---

## Cómo regenerar todo desde cero

Si por alguna razón querés volver a producir todos los derivados (por ejemplo, para extender
las series hasta una fecha más reciente), el orden del pipeline es:

```bash
# 1. Datos externos
python scripts/download_adr.py
python scripts/download_tbills.py
python scripts/download_ccl.py
python scripts/download_merval.py

# 2. Opciones sintéticas 2019–2023
python scripts/generate_synthetic_options.py

# 3. Pipeline de procesamiento
python -m src.data_loader
python -m src.fx
```

---

## Fuentes y licencias

| Dataset | Fuente | Licencia / Condiciones |
|---|---|---|
| Opciones GGAL local | Fuente pública | Acceso libre |
| ADR GGAL | Yahoo Finance vía `yfinance` | Solo uso personal/académico |
| Merval | Yahoo Finance vía `yfinance` | Solo uso personal/académico |
| T-Bills 3M | FRED (Federal Reserve) | Público, uso libre |
| Opciones sintéticas 2019–2023 | Black-Scholes calibrado (este repo) | Ver `CLAUDE.md §5.5` |
