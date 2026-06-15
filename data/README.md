# data/ — Guía de datos del proyecto

## Por qué `data/raw/` está vacía en el repo

Los archivos de datos **no se versionan en Git** por dos razones:

1. **Tamaño:** las planillas CSV de opciones, el histórico del ADR y los datos de FRED superan el límite razonable para un repositorio de código.
2. **Privacidad:** las planillas de opciones fueron recopiladas manualmente por los autores y no son de distribución pública.

El `.gitignore` excluye explícitamente `data/raw/**/*.csv`, `data/processed/` y `*.parquet`.

---

## Cómo poblar `data/raw/`

### 1. Opciones locales de GGAL (`data/raw/options/`)

Copiar manualmente las planillas CSV de vencimiento con el siguiente nombre de archivo:

```
GGAL_OPEX_YYYY-MM.csv
```

Ejemplos: `GGAL_OPEX_2023-10.csv`, `GGAL_OPEX_2024-02.csv`, etc.

La lista completa de archivos esperados está en `config.yaml` bajo la clave `expected_opex_files`.

> **Formato:** decimales argentinos (coma), separador de miles (punto). El parser `src/data_loader.py` los convierte automáticamente.

### 2. ADR GGAL — diario 2015–2026 (`data/raw/adr/`)

```bash
python scripts/download_adr.py
```

Descarga el histórico del ADR GGAL (NYSE) desde Yahoo Finance (ticker `GGAL`) y lo guarda en esta carpeta.

### 3. Merval en ARS — diario 2015–2026 (`data/raw/merval/`)

```bash
python scripts/download_merval.py
```

Descarga el Merval desde Yahoo Finance (ticker `M.BA`) y lo guarda en esta carpeta. La conversión a USD se hace por separado en `src/fx.py`.

### 4. CCL histórico (`data/raw/ccl/`)

El CCL se aproxima a partir del ratio `precio_GGAL_local / (precio_ADR × 10)`. Generado automáticamente por `src/fx.py` al correr el pipeline principal.

### 5. T-Bills 3M — diario 2015–2026 (`data/raw/tbills/`)

```bash
python scripts/download_tbills.py
```

Descarga la tasa T-Bills 3M desde FRED (serie `DTB3`) vía `pandas-datareader`.

---

## Outputs procesados (`data/processed/`)

Esta carpeta también está excluida de Git. Se genera al correr `main.py` o `src/data_loader.py`:

| Archivo | Descripción |
|---|---|
| `options_tidy.parquet` | Planillas Opex en formato tidy (ver `CLAUDE.md §7`) |
| `adr_usd_daily.parquet` | ADR GGAL en USD, frecuencia diaria |
| `merval_usd_daily.parquet` | Merval convertido a USD vía CCL |
| `tbills_daily.parquet` | T-Bills 3M diarios desde FRED |

---

## Fuentes y licencias

| Dataset | Fuente | Licencia / Condiciones |
|---|---|---|
| Opciones GGAL local | Recopilación manual (BYMA) | Datos privados, no redistribuir |
| ADR GGAL | Yahoo Finance vía `yfinance` | Solo uso personal/académico |
| Merval | Yahoo Finance vía `yfinance` | Solo uso personal/académico |
| T-Bills 3M | FRED (Federal Reserve) | Público, uso libre |
| Opciones históricas 2015–2022 | Pendiente (Databento / WRDS / sintético) | Ver `CLAUDE.md §5.3` |
