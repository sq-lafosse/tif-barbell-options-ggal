# tif-barbell-options-ggal

![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![Licencia](https://img.shields.io/badge/licencia-académica-lightgrey)

**Backtest empírico de la Estrategia Barbell sobre opciones de GGAL (2015–2026)**

Trabajo de Investigación Final — Licenciatura en Finanzas
Autores: Estudiante 1 y Estudiante 2

> La estrategia evalúa si una cartera bimodal (≈90% en T-Bills + ≈10% en opciones OTM de GGAL)
> mitiga el riesgo de cola izquierda del equity argentino, dominado por azar salvaje y eventos
> políticos extremos (PASO 2019, cambios de régimen). Se mide en USD contra dos benchmarks:
> ADR GGAL buy-and-hold y Merval en USD.

---

## Setup

### Requisitos previos

- Python ≥ 3.11
- Git

### Instalación paso a paso

```bash
# 1. Clonar el repositorio (los datos crudos ya vienen incluidos)
git clone https://github.com/<usuario>/tif-barbell-options-ggal.git
cd tif-barbell-options-ggal

# 2. Crear y activar el entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

Los archivos de datos (CSVs de opciones y parquets) están versionados en el repo.
No se requiere ninguna descarga manual. Ver [data/README.md](data/README.md) para
el detalle de cada fuente y cómo regenerar los derivados desde cero.

> Los datos crudos de opciones son de acceso público.

---

## Estructura

```
tif-barbell-options-ggal/
│
├── CLAUDE.md                   ← contexto del proyecto para Claude (leer primero)
├── README.md                   ← este archivo
├── requirements.txt            ← dependencias Python
├── config.yaml                 ← parámetros de la estrategia (editable)
├── .gitignore
├── main.py                     ← punto de entrada del backtest
│
├── data/
│   ├── raw/                    ← datos crudos (versionados)
│   │   ├── options/            ← 17 CSV Historial GGAL por Opex + sintético .parquet
│   │   ├── adr/                ← histórico ADR GGAL (.parquet)
│   │   ├── merval/             ← histórico Merval en ARS (.parquet)
│   │   ├── ccl/                ← histórico CCL (.parquet)
│   │   └── tbills/             ← histórico T-Bills 3M (.parquet)
│   ├── processed/              ← outputs del pipeline en .parquet (versionados)
│   └── README.md               ← detalle de fuentes y cómo regenerar
│
├── src/                        ← módulos del pipeline
│   ├── data_loader.py          ← parsea planillas wide → formato tidy
│   ├── data_audit.py           ← validaciones: paridad put-call, monotonía, VI
│   ├── fx.py                   ← conversión ARS↔USD vía CCL
│   ├── greeks.py               ← Black-Scholes, griegas, VI por strike
│   ├── strategy.py             ← lógica de la Barbell
│   ├── backtest.py             ← motor de simulación
│   ├── metrics.py              ← MDD, Expected Shortfall, Sortino, Sharpe
│   └── report.py               ← gráficos y tablas finales
│
├── scripts/                    ← descarga de datos de fuentes externas
│   ├── download_adr.py
│   ├── download_merval.py
│   ├── download_tbills.py
│   └── download_options_historical.py
│
├── notebooks/                  ← exploración y validación
│   ├── 01_audit_datos.ipynb
│   ├── 02_explore_skew.ipynb
│   └── 03_validate_barbell.ipynb
│
├── tests/                      ← suite de tests unitarios
│   ├── test_data_loader.py
│   ├── test_greeks.py
│   ├── test_strategy.py
│   └── test_metrics.py
│
└── agents/                     ← prompts persistentes para Claude.ai
```

---

## Cómo correr el backtest

> **Estado actual:** el pipeline está en construcción. El siguiente comando ejecuta el skeleton
> pero sin lógica implementada todavía. Ver [Estado del proyecto](#estado-del-proyecto).

```bash
# Activar el entorno virtual primero (ver Setup)
python main.py
```

Cuando el pipeline esté completo, este comando:
1. Leerá la configuración de `config.yaml`
2. Cargará y auditará los datos de `data/raw/`
3. Convertirá todo a USD vía CCL
4. Ejecutará el motor de backtest
5. Calculará métricas (MDD, ES, Sortino, Sharpe)
6. Generará un reporte con gráficos y tablas

Para correr los tests:

```bash
pytest tests/
```

---

## Estado del proyecto

Ver [CLAUDE.md — Sección 10: Estado actual y próximos pasos](CLAUDE.md#10-estado-actual-y-próximos-pasos) para el estado detallado de cada módulo y la división de trabajo entre autores.

**Resumen:**
- ✅ Marco teórico y decisiones metodológicas cerradas con el tutor
- ✅ Estructura del repositorio
- 🔄 En curso: capa de datos (Estudiante 1)
- ⏳ Bloqueado hasta tener `data_loader.py`: lógica de la estrategia (Estudiante 2)
- ❓ Pendiente: resolución de la fuente de opciones históricas 2015–2023

---

## Equipo

| Rol | Responsabilidad |
|---|---|
| Estudiante 1 — Datos | Capa de datos: `data_loader.py`, `data_audit.py`, `fx.py`, scripts de descarga |
| Estudiante 2 — Estrategia | Estrategia: `greeks.py`, `strategy.py`, `backtest.py`, `metrics.py`, `report.py` |
| Tutor académico | Dirección académica — Licenciatura en Finanzas (2026) |
