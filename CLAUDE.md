# CLAUDE.md — Contexto del proyecto

> Archivo de contexto persistente para Claude (VS Code).
> Leer esto antes de cualquier tarea de código. Actualizar a medida que se cierren decisiones.

---

## 1. Identidad del proyecto

**Repositorio:** `tif-barbell-options-ggal`
**Tipo:** Trabajo de Investigación Final (TIF) — Licenciatura en Finanzas, UADE
**Autores:** Santiago Quintero (LU 1176122) · Matías Malo Medrano (LU 1147831)
**Tutor:** Mauro Natalucci
**Año académico:** 2025–2026 (1C 2026)

**Tesis:** *Mitigación del Riesgo de Cola Izquierda en el Equity Argentino: Un Enfoque Bimodal mediante la Estrategia Barbell y Derivados de ADRs (2015–2025).*

**Objetivo del código:** Implementar un backtest empírico de la Estrategia Barbell sobre opciones de GGAL, midiendo desempeño en USD, contra dos benchmarks (ADR GGAL buy-and-hold y Merval en USD).

---

## 2. División de trabajo

| Persona | Responsabilidad | Módulos |
|---|---|---|
| **Santiago (datos)** | Construir la capa de datos limpia y reproducible | `data_loader.py`, `data_audit.py`, `fx.py`, descarga de benchmarks |
| **Matías (estrategia)** | Implementar lógica cuantitativa y motor de simulación | `greeks.py`, `strategy.py`, `backtest.py`, `metrics.py`, `report.py` |

Trabajamos con **ramas de Git separadas** (`feature/data-*` y `feature/strategy-*`) y mergeamos a `main` vía Pull Request.

---

## 3. La Estrategia Barbell — versión operativa

Asignación **bimodal** del capital, evitando el "medio frágil":

- **Polo seguro (≈90%):** posición en USD libre de riesgo (T-Bills 3M). Inmuniza el capital ante eventos de cola izquierda.
- **Polo agresivo (≈10%):** compra sistemática de opciones OTM sobre GGAL. Pérdida máxima = prima pagada; ganancia potencialmente no lineal vía Gamma.

La tesis sostiene que en el mercado argentino, dominado por *azar salvaje* (Mandelbrot) y *Cisnes Negros* políticos (PASO 2019, cambios de régimen), una arquitectura convexa supera estructuralmente al buy-and-hold lineal.

---

## 4. Decisiones metodológicas cerradas con el tutor

> Estas decisiones surgieron de la devolución del Prof. Natalucci sobre la 2da entrega (40%). **No cambiar sin discutirlas con él.**

| # | Decisión | Razón |
|---|---|---|
| 5 | **Subyacente:** opción local de GGAL convertida a USD vía CCL (no derivado sintético sobre ADR) | Data-driven, reproducible, defendible. Evita el problema de "derivados teóricos del ADR sin mercado observable" |
| 7 | (ok — sin cambios) | — |
| 8 | **Moneyness:** % fijo de distancia (spot vs strike), no delta objetivo | Observable directo, no depende de Black-Scholes, robusto a errores de VI |
| 9 | **Volatilidad:** VI **por strike** (no promedio) | Permite analizar skew/sonrisa; el promedio aplasta la información del riesgo asimétrico |
| 10 | **Costos de transacción / bid-ask spread:** modelado por liquidez | Volumen alto → spread bajo (1–2%); volumen bajo → spread alto (8–10%). Usar bid/ask explícito si está disponible, sino proxy por volumen |
| 11 | **Benchmarks (ambos):** ADR GGAL buy-and-hold + Merval en USD | El primero mide efecto activo (vs el subyacente); el segundo mide efecto mercado (vs el equity argentino completo) |

---

## 5. Universo de datos

### 5.1 Datos que ya tenemos

**Opciones locales de GGAL** — planillas CSV por vencimiento (Opex), meses pares desde octubre 2023:

- 2023: oct, dic
- 2024: feb, abr, jun, ago, oct, dic
- 2025: feb, abr, jun, ago, oct, dic
- 2026: feb, abr (en adelante, según se agreguen)

**Estructura de cada planilla:**
- 19 columnas de metadata (fecha, GGAL local en ARS, CCL, ADR USD, días al vencimiento, TLR, volúmenes calls/puts, VI promedio calls/puts, ADR%)
- Matriz de strikes a partir de columna 20: cada strike ocupa **2 columnas** (CALL / PUT)
- Decimales en formato argentino (coma)
- El encabezado de strikes **puede cambiar dentro del mismo archivo** cuando cambia la grilla cotizada — el parser debe re-detectar encabezados

**Punto crítico a auditar:** En la planilla de octubre 2023, la columna del PUT se comporta como prima correctamente (crece monótonamente con strike). La columna del CALL tiene valores inconsistentes en muchos strikes — posiblemente mezcla prima con volumen/nominal operado, y la paridad put-call no cierra. **Resolver esto en `data_audit.py` antes de simular.**

### 5.2 Datos que faltan traer

#### a) ADR de GGAL (NYSE) — diario 2015–2026
- **Fuente recomendada:** Yahoo Finance (`yfinance`, ticker `GGAL`)
- **Frecuencia:** diaria
- **Uso:** benchmark idiosincrático + cross-check de la columna ADR de las planillas + insumo para spot histórico de opciones sintéticas (si se decide ese camino)

#### b) Merval en pesos → convertido a USD CCL — diario 2015–2026
- **Fuente recomendada:** BYMA histórico o Investing.com (ticker `M.BA` en Yahoo Finance también funciona pero a veces tiene gaps)
- **Conversión:** dividir cada cierre del Merval (ARS) por el CCL del mismo día
- **CCL histórico:** se puede aproximar como `precio_AY24_ARS / precio_AY24_USD` o `precio_GGAL_local / precio_ADR × ratio_ADR` (donde ratio_ADR de GGAL = 10 acciones por ADR)
- **Uso:** benchmark de mercado

#### c) T-Bills 3M — diario 2015–2026
- **Fuente recomendada:** FRED (`DGS3MO` o `DTB3`), gratis vía API
- **Uso:** polo seguro del Barbell + tasa libre de riesgo para griegas (cross-check vs la columna `TLR` de las planillas)

### 5.3 El problema de las opciones OTM históricas 2015–2023 (CRÍTICO)

**Decisión del usuario:** investigar fuentes pagas/gratuitas antes de generar sintéticos. Estado actual del análisis:

**Opciones reales del ADR de GGAL (NYSE) 2015–2026:**
- **OptionMetrics IvyDB US** — gold standard académico, contiene GGAL desde 1996. Acceso vía Wharton Research Data Services (WRDS). UADE **no parece tener convenio con WRDS** (verificar con biblioteca/dirección de carrera). Sin convenio: producto institucional, miles de USD.
- **ORATS** — datos EOD desde 2007, calidad alta, planes pagos.
- **CBOE DataShop** — opciones EOD desde 2010+, paga.
- **Databento** — pay-per-use con $125 USD de crédito inicial gratis; puede alcanzar para descargar GGAL OTM histórico si se filtra bien.
- **Yahoo Finance / Webull / Investing** — opciones **actuales únicamente**, no histórico profundo.

**Opciones locales de GGAL en BYMA 2015–2023:**
- No hay fuente pública con histórico granular. Las planillas del usuario (desde oct 2023) son justamente lo que falta hacia atrás.
- BYMA publica boletines diarios en PDF, pero parsearlos para todo el periodo sería un proyecto en sí mismo.

**Caminos posibles (a discutir con el tutor antes de codear):**

1. **Pedir acceso a WRDS por UADE** (la Lic. en Finanzas está afiliada a CFA Institute — chequear si hay convenio académico de datos).
2. **Probar Databento con los $125 USD gratis** para descargar opciones del ADR GGAL filtradas a OTM puts/calls a una distancia fija de moneyness. Si alcanza, este es el dataset óptimo.
3. **Generar precios sintéticos con Black-Scholes calibrado al skew observado en 2023–2025.** Riesgo metodológico serio: si la VI usada para sintetizar es la histórica realizada (HV), se subestiman las primas de los puts OTM y **se invalida la propia tesis** (que afirma que esos puts están sobreprecio por crashophobia, no subpreciados). Solución parcial: calibrar la skew a la observada en las planillas reales y aplicarla hacia atrás, pero **se vuelve circular**.
4. **Reducir el backtest a 2023–2026 con datos reales** y mantener 2015–2025 como análisis narrativo del subyacente (drawdowns del ADR, eventos políticos). **El usuario rechazó esta opción**: quiere backtest completo 2015–2025.

**Mi recomendación para discusión:** intentar Databento primero (camino 2). Si no alcanza el crédito, calibrar sintéticos con la skew de 2023–2025 y dejar explícito en el capítulo metodológico el supuesto y sus limitaciones (camino 3). Mantener 2023–2026 con datos reales como **validación out-of-sample** del modelo sintético — eso es defendible académicamente y le da rigor.

---

## 6. Estructura del repositorio

```
tif-barbell-options-ggal/
│
├── CLAUDE.md                       ← este archivo
├── README.md                       ← instrucciones de instalación y uso
├── requirements.txt                ← dependencias Python
├── config.yaml                     ← parámetros de la estrategia (editable)
├── .gitignore
│
├── data/
│   ├── raw/                        ← CSVs originales sin tocar
│   │   ├── options/                ← planillas de opciones GGAL por Opex
│   │   │   ├── GGAL_OPEX_2023-10.csv
│   │   │   ├── GGAL_OPEX_2023-12.csv
│   │   │   └── ...
│   │   ├── adr/                    ← histórico diario ADR GGAL
│   │   ├── merval/                 ← histórico Merval en ARS
│   │   ├── ccl/                    ← histórico CCL
│   │   └── tbills/                 ← histórico T-Bills 3M (FRED)
│   ├── processed/                  ← outputs de data_loader (tidy format)
│   │   ├── options_tidy.parquet
│   │   ├── adr_usd_daily.parquet
│   │   ├── merval_usd_daily.parquet
│   │   └── tbills_daily.parquet
│   └── README.md                   ← documenta cada fuente, formato, columnas
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              ← parsea planillas wide → formato tidy
│   ├── data_audit.py               ← validaciones: paridad put-call, monotonía, VI
│   ├── fx.py                       ← conversión ARS↔USD vía CCL
│   ├── greeks.py                   ← Black-Scholes, griegas, VI por strike
│   ├── strategy.py                 ← lógica de la Barbell
│   ├── backtest.py                 ← motor de simulación
│   ├── metrics.py                  ← MDD, Expected Shortfall, Sortino, Sharpe
│   └── report.py                   ← gráficos y tablas finales
│
├── scripts/
│   ├── download_adr.py             ← descarga ADR GGAL desde Yahoo
│   ├── download_merval.py          ← descarga Merval desde Yahoo / BYMA
│   ├── download_tbills.py          ← descarga T-Bills desde FRED
│   └── download_options_historical.py  ← TBD: Databento / WRDS / sintético
│
├── notebooks/
│   ├── 01_audit_datos.ipynb        ← exploración inicial de la planilla oct 2023
│   ├── 02_explore_skew.ipynb       ← análisis del skew/sonrisa de VI
│   └── 03_validate_barbell.ipynb   ← validación de la estrategia
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_greeks.py
│   ├── test_strategy.py
│   └── test_metrics.py
│
├── agents/                         ← prompts persistentes para chats Claude.ai
│   ├── agent-a-code-reviewer.md
│   ├── agent-b-theory-checker.md
│   ├── agent-c-developer.md
│   └── agent-d-integration.md
│
└── main.py                         ← punto de entrada del backtest
```

---

## 7. Formato tidy (contrato entre módulos)

`data_loader.py` debe producir un DataFrame con este esquema. Todo lo demás consume esto, sin importar las rarezas del CSV original:

```
columna           | tipo         | descripción
------------------|--------------|----------------------------------------
fecha             | datetime     | fecha de rueda
opex              | str          | identificador de la serie (ej. "2023-10")
tipo              | category     | "CALL" o "PUT"
strike            | float        | precio de ejercicio (ARS)
prima             | float        | precio de cierre de la opción (ARS)
volumen           | float        | volumen nominal operado
vi_implicita      | float        | VI estimada (NaN si data_loader no la calcula)
dias_vto          | int          | días hasta vencimiento
ggal_local        | float        | spot GGAL en ARS
adr_usd           | float        | precio del ADR en USD
ccl               | float        | CCL del día
tlr               | float        | tasa libre de riesgo de la planilla
```

**Reglas:**
- Una fila por (fecha, opex, strike, tipo) con cotización válida ese día.
- `0,000` → `NaN` (strike sin cotización).
- Decimales argentinos (`"1.234,56"`) → float (`1234.56`).
- Conversión a USD se hace en `fx.py`, **no** en `data_loader.py` (separation of concerns).

---

## 8. Stack y dependencias

**Python ≥ 3.11**

```
pandas
numpy
scipy            # optimización para VI implícita
matplotlib
seaborn
yfinance         # ADR, Merval
pandas-datareader # FRED T-Bills
pyyaml           # config
pytest           # testing
jupyter          # notebooks
```

Mantener `requirements.txt` versionado.

---

## 9. Convenciones de código

- **Estilo:** PEP 8, líneas ≤ 100 chars
- **Docstrings:** Google style en cada función pública
- **Type hints:** obligatorios en firmas de funciones
- **Logging:** módulo `logging`, no `print` en código de producción (solo en notebooks)
- **Configuración:** todo parámetro económico vive en `config.yaml`, **no hardcodear**
- **Tests:** cada función no trivial tiene su test en `tests/`
- **Idioma:** comentarios y docstrings en español (consistente con la tesis); nombres de variables y funciones en inglés (estándar de la comunidad)

---

## 10. Estado actual y próximos pasos

### Hecho
- [x] Marco teórico de la tesis (2da entrega 40%, aprobada)
- [x] Decisiones metodológicas cerradas con el tutor
- [x] Repo en GitHub creado y clonado localmente
- [x] Documento preliminar de diseño del backtest

### En curso (Santiago — datos)
- [ ] Estructura de carpetas `data/raw/`, `data/processed/`, `data/external/`
- [ ] `data_loader.py` — parser de planillas wide → tidy
- [ ] `data_audit.py` — resolver el bloqueante de la columna CALL (prima vs volumen)
- [ ] `scripts/download_adr.py` — bajar ADR GGAL 2015–2026 (Yahoo)
- [ ] `scripts/download_tbills.py` — bajar T-Bills 3M 2015–2026 (FRED)
- [ ] `scripts/download_merval.py` — bajar Merval 2015–2026 + conversión a USD CCL
- [ ] Notebook `01_audit_datos.ipynb`

### Pendiente (decisión metodológica)
- [ ] Resolver fuente de opciones históricas 2015–2023 (Databento vs sintético vs reducir backtest)

### Bloqueado (Matías — estrategia)
- Empieza cuando Santiago entregue `data_loader.py` + `fx.py` funcionando sobre al menos una planilla.

---

## 11. Cómo usar este archivo

**Cuando trabajes con Claude (este agente en VS Code):**
- Asumí que ya leyó este archivo al inicio de la sesión. Si no, pedile que lo lea.
- Cuando se cierre una decisión nueva, actualizar la sección correspondiente y commitear.
- Para tareas complejas que cruzan módulos, abrí un chat en Claude.ai con uno de los agentes especializados (ver `agents/`) y referencias este CLAUDE.md como contexto.

**Cuando arranque una nueva conversación con Claude:**
> "Leé `CLAUDE.md`. Estamos trabajando en [X]. Necesito que [Y]."
