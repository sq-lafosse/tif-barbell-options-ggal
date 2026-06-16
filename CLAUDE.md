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

**Visión a largo plazo:** Este repositorio se diseña también como **base reutilizable** para futuras estrategias de opciones sobre activos argentinos. Por eso los datos crudos se mantienen intactos y la arquitectura es modular.

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

### 5.1 Fuente principal: archivos Historial de opciones GGAL

Los datos crudos son **17 archivos CSV** llamados `Historial`, uno por ciclo Opex (vencimiento) de opciones de GGAL desde octubre 2023 hasta junio 2026. Cobertura: **18/08/2023 → 12/06/2026, sin gaps temporales** (cada archivo cubre el ciclo del Opex anterior + ruedas iniciales del siguiente).

**Formato del archivo Historial:**
- Separador: `;` (punto y coma)
- Encoding: UTF-8 con BOM
- Decimales: formato argentino (coma)
- Estructura: **una fila por contrato por fecha** (formato ya casi tidy — no es wide)
- Cada fila identifica explícitamente `ESPECIE`, `BASE` (strike), `TIPO` (Call/Put). No hay ambigüedad de columnas.

**Convención de nombres en el repo:** `GGAL_HIST_YYYY-MM.csv` (17 archivos esperados).

### 5.2 Dos esquemas de columnas dentro del Historial

La fuente cambió el formato en **junio 2025**. Hay dos esquemas a manejar:

#### Esquema A (viejo) — 20 columnas — Archivos 2023-10 a 2025-04 (10 archivos)
```
FECHA, ESPECIE, BASE, TIPO, ÚLTIMO, %, MONTO, HORA, APE., MAX., MIN.,
C. ANT., NOMINAL, PRECIO GGAL, VAR. % GGAL, TLR, VI %, VE %,
DÍAS AL VTO., PLAZO (años)
```

#### Esquema B (nuevo) — 24 columnas — Archivos 2025-06 a 2026-06 (7 archivos)
Las 20 columnas de A + **DELTA, GAMMA, VEGA, THETA** (griegas pre-calculadas por la fuente).

**Implicancia:** los archivos del esquema A no traen griegas. La VI sí está en los 17.

### 5.3 Decisión sobre griegas

**Política adoptada:** Recalcular DELTA, GAMMA, VEGA, THETA **únicamente** para los 10 archivos del esquema A, usando `greeks.py` con la `VI %` del propio archivo como input. Para los 7 archivos del esquema B, **confiamos en las griegas tal como vienen** (la fuente las publica).

**Protocolo de calibración** (importante para Matías cuando programe `greeks.py`):
1. Implementar Black-Scholes estándar (con la convención que use la fuente — probablemente call/put europea, spot del subyacente local en ARS, TLR del archivo, días al vto del archivo).
2. **Validación cruzada:** correr `greeks.py` sobre los archivos del **esquema B** y comparar columna por columna contra DELTA/GAMMA/VEGA/THETA del archivo. Si coinciden con tolerancia razonable (ej. < 1% de error relativo), la implementación está calibrada con la fuente.
3. Una vez calibrado, aplicar `greeks.py` a los archivos del esquema A para llenar las 4 columnas faltantes.
4. Resultado: dataset final homogéneo con griegas consistentes en los 17 archivos.

### 5.4 Otros datos necesarios (a descargar después)

#### a) ADR de GGAL (NYSE) — diario 2015–2026
- **Fuente recomendada:** Yahoo Finance (`yfinance`, ticker `GGAL`)
- **Frecuencia:** diaria
- **Uso:** benchmark idiosincrático + cross-check de la columna `PRECIO GGAL` de las planillas (las planillas tienen GGAL local en ARS; el ADR es la referencia en USD) + spot histórico para opciones sintéticas si se decide ese camino

#### b) Merval en pesos → convertido a USD CCL — diario 2015–2026
- **Fuente recomendada:** BYMA histórico o Investing.com (ticker `M.BA` en Yahoo Finance también funciona pero a veces tiene gaps)
- **Conversión:** dividir cada cierre del Merval (ARS) por el CCL del mismo día
- **CCL histórico:** se puede aproximar como `precio_AY24_ARS / precio_AY24_USD` o `precio_GGAL_local / precio_ADR × ratio_ADR` (donde ratio_ADR de GGAL = 10 acciones por ADR)
- **Uso:** benchmark de mercado

#### c) T-Bills 3M — diario 2015–2026
- **Fuente recomendada:** FRED (`DGS3MO` o `DTB3`), gratis vía API
- **Uso:** polo seguro del Barbell + tasa libre de riesgo para griegas (cross-check vs la columna `TLR` de las planillas)

### 5.5 El problema de las opciones OTM históricas 2015–2023 (CRÍTICO)

Los archivos Historial empiezan en oct 2023. La tesis abarca 2015–2025. **Faltan opciones para 2015–2023**.

**Decisión del usuario:** investigar fuentes pagas/gratuitas antes de generar sintéticos. Estado actual del análisis:

**Opciones reales del ADR de GGAL (NYSE) 2015–2026:**
- **OptionMetrics IvyDB US** — gold standard académico, contiene GGAL desde 1996. Acceso vía Wharton Research Data Services (WRDS). UADE **no parece tener convenio con WRDS** (verificar con biblioteca/dirección de carrera). Sin convenio: producto institucional, miles de USD.
- **ORATS** — datos EOD desde 2007, calidad alta, planes pagos.
- **CBOE DataShop** — opciones EOD desde 2010+, paga.
- **Databento** — pay-per-use con $125 USD de crédito inicial gratis; puede alcanzar para descargar GGAL OTM histórico si se filtra bien.
- **Yahoo Finance / Webull / Investing** — opciones **actuales únicamente**, no histórico profundo.

**Opciones locales de GGAL en BYMA 2015–2023:**
- No hay fuente pública con histórico granular. Los archivos Historial (desde oct 2023) son justamente lo que falta hacia atrás.
- BYMA publica boletines diarios en PDF, pero parsearlos para todo el periodo sería un proyecto en sí mismo.

**Caminos posibles (a discutir con el tutor antes de codear):**

1. **Pedir acceso a WRDS por UADE** (la Lic. en Finanzas está afiliada a CFA Institute — chequear si hay convenio académico de datos).
2. **Probar Databento con los $125 USD gratis** para descargar opciones del ADR GGAL filtradas a OTM puts/calls a una distancia fija de moneyness. Si alcanza, este es el dataset óptimo.
3. **Generar precios sintéticos con Black-Scholes calibrado al skew observado en 2023–2025.** Riesgo metodológico serio: si la VI usada para sintetizar es la histórica realizada (HV), se subestiman las primas de los puts OTM y **se invalida la propia tesis** (que afirma que esos puts están sobreprecio por crashophobia, no subpreciados). Solución parcial: calibrar la skew a la observada en los archivos reales y aplicarla hacia atrás, pero **se vuelve circular**.
4. **Reducir el backtest a 2023–2026 con datos reales** y mantener 2015–2025 como análisis narrativo del subyacente (drawdowns del ADR, eventos políticos). **El usuario rechazó esta opción**: quiere backtest completo 2015–2025.

**Recomendación para discusión:** intentar Databento primero (camino 2). Si no alcanza el crédito, calibrar sintéticos con la skew de 2023–2025 y dejar explícito en el capítulo metodológico el supuesto y sus limitaciones (camino 3). Mantener 2023–2026 con datos reales como **validación out-of-sample** del modelo sintético — eso es defendible académicamente y le da rigor.

### 5.6 Información del formato Opex (legacy, descartado)

> **Nota contextual:** además del Historial, la fuente también publica una pestaña llamada Opex con el mismo periodo. Ese formato fue evaluado y **descartado** porque el Historial es estructuralmente superior (formato tidy, tipo explícito, sin ambigüedad). Se documenta aquí para futuras estrategias que puedan querer evaluar esa fuente.

El formato Opex es **wide**:
- 19 columnas de metadata (similares a las del Historial)
- Matriz de strikes a partir de la columna 20: cada strike ocupa **2 columnas** (CALL / PUT)
- El encabezado de strikes puede cambiar dentro del mismo archivo cuando cambia la grilla cotizada
- Problema detectado: en la planilla de octubre 2023, la columna del PUT se comporta como prima correctamente (crece monótonamente con strike), pero la columna del CALL tiene valores inconsistentes en muchos strikes — posiblemente mezcla prima con volumen/nominal operado, y la paridad put-call no cierra
- **Decisión:** no usar Opex en este proyecto. Si una estrategia futura lo necesita, primero hay que resolver ese bloqueante con la fuente original

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
│   ├── raw/                        ← CSVs originales sin tocar (intactos)
│   │   ├── options/                ← archivos Historial GGAL_HIST_YYYY-MM.csv
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
│   ├── data_loader.py              ← lee los 17 Historial → formato tidy unificado
│   ├── data_audit.py               ← validaciones: paridad, monotonía, esquemas
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
│   ├── 01_audit_datos.ipynb        ← exploración inicial de los 17 archivos
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

`data_loader.py` debe producir un DataFrame con este esquema. Todo lo demás consume esto:

```
columna           | tipo         | descripción / origen
------------------|--------------|----------------------------------------------------
fecha             | datetime     | FECHA del archivo, parseada de dd/mm/yyyy
opex              | str          | identificador de la serie (ej. "2025-10"), derivado del nombre del archivo
especie           | str          | ESPECIE del archivo (ej. "GFGC38785O")
tipo              | category     | TIPO del archivo, "Call" o "Put"
strike            | float        | BASE del archivo (ARS)
prima             | float        | ÚLTIMO del archivo (precio de cierre de la opción, ARS)
monto             | float        | MONTO del archivo (ARS operados)
nominal           | int          | NOMINAL del archivo (contratos operados)
ggal_local        | float        | PRECIO GGAL del archivo (ARS)
var_ggal          | float        | VAR. % GGAL del archivo (decimal, no %)
tlr               | float        | TLR del archivo (decimal anualizado)
vi_implicita      | float        | VI % del archivo (decimal)
valor_extrinseco  | float        | VE % del archivo (decimal)
dias_vto          | int          | DÍAS AL VTO. del archivo
plazo_anios       | float        | PLAZO (años) del archivo
delta             | float        | DELTA del archivo si existe (esquema B); NaN si esquema A → llenar con greeks.py
gamma             | float        | GAMMA del archivo si existe (esquema B); NaN si esquema A → llenar con greeks.py
vega              | float        | VEGA del archivo si existe (esquema B); NaN si esquema A → llenar con greeks.py
theta             | float        | THETA del archivo si existe (esquema B); NaN si esquema A → llenar con greeks.py
esquema           | category     | "A" o "B" — para trazabilidad metodológica
fuente_archivo    | str          | nombre del archivo original (ej. "GGAL_HIST_2025-10.csv")
```

**Reglas críticas:**
- **Los CSV crudos en `data/raw/` no se tocan jamás.** El parser lee, transforma en memoria, y escribe a `data/processed/`.
- `0,000` y string vacío en numéricos → `NaN`.
- Decimales argentinos (`"1.234,56"`, `"40,03%"`) → float (`1234.56`, `0.4003`).
- Porcentajes (TLR, VI %, VE %, VAR. % GGAL) se convierten a decimales (dividir por 100).
- Conversión a USD se hace en `fx.py`, **no** en `data_loader.py` (separation of concerns).
- El campo `esquema` permite filtrar después por origen de las griegas — útil para análisis metodológico.

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
pyarrow          # parquet
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
- **Datos crudos:** intocables. Cualquier transformación deja el original sin modificar y produce un nuevo artefacto.

---

## 10. Estado actual y próximos pasos

### Hecho
- [x] Marco teórico de la tesis (2da entrega 40%, aprobada)
- [x] Decisiones metodológicas cerradas con el tutor
- [x] Repo en GitHub creado y clonado localmente
- [x] Esqueleto del repo (carpetas, config.yaml, README.md, requirements.txt, main.py)
- [x] Documento preliminar de diseño del backtest
- [x] Auditoría completa de los 17 archivos Historial — cobertura continua 2023-08 a 2026-06
- [x] Detección de dos esquemas (A: 20 cols sin griegas; B: 24 cols con griegas)
- [x] Decisión sobre griegas: recalcular las del esquema A con `greeks.py` calibrado contra B

### En curso (Santiago — datos)
- [ ] Subir los 17 archivos `GGAL_HIST_YYYY-MM.csv` a `data/raw/options/` (local, no al repo)
- [ ] `data_loader.py` — parser unificado de ambos esquemas → tidy
- [ ] `data_audit.py` — validaciones de esquema, paridad, consistencia
- [ ] `scripts/download_adr.py` — bajar ADR GGAL 2015–2026 (Yahoo)
- [ ] `scripts/download_tbills.py` — bajar T-Bills 3M 2015–2026 (FRED)
- [ ] `scripts/download_merval.py` — bajar Merval 2015–2026 + conversión a USD CCL
- [ ] Notebook `01_audit_datos.ipynb`

### Pendiente (decisión metodológica)
- [ ] Resolver fuente de opciones históricas 2015–2023 (Databento vs sintético vs reducir backtest)

### Bloqueado (Matías — estrategia)
- Empieza cuando Santiago entregue `data_loader.py` + `fx.py` funcionando sobre los 17 archivos.
- Primera tarea: `greeks.py` calibrado contra esquema B (ver protocolo en sección 5.3).

---

## 11. Cómo usar este archivo

**Cuando trabajes con Claude (este agente en VS Code):**
- Asumí que ya leyó este archivo al inicio de la sesión. Si no, pedile que lo lea.
- Cuando se cierre una decisión nueva, actualizar la sección correspondiente y commitear.
- Para tareas complejas que cruzan módulos, abrí un chat en Claude.ai con uno de los agentes especializados (ver `agents/`) y referencias este CLAUDE.md como contexto.

**Cuando arranque una nueva conversación con Claude:**
> "Leé `CLAUDE.md`. Estamos trabajando en [X]. Necesito que [Y]."