"""
main.py — Punto de entrada del backtest de la Estrategia Barbell sobre opciones GGAL.

Flujo esperado del pipeline:
    1. Cargar configuración  (config.yaml)
    2. Descargar / leer datos crudos
    3. Limpiar y auditar datos
    4. Convertir a USD vía CCL
    5. Correr el motor de backtest
    6. Calcular métricas de desempeño
    7. Generar reporte final

Autores: Santiago Quintero (LU 1176122) · Matías Malo Medrano (LU 1147831)
Tutor:   Mauro Natalucci — TIF Licenciatura en Finanzas, UADE 2025-2026
"""

import logging
import yaml

# --- Importaciones de módulos propios (aún no implementados) ---
# from src.data_loader import load_options, load_adr, load_merval, load_tbills
# from src.data_audit import audit_options
# from src.fx import convert_to_usd
# from src.greeks import compute_implied_vol
# from src.strategy import build_barbell_portfolio
# from src.backtest import run_backtest
# from src.metrics import compute_metrics
# from src.report import generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Carga la configuración del backtest desde el archivo YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Orquesta el pipeline completo del backtest."""

    logger.info("Iniciando pipeline — Estrategia Barbell sobre opciones GGAL")

    # 1. Cargar configuración
    config = load_config()
    logger.info("Configuración cargada: backtest %s → %s",
                config["backtest"]["start_date"],
                config["backtest"]["end_date"])

    # 2. Cargar datos crudos
    # TODO (Santiago): implementar src/data_loader.py
    # options_raw = load_options(config["data"]["raw_options_dir"])
    # adr_raw     = load_adr(config["data"]["raw_adr_dir"])
    # merval_raw  = load_merval(config["data"]["raw_merval_dir"])
    # tbills_raw  = load_tbills(config["data"]["raw_tbills_dir"])
    logger.info("[PLACEHOLDER] Paso 2: carga de datos crudos — pendiente de implementación")

    # 3. Auditar y limpiar datos de opciones
    # TODO (Santiago): implementar src/data_audit.py
    # Punto crítico: resolver la inconsistencia en la columna CALL (prima vs volumen)
    # options_clean = audit_options(options_raw)
    logger.info("[PLACEHOLDER] Paso 3: auditoría de datos — pendiente de implementación")

    # 4. Convertir ARS → USD vía CCL
    # TODO (Santiago): implementar src/fx.py
    # options_usd = convert_to_usd(options_clean, ccl_series)
    # merval_usd  = convert_to_usd(merval_raw, ccl_series)
    logger.info("[PLACEHOLDER] Paso 4: conversión FX — pendiente de implementación")

    # 5. Calcular volatilidad implícita por strike (si no viene de la planilla)
    # TODO (Matías): implementar src/greeks.py
    # options_with_vi = compute_implied_vol(options_usd, config["backtest"])
    logger.info("[PLACEHOLDER] Paso 5: griegas / VI por strike — pendiente de implementación")

    # 6. Construir la cartera Barbell y correr el backtest
    # TODO (Matías): implementar src/strategy.py + src/backtest.py
    # portfolio   = build_barbell_portfolio(options_with_vi, tbills_raw, config["portfolio"])
    # results     = run_backtest(portfolio, config["backtest"])
    logger.info("[PLACEHOLDER] Paso 6: backtest Barbell — pendiente de implementación")

    # 7. Calcular métricas vs benchmarks
    # TODO (Matías): implementar src/metrics.py
    # metrics = compute_metrics(results, benchmarks={
    #     "adr":    adr_raw,
    #     "merval": merval_usd,
    # })
    logger.info("[PLACEHOLDER] Paso 7: métricas (MDD, ES, Sortino, Sharpe) — pendiente de implementación")

    # 8. Generar reporte con gráficos y tablas
    # TODO (Matías): implementar src/report.py
    # generate_report(results, metrics, config)
    logger.info("[PLACEHOLDER] Paso 8: reporte final — pendiente de implementación")

    logger.info("Pipeline finalizado (modo skeleton — sin lógica implementada todavía)")


if __name__ == "__main__":
    main()
