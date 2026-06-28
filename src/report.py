"""report.py — Gráficos comparativos: Barbell vs. benchmarks (CLAUDE.md §6).

Construye 3 figuras a partir de los outputs ya generados por `backtest.py` y
`benchmark.py`:
  1. Curvas de equity (capital en USD a lo largo del tiempo), escala lineal y log.
  2. Curvas de drawdown (% desde el máximo previo) — la evidencia visual central de la
     tesis: la Barbell debería caer mucho menos que los benchmarks lineales en los
     eventos de cola izquierda (PASO 2019, etc.).
  3. Barras resumen: retorno total, CAGR y max drawdown lado a lado.

La curva de equity de la Barbell es por construcción una serie de PUNTOS (un valor por
ciclo de ~2 meses, en `entry_date`/`expiry_date` — no hay observación diaria entre
medio, porque el polo agresivo se liquida a valor intrínseco solo al vencimiento). Se
grafica conectando esos puntos con una línea recta, lo cual es una aproximación
explícita: subestima la volatilidad intra-ciclo real del polo seguro (que sí acumula
interés día a día) pero es la única curva que el ledger permite reconstruir sin
inventar datos. Esta limitación debe documentarse en el capítulo metodológico si se usa
la figura en la tesis.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import yaml

from src.benchmark import buy_and_hold_equity, load_adr_price, load_merval_usd_price

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("data/processed/backtest_hybrid.parquet")
DEFAULT_ADR_PATH = Path("data/raw/adr/GGAL_ADR_daily.parquet")
DEFAULT_MERVAL_PATH = Path("data/raw/merval/MERVAL_daily.parquet")
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/figures")

LABEL_BARBELL = "Barbell (hybrid)"
LABEL_ADR = "GGAL ADR buy-and-hold"
LABEL_MERVAL = "Merval en USD"
COLORS = {
    LABEL_BARBELL: "#1B7A5C",
    LABEL_ADR: "#5B4D9E",
    LABEL_MERVAL: "#C45B12",
}
SAVEFIG_DPI = 300

# Estilo profesional consistente en las 3 figuras: tipografía sans-serif, ejes sin
# spines superior/derecho, grilla horizontal sutil — pensado para impresión en la tesis.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#444444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
})


def _usd_formatter(value: float, _pos: int) -> str:
    return f"USD {value:,.0f}"


# ---------------------------------------------------------------------------
# Curva de equity de la Barbell (puntos por ciclo)
# ---------------------------------------------------------------------------

def load_barbell_equity(ledger_path: Path, initial_capital: float) -> pd.Series:
    """Reconstruye la curva de equity de un ledger de `backtest.py` como serie de puntos.

    Args:
        ledger_path:      path a `backtest_{model}.parquet`.
        initial_capital:  capital al inicio del primer ciclo.

    Returns:
        Series de equity en USD, indexada por fecha (`entry_date` del primer ciclo +
        `expiry_date` de cada ciclo subsiguiente).
    """
    ledger = pd.read_parquet(ledger_path)
    dates = pd.concat(
        [pd.Series([ledger["entry_date"].iloc[0]]), ledger["expiry_date"]], ignore_index=True
    )
    values = pd.concat(
        [pd.Series([initial_capital]), ledger["safe_capital_end"]], ignore_index=True
    )
    equity = pd.Series(values.values, index=pd.to_datetime(dates.values)).sort_index()
    equity.index.name = "fecha"
    return equity


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def plot_equity_curves(equity_curves: dict[str, pd.Series], output_path: Path) -> None:
    """Grafica las curvas de equity en escala lineal y logarítmica (2 paneles).

    Args:
        equity_curves: dict {nombre: Series de equity}, ya alineadas a la misma ventana.
        output_path:   path del PNG de salida.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, equity in equity_curves.items():
        color = COLORS.get(name)
        is_barbell = name == LABEL_BARBELL
        axes[0].plot(
            equity.index, equity.values, label=name, color=color,
            linewidth=2.2 if is_barbell else 1.4, zorder=3 if is_barbell else 2,
        )
        axes[1].plot(
            equity.index, equity.values, label=name, color=color,
            linewidth=2.2 if is_barbell else 1.4, zorder=3 if is_barbell else 2,
        )

    axes[0].set_title("Curva de equity — escala lineal")
    axes[1].set_title("Curva de equity — escala logarítmica")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Capital")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))
        ax.legend(loc="upper left")
        ax.grid(axis="y")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Barbell (hybrid) vs. ADR GGAL vs. Merval en USD — Curva de equity",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVEFIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada en: %s", output_path)


def plot_drawdown_curves(equity_curves: dict[str, pd.Series], output_path: Path) -> None:
    """Grafica el drawdown (% desde el máximo previo) de cada curva de equity.

    Es la evidencia visual central de la tesis: la Barbell debería caer mucho menos
    que los benchmarks lineales en los eventos de cola izquierda (PASO 2019, etc.).

    Args:
        equity_curves: dict {nombre: Series de equity}.
        output_path:   path del PNG de salida.
    """
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for name, equity in equity_curves.items():
        color = COLORS.get(name)
        is_barbell = name == LABEL_BARBELL
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100.0
        ax.plot(
            drawdown.index, drawdown.values, label=name, color=color,
            linewidth=2.2 if is_barbell else 1.4, zorder=3 if is_barbell else 2,
        )
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.10, color=color)

    ax.set_title("Drawdown desde el máximo previo")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="lower left")
    ax.grid(axis="y")
    ax.grid(axis="x", alpha=0.3)
    ax.axhline(0, color="#444444", linewidth=0.8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVEFIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada en: %s", output_path)


def build_full_comparison_table(
    backtest_path: Path, risk_path: Path, benchmark_path: Path,
) -> pd.DataFrame:
    """Unifica retorno (`backtest.py`), riesgo (`metrics.py`) y benchmarks (`benchmark.py`).

    `backtest_comparison.csv` (3 modelos Barbell) y `risk_metrics_comparison.csv`
    (mismos 3 modelos) llevan métricas complementarias en archivos separados porque
    surgieron en iteraciones distintas; `benchmark_comparison.csv` ya las tiene todas
    juntas para los 2 benchmarks. Esta función las junta en una sola tabla, sin volver
    a calcular nada, para que el capítulo de resultados de la tesis tenga un solo cuadro
    con retorno, drawdown, Calmar, Sharpe y Sortino lado a lado.

    Args:
        backtest_path:  path a `backtest_comparison.csv` (modelo, final_capital,
                         total_return_pct, cagr, max_drawdown_pct, calmar, ...).
        risk_path:       path a `risk_metrics_comparison.csv` (modelo, sharpe, sortino,
                         expected_shortfall, max_drawdown_pct, calmar, periods_per_year).
        benchmark_path:  path a `benchmark_comparison.csv` (ya con todas las columnas).

    Returns:
        DataFrame indexado por nombre de modelo/benchmark, con las columnas
        `final_capital, total_return_pct, cagr, max_drawdown_pct, calmar, sharpe,
        sortino, expected_shortfall, periods_per_year`.
    """
    cols = [
        "final_capital", "total_return_pct", "cagr", "max_drawdown_pct", "calmar",
        "sharpe", "sortino", "expected_shortfall", "periods_per_year",
    ]

    backtest_df = pd.read_csv(backtest_path, index_col="model")
    risk_df = pd.read_csv(risk_path, index_col="model").drop(columns=["max_drawdown_pct", "calmar"])
    barbell_df = backtest_df.join(risk_df)

    benchmark_df = pd.read_csv(benchmark_path, index_col="model")

    full = pd.concat([barbell_df, benchmark_df])
    return full[cols]


def plot_summary_bars(summary: pd.DataFrame, output_path: Path) -> None:
    """Grafica barras lado a lado de retorno total, CAGR, max drawdown y Calmar.

    El panel de Calmar (CAGR / |Max Drawdown|) es el resumen visual del argumento
    central de la tesis: retorno ajustado por el riesgo de cola que la Barbell busca
    evitar, no solo retorno nominal.

    Args:
        summary:     DataFrame indexado por nombre de modelo/benchmark, con columnas
                     `total_return_pct`, `cagr`, `max_drawdown_pct` y, si está
                     disponible, `calmar`.
        output_path: path del PNG de salida.
    """
    metrics = [
        ("total_return_pct", "Retorno total", "%"),
        ("cagr", "CAGR", "%"),
        ("max_drawdown_pct", "Max Drawdown", "%"),
    ]
    if "calmar" in summary.columns:
        metrics.append(("calmar", "Calmar ratio", "x"))

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 5.3))
    colors = [COLORS.get(name, "#999999") for name in summary.index]
    short_labels = [name.replace(" (hybrid)", "\n(hybrid)").replace(" en ", "\n en ").replace(" buy-and-hold", "\nbuy-and-hold") for name in summary.index]

    for ax, (col, title, unit) in zip(axes, metrics):
        bars = ax.bar(short_labels, summary[col], color=colors, width=0.62, edgecolor="white", linewidth=0.8)
        ax.set_title(title, pad=10)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color="#444444", linewidth=0.8)

        span = summary[col].max() - summary[col].min()
        offset = max(span, 1e-9) * 0.03
        for bar, value in zip(bars, summary[col]):
            label = f"{value:,.2f}{unit}" if unit == "x" else f"{value:,.1f}{unit}"
            va = "bottom" if value >= 0 else "top"
            y = value + offset if value >= 0 else value - offset
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, y), ha="center", va=va, fontsize=9.5)

    fig.suptitle(
        "Barbell (hybrid) vs. ADR GGAL vs. Merval en USD — Resumen de desempeño",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVEFIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada en: %s", output_path)


# ---------------------------------------------------------------------------
# CLI standalone (python -m src.report)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera las figuras comparativas Barbell (hybrid) vs. ADR vs. Merval USD."
    )
    parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--adr-path", default=str(DEFAULT_ADR_PATH))
    parser.add_argument("--merval-path", default=str(DEFAULT_MERVAL_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    """Orquesta la generación de las 3 figuras comparativas.

    Returns:
        0 si todo OK, 1 si falta algún archivo de entrada.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    args = _parse_args()

    config_path = Path(args.config_path)
    ledger_path = Path(args.ledger_path)
    adr_path = Path(args.adr_path)
    merval_path = Path(args.merval_path)

    for path, cmd in [
        (config_path, None),
        (ledger_path, "python -m src.backtest"),
        (adr_path, "python scripts/download_adr.py"),
        (merval_path, "python scripts/download_merval.py"),
    ]:
        if not path.exists():
            logger.error("No se encontró '%s'. Correr primero `%s`.", path, cmd)
            return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]
    initial_capital = config["capital_models"]["initial_capital"]

    barbell_equity = load_barbell_equity(ledger_path, initial_capital)
    adr_equity = buy_and_hold_equity(load_adr_price(adr_path), start_date, end_date, initial_capital)
    merval_equity = buy_and_hold_equity(
        load_merval_usd_price(merval_path), start_date, end_date, initial_capital
    )

    equity_curves = {
        LABEL_BARBELL: barbell_equity,
        LABEL_ADR: adr_equity,
        LABEL_MERVAL: merval_equity,
    }

    output_dir = Path(args.output_dir)
    plot_equity_curves(equity_curves, output_dir / "equity_curves.png")
    plot_drawdown_curves(equity_curves, output_dir / "drawdown_curves.png")

    comparison_path = Path("data/processed/backtest_comparison.csv")
    risk_path = Path("data/processed/risk_metrics_comparison.csv")
    benchmark_path = Path("data/processed/benchmark_comparison.csv")
    if comparison_path.exists() and risk_path.exists() and benchmark_path.exists():
        full_comparison = build_full_comparison_table(comparison_path, risk_path, benchmark_path)
        full_comparison = full_comparison.rename(index={"hybrid": LABEL_BARBELL})

        full_comparison_path = Path("data/processed/full_comparison.csv")
        full_comparison.to_csv(full_comparison_path)
        logger.info("Tabla unificada guardada en: %s", full_comparison_path)

        summary = full_comparison.loc[
            [LABEL_BARBELL, LABEL_ADR, LABEL_MERVAL],
            ["total_return_pct", "cagr", "max_drawdown_pct", "calmar"],
        ]
        plot_summary_bars(summary, output_dir / "summary_bars.png")
    else:
        logger.warning(
            "Faltan %s, %s o %s — se omite la tabla unificada y el gráfico de barras resumen.",
            comparison_path, risk_path, benchmark_path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
