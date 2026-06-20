"""greeks.py — Pricing Black-Scholes y griegas para opciones europeas.

Implementa el modelo Black-Scholes estándar para opciones europeas. Se usa en dos contextos:
  1. Recalcular DELTA, GAMMA, VEGA, THETA para los archivos Historial del esquema A
     (2023-10 → 2025-04), que no traen griegas pre-calculadas (ver CLAUDE.md §5.3).
  2. Generar precios sintéticos para el tramo 2019-2023 (ver generate_synthetic_options.py).

Protocolo de calibración (ver CLAUDE.md §5.3):
  Antes de aplicar a los archivos del esquema A, validar contra los del esquema B
  (2025-06 → 2026-06), donde la fuente publica DELTA/GAMMA/VEGA/THETA. Si los valores
  difieren menos del 1% relativo, la implementación está calibrada. Esta validación es
  responsabilidad de la iteración siguiente (Matías).

Convenciones de unidades:
  - T    : tiempo al vencimiento en años (columna ``plazo_anios`` del formato tidy, §7)
  - r    : tasa libre de riesgo anualizada como decimal (columna ``tlr``, ej. 0.05 = 5%)
  - sigma: volatilidad implícita anualizada como decimal (columna ``vi_implicita``, ej. 0.40 = 40%)
  - vega : por cada 1% de cambio en sigma (resultado BS / 100)
  - theta: decaimiento diario (resultado BS anualizado / 365)
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _prepare(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Convierte inputs a arrays broadcast-alineados y detecta si todos eran escalares."""
    raw = [np.asarray(x, dtype=float) for x in (S, K, T, r, sigma)]
    scalar = all(a.ndim == 0 for a in raw)
    S_, K_, T_, r_, sig_ = np.broadcast_arrays(*[np.atleast_1d(a) for a in raw])
    return (
        np.ascontiguousarray(S_),
        np.ascontiguousarray(K_),
        np.ascontiguousarray(T_),
        np.ascontiguousarray(r_),
        np.ascontiguousarray(sig_),
        scalar,
    )


def _out(arr: np.ndarray, scalar: bool) -> float | np.ndarray:
    """Retorna float si el input fue escalar; array en caso contrario."""
    return float(arr.flat[0]) if scalar else arr


def _d1_d2(
    S_: np.ndarray,
    K_: np.ndarray,
    T_: np.ndarray,
    r_: np.ndarray,
    sigma_: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula d1 y d2 de Black-Scholes con valores de respaldo para evitar división por cero.

    Cuando T=0 o sigma=0, substituye con T=1 y sigma=1 respectivamente. Los resultados
    intermedios no se usan en esos casos; la función pública aplica una máscara que los
    reemplaza con el valor intrínseco o 0.
    """
    _T = np.where(T_ > 0, T_, 1.0)
    _s = np.where(sigma_ > 0, sigma_, 1.0)
    sqrt_T = np.sqrt(_T)
    d1 = (np.log(S_ / K_) + (r_ + 0.5 * _s ** 2) * _T) / (_s * sqrt_T)
    d2 = d1 - _s * sqrt_T
    return d1, d2


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def validate_inputs(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> None:
    """Valida que los inputs estén en rangos razonables para Black-Scholes.

    Los NaN en los inputs pasan la validación y se propagan en el cálculo posterior.

    Args:
        S:     precio del subyacente.
        K:     precio de ejercicio (strike).
        T:     tiempo al vencimiento en años. Puede ser 0 (opción vencida).
        r:     tasa libre de riesgo anualizada como decimal. Puede ser negativa
               (ej. T-Bills en periodos de tasas negativas).
        sigma: volatilidad implícita anualizada como decimal. Puede ser 0.

    Raises:
        ValueError: si S o K contienen valores no positivos, o si T o sigma
            contienen valores estrictamente negativos.
    """
    checks = [
        ("S",     S,     False),   # debe ser S > 0
        ("K",     K,     False),   # debe ser K > 0
        ("T",     T,     True),    # debe ser T >= 0
        ("sigma", sigma, True),    # debe ser sigma >= 0
    ]
    for nombre, val, permite_cero in checks:
        arr = np.atleast_1d(np.asarray(val, dtype=float))
        validos = arr[~np.isnan(arr)]
        if validos.size == 0:
            continue
        if not permite_cero and validos.min() <= 0:
            raise ValueError(
                f"'{nombre}' debe ser estrictamente positivo; "
                f"mínimo recibido: {validos.min():.6g}"
            )
        if permite_cero and validos.min() < 0:
            raise ValueError(
                f"'{nombre}' no puede ser negativo; "
                f"mínimo recibido: {validos.min():.6g}"
            )
    # 'r' no se valida: puede ser negativo (tasas de política monetaria negativas o
    # FRED DTB3 en eventos de liquidez).


# ---------------------------------------------------------------------------
# Precio
# ---------------------------------------------------------------------------

def black_scholes_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str,
) -> float | np.ndarray:
    """Precio Black-Scholes de una opción europea.

    Args:
        S:           precio del subyacente.
        K:           precio de ejercicio (strike).
        T:           tiempo al vencimiento en años.
        r:           tasa libre de riesgo anualizada como decimal.
        sigma:       volatilidad implícita anualizada como decimal.
        option_type: ``"call"`` o ``"put"`` (insensible a mayúsculas).

    Returns:
        Precio teórico de la opción. Si T <= 0 o sigma <= 0, retorna el valor
        intrínseco: max(S-K, 0) para call, max(K-S, 0) para put. Propaga NaN
        si algún input es NaN.

    Raises:
        ValueError: si los inputs no pasan ``validate_inputs()`` o si
            ``option_type`` no es ``"call"`` ni ``"put"``.
    """
    validate_inputs(S, K, T, r, sigma)
    ot = option_type.strip().lower()
    if ot not in ("call", "put"):
        raise ValueError(f"option_type debe ser 'call' o 'put', no '{option_type}'")

    S_, K_, T_, r_, sig_, scalar = _prepare(S, K, T, r, sigma)
    nan_mask = np.isnan(S_) | np.isnan(K_) | np.isnan(T_) | np.isnan(r_) | np.isnan(sig_)
    degenerate = (T_ <= 0) | (sig_ <= 0)

    d1, d2 = _d1_d2(S_, K_, T_, r_, sig_)

    if ot == "call":
        normal_price = S_ * norm.cdf(d1) - K_ * np.exp(-r_ * T_) * norm.cdf(d2)
        intrinsic = np.maximum(S_ - K_, 0.0)
    else:  # put
        normal_price = K_ * np.exp(-r_ * T_) * norm.cdf(-d2) - S_ * norm.cdf(-d1)
        intrinsic = np.maximum(K_ - S_, 0.0)

    result = np.where(degenerate, intrinsic, normal_price)
    result = np.where(nan_mask, np.nan, result)
    return _out(result, scalar)


# ---------------------------------------------------------------------------
# Griegas
# ---------------------------------------------------------------------------

def delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str,
) -> float | np.ndarray:
    """Delta: sensibilidad del precio de la opción al precio del subyacente (dV/dS).

    Args:
        S:           precio del subyacente.
        K:           precio de ejercicio.
        T:           tiempo al vencimiento en años.
        r:           tasa libre de riesgo anualizada como decimal.
        sigma:       volatilidad implícita anualizada como decimal.
        option_type: ``"call"`` o ``"put"``.

    Returns:
        Delta de la opción. Si T <= 0 o sigma <= 0, retorna el delta degenerado:
        1 si call ITM, 0 si call OTM, 0.5 si call ATM (y equivalentes negativos
        para put). Propaga NaN si algún input es NaN.

    Raises:
        ValueError: si los inputs no pasan ``validate_inputs()`` o si
            ``option_type`` es inválido.
    """
    validate_inputs(S, K, T, r, sigma)
    ot = option_type.strip().lower()
    if ot not in ("call", "put"):
        raise ValueError(f"option_type debe ser 'call' o 'put', no '{option_type}'")

    S_, K_, T_, r_, sig_, scalar = _prepare(S, K, T, r, sigma)
    nan_mask = np.isnan(S_) | np.isnan(K_) | np.isnan(T_) | np.isnan(r_) | np.isnan(sig_)
    degenerate = (T_ <= 0) | (sig_ <= 0)

    d1, _ = _d1_d2(S_, K_, T_, r_, sig_)

    if ot == "call":
        normal_delta = norm.cdf(d1)
        # Convención ATM: 0.5 (precio del subyacente al vencimiento es exactamente K)
        degen_delta = np.where(S_ > K_, 1.0, np.where(S_ == K_, 0.5, 0.0))
    else:  # put
        normal_delta = norm.cdf(d1) - 1.0
        degen_delta = np.where(S_ < K_, -1.0, np.where(S_ == K_, -0.5, 0.0))

    result = np.where(degenerate, degen_delta, normal_delta)
    result = np.where(nan_mask, np.nan, result)
    return _out(result, scalar)


def gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Gamma: segunda derivada del precio respecto al precio del subyacente (d²V/dS²).

    Gamma es igual para calls y puts (se puede demostrar por paridad put-call).

    Args:
        S:     precio del subyacente.
        K:     precio de ejercicio.
        T:     tiempo al vencimiento en años.
        r:     tasa libre de riesgo anualizada como decimal.
        sigma: volatilidad implícita anualizada como decimal.

    Returns:
        Gamma de la opción. Si T <= 0 o sigma <= 0, retorna 0.
        Propaga NaN si algún input es NaN.

    Raises:
        ValueError: si los inputs no pasan ``validate_inputs()``.
    """
    validate_inputs(S, K, T, r, sigma)
    S_, K_, T_, r_, sig_, scalar = _prepare(S, K, T, r, sigma)
    nan_mask = np.isnan(S_) | np.isnan(K_) | np.isnan(T_) | np.isnan(r_) | np.isnan(sig_)
    degenerate = (T_ <= 0) | (sig_ <= 0)

    d1, _ = _d1_d2(S_, K_, T_, r_, sig_)
    _T = np.where(T_ > 0, T_, 1.0)
    _s = np.where(sig_ > 0, sig_, 1.0)

    normal_gamma = norm.pdf(d1) / (S_ * _s * np.sqrt(_T))

    result = np.where(degenerate, 0.0, normal_gamma)
    result = np.where(nan_mask, np.nan, result)
    return _out(result, scalar)


def vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Vega: sensibilidad del precio a la volatilidad implícita (dV/dσ).

    Vega es igual para calls y puts. Convención: expresada por cada 1% de cambio en
    sigma (resultado Black-Scholes estándar / 100).

    Args:
        S:     precio del subyacente.
        K:     precio de ejercicio.
        T:     tiempo al vencimiento en años.
        r:     tasa libre de riesgo anualizada como decimal.
        sigma: volatilidad implícita anualizada como decimal.

    Returns:
        Vega de la opción por 1% de cambio en sigma. Si T <= 0 o sigma <= 0,
        retorna 0. Propaga NaN si algún input es NaN.

    Raises:
        ValueError: si los inputs no pasan ``validate_inputs()``.
    """
    validate_inputs(S, K, T, r, sigma)
    S_, K_, T_, r_, sig_, scalar = _prepare(S, K, T, r, sigma)
    nan_mask = np.isnan(S_) | np.isnan(K_) | np.isnan(T_) | np.isnan(r_) | np.isnan(sig_)
    degenerate = (T_ <= 0) | (sig_ <= 0)

    d1, _ = _d1_d2(S_, K_, T_, r_, sig_)
    _T = np.where(T_ > 0, T_, 1.0)

    # División por 100: convención "vega por 1% de cambio en sigma"
    normal_vega = S_ * norm.pdf(d1) * np.sqrt(_T) / 100.0

    result = np.where(degenerate, 0.0, normal_vega)
    result = np.where(nan_mask, np.nan, result)
    return _out(result, scalar)


def theta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str,
) -> float | np.ndarray:
    """Theta: tasa de decaimiento temporal del precio de la opción (dV/dt, por día).

    Convención: theta diario (resultado Black-Scholes anualizado / 365). Es típicamente
    negativo para posiciones largas: el valor de la opción cae con el paso del tiempo.

    Args:
        S:           precio del subyacente.
        K:           precio de ejercicio.
        T:           tiempo al vencimiento en años.
        r:           tasa libre de riesgo anualizada como decimal.
        sigma:       volatilidad implícita anualizada como decimal.
        option_type: ``"call"`` o ``"put"``.

    Returns:
        Theta diario de la opción. Si T <= 0 o sigma <= 0, retorna 0.
        Propaga NaN si algún input es NaN.

    Raises:
        ValueError: si los inputs no pasan ``validate_inputs()`` o si
            ``option_type`` es inválido.
    """
    validate_inputs(S, K, T, r, sigma)
    ot = option_type.strip().lower()
    if ot not in ("call", "put"):
        raise ValueError(f"option_type debe ser 'call' o 'put', no '{option_type}'")

    S_, K_, T_, r_, sig_, scalar = _prepare(S, K, T, r, sigma)
    nan_mask = np.isnan(S_) | np.isnan(K_) | np.isnan(T_) | np.isnan(r_) | np.isnan(sig_)
    degenerate = (T_ <= 0) | (sig_ <= 0)

    d1, d2 = _d1_d2(S_, K_, T_, r_, sig_)
    _T = np.where(T_ > 0, T_, 1.0)
    _s = np.where(sig_ > 0, sig_, 1.0)

    # Componente compartida: pérdida de valor por reducción de volatilidad temporal
    vol_decay = -(S_ * norm.pdf(d1) * _s) / (2.0 * np.sqrt(_T))

    if ot == "call":
        normal_theta = (vol_decay - r_ * K_ * np.exp(-r_ * T_) * norm.cdf(d2)) / 365.0
    else:  # put
        normal_theta = (vol_decay + r_ * K_ * np.exp(-r_ * T_) * norm.cdf(-d2)) / 365.0

    result = np.where(degenerate, 0.0, normal_theta)
    result = np.where(nan_mask, np.nan, result)
    return _out(result, scalar)
