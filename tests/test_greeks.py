"""tests/test_greeks.py — Tests de Black-Scholes y griegas (src/greeks.py)."""

import numpy as np
import pytest

from src.greeks import (
    black_scholes_price,
    delta,
    gamma,
    theta,
    validate_inputs,
    vega,
)


# ---------------------------------------------------------------------------
# 1. Paridad put-call: C - P = S - K * exp(-r*T)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S, K, T, r, sigma", [
    (100.0, 100.0, 1.00, 0.05, 0.20),
    (100.0,  90.0, 0.50, 0.03, 0.30),
    (110.0, 100.0, 2.00, 0.02, 0.15),
    ( 80.0, 100.0, 0.25, 0.04, 0.50),
    ( 50.0,  50.0, 0.10, 0.08, 0.60),
])
def test_put_call_parity(S, K, T, r, sigma):
    """C - P = S - K·exp(-r·T) con tolerancia de 1e-8."""
    C = black_scholes_price(S, K, T, r, sigma, "call")
    P = black_scholes_price(S, K, T, r, sigma, "put")
    forward = S - K * np.exp(-r * T)
    assert abs(C - P - forward) < 1e-8, (
        f"Paridad rota para S={S}, K={K}, T={T}, r={r}, sigma={sigma}: "
        f"C-P={C - P:.8f}, S-Ke^(-rT)={forward:.8f}"
    )


# ---------------------------------------------------------------------------
# 2. Límites ATM
# ---------------------------------------------------------------------------

def test_atm_t_zero_vale_cero():
    """Opción ATM con T=0 debe valer 0 (valor intrínseco nulo)."""
    C = black_scholes_price(100.0, 100.0, 0.0, 0.05, 0.20, "call")
    P = black_scholes_price(100.0, 100.0, 0.0, 0.05, 0.20, "put")
    assert C == 0.0
    assert P == 0.0


def test_atm_valor_conocido():
    """Call ATM con r=0, sigma=20%, T=1 debe ser ≈ 7.966.

    Resultado analítico: d1=0.10, d2=-0.10
    C = 100 * (N(0.10) - N(-0.10)) = 100 * (0.53983 - 0.46017) ≈ 7.966
    """
    price = black_scholes_price(100.0, 100.0, 1.0, 0.0, 0.20, "call")
    assert abs(price - 7.966) < 0.005, f"Precio ATM inesperado: {price:.4f}"


def test_precio_crece_con_t():
    """Para una call ATM, el precio debe crecer con T (más tiempo = más valor)."""
    precios = [
        black_scholes_price(100.0, 100.0, T, 0.05, 0.20, "call")
        for T in [0.25, 0.5, 1.0, 2.0]
    ]
    assert all(precios[i] < precios[i + 1] for i in range(len(precios) - 1))


# ---------------------------------------------------------------------------
# 3. Griegas en ATM
# ---------------------------------------------------------------------------

def test_delta_call_atm_aprox_cero_punto_cinco():
    """Delta de call ATM con T>0 debe estar próxima a 0.5."""
    d = delta(100.0, 100.0, 1.0, 0.0, 0.20, "call")
    assert abs(d - 0.5) < 0.05, f"Delta call ATM inesperado: {d:.4f}"


def test_delta_put_call_paridad():
    """Delta(call) - Delta(put) = 1 para cualquier (S, K, T, r, sigma)."""
    for S, K in [(100, 100), (110, 100), (90, 100)]:
        d_call = delta(float(S), float(K), 1.0, 0.05, 0.20, "call")
        d_put  = delta(float(S), float(K), 1.0, 0.05, 0.20, "put")
        assert abs(d_call - d_put - 1.0) < 1e-10, (
            f"Delta no satisface paridad para S={S}, K={K}"
        )


def test_gamma_positivo():
    """Gamma debe ser estrictamente positivo para T > 0 y sigma > 0."""
    g = gamma(100.0, 100.0, 1.0, 0.05, 0.20)
    assert g > 0


def test_gamma_maximo_atm():
    """Gamma debe ser mayor en ATM que en ITM o OTM para igual T y sigma."""
    g_otm = gamma( 80.0, 100.0, 1.0, 0.05, 0.20)
    g_atm = gamma(100.0, 100.0, 1.0, 0.05, 0.20)
    g_itm = gamma(120.0, 100.0, 1.0, 0.05, 0.20)
    assert g_atm > g_otm
    assert g_atm > g_itm


def test_vega_positivo():
    """Vega debe ser estrictamente positivo para T > 0 y sigma > 0."""
    v = vega(100.0, 100.0, 1.0, 0.05, 0.20)
    assert v > 0


def test_theta_negativo_call_put():
    """Theta debe ser negativo para posiciones largas (call y put) cuando T > 0."""
    t_call = theta(100.0, 100.0, 1.0, 0.05, 0.20, "call")
    t_put  = theta(100.0, 100.0, 1.0, 0.05, 0.20, "put")
    assert t_call < 0, f"Theta call positivo: {t_call:.6f}"
    assert t_put  < 0, f"Theta put positivo: {t_put:.6f}"


# ---------------------------------------------------------------------------
# 4. T=0: valor intrínseco y griegas degeneradas
# ---------------------------------------------------------------------------

def test_t_zero_call_itm():
    """Call ITM con T=0: precio = intrínseco, delta = 1, gamma = vega = theta = 0."""
    S, K = 110.0, 100.0
    assert black_scholes_price(S, K, 0.0, 0.05, 0.20, "call") == 10.0
    assert delta(S, K, 0.0, 0.05, 0.20, "call") == 1.0
    assert gamma(S, K, 0.0, 0.05, 0.20) == 0.0
    assert vega(S, K, 0.0, 0.05, 0.20) == 0.0
    assert theta(S, K, 0.0, 0.05, 0.20, "call") == 0.0


def test_t_zero_call_otm():
    """Call OTM con T=0: precio = 0, delta = 0."""
    S, K = 90.0, 100.0
    assert black_scholes_price(S, K, 0.0, 0.05, 0.20, "call") == 0.0
    assert delta(S, K, 0.0, 0.05, 0.20, "call") == 0.0


def test_t_zero_put_itm():
    """Put ITM con T=0: precio = intrínseco, delta = -1."""
    S, K = 90.0, 100.0
    assert black_scholes_price(S, K, 0.0, 0.05, 0.20, "put") == 10.0
    assert delta(S, K, 0.0, 0.05, 0.20, "put") == -1.0


def test_t_zero_put_otm():
    """Put OTM con T=0: precio = 0, delta = 0."""
    S, K = 110.0, 100.0
    assert black_scholes_price(S, K, 0.0, 0.05, 0.20, "put") == 0.0
    assert delta(S, K, 0.0, 0.05, 0.20, "put") == 0.0


# ---------------------------------------------------------------------------
# 5. Vectorización
# ---------------------------------------------------------------------------

def test_vectorized_precio():
    """Con array de S, debe retornar array de igual forma con valores monótonos."""
    S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = black_scholes_price(S, 100.0, 1.0, 0.05, 0.20, "call")
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (5,)
    assert np.all(prices >= 0)
    # Call: precio crece con S
    assert np.all(np.diff(prices) > 0)


def test_vectorized_griegas():
    """Griegas vectorizadas deben retornar arrays del mismo tamaño que los inputs."""
    S = np.array([90.0, 100.0, 110.0])
    d = delta(S, 100.0, 1.0, 0.05, 0.20, "call")
    g = gamma(S, 100.0, 1.0, 0.05, 0.20)
    v = vega(S, 100.0, 1.0, 0.05, 0.20)
    t = theta(S, 100.0, 1.0, 0.05, 0.20, "call")
    assert d.shape == g.shape == v.shape == t.shape == (3,)


def test_vectorized_broadcast_sk():
    """Arrays de S y K deben funcionar por broadcast."""
    S = np.array([90.0, 100.0, 110.0])
    K = np.array([95.0, 100.0, 105.0])
    prices = black_scholes_price(S, K, 1.0, 0.05, 0.20, "call")
    assert prices.shape == (3,)
    assert np.all(prices >= 0)


# ---------------------------------------------------------------------------
# 6. Tasa libre de riesgo negativa
# ---------------------------------------------------------------------------

def test_r_negativo_no_lanza():
    """Con r negativo, las funciones deben ejecutar sin error y retornar valores válidos."""
    C = black_scholes_price(100.0, 100.0, 1.0, -0.01, 0.20, "call")
    P = black_scholes_price(100.0, 100.0, 1.0, -0.01, 0.20, "put")
    assert np.isfinite(C)
    assert np.isfinite(P)


def test_r_negativo_paridad():
    """Paridad put-call debe seguir valiendo con r negativo."""
    r = -0.01
    C = black_scholes_price(100.0, 100.0, 1.0, r, 0.20, "call")
    P = black_scholes_price(100.0, 100.0, 1.0, r, 0.20, "put")
    forward = 100.0 - 100.0 * np.exp(-r * 1.0)
    assert abs(C - P - forward) < 1e-8


# ---------------------------------------------------------------------------
# 7. Propagación de NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"S": np.nan},
    {"K": np.nan},
    {"T": np.nan},
    {"r": np.nan},
    {"sigma": np.nan},
])
def test_nan_propagacion_precio(kwargs):
    """NaN en cualquier input debe dar NaN en el precio."""
    base = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20}
    base.update(kwargs)
    result = black_scholes_price(**base, option_type="call")
    assert np.isnan(result), f"Se esperaba NaN con kwargs={kwargs}, se obtuvo {result}"


def test_nan_propagacion_griegas():
    """NaN en r o sigma debe propagarse a todas las griegas."""
    assert np.isnan(delta(100.0, 100.0, 1.0, np.nan, 0.20, "call"))
    assert np.isnan(gamma(100.0, 100.0, 1.0, 0.05, np.nan))
    assert np.isnan(vega(100.0, 100.0, 1.0, np.nan, 0.20))
    assert np.isnan(theta(100.0, 100.0, 1.0, 0.05, np.nan, "put"))


# ---------------------------------------------------------------------------
# 8. validate_inputs
# ---------------------------------------------------------------------------

def test_validate_s_no_positivo():
    """S <= 0 debe lanzar ValueError."""
    with pytest.raises(ValueError, match="'S'"):
        validate_inputs(-1.0, 100.0, 1.0, 0.05, 0.20)
    with pytest.raises(ValueError, match="'S'"):
        validate_inputs(0.0, 100.0, 1.0, 0.05, 0.20)


def test_validate_k_no_positivo():
    """K <= 0 debe lanzar ValueError."""
    with pytest.raises(ValueError, match="'K'"):
        validate_inputs(100.0, 0.0, 1.0, 0.05, 0.20)


def test_validate_t_negativo():
    """T < 0 debe lanzar ValueError."""
    with pytest.raises(ValueError, match="'T'"):
        validate_inputs(100.0, 100.0, -0.1, 0.05, 0.20)


def test_validate_sigma_negativa():
    """sigma < 0 debe lanzar ValueError."""
    with pytest.raises(ValueError, match="'sigma'"):
        validate_inputs(100.0, 100.0, 1.0, 0.05, -0.01)


def test_validate_r_negativo_no_lanza():
    """r negativo es válido y no debe lanzar."""
    validate_inputs(100.0, 100.0, 1.0, -0.05, 0.20)  # no debe lanzar


def test_validate_t_cero_no_lanza():
    """T = 0 es válido (opción vencida)."""
    validate_inputs(100.0, 100.0, 0.0, 0.05, 0.20)  # no debe lanzar


def test_validate_sigma_cero_no_lanza():
    """sigma = 0 es válido."""
    validate_inputs(100.0, 100.0, 1.0, 0.05, 0.0)   # no debe lanzar


def test_validate_nan_pasa():
    """NaN en inputs pasa la validación (se propaga luego en el cálculo)."""
    validate_inputs(np.nan, 100.0, 1.0, 0.05, 0.20)  # no debe lanzar
    validate_inputs(100.0, 100.0, np.nan, 0.05, 0.20)


# ---------------------------------------------------------------------------
# 9. option_type inválido
# ---------------------------------------------------------------------------

def test_option_type_invalido():
    """option_type distinto de 'call' o 'put' debe lanzar ValueError."""
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.20, "forward")
    with pytest.raises(ValueError, match="option_type"):
        delta(100.0, 100.0, 1.0, 0.05, 0.20, "")
    with pytest.raises(ValueError, match="option_type"):
        theta(100.0, 100.0, 1.0, 0.05, 0.20, "C")
