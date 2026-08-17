import pytest
import numpy as np
from dpp.calibration.svi import get_svi_surface
from dpp.arbitrage.checks import (
    calendar_arbitrage_violations,
    butterfly_arbitrage_violations,
    price_bounds_check,
    put_call_parity_residual
)

def test_calendar_arbitrage_violation():
    T_grid = np.array([0.5, 1.0])
    k_grid = np.linspace(-0.2, 0.2, 5)
    
    # 1. Arbitrage-free surface (total variance increases in T)
    svi_free = [
        (0.04, 0.1, 0.0, 0.0, 0.1), # a=0.04 at T=0.5
        (0.08, 0.1, 0.0, 0.0, 0.1)  # a=0.08 at T=1.0
    ]
    w_free = get_svi_surface(T_grid, np.array(svi_free))
    assert len(calendar_arbitrage_violations(w_free, k_grid, T_grid)) == 0
    
    # 2. Arbitrage-present surface (total variance decreases in T)
    svi_arb = [
        (0.08, 0.1, 0.0, 0.0, 0.1), # a=0.08 at T=0.5
        (0.04, 0.1, 0.0, 0.0, 0.1)  # a=0.04 at T=1.0
    ]
    w_arb = get_svi_surface(T_grid, np.array(svi_arb))
    violations = calendar_arbitrage_violations(w_arb, k_grid, T_grid)
    assert len(violations) > 0

def test_butterfly_arbitrage_violation():
    T_grid = np.array([0.5])
    k_grid = np.linspace(-0.5, 0.5, 21)
    
    # 1. Arbitrage-free SVI params
    svi_free = [
        (0.04, 0.05, 0.0, 0.0, 0.1)
    ]
    w_free = get_svi_surface(T_grid, np.array(svi_free))
    assert len(butterfly_arbitrage_violations(w_free, k_grid, T_grid)) == 0
    
    # 2. Arbitrage-present SVI params: extremely large b (wing slope) causing curvature violation
    svi_arb = [
        (0.04, 2.0, 0.0, 0.0, 0.001)
    ]
    w_arb = get_svi_surface(T_grid, np.array(svi_arb))
    violations = butterfly_arbitrage_violations(w_arb, k_grid, T_grid)
    assert len(violations) > 0

def test_price_bounds_check():
    # S=100, K=100, r=0.05, q=0.0, T=1.0
    # lower_bound = max(0, 100 - 100 * e^{-0.05}) = 100 * (1 - 0.9512) = 4.88
    # upper_bound = 100
    res = price_bounds_check(price=6.0, spot=100.0, strike=100.0, rate=0.05, div_yield=0.0, maturity=1.0, option_type="call")
    assert res["is_valid"] is True
    
    # Arbitrarily cheap price (below lower bound)
    res_cheap = price_bounds_check(price=3.0, spot=100.0, strike=100.0, rate=0.05, div_yield=0.0, maturity=1.0, option_type="call")
    assert res_cheap["is_valid"] is False
    assert res_cheap["violation"] == "below_lower"
    
    # Arbitrarily expensive price (above upper bound)
    res_expensive = price_bounds_check(price=105.0, spot=100.0, strike=100.0, rate=0.05, div_yield=0.0, maturity=1.0, option_type="call")
    assert res_expensive["is_valid"] is False
    assert res_expensive["violation"] == "above_upper"

def test_put_call_parity_residual():
    # S=100, K=100, r=0.05, q=0.0, T=1.0
    # Expected C - P = S - K * e^{-rT} = 4.877
    spot = 100.0
    strike = 100.0
    rate = 0.05
    div_yield = 0.0
    maturity = 1.0
    
    # Price call and put
    expected_diff = spot - strike * np.exp(-rate * maturity)
    
    call_px = 10.0
    put_px = call_px - expected_diff
    
    residual = put_call_parity_residual(call_px, put_px, spot, strike, rate, div_yield, maturity)
    assert np.isclose(residual, 0.0, atol=1e-5)
