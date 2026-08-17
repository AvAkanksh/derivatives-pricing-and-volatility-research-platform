import pytest
import jax.numpy as jnp
import numpy as np
from dpp.core.models import PricingParams
from dpp.pricers.black_scholes import BlackScholesModel

def test_black_scholes_correctness():
    # Test values from standard textbook examples (e.g., Hull)
    # S0 = 42, K = 40, r = 0.10, q = 0.0, sigma = 0.20, T = 0.5 (6 months)
    # Expected call price: 4.76
    # Expected put price: 0.81
    
    bs = BlackScholesModel()
    
    params_call = PricingParams(
        spot=42.0,
        strike=40.0,
        maturity=0.5,
        rate=0.10,
        div_yield=0.0,
        sigma=0.20,
        option_type="call"
    )
    
    params_put = PricingParams(
        spot=42.0,
        strike=40.0,
        maturity=0.5,
        rate=0.10,
        div_yield=0.0,
        sigma=0.20,
        option_type="put"
    )
    
    price_call = bs.price(params_call)
    price_put = bs.price(params_put)
    
    # Check price match Hull values (tolerance ~ 1e-2 since Hull uses rounded figures, or exact analytical formulas)
    assert np.isclose(price_call, 4.75942, atol=1e-5)
    assert np.isclose(price_put, 0.808599, atol=1e-5)
    
    # Put-call parity check: C - P = S - K * e^{-rT}
    parity_diff = price_call - price_put - (42.0 - 40.0 * np.exp(-0.10 * 0.5))
    assert np.isclose(parity_diff, 0.0, atol=1e-6)

def test_black_scholes_greeks():
    bs = BlackScholesModel()
    
    params = PricingParams(
        spot=49.0,
        strike=50.0,
        maturity=0.3846, # 20 weeks
        rate=0.05,
        div_yield=0.0,
        sigma=0.20,
        option_type="call"
    )
    
    greeks = bs.greeks(params)
    
    # Compare with known Greeks for Call:
    # d1 = (ln(49/50) + (0.05 + 0.02)*0.3846) / (0.2 * sqrt(0.3846))
    #    = (-0.02020 + 0.02692) / (0.12403) = 0.0542
    # N(d1) ~ 0.5216 (Delta)
    
    # Let's verify our analytical Greeks formula using finite differences
    epsilon = 1e-5
    
    # Delta = dV/dS
    params_up = PricingParams(spot=params.spot + epsilon, strike=params.strike, maturity=params.maturity, rate=params.rate, div_yield=params.div_yield, sigma=params.sigma, option_type=params.option_type)
    params_down = PricingParams(spot=params.spot - epsilon, strike=params.strike, maturity=params.maturity, rate=params.rate, div_yield=params.div_yield, sigma=params.sigma, option_type=params.option_type)
    fd_delta = (bs.price(params_up) - bs.price(params_down)) / (2 * epsilon)
    assert np.isclose(greeks.delta, fd_delta, atol=1e-5)
    
    # Gamma = d2V/dS2
    fd_gamma = (bs.price(params_up) - 2 * bs.price(params) + bs.price(params_down)) / (epsilon**2)
    assert np.isclose(greeks.gamma, fd_gamma, atol=1e-4)
    
    # Vega = dV/dsigma
    params_v_up = PricingParams(spot=params.spot, strike=params.strike, maturity=params.maturity, rate=params.rate, div_yield=params.div_yield, sigma=params.sigma + epsilon, option_type=params.option_type)
    params_v_down = PricingParams(spot=params.spot, strike=params.strike, maturity=params.maturity, rate=params.rate, div_yield=params.div_yield, sigma=params.sigma - epsilon, option_type=params.option_type)
    fd_vega = (bs.price(params_v_up) - bs.price(params_v_down)) / (2 * epsilon)
    assert np.isclose(greeks.vega, fd_vega, atol=1e-5)
    
    # Theta = dV/dt (where t is current time, T_maturity = T - t, so dV/dt = -dV/dT)
    params_t_up = PricingParams(spot=params.spot, strike=params.strike, maturity=params.maturity + epsilon, rate=params.rate, div_yield=params.div_yield, sigma=params.sigma, option_type=params.option_type)
    params_t_down = PricingParams(spot=params.spot, strike=params.strike, maturity=params.maturity - epsilon, rate=params.rate, div_yield=params.div_yield, sigma=params.sigma, option_type=params.option_type)
    fd_theta = -(bs.price(params_t_up) - bs.price(params_t_down)) / (2 * epsilon)
    assert np.isclose(greeks.theta, fd_theta, atol=1e-5)

def test_black_scholes_batch():
    bs = BlackScholesModel()
    
    spots = jnp.array([40.0, 42.0, 44.0])
    strikes = jnp.array([40.0, 40.0, 40.0])
    maturities = jnp.array([0.5, 0.5, 0.5])
    rates = jnp.array([0.1, 0.1, 0.1])
    div_yields = jnp.array([0.0, 0.0, 0.0])
    sigmas = jnp.array([0.2, 0.2, 0.2])
    option_types = np.array(["call", "call", "call"]) # Can use object array for strings or custom JAX types
    
    params_batch = PricingParams(
        spot=spots,
        strike=strikes,
        maturity=maturities,
        rate=rates,
        div_yield=div_yields,
        sigma=sigmas,
        option_type=option_types
    )
    
    prices = bs.price_batch(params_batch)
    
    assert prices.shape == (3,)
    assert np.isclose(prices[1], 4.75942, atol=1e-5)
