import pytest
import numpy as np
import jax.numpy as jnp
from dpp.core.models import PricingParams
from dpp.core.instruments import DividendSchedule
from dpp.pricers.binomial_tree import BinomialTreeModel
from dpp.pricers.black_scholes import BlackScholesModel

def test_binomial_vs_black_scholes_no_div():
    # Binomial tree should converge to Black-Scholes price as steps increase
    spot = 100.0
    strike = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    
    params_call = PricingParams(spot=spot, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="call")
    params_put = PricingParams(spot=spot, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="put")
    
    bs = BlackScholesModel()
    bs_call = bs.price(params_call)
    bs_put = bs.price(params_put)
    
    # 200 steps binomial tree
    tree = BinomialTreeModel(n_steps=200)
    tree_call = tree.price(params_call)
    tree_put = tree.price(params_put)
    
    # Check convergence: difference should be small (usually within 0.05 for 200 steps)
    assert np.abs(tree_call - bs_call) < 0.05
    assert np.abs(tree_put - bs_put) < 0.05

def test_binomial_dividends():
    # S0 = 100, K = 100, T = 1.0, r = 0.05, q = 0.0, sigma = 0.20
    # Discrete dividends: 0.50 paid at t=0.25, and 0.50 paid at t=0.75
    spot = 100.0
    strike = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20
    
    div_schedule = DividendSchedule(dividends=[(0.25, 0.50), (0.75, 0.50)])
    
    params = PricingParams(spot=spot, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="call")
    
    # Price using escrowed dividend model
    tree_escrowed = BinomialTreeModel(n_steps=100, treatment="escrowed")
    px_escrowed = tree_escrowed.price(params, dividend_schedule=div_schedule)
    
    # Price using piecewise lognormal Vellekoop-Nieuwenhuis interpolation model
    tree_pw = BinomialTreeModel(n_steps=100, treatment="piecewise_lognormal")
    px_pw = tree_pw.price(params, dividend_schedule=div_schedule)
    
    # With dividends, the price should be lower than without dividends
    tree_no_div = BinomialTreeModel(n_steps=100)
    px_no_div = tree_no_div.price(params)
    
    assert px_escrowed < px_no_div
    assert px_pw < px_no_div
    
    # Escrowed vs PW lognormal should be close but not identical
    assert np.abs(px_escrowed - px_pw) < 0.2

def test_binomial_greeks():
    spot = 100.5
    strike = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20
    
    params = PricingParams(spot=spot, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="call")
    
    tree = BinomialTreeModel(n_steps=50)
    greeks = tree.greeks(params)
    
    # Verify using finite differences
    epsilon = 1e-4
    
    # Delta
    p_up = PricingParams(spot=spot + epsilon, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="call")
    p_down = PricingParams(spot=spot - epsilon, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="call")
    fd_delta = (tree.price(p_up) - tree.price(p_down)) / (2 * epsilon)
    
    # Binomial greeks calculated via autodiff should match finite differences
    assert np.isclose(greeks.delta, fd_delta, rtol=1e-2)
