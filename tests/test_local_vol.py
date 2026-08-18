import pytest
import numpy as np
import jax
import jax.numpy as jnp
from dpp.core.models import PricingParams
from dpp.pricers.local_vol import LocalVolatilityModel
from dpp.pricers.black_scholes import BlackScholesModel
from dpp.calibration.svi import fit_svi_slice, get_svi_surface
from dpp.calibration.dupire import local_vol_surface

def test_pde_vs_black_scholes_flat_vol():
    # PDE solver with flat volatility should match Black-Scholes closed form price
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
    
    # Create PDE model (with 201 space steps and 200 time steps for high resolution)
    pde_model = LocalVolatilityModel(n_space=201, n_time=200)
    pde_call = pde_model.price(params_call)
    pde_put = pde_model.price(params_put)
    
    # Check match within O(2e-3)
    assert np.abs(pde_call - bs_call) < 2e-3
    assert np.abs(pde_put - bs_put) < 2e-3

def test_pde_greeks_flat_vol():
    # PDE Greeks under flat vol should match finite differences
    spot = 100.005
    strike = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20
    
    params = PricingParams(spot=spot, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="call")
    
    pde_model = LocalVolatilityModel(n_space=101, n_time=100)
    greeks = pde_model.greeks(params, ref_spot=100.0)
    
    # central finite differences
    epsilon = 1e-5
    
    p_up = PricingParams(spot=spot + epsilon, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="call")
    p_down = PricingParams(spot=spot - epsilon, strike=strike, maturity=T, rate=r, div_yield=q, sigma=sigma, option_type="call")
    fd_delta = (pde_model.price(p_up, ref_spot=100.0) - pde_model.price(p_down, ref_spot=100.0)) / (2 * epsilon)
    
    # Matches to ~1% relative error
    assert np.isclose(greeks.delta, fd_delta, rtol=1e-2)

def test_svi_and_dupire_flow():
    # Build a synthetic SVI surface and compute Dupire local volatility
    T_expiries = np.array([0.25, 0.5, 1.0])
    
    # Strikes and implied vols for each expiry
    # we represent them as a function of log-moneyness k
    k_grid = np.linspace(-0.5, 0.5, 11)
    
    # SVI parameters per expiry
    # a, b, rho, m, sigma
    svi_params_by_expiry = [
        (0.04, 0.1, -0.2, 0.0, 0.1),  # T = 0.25
        (0.05, 0.1, -0.2, 0.0, 0.1),  # T = 0.5
        (0.06, 0.1, -0.2, 0.0, 0.1)   # T = 1.0
    ]
    
    w_surface = get_svi_surface(T_expiries, np.array(svi_params_by_expiry))
    loc_vol_single, loc_vol_vec = local_vol_surface(w_surface)
    
    # Check local vol values at spot=100.0, strike=100.0 (k=0.0) at T=0.5
    vol = loc_vol_single(100.0, 100.0, 0.5)
    
    # Implied variance at k=0 is: w_left = 0.05 + 0.1 * 0.1 = 0.06.
    # Total variance is w = 0.06. Implied vol is sqrt(0.06 / 0.5) = sqrt(0.12) = 0.3464.
    # Local vol should be positive and reasonable
    assert vol > 0.0
    assert 0.1 < vol < 0.6

def test_pde_pricing_non_flat_local_vol():
    # Setup a simple SVI surface and price using LocalVolatilityModel
    T_expiries = np.array([0.5, 1.0])
    svi_params = np.array([
        [0.04, 0.1, -0.2, 0.0, 0.1],  # T = 0.5
        [0.06, 0.1, -0.2, 0.0, 0.1]   # T = 1.0
    ])
    w_surface = get_svi_surface(T_expiries, svi_params)
    loc_vol_single, _ = local_vol_surface(w_surface)
    
    pde_model = LocalVolatilityModel(n_space=51, n_time=50)
    params = PricingParams(spot=100.0, strike=100.0, maturity=0.75, rate=0.05, div_yield=0.0, sigma=0.20, option_type="call")
    
    def loc_vol_fn(S, t):
        if jnp.ndim(S) == 0:
            return loc_vol_single(100.0, S, t)
        else:
            return jax.vmap(lambda s: loc_vol_single(100.0, s, t))(S)
            
    p = pde_model.price(params, local_vol_fn=loc_vol_fn, ref_spot=100.0)
    assert p > 0.0

