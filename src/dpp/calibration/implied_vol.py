import numpy as np
from scipy.optimize import brentq
from dpp.core.models import PricingParams
from dpp.pricers.black_scholes import BlackScholesModel

def implied_volatility_bs(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
    option_type: str
) -> float:
    """
    Calculate Black-Scholes implied volatility from option price using Brent's root-finding method.
    """
    bs = BlackScholesModel()
    
    # Target function for root finder
    def diff_fn(sigma: float) -> float:
        params = PricingParams(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            div_yield=div_yield,
            sigma=sigma,
            option_type=option_type
        )
        return float(bs.price(params) - price)
    
    # Check lower bound: option price must be greater than intrinsic value
    df_q = np.exp(-div_yield * maturity)
    df_r = np.exp(-rate * maturity)
    intrinsic = spot * df_q - strike * df_r if option_type.lower() == "call" else strike * df_r - spot * df_q
    intrinsic = max(0.0, intrinsic)
    
    if price <= intrinsic + 1e-6:
        return 0.0
        
    # Check upper bound: call must be less than spot, put less than strike * df_r
    max_value = spot * df_q if option_type.lower() == "call" else strike * df_r
    if price >= max_value - 1e-6:
        return 5.0  # Cap at high volatility limit
        
    try:
        # Brent's method root-finding in range [1e-4, 5.0] (0.01% to 500%)
        return float(brentq(diff_fn, 1e-4, 5.0, xtol=1e-5))
    except (ValueError, RuntimeError):
        # Fallback or return nan if out of bounds
        return np.nan
