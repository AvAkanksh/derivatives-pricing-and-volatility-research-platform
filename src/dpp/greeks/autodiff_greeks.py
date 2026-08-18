import jax
from typing import Callable, Any
from dpp.core.models import PricingParams, Greeks

def make_autodiff_greeks(
    raw_price_fn: Callable[[Any, Any, Any, Any, Any, Any, Any], Any]
) -> Callable[[Any, Any, Any, Any, Any, Any, Any], Tuple[Any, Any, Any, Any, Any]]:
    """
    Given a pure JAX pricing function of signature:
        price(spot, strike, maturity, rate, div_yield, sigma, is_call)
    Returns a JITted function that computes delta, gamma, vega, theta, rho
    using JAX automatic differentiation.
    """
    
    @jax.jit
    def greeks_fn(spot, strike, T, r, q, sigma, is_call):
        p_fn = lambda s: raw_price_fn(s, strike, T, r, q, sigma, is_call)
        delta_fn = jax.grad(p_fn)
        gamma_fn = jax.grad(delta_fn)
        
        delta = delta_fn(spot)
        gamma = gamma_fn(spot)
        
        vega = jax.grad(lambda sig_val: raw_price_fn(spot, strike, T, r, q, sig_val, is_call))(sigma)
        theta = -jax.grad(lambda t_val: raw_price_fn(spot, strike, t_val, r, q, sigma, is_call))(T)
        rho = jax.grad(lambda r_val: raw_price_fn(spot, strike, T, r_val, q, sigma, is_call))(r)
        
        return delta, gamma, vega, theta, rho
        
    return greeks_fn
