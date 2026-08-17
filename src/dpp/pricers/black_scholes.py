import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Any
from dpp.core.models import PricingModel, PricingParams, Greeks
from dpp.core.registry import register_model

@jax.jit
def bs_price_jax(spot, strike, T, r, q, sigma, is_call):
    """
    Calculate Black-Scholes price using JAX.
    is_call should be 1.0 for call, 0.0 for put.
    """
    # Safe handling of T -> 0 to avoid NaN in gradients
    T_safe = jnp.where(T > 0, T, 1e-10)
    sqrt_T = jnp.sqrt(T_safe)
    
    d1 = (jnp.log(spot / strike) + (r - q + 0.5 * sigma**2) * T_safe) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Standard BS prices
    call_price = spot * jnp.exp(-q * T_safe) * norm.cdf(d1) - strike * jnp.exp(-r * T_safe) * norm.cdf(d2)
    put_price = strike * jnp.exp(-r * T_safe) * norm.cdf(-d2) - spot * jnp.exp(-q * T_safe) * norm.cdf(-d1)
    
    # Intrinsic values for T <= 0
    intrinsic_call = jnp.maximum(spot - strike, 0.0)
    intrinsic_put = jnp.maximum(strike - spot, 0.0)
    
    price = jnp.where(is_call > 0.5, call_price, put_price)
    intrinsic = jnp.where(is_call > 0.5, intrinsic_call, intrinsic_put)
    
    return jnp.where(T > 0, price, intrinsic)

@jax.jit
def bs_greeks_jax(spot, strike, T, r, q, sigma, is_call):
    """
    Calculate Black-Scholes Greeks using JAX automatic differentiation.
    """
    # Delta & Gamma via nested gradients on spot
    p_fn = lambda s: bs_price_jax(s, strike, T, r, q, sigma, is_call)
    delta_fn = jax.grad(p_fn)
    gamma_fn = jax.grad(delta_fn)
    
    delta = delta_fn(spot)
    gamma = gamma_fn(spot)
    
    # Vega, Theta, Rho via gradients on sigma, T, r
    vega = jax.grad(lambda s_val: bs_price_jax(spot, strike, T, r, q, s_val, is_call))(sigma)
    theta = -jax.grad(lambda t_val: bs_price_jax(spot, strike, t_val, r, q, sigma, is_call))(T)
    rho = jax.grad(lambda r_val: bs_price_jax(spot, strike, T, r_val, q, sigma, is_call))(r)
    
    return delta, gamma, vega, theta, rho

class BlackScholesModel(PricingModel):
    def price(self, params: PricingParams, **kwargs: Any) -> float | jnp.ndarray:
        is_call = 1.0 if params.option_type.lower() == "call" else 0.0
        return bs_price_jax(params.spot, params.strike, params.maturity, params.rate, params.div_yield, params.sigma, is_call)

    def price_batch(self, params_batch: PricingParams, **kwargs: Any) -> jnp.ndarray:
        # Vectorized price using jax.vmap
        is_call_batch = jnp.where(params_batch.option_type == "call", 1.0, 0.0)
        vmap_price = jax.vmap(bs_price_jax)
        return vmap_price(
            params_batch.spot,
            params_batch.strike,
            params_batch.maturity,
            params_batch.rate,
            params_batch.div_yield,
            params_batch.sigma,
            is_call_batch
        )

    def greeks(self, params: PricingParams, **kwargs: Any) -> Greeks:
        is_call = 1.0 if params.option_type.lower() == "call" else 0.0
        delta, gamma, vega, theta, rho = bs_greeks_jax(
            params.spot, params.strike, params.maturity, params.rate, params.div_yield, params.sigma, is_call
        )
        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

# Register the model
register_model("black_scholes", BlackScholesModel())
