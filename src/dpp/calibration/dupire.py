import jax
import jax.numpy as jnp
from typing import Any

def local_vol_surface(w_surface: Any) -> Any:
    """
    Computes the Dupire local volatility surface from a JAX-differentiable 
    total variance surface w_surface(k, T) using Gatheral's formulation.
    Returns a function local_vol(spot, strike, T) -> local_volatility_value
    """
    
    # 1. First-order derivative wrt T
    dw_dT_fn = jax.grad(w_surface, argnums=1)
    
    # 2. First-order derivative wrt k
    dw_dk_fn = jax.grad(w_surface, argnums=0)
    
    # 3. Second-order derivative wrt k
    d2w_dk2_fn = jax.grad(dw_dk_fn, argnums=0)
    
    @jax.jit
    def loc_vol_single(spot: Any, strike: Any, T: Any) -> Any:
        # Avoid division by zero at T=0 or strike=0
        T_safe = jnp.where(T > 0, T, 1e-5)
        strike_safe = jnp.where(strike > 0, strike, 1e-5)
        
        # log-moneyness k = log(strike / spot)
        # Note: Gatheral's formulation uses k = log(K/F)
        # For q=0 and r=0, F = S. If r and q are non-zero, F = S * e^{(r-q)T}.
        # Let's write the formula in terms of k = log(K/S).
        k = jnp.log(strike_safe / spot)
        
        w = w_surface(k, T_safe)
        dw_dT = dw_dT_fn(k, T_safe)
        dw_dk = dw_dk_fn(k, T_safe)
        d2w_dk2 = d2w_dk2_fn(k, T_safe)
        
        # Handle small total variance to avoid division by zero
        w_safe = jnp.where(w > 1e-6, w, 1e-6)
        
        # Dupire denominator (Gatheral's formulation)
        term1 = 1.0 - (k / w_safe) * dw_dk
        term2 = 0.25 * (-0.25 - 1.0 / w_safe + (k**2) / (w_safe**2)) * (dw_dk**2)
        term3 = 0.5 * d2w_dk2
        
        denom = term1 + term2 + term3
        
        # Ensure denominator is positive to prevent negative local variance (butterfly arbitrage)
        denom_safe = jnp.where(denom > 1e-5, denom, 1e-5)
        
        # Local variance must be non-negative
        loc_var = dw_dT / denom_safe
        loc_var_safe = jnp.where(loc_var > 1e-6, loc_var, 1e-6)
        
        return jnp.sqrt(loc_var_safe)
        
    # We vmap over the spots to allow vectorized evaluation
    @jax.jit
    def loc_vol_vectorized(spots: Any, strikes: Any, Ts: Any) -> Any:
        vmap_fn = jax.vmap(loc_vol_single)
        return vmap_fn(spots, strikes, Ts)
        
    return loc_vol_single, loc_vol_vectorized
