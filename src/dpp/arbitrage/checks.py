import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, List, Tuple, Callable, Dict
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def butterfly_condition_jax(w_surface: Callable[[Any, Any], Any], k: Any, T: Any) -> Any:
    """
    Computes Gatheral-Jacquier butterfly condition g(k) at (k, T).
    g(k) = (1 - k * w'/ (2w))^2 - (w'^2 / 4)*(1/w + 1/4) + w''/2
    A violation occurs if g(k) < 0.
    """
    # derivatives wrt k
    w_fn = lambda x: w_surface(x, T)
    dw_dk_fn = jax.grad(w_fn)
    d2w_dk2_fn = jax.grad(dw_dk_fn)
    
    w_val = w_fn(k)
    dw_dk = dw_dk_fn(k)
    d2w_dk2 = d2w_dk2_fn(k)
    
    w_safe = jnp.where(w_val > 1e-6, w_val, 1e-6)
    
    term1 = (1.0 - k * dw_dk / (2.0 * w_safe))**2
    term2 = (dw_dk**2 / 4.0) * (1.0 / w_safe + 0.25)
    term3 = 0.5 * d2w_dk2
    
    return term1 - term2 + term3

def calendar_arbitrage_violations(
    w_surface: Callable[[Any, Any], Any],
    k_grid: np.ndarray,
    T_grid: np.ndarray
) -> List[Tuple[float, float, float]]:
    """
    Check for calendar spread arbitrage: w(k, T) must be non-decreasing in T.
    Returns list of violations: (k, T_prev, T_curr) where variance decreased.
    """
    violations = []
    # Sort T_grid to ensure increasing order
    T_sorted = np.sort(T_grid)
    
    for k in k_grid:
        w_prev = float(w_surface(k, T_sorted[0]))
        for i in range(1, len(T_sorted)):
            T_prev = T_sorted[i-1]
            T_curr = T_sorted[i]
            w_curr = float(w_surface(k, T_curr))
            
            # Toleration buffer for small numerical noise
            if w_curr < w_prev - 1e-6:
                violations.append((k, T_prev, T_curr))
            w_prev = w_curr
            
    return violations

def butterfly_arbitrage_violations(
    w_surface: Callable[[Any, Any], Any],
    k_grid: np.ndarray,
    T_grid: np.ndarray
) -> List[Tuple[float, float, float]]:
    """
    Check for butterfly arbitrage: g(k) >= 0.
    Returns list of violations: (k, T, g_val) where g_val < 0.
    """
    violations = []
    
    # We compile the JAX function to check butterfly condition
    vmap_check = jax.vmap(lambda k_val, T_val: butterfly_condition_jax(w_surface, k_val, T_val))
    
    for T in T_grid:
        g_vals = np.array(vmap_check(jnp.array(k_grid), jnp.ones(len(k_grid)) * T))
        for k, g in zip(k_grid, g_vals):
            if g < -1e-6:
                violations.append((k, T, float(g)))
                
    return violations

def price_bounds_check(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    div_yield: float,
    maturity: float,
    option_type: str
) -> Dict[str, Any]:
    """
    Check no-arbitrage bounds on individual option price.
    Call: max(0, S * e^{-qT} - K * e^{-rT}) <= C <= S * e^{-qT}
    Put: max(0, K * e^{-rT} - S * e^{-qT}) <= P <= K * e^{-rT}
    """
    df_q = np.exp(-div_yield * maturity)
    df_r = np.exp(-rate * maturity)
    
    lower_bound = 0.0
    upper_bound = 0.0
    
    if option_type.lower() == "call":
        lower_bound = max(0.0, spot * df_q - strike * df_r)
        upper_bound = spot * df_q
    elif option_type.lower() == "put":
        lower_bound = max(0.0, strike * df_r - spot * df_q)
        upper_bound = strike * df_r
        
    is_valid = bool((price >= lower_bound - 1e-5) and (price <= upper_bound + 1e-5))
    
    return {
        "is_valid": is_valid,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "violation": "none" if is_valid else ("below_lower" if price < lower_bound - 1e-5 else "above_upper")
    }

def put_call_parity_residual(
    call_px: float,
    put_px: float,
    spot: float,
    strike: float,
    rate: float,
    div_yield: float,
    maturity: float
) -> float:
    """
    Computes put-call parity residual:
    C - P - (S * e^{-qT} - K * e^{-rT})
    """
    df_q = np.exp(-div_yield * maturity)
    df_r = np.exp(-rate * maturity)
    expected_diff = spot * df_q - strike * df_r
    actual_diff = call_px - put_px
    return float(actual_diff - expected_diff)
