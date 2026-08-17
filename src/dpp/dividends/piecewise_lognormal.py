import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Any

def escrowed_spot(S0: float, dividends: List[Tuple[float, float]], r: float, T: float) -> float:
    """
    Calculate the escrowed (adjusted) stock price.
    S_adj = S0 - sum(D_i * e^{-r * t_i}) for all t_i < T.
    """
    pv_divs = 0.0
    for t_i, D_i in dividends:
        if t_i < T:
            pv_divs += D_i * np.exp(-r * t_i)
    return S0 - pv_divs

@jax.jit
def escrowed_spot_jax(S0: Any, dividend_times: Any, dividend_amounts: Any, r: Any, T: Any) -> Any:
    """
    Calculate the escrowed stock price using JAX.
    dividend_times and dividend_amounts must be arrays of the same length (padded with zeros if needed).
    """
    mask = (dividend_times < T) & (dividend_times > 0)
    pv_divs = jnp.sum(jnp.where(mask, dividend_amounts * jnp.exp(-r * dividend_times), 0.0))
    return S0 - pv_divs

@jax.jit
def interpolate_continuation_value_jax(
    target_spots: Any,
    grid_spots: Any,
    grid_values: Any,
    is_call: Any,
    strike: Any,
    df: Any
) -> Any:
    """
    Interpolate continuation values for off-tree nodes (Vellekoop-Nieuwenhuis method).
    """
    # For values below the grid:
    # Call option value approaches 0
    # Put option value approaches strike * df - spot
    left_val = jnp.where(is_call > 0.5, 0.0, jnp.maximum(strike * df - target_spots, 0.0))
    
    # For values above the grid:
    # Call option value approaches spot - strike * df
    # Put option value approaches 0
    right_val = jnp.where(is_call > 0.5, jnp.maximum(target_spots - strike * df, 0.0), 0.0)
    
    return jnp.interp(target_spots, grid_spots, grid_values, left=left_val, right=right_val)
