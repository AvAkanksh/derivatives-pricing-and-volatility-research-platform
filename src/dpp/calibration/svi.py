import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Any

@jax.jit
def svi_variance_jax(k: Any, a: Any, b: Any, rho: Any, m: Any, sigma: Any) -> Any:
    """
    Gatheral's SVI formulation for total variance w(k).
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    return a + b * (rho * (k - m) + jnp.sqrt((k - m)**2 + sigma**2))

def fit_svi_slice(k_array: np.ndarray, w_array: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Fit SVI parameters (a, b, rho, m, sigma) for a single expiry slice using SciPy least squares.
    """
    # Objective function
    def loss_fn(params):
        a, b, rho, m, sigma = params
        w_pred = a + b * (rho * (k_array - m) + np.sqrt((k_array - m)**2 + sigma**2))
        return np.sum((w_pred - w_array)**2)
    
    # Constraints and bounds
    # b >= 0, rho in [-1, 1], sigma > 0
    # To avoid exact boundary issues, we restrict slightly
    bounds = [
        (-10.0, 10.0),       # a: variance level (can be slightly negative in SVI if spot is fitted, but usually positive)
        (0.0, 5.0),          # b: slope of wings
        (-0.999, 0.999),     # rho: skew
        (-5.0, 5.0),         # m: smile shift
        (1e-4, 5.0)          # sigma: vertex smoothing
    ]
    
    # Initial guess
    # a ~ min of variance, b ~ slope, rho ~ skew, m ~ argmin, sigma ~ 0.1
    min_w = np.min(w_array)
    idx_min = np.argmin(w_array)
    m_guess = k_array[idx_min]
    a_guess = max(1e-4, min_w)
    
    initial_guess = [a_guess, 0.1, -0.1, m_guess, 0.1]
    
    res = minimize(loss_fn, initial_guess, bounds=bounds, method="L-BFGS-B")
    return tuple(res.x)

def get_svi_surface(T_expiries: np.ndarray, svi_params: np.ndarray) -> Any:
    """
    Returns a JAX-compatible, differentiable function w_surface(k, T)
    which interpolates/extrapolates total variance across the expiry slices.
    """
    T_exp_jax = jnp.array(T_expiries)
    params_jax = jnp.array(svi_params)
    
    @jax.jit
    def w_surface(k: Any, T: Any) -> Any:
        # Safe T to avoid division by zero
        T_safe = jnp.where(T > 0, T, 1e-10)
        
        # Find neighboring expiries
        idx = jnp.searchsorted(T_exp_jax, T_safe)
        idx = jnp.clip(idx, 1, len(T_expiries) - 1)
        
        T_left = T_exp_jax[idx - 1]
        T_right = T_exp_jax[idx]
        
        p_left = params_jax[idx - 1]
        p_right = params_jax[idx]
        
        # SVI variance for left and right expiries
        w_left = svi_variance_jax(k, p_left[0], p_left[1], p_left[2], p_left[3], p_left[4])
        w_right = svi_variance_jax(k, p_right[0], p_right[1], p_right[2], p_right[3], p_right[4])
        
        # Interpolation weight
        w_interp = w_left + (w_right - w_left) * (T_safe - T_left) / (T_right - T_left)
        
        # Extrapolation below the first expiry
        w_below = w_left * (T_safe / T_left)
        
        # Extrapolation above the last expiry (constant volatility slope)
        w_above = w_right + (T_safe - T_right) * (w_right / T_right)
        
        # Combine branches based on T_safe location
        result = jnp.where(T_safe < T_left, w_below, w_interp)
        result = jnp.where(T_safe > T_right, w_above, result)
        
        return result
        
    return w_surface
