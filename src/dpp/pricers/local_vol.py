import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Callable
from dpp.core.models import PricingModel, PricingParams, Greeks
from dpp.core.registry import register_model

@jax.jit
def thomas_solve_jax(a: Any, b: Any, c: Any, d: Any) -> Any:
    """
    Solve tridiagonal linear system using JAX-compatible Thomas algorithm.
    a, b, c, d are arrays of size n.
    """
    n = b.shape[0]
    
    # Forward sweep
    c0 = c[0] / b[0]
    d0 = d[0] / b[0]
    
    a_slice = a[1:]
    b_slice = b[1:]
    c_slice = c[1:]
    d_slice = d[1:]
    
    def forward_step(carry, inputs):
        c_prev, d_prev = carry
        a_i, b_i, c_i, d_i = inputs
        denom = b_i - a_i * c_prev
        c_curr = c_i / denom
        d_curr = (d_i - a_i * d_prev) / denom
        return (c_curr, d_curr), (c_curr, d_curr)
        
    _, (c_primes, d_primes) = jax.lax.scan(
        forward_step,
        (c0, d0),
        (a_slice, b_slice, c_slice, d_slice)
    )
    
    c_prime_full = jnp.concatenate([jnp.array([c0]), c_primes])
    d_prime_full = jnp.concatenate([jnp.array([d0]), d_primes])
    
    # Backward substitution
    x_last = d_prime_full[-1]
    
    c_prime_rev = c_prime_full[:-1][::-1]
    d_prime_rev = d_prime_full[:-1][::-1]
    
    def backward_step(x_next, inputs):
        c_i, d_i = inputs
        x_curr = d_i - c_i * x_next
        return x_curr, x_curr
        
    _, x_rev = jax.lax.scan(
        backward_step,
        x_last,
        (c_prime_rev, d_prime_rev)
    )
    
    x_full = jnp.concatenate([x_rev[::-1], jnp.array([x_last])])
    return x_full

@partial(jax.jit, static_argnums=(5, 7, 8))
def pde_price_jax(
    spot: Any,
    strike: Any,
    T: Any,
    r: Any,
    q: Any,
    local_vol_fn: Callable[[Any, Any], Any],
    is_call: Any,
    n_space: int,
    n_time: int,
    ref_spot: Any
) -> Any:
    """
    Solve Crank-Nicolson PDE on log-spot grid using JAX.
    local_vol_fn: callable (S, t) -> vol
    """
    T_safe = jnp.where(T > 0, T, 1e-10)
    dt = T_safe / n_time
    
    # Use stop_gradient on the passed ref_spot to fix the grid coordinates
    S_ref = jax.lax.stop_gradient(ref_spot)
    char_vol = local_vol_fn(S_ref, 0.0)
    
    x_center = jnp.log(S_ref)
    x_min = x_center - 4.0 * char_vol * jnp.sqrt(T_safe)
    x_max = x_center + 4.0 * char_vol * jnp.sqrt(T_safe)
    dx = (x_max - x_min) / (n_space - 1)
    
    x_grid = jnp.linspace(x_min, x_max, n_space)
    S_grid = jnp.exp(x_grid)
    
    # Initial conditions at tau = 0 (maturity)
    intrinsic_call = jnp.maximum(S_grid - strike, 0.0)
    intrinsic_put = jnp.maximum(strike - S_grid, 0.0)
    V = jnp.where(is_call > 0.5, intrinsic_call, intrinsic_put)
    
    # Scan backward in time
    step_indices = jnp.arange(n_time)
    
    def step_fn(carry_V, k_step):
        V_prev = carry_V
        tau_curr = k_step * dt
        tau_next = (k_step + 1) * dt
        
        # t_curr is actual time = T - tau_curr
        t_curr = T_safe - tau_curr
        
        # Evaluate local volatility at each spot grid point
        sig = local_vol_fn(S_grid, t_curr)
        
        a = 0.5 * sig**2
        b = r - q - a
        
        alpha = a / dx**2 - b / (2.0 * dx)
        beta = -2.0 * a / dx**2
        gamma = a / dx**2 + b / (2.0 * dx)
        
        # Interior coefficients
        A_int = -0.5 * dt * alpha[1:-1]
        B_int = 1.0 - 0.5 * dt * (beta[1:-1] - r)
        C_int = -0.5 * dt * gamma[1:-1]
        
        D_int = -A_int * V_prev[:-2] + (2.0 - B_int) * V_prev[1:-1] - C_int * V_prev[2:]
        
        # Boundary conditions at tau_next
        D_0_call = jnp.maximum(0.0, S_grid[0] * jnp.exp(-q * tau_next) - strike * jnp.exp(-r * tau_next))
        D_N_call = jnp.maximum(0.0, S_grid[-1] * jnp.exp(-q * tau_next) - strike * jnp.exp(-r * tau_next))
        
        D_0_put = jnp.maximum(0.0, strike * jnp.exp(-r * tau_next) - S_grid[0] * jnp.exp(-q * tau_next))
        D_N_put = jnp.maximum(0.0, strike * jnp.exp(-r * tau_next) - S_grid[-1] * jnp.exp(-q * tau_next))
        
        D_0 = jnp.where(is_call > 0.5, D_0_call, D_0_put)
        D_N = jnp.where(is_call > 0.5, D_N_call, D_N_put)
        
        A = jnp.concatenate([jnp.zeros(1), A_int, jnp.zeros(1)])
        B = jnp.concatenate([jnp.ones(1), B_int, jnp.ones(1)])
        C = jnp.concatenate([jnp.zeros(1), C_int, jnp.zeros(1)])
        D = jnp.concatenate([jnp.array([D_0]), D_int, jnp.array([D_N])])
        
        V_new = thomas_solve_jax(A, B, C, D)
        return V_new, None
        
    final_V, _ = jax.lax.scan(step_fn, V, step_indices)
    
    # Interpolate the final price at the variable spot using the fixed grid
    return jnp.interp(jnp.log(spot), x_grid, final_V)

@partial(jax.jit, static_argnums=(5, 7, 8))
def pde_greeks_jax(
    spot: Any,
    strike: Any,
    T: Any,
    r: Any,
    q: Any,
    local_vol_fn: Callable[[Any, Any], Any],
    is_call: Any,
    n_space: int,
    n_time: int,
    ref_spot: Any
) -> Any:
    """
    Calculate PDE Greeks using JAX automatic differentiation.
    """
    p_fn = lambda s: pde_price_jax(s, strike, T, r, q, local_vol_fn, is_call, n_space, n_time, ref_spot)
    delta_fn = jax.grad(p_fn)
    gamma_fn = jax.grad(delta_fn)
    
    delta = delta_fn(spot)
    gamma = gamma_fn(spot)
    
    # Vega, Theta, Rho
    # Since vega depends on shifting the local vol surface, we can wrap local_vol_fn:
    # sig_shifted = lambda S, t: local_vol_fn(S, t) + shift
    # And differentiate wrt shift!
    vega = jax.grad(
        lambda shift_val: pde_price_jax(
            spot, strike, T, r, q, lambda S, t: local_vol_fn(S, t) + shift_val, is_call, n_space, n_time, ref_spot
        )
    )(0.0)
    
    theta = -jax.grad(
        lambda t_val: pde_price_jax(spot, strike, t_val, r, q, local_vol_fn, is_call, n_space, n_time, ref_spot)
    )(T)
    
    rho = jax.grad(
        lambda r_val: pde_price_jax(spot, strike, T, r_val, q, local_vol_fn, is_call, n_space, n_time, ref_spot)
    )(r)
    
    return delta, gamma, vega, theta, rho

class LocalVolatilityModel(PricingModel):
    def __init__(self, n_space: int = 101, n_time: int = 100, local_vol_fn: Optional[Callable[[Any, Any], Any]] = None):
        self.n_space = n_space
        self.n_time = n_time
        self.local_vol_fn = local_vol_fn if local_vol_fn is not None else (lambda S, t: 0.20)

    def price(self, params: PricingParams, **kwargs: Any) -> float | jnp.ndarray:
        n_space = kwargs.get("n_space", self.n_space)
        n_time = kwargs.get("n_time", self.n_time)
        local_vol_fn = kwargs.get("local_vol_fn", self.local_vol_fn)
        ref_spot = kwargs.get("ref_spot", params.spot)
        
        is_call = 1.0 if params.option_type.lower() == "call" else 0.0
        
        # If flat vol is passed in params, we can override or wrap it
        # Actually, let's use the local_vol_fn if available, otherwise construct flat vol from params.sigma
        if "local_vol_fn" not in kwargs and params.sigma is not None:
            active_vol_fn = lambda S, t: jnp.where(jnp.shape(S) == (), params.sigma, jnp.ones_like(S) * params.sigma)
        else:
            active_vol_fn = local_vol_fn
            
        return pde_price_jax(
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            params.div_yield,
            active_vol_fn,
            is_call,
            n_space,
            n_time,
            ref_spot
        )

    def price_batch(self, params_batch: PricingParams, **kwargs: Any) -> jnp.ndarray:
        n_space = kwargs.get("n_space", self.n_space)
        n_time = kwargs.get("n_time", self.n_time)
        local_vol_fn = kwargs.get("local_vol_fn", self.local_vol_fn)
        
        is_call_batch = jnp.where(params_batch.option_type == "call", 1.0, 0.0)
        
        # If ref_spot is not in kwargs, default to batch spot (vectorized)
        ref_spot_batch = kwargs.get("ref_spot", params_batch.spot)
        
        # Define vectorized PDE pricer
        vmap_price = jax.vmap(
            lambda s, k, t, r, q, sig, ic, rs: pde_price_jax(
                s, k, t, r, q,
                (lambda S, time: jnp.where(jnp.shape(S) == (), sig, jnp.ones_like(S) * sig) if local_vol_fn is None else local_vol_fn),
                ic, n_space, n_time, rs
            )
        )
        return vmap_price(
            params_batch.spot,
            params_batch.strike,
            params_batch.maturity,
            params_batch.rate,
            params_batch.div_yield,
            params_batch.sigma,
            is_call_batch,
            ref_spot_batch
        )

    def greeks(self, params: PricingParams, **kwargs: Any) -> Greeks:
        n_space = kwargs.get("n_space", self.n_space)
        n_time = kwargs.get("n_time", self.n_time)
        local_vol_fn = kwargs.get("local_vol_fn", self.local_vol_fn)
        ref_spot = kwargs.get("ref_spot", params.spot)
        
        is_call = 1.0 if params.option_type.lower() == "call" else 0.0
        
        if "local_vol_fn" not in kwargs and params.sigma is not None:
            active_vol_fn = lambda S, t: jnp.where(jnp.shape(S) == (), params.sigma, jnp.ones_like(S) * params.sigma)
        else:
            active_vol_fn = local_vol_fn
            
        delta, gamma, vega, theta, rho = pde_greeks_jax(
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            params.div_yield,
            active_vol_fn,
            is_call,
            n_space,
            n_time,
            ref_spot
        )
        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

# Register the model
register_model("local_vol", LocalVolatilityModel())
