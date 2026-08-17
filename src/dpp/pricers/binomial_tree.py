import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Optional
from dpp.core.models import PricingModel, PricingParams, Greeks
from dpp.core.registry import register_model
from dpp.dividends.piecewise_lognormal import interpolate_continuation_value_jax, escrowed_spot

@partial(jax.jit, static_argnums=(7,))
def binomial_price_jax(
    spot: Any,
    strike: Any,
    T: Any,
    r: Any,
    q: Any,
    sigma: Any,
    is_call: Any,
    n_steps: int,
    is_dividend_step: Any,
    dividend_amount_step: Any
) -> Any:
    """
    Price an option using the Cox-Ross-Rubinstein (CRR) binomial tree with JAX.
    Supports discrete dividend Vellekoop-Nieuwenhuis interpolation.
    """
    dt = T / n_steps
    u = jnp.exp(sigma * jnp.sqrt(dt))
    d = 1.0 / u
    p = (jnp.exp((r - q) * dt) - d) / (u - d)
    # Clip probability to [0, 1] for safety in extreme cases
    p = jnp.clip(p, 0.0, 1.0)
    discount = jnp.exp(-r * dt)
    
    # State values grid at step N
    i_arr = jnp.arange(n_steps + 1)
    S_N = spot * (u ** i_arr) * (d ** (n_steps - i_arr))
    
    # Intrinsic values at maturity (step N)
    intrinsic_call = jnp.maximum(S_N - strike, 0.0)
    intrinsic_put = jnp.maximum(strike - S_N, 0.0)
    V = jnp.where(is_call > 0.5, intrinsic_call, intrinsic_put)
    
    # Scan backward from step N-1 to step 0
    step_indices = jnp.arange(n_steps - 1, -1, -1)
    is_div_reversed = is_dividend_step[::-1]
    div_amt_reversed = dividend_amount_step[::-1]
    
    def scan_fn(carry_V, xs):
        j_step, is_div, div_amt = xs
        
        # 1. Compute continuation value at step j
        V_next = discount * (p * carry_V[1:] + (1.0 - p) * carry_V[:-1])
        # Pad V_next to size n_steps + 1 using jnp.concatenate
        V_next = jnp.concatenate([V_next, jnp.zeros(1)])
        
        # 2. Apply dividend adjustment if this step contains a dividend
        S_grid = spot * (u ** i_arr) * (d ** (j_step - i_arr))
        df_j = jnp.exp(-r * (T - j_step * dt)) # discount factor to maturity
        
        target_spots = S_grid - div_amt
        # Interpolate continuation values
        V_cum = interpolate_continuation_value_jax(target_spots, S_grid, V_next, is_call, strike, df_j)
        
        V_next = jnp.where(is_div, V_cum, V_next)
        
        return V_next, None

    final_V, _ = jax.lax.scan(scan_fn, V, (step_indices, is_div_reversed, div_amt_reversed))
    return final_V[0]

@partial(jax.jit, static_argnums=(7,))
def binomial_greeks_jax(
    spot: Any,
    strike: Any,
    T: Any,
    r: Any,
    q: Any,
    sigma: Any,
    is_call: Any,
    n_steps: int,
    is_dividend_step: Any,
    dividend_amount_step: Any
) -> Any:
    """
    Calculate binomial tree Greeks using JAX automatic differentiation.
    """
    p_fn = lambda s: binomial_price_jax(s, strike, T, r, q, sigma, is_call, n_steps, is_dividend_step, dividend_amount_step)
    delta_fn = jax.grad(p_fn)
    gamma_fn = jax.grad(delta_fn)
    
    delta = delta_fn(spot)
    gamma = gamma_fn(spot)
    
    vega = jax.grad(lambda sig_val: binomial_price_jax(spot, strike, T, r, q, sig_val, is_call, n_steps, is_dividend_step, dividend_amount_step))(sigma)
    theta = -jax.grad(lambda t_val: binomial_price_jax(spot, strike, t_val, r, q, sigma, is_call, n_steps, is_dividend_step, dividend_amount_step))(T)
    rho = jax.grad(lambda r_val: binomial_price_jax(spot, strike, T, r_val, q, sigma, is_call, n_steps, is_dividend_step, dividend_amount_step))(r)
    
    return delta, gamma, vega, theta, rho

class BinomialTreeModel(PricingModel):
    def __init__(self, n_steps: int = 100, treatment: str = "piecewise_lognormal"):
        self.n_steps = n_steps
        self.treatment = treatment # "piecewise_lognormal" (Vellekoop-Nieuwenhuis) or "escrowed"

    def price(self, params: PricingParams, **kwargs: Any) -> float | jnp.ndarray:
        n_steps = kwargs.get("n_steps", self.n_steps)
        treatment = kwargs.get("treatment", self.treatment)
        
        is_call = 1.0 if params.option_type.lower() == "call" else 0.0
        
        # Extract dividend schedule
        dividend_schedule = kwargs.get("dividend_schedule", None)
        
        # Handle treatments
        if treatment == "escrowed" and dividend_schedule is not None:
            # 1. Spot/escrowed dividend model
            S_adj = escrowed_spot(params.spot, dividend_schedule.dividends, params.rate, params.maturity)
            
            # Setup dummy zero-dividend arrays
            is_div_step = jnp.zeros(n_steps, dtype=bool)
            div_amt_step = jnp.zeros(n_steps)
            
            return binomial_price_jax(S_adj, params.strike, params.maturity, params.rate, params.div_yield, params.sigma, is_call, n_steps, is_div_step, div_amt_step)
            
        else:
            # 2. Piecewise lognormal Vellekoop-Nieuwenhuis
            is_div_step = np.zeros(n_steps, dtype=bool)
            div_amt_step = np.zeros(n_steps)
            
            if dividend_schedule is not None:
                dt = params.maturity / n_steps
                for t_d, D in dividend_schedule.dividends:
                    j = int(np.floor(t_d / dt))
                    if 0 <= j < n_steps:
                        is_div_step[j] = True
                        div_amt_step[j] = D
            
            is_div_step_jax = jnp.array(is_div_step)
            div_amt_step_jax = jnp.array(div_amt_step)
            
            return binomial_price_jax(
                params.spot,
                params.strike,
                params.maturity,
                params.rate,
                params.div_yield,
                params.sigma,
                is_call,
                n_steps,
                is_div_step_jax,
                div_amt_step_jax
            )

    def price_batch(self, params_batch: PricingParams, **kwargs: Any) -> jnp.ndarray:
        # Batch pricing with vmap
        # NOTE: dividend_schedule cannot be easily batch-varied if shapes change,
        # but for identical schedules we can pass constant arrays.
        n_steps = kwargs.get("n_steps", self.n_steps)
        is_call_batch = jnp.where(params_batch.option_type == "call", 1.0, 0.0)
        
        is_div_step_jax = kwargs.get("is_dividend_step", jnp.zeros(n_steps, dtype=bool))
        div_amt_step_jax = kwargs.get("dividend_amount_step", jnp.zeros(n_steps))
        
        vmap_price = jax.vmap(
            lambda s, k, t, r, q, sig, ic: binomial_price_jax(s, k, t, r, q, sig, ic, n_steps, is_div_step_jax, div_amt_step_jax)
        )
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
        n_steps = kwargs.get("n_steps", self.n_steps)
        treatment = kwargs.get("treatment", self.treatment)
        is_call = 1.0 if params.option_type.lower() == "call" else 0.0
        dividend_schedule = kwargs.get("dividend_schedule", None)
        
        if treatment == "escrowed" and dividend_schedule is not None:
            # For escrowed, S is adjusted, so Greeks are calculated on adjusted price
            S_adj = escrowed_spot(params.spot, dividend_schedule.dividends, params.rate, params.maturity)
            is_div_step = jnp.zeros(n_steps, dtype=bool)
            div_amt_step = jnp.zeros(n_steps)
            
            delta, gamma, vega, theta, rho = binomial_greeks_jax(
                S_adj, params.strike, params.maturity, params.rate, params.div_yield, params.sigma, is_call, n_steps, is_div_step, div_amt_step
            )
            # Note: since S_adj = S - PV_div, dS_adj / dS = 1. So delta and gamma are correct!
            return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
        else:
            is_div_step = np.zeros(n_steps, dtype=bool)
            div_amt_step = np.zeros(n_steps)
            if dividend_schedule is not None:
                dt = params.maturity / n_steps
                for t_d, D in dividend_schedule.dividends:
                    j = int(np.floor(t_d / dt))
                    if 0 <= j < n_steps:
                        is_div_step[j] = True
                        div_amt_step[j] = D
            
            is_div_step_jax = jnp.array(is_div_step)
            div_amt_step_jax = jnp.array(div_amt_step)
            
            delta, gamma, vega, theta, rho = binomial_greeks_jax(
                params.spot, params.strike, params.maturity, params.rate, params.div_yield, params.sigma, is_call, n_steps, is_div_step_jax, div_amt_step_jax
            )
            return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

# Register the model
register_model("binomial", BinomialTreeModel())
