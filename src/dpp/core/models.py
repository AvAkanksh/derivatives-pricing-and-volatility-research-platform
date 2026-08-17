from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass(frozen=True)
class PricingParams:
    spot: Any
    strike: Any
    maturity: Any      # T
    rate: Any          # r
    div_yield: Any     # q
    sigma: Any         # volatility
    option_type: str = "call"  # "call" or "put"

    def tree_flatten(self):
        children = (self.spot, self.strike, self.maturity, self.rate, self.div_yield, self.sigma)
        aux_data = (self.option_type,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, option_type=aux_data[0])

@dataclass(frozen=True)
class Greeks:
    delta: float | jnp.ndarray
    gamma: float | jnp.ndarray
    vega: float | jnp.ndarray
    theta: float | jnp.ndarray
    rho: float | jnp.ndarray

@runtime_checkable
class PricingModel(Protocol):
    def price(self, params: PricingParams, **kwargs: Any) -> float | jnp.ndarray:
        """Calculate the option price."""
        ...

    def price_batch(self, params_batch: PricingParams, **kwargs: Any) -> jnp.ndarray:
        """Calculate batch option prices (vectorized using vmap)."""
        ...

    def greeks(self, params: PricingParams, **kwargs: Any) -> Greeks:
        """Calculate the option Greeks (delta, gamma, vega, theta, rho) using automatic differentiation."""
        ...
