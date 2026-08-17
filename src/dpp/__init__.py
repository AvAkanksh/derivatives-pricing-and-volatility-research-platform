import jax

# Enable float64 (double precision) globally in JAX for quantitative finance accuracy
jax.config.update("jax_enable_x64", True)
