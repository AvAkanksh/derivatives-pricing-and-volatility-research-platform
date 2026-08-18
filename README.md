# Derivatives Pricing & Volatility Research Platform

A high-performance, model-agnostic derivatives pricing and volatility research engine written in Python and JAX. 

The platform supports closed-form Black-Scholes, discrete-dividend-aware binomial trees, and local volatility pricing via a Crank-Nicolson PDE solver. It extracts continuous implied volatility surfaces using Gatheral's Stochastic Volatility Inspired (SVI) smile model, reconstructs Dupire local volatility surfaces using JAX automatic differentiation, and enforces static no-arbitrage constraints.

---

## Key Features

- **Model-Agnostic Core**: Pricing logic is decoupled from instrument definitions. Any model (Black-Scholes, Binomial Tree, PDE solver) can price any vanilla instrument via a unified structural interface.
- **JAX-Accelerated Engine**: Numerical kernels are JIT-compiled to XLA via `@jax.jit` for near-native performance. Vectorized pricing across strike-expiry grids is handled via `jax.vmap` to avoid Python loops.
- **Discrete Dividend Treatment**: Supports both standard Spot/Escrowed dividend adjustment and the **Vellekoop-Nieuwenhuis (VN) piecewise lognormal method** (shifting binomial tree nodes and interpolating continuation values at ex-dividend dates).
- **Auto-Differentiated Greeks**: Greek sensitivities (Delta, Gamma, Vega, Theta, Rho) are computed using exact automatic differentiation (`jax.grad` and `jax.hessian`) rather than finite-difference approximations.
- **Dupire Local Volatility**: Calculates Dupire local volatility surfaces using JAX differentiation (`jax.grad`/`jax.hessian`) on the SVI-fitted surface.
- **Static No-Arbitrage Verification**: Automatically checks SVI surface slices for **Calendar Spread** (non-decreasing variance in maturity) and **Butterfly Arbitrage** (non-negative implied density via Gatheral-Jacquier $g(k) \ge 0$).

---

## Directory Architecture

```
derivatives-pricing-platform/
├── pyproject.toml
├── README.md
├── agent.md
├── src/dpp/
│   ├── __init__.py
│   ├── core/
│   │   ├── instruments.py       # EuropeanOption & DividendSchedule dataclasses
│   │   ├── models.py            # PricingModel protocol & Greeks dataclasses
│   │   └── registry.py          # Decoupled pricing model registry
│   ├── pricers/
│   │   ├── black_scholes.py     # Closed-form JIT-compiled pricer
│   │   ├── binomial_tree.py     # CRR tree with piecewise lognormal dividend treatment
│   │   └── local_vol.py         # Crank-Nicolson PDE solver on local vol surface
│   ├── dividends/
│   │   └── piecewise_lognormal.py   # Discrete dividend node adjustment & interpolation
│   ├── calibration/
│   │   ├── implied_vol.py       # BS implied volatility inversion (Brent's method)
│   │   ├── svi.py               # Gatheral SVI fit per expiry slice
│   │   └── dupire.py            # Dupire local volatility reconstruction
│   ├── arbitrage/
│   │   └── checks.py            # Static calendar-spread & butterfly checks
│   ├── data/
│   │   └── market_data.py       # Live option chain loader and OTM filter
│   ├── benchmark/
│   │   └── bench_pricers.py     # Timing harness and market spread hit-rate backtest
│   └── greeks/
│       └── autodiff_greeks.py   # Generic autodiff Greeks wrapper
├── tests/
│   ├── test_black_scholes.py
│   ├── test_binomial_tree.py
│   ├── test_local_vol.py
│   ├── test_arbitrage.py
│   └── test_benchmarks.py
├── notebooks/
│   ├── 01_vol_surface_construction.ipynb
│   ├── 02_dividend_treatment_comparison.ipynb
│   ├── 03_arbitrage_analysis.ipynb
│   └── 04_benchmark_results.ipynb
└── benchmarks/
    └── results.md                # Timings, hit rates, and environment details
```

---

## Theoretical Details & Mathematical Formulations

### 1. Gatheral SVI Implied Variance Formulation
Implied variance slices are parameterized using Gatheral's raw SVI formulation:
$$w(k; a, b, \rho, m, \sigma) = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)$$
where $k = \log(K/S_0)$ is the log-moneyness. Calibration is performed using SciPy's L-BFGS-B optimizer on Out-of-the-Money (OTM) options to ensure clean, stable convergence.

### 2. Dupire Local Volatility Surface
Dupire local volatility is reconstructed from the continuous total variance surface $w(k, T)$ via:
$$\sigma_{\text{loc}}(k, T)^2 = \frac{\frac{\partial w}{\partial T}}{\left( 1 - \frac{k}{w} \frac{\partial w}{\partial k} \right) + \frac{1}{4}\left( -\frac{1}{4} - \frac{1}{w} + \frac{k^2}{w^2} \right) \left( \frac{\partial w}{\partial k} \right)^2 + \frac{1}{2} \frac{\partial^2 w}{\partial k^2}}$$
Partial derivatives ($\partial w / \partial T$, $\partial w / \partial k$, and $\partial^2 w / \partial k^2$) are computed exactly using JAX autodiff.

### 3. Crank-Nicolson PDE Solver
The solver discretizes the Black-Scholes PDE on a uniform log-spot grid using Crank-Nicolson (which is second-order accurate in space and time):
$$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma_{\text{loc}}(S, t)^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - q) S \frac{\partial V}{\partial S} - r V = 0$$
The implicit tridiagonal system at each time step is solved in $O(N)$ using a JAX-compiled Thomas algorithm solver wrapped inside `jax.lax.scan`. Robust boundary conditions prevent numerical pricing anomalies:
- **Call Option boundary**: $V(S_{\max}) = \max(0, S_{\max} e^{-q \tau} - K e^{-r \tau})$, $V(S_{\min}) = 0$.
- **Put Option boundary**: $V(S_{\min}) = \max(0, K e^{-r \tau} - S_{\min} e^{-q \tau})$, $V(S_{\max}) = 0$.

---

## Timing and Calibration Benchmarks

Timing and market benchmarks were evaluated on a Linux CPU environment:

| Benchmark Metric | Grid / Tree Size | Steady-State Latency | Warmup Time (JIT) |
|---|---|---|---|
| **Binomial Tree (CRR)** | 100 steps | **909.72 μs** | 432.27 ms |
| **PDE Solver (Crank-Nicolson)** | 101 x 100 grid | **252.06 ms** | 364.71 ms |

### Live Market Replication (AAPL)
Using SVI calibrated surfaces on OTM options only (Average Volatility RMSE: **0.65%**), PDE local volatility prices are compared against live AAPL spreads:
- **Overall Bid-Ask Spread Hit Rate**: **34.00%**
- **ITM Contracts Hit Rate**: 41.18%
- **OTM Contracts Hit Rate**: 32.00%
- **NTM Contracts Hit Rate**: 25.00%

*Note: Tight spreads of liquid options (often $0.01 - $0.05) make hit rate metrics extremely sensitive to minor calibration residuals.*

---

## Setup & Running Guide

### 1. Installation
Ensure Python 3.11+ is installed. Clone the repository and set up a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
*(If a `requirements.txt` is not present, install the dependencies manually: `pip install jax jaxlib numpy scipy pandas yfinance pytest pytest-benchmark matplotlib ipykernel`)*

### 2. Running Tests
Run the entire unit and benchmark test suite with:
```bash
PYTHONPATH=src pytest
```

### 3. Running Benchmarks
Execute the timing and live market calibration benchmarks:
```bash
PYTHONPATH=src python3 src/dpp/benchmark/bench_pricers.py
```

### 4. Running Research Notebooks
Open the notebooks to visualize calibration smiles, dividend treatments, and arbitrage violations:
```bash
jupyter notebook notebooks/
```
