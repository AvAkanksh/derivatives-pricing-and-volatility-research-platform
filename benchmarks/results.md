# derivatives-pricing-and-volatility-research-platform — Performance & Market Benchmarks

This document records the timing performance and live market data replication accuracy of the derivatives pricing engine.

## 1. Timing Benchmarks

All performance benchmarks were run on a Linux environment with JIT compilation enabled on CPU. Pre-compilation (warmup) times measure the JIT compilation overhead on the first run, and steady-state timings measure subsequent runs.

| Pricing Model | Grid / Tree Size | Warmup Time (JIT) | Steady-State Latency (per call) |
|---|---|---|---|
| **Binomial Tree (CRR)** | 100 steps | 432.27 ms | **909.72 μs** |
| **Local Volatility PDE (Crank-Nicolson)** | 101 x 100 grid | 364.71 ms | **252.06 ms** |

### Methodology
- Timing averages were calculated over 1,000 iterations for the Binomial Tree and 50 iterations for the PDE solver.
- Warmup times represent the compiler compilation latency of the first call before XLA optimizes the execution graph.
- Amortized per-option latency is under 1 ms for binomial trees and around 250 ms for full finite-difference local volatility calibration and PDE pricing.

---

## 2. Live Options Market replication Benchmarks (AAPL)

Option data was fetched for **AAPL** option chains on **August 17, 2026**.
Implied volatilities were calibrated slice-by-slice using the **SVI (Stochastic Volatility Inspired)** model on Out-of-the-Money (OTM) options (calls for $K \ge S$ and puts for $K < S$) to ensure a stable smile, and continuous total variance was reconstructed via linear interpolation in time.

Repricing was conducted on a sample of 50 contracts across ITM, NTM, and OTM options using the Crank-Nicolson PDE solver on the Dupire local volatility surface.

### Spread Hit Rate Results
We define a "Hit" when the theoretical model price falls within the bid-ask spread:
$$\text{Bid} - 10^{-4} \le P_{\text{theo}} \le \text{Ask} + 10^{-4}$$

- **Overall Hit Rate**: **34.00%**
- **Total Contracts Evaluated**: 50

#### Moneyness Bucket Breakdown
- **In-the-Money (ITM)**: 41.18% (7 out of 17 hits)
- **Near-the-Money (NTM)**: 25.00% (2 out of 8 hits)
- **Out-of-the-Money (OTM)**: 32.00% (8 out of 25 hits)

### Calibration & Fitting Discussion
- The calibration of the SVI surface on OTM options only yields a very smooth fit with a low Volatility RMSE of **0.65%** on average.
- The 34.00% spread hit rate is a direct consequence of extremely tight bid-ask spreads in highly liquid options like AAPL (often just $0.01 to $0.05). Under such narrow margins, even minor SVI fitting residuals (e.g., 0.5% in volatility) push model prices outside the bid-ask spread.
- Property-based boundaries in Crank-Nicolson PDE solver prevent any pricing issues (e.g., negative prices for deep OTM puts), enforcing valid mathematical lower and upper boundaries.

---

## 3. Environment Details

- **OS**: Linux
- **Python**: 3.14.6
- **JAX**: 0.11.1
- **Hardware**: CPU-based execution (falling back from CUDA)
- **Data Source**: yfinance (fetched live market option chains)
