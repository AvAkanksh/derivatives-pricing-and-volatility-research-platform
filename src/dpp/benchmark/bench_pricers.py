import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple
from dpp.core.models import PricingParams
from dpp.pricers.black_scholes import BlackScholesModel
from dpp.pricers.binomial_tree import BinomialTreeModel
from dpp.pricers.local_vol import LocalVolatilityModel
from dpp.calibration.implied_vol import implied_volatility_bs
from dpp.calibration.svi import fit_svi_slice, get_svi_surface
from dpp.calibration.dupire import local_vol_surface
from dpp.data.market_data import fetch_option_chain_data

def run_performance_benchmarks() -> Dict[str, Any]:
    """
    Run timing harness for Binomial Tree and PDE pricers.
    Measures warmup vs steady-state latency.
    """
    results = {}
    
    # 1. Binomial Tree Benchmark
    tree = BinomialTreeModel(n_steps=100)
    params = PricingParams(spot=100.5, strike=100.0, maturity=1.0, rate=0.05, div_yield=0.0, sigma=0.20, option_type="call")
    
    # Warmup (first compile)
    t0 = time.perf_counter()
    tree.price(params)
    t_warmup_tree = (time.perf_counter() - t0) * 1000.0 # ms
    
    # Steady state (repeat 1000 times)
    n_runs = 1000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        # We call the JAX pricing function directly to measure pure XLA performance
        # For BinomialTreeModel, price calls binomial_price_jax
        tree.price(params)
    t_steady_tree = ((time.perf_counter() - t0) / n_runs) * 1000000.0 # microseconds (μs)
    
    results["binomial_tree_warmup_ms"] = t_warmup_tree
    results["binomial_tree_steady_us"] = t_steady_tree
    
    # 2. Local Volatility PDE Benchmark
    pde = LocalVolatilityModel(n_space=101, n_time=100)
    
    # Warmup (first compile)
    t0 = time.perf_counter()
    pde.price(params)
    t_warmup_pde = (time.perf_counter() - t0) * 1000.0 # ms
    
    # Steady state (repeat 50 times)
    n_runs_pde = 50
    t0 = time.perf_counter()
    for _ in range(n_runs_pde):
        pde.price(params)
    t_steady_pde = ((time.perf_counter() - t0) / n_runs_pde) * 1000.0 # ms
    
    results["pde_warmup_ms"] = t_warmup_pde
    results["pde_steady_ms"] = t_steady_pde
    
    return results

def calibrate_and_price_market(ticker_symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pulls live options chain data, calibrates SVI and Dupire Local Volatility,
    prices options using PDE solver, and checks bid-ask spread hit rate.
    """
    print(f"Fetching market data for {ticker_symbol}...")
    df = fetch_option_chain_data(ticker_symbol)
    if df.empty:
        raise ValueError(f"No option chain data available for {ticker_symbol}")
        
    print(f"Fetched {len(df)} option contracts. Calculating implied volatilities...")
    # Calculate implied vols from mid prices
    df["implied_vol_calc"] = df.apply(
        lambda r: implied_volatility_bs(r["mid"], r["spot"], r["strike"], r["maturity"], 0.05, 0.0, r["option_type"]),
        axis=1
    )
    
    # Filter out contracts where IV could not be calculated (NaN or 0.0)
    df = df[df["implied_vol_calc"] > 0.01].dropna().copy()
    print(f"After IV cleaning: {len(df)} contracts.")
    
    if len(df) < 10:
        raise ValueError("Too few valid options contract data after cleaning.")
        
    # Filter OTM only for SVI calibration
    df_otm = df[
        ((df["option_type"] == "call") & (df["strike"] >= df["spot"])) |
        ((df["option_type"] == "put") & (df["strike"] < df["spot"]))
    ].copy()
    
    # Calibrate SVI per expiry slice (must have at least 3 strikes in each slice)
    expiries = df_otm["expiry_str"].unique()
    svi_params_list = []
    valid_expiries = []
    
    print("Fitting SVI smiles per expiry...")
    for expiry in expiries:
        slice_df = df_otm[df_otm["expiry_str"] == expiry].sort_values("strike")
        if len(slice_df) < 4:
            continue # Skip slice if not enough strikes
            
        strikes = slice_df["strike"].values
        spot = slice_df["spot"].iloc[0]
        k = np.log(strikes / spot) # log-moneyness
        T = slice_df["maturity"].iloc[0]
        w_mkt = (slice_df["implied_vol_calc"].values ** 2) * T
        
        try:
            params = fit_svi_slice(k, w_mkt)
            svi_params_list.append(params)
            valid_expiries.append(T)
        except Exception as e:
            print(f"Failed to fit SVI for {expiry}: {e}")
            
    if len(valid_expiries) < 2:
        raise ValueError("Not enough valid expiry slices calibrated to build a surface.")
        
    # Sort expiries and params
    sort_idx = np.argsort(valid_expiries)
    valid_expiries = np.array(valid_expiries)[sort_idx]
    svi_params = np.array(svi_params_list)[sort_idx]
    
    # Build continuous SVI surface and Dupire Local Volatility surface
    w_surface = get_svi_surface(valid_expiries, svi_params)
    loc_vol_single, loc_vol_vec = local_vol_surface(w_surface)
    
    # Price a representative sample of options using PDE solver on Local Vol surface (to keep benchmarks fast)
    pde_model = LocalVolatilityModel(n_space=101, n_time=100)
    
    df_sample = df.sample(min(50, len(df)), random_state=42).copy()
    spot_val = df_sample["spot"].iloc[0]
    
    # Define loc_vol_fn once to avoid JAX JIT recompilation overhead
    def loc_vol_fn(S, t):
        if jnp.ndim(S) == 0:
            return loc_vol_single(spot_val, S, t)
        else:
            return jax.vmap(lambda s: loc_vol_single(spot_val, s, t))(S)
            
    print(f"Pricing a representative sample of {len(df_sample)} options on local volatility surface...")
    theo_prices = []
    for idx, row in df_sample.iterrows():
        # Setup PricingParams (we pass flat market IV as parameter, but override with local vol fn)
        params = PricingParams(
            spot=row["spot"],
            strike=row["strike"],
            maturity=row["maturity"],
            rate=0.05,
            div_yield=0.0,
            sigma=row["implied_vol_calc"],
            option_type=row["option_type"]
        )
        
        try:
            # Price using PDE pricer
            p_theo = float(pde_model.price(params, local_vol_fn=loc_vol_fn, ref_spot=row["spot"]))
        except Exception:
            p_theo = np.nan
        theo_prices.append(p_theo)
        
    df_sample["theoretical_price"] = theo_prices
    df_sample = df_sample.dropna(subset=["theoretical_price"]).copy()
    
    # Check hit rate against bid-ask spreads
    # bid <= theoretical <= ask (with small numerical tolerance 1e-4)
    df_sample["within_spread"] = (df_sample["theoretical_price"] >= df_sample["bid"] - 1e-4) & (df_sample["theoretical_price"] <= df_sample["ask"] + 1e-4)
    
    # Compute hit rate overall
    hit_rate = df_sample["within_spread"].mean()
    
    # Compute hit rate by moneyness bucket
    # Near-the-money (NTM): 0.95 <= S/K <= 1.05
    # Out-of-the-money (OTM): S/K < 0.95 (calls) or S/K > 1.05 (puts)
    # In-the-money (ITM): S/K > 1.05 (calls) or S/K < 0.95 (puts)
    moneyness = df_sample["spot"] / df_sample["strike"]
    
    df_sample["moneyness_bucket"] = "NTM"
    # Call OTM/ITM
    df_sample.loc[(df_sample["option_type"] == "call") & (moneyness > 1.05), "moneyness_bucket"] = "ITM"
    df_sample.loc[(df_sample["option_type"] == "call") & (moneyness < 0.95), "moneyness_bucket"] = "OTM"
    # Put OTM/ITM
    df_sample.loc[(df_sample["option_type"] == "put") & (moneyness > 1.05), "moneyness_bucket"] = "OTM"
    df_sample.loc[(df_sample["option_type"] == "put") & (moneyness < 0.95), "moneyness_bucket"] = "ITM"
    
    bucket_stats = df_sample.groupby("moneyness_bucket")["within_spread"].agg(["count", "mean"])
    
    stats = {
        "hit_rate_overall": float(hit_rate),
        "total_contracts": len(df_sample),
        "bucket_stats": bucket_stats.to_dict(orient="index")
    }
    
    return df_sample, stats

if __name__ == "__main__":
    print("Running Pricing Engine Performance Benchmarks...")
    perf = run_performance_benchmarks()
    print("-" * 50)
    print(f"Binomial Tree (100 steps) Warmup: {perf['binomial_tree_warmup_ms']:.2f} ms")
    print(f"Binomial Tree (100 steps) Steady State: {perf['binomial_tree_steady_us']:.2f} us")
    print(f"PDE Pricer (101x100 grid) Warmup: {perf['pde_warmup_ms']:.2f} ms")
    print(f"PDE Pricer (101x100 grid) Steady State: {perf['pde_steady_ms']:.2f} ms")
    print("-" * 50)
    
    try:
        df, stats = calibrate_and_price_market("AAPL")
        print(f"Live Market Option Spread Hit Rate (AAPL): {stats['hit_rate_overall']*100:.2f}%")
        print("Moneyness Bucket Breakdown:")
        for bucket, data in stats["bucket_stats"].items():
            print(f"  {bucket}: Count = {data['count']}, Hit Rate = {data['mean']*100:.2f}%")
    except Exception as e:
        print(f"Market calibration check failed: {e}")
