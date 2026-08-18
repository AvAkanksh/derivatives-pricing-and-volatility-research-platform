import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any

def fetch_option_chain_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetch and clean option chain data from yfinance for a given ticker.
    Computes time-to-maturity T in years, option mid price, and filters garbage data.
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # Fetch spot price
    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"Could not fetch spot price for {ticker_symbol}")
    spot = float(history['Close'].iloc[-1])
    
    expiries = ticker.options
    if not expiries:
        raise ValueError(f"No option chains found for {ticker_symbol}")
        
    records = []
    current_date = pd.Timestamp.now(tz='UTC')
    
    for expiry_str in expiries:
        # Parse expiry date
        expiry_date = pd.to_datetime(expiry_str).tz_localize('UTC')
        T = (expiry_date - current_date).days / 365.25
        
        # Exclude already expired or near-expired options (T < 1 day)
        if T <= 0.003:
            continue
            
        try:
            opt_chain = ticker.option_chain(expiry_str)
        except Exception:
            # Skip if API call fails for this expiry
            continue
            
        for opt_type, chain_df in [('call', opt_chain.calls), ('put', opt_chain.puts)]:
            for _, row in chain_df.iterrows():
                bid = float(row.get('bid', 0.0))
                ask = float(row.get('ask', 0.0))
                strike = float(row.get('strike', 0.0))
                implied_vol = float(row.get('impliedVolatility', 0.0))
                last_price = float(row.get('lastPrice', 0.0))
                
                # Calculate mid price
                mid = (bid + ask) / 2.0 if (bid > 0.0 and ask > 0.0) else last_price
                
                # Data cleaning filters:
                # 1. Bid and ask must be positive (prevents stale/non-quoted options)
                # 2. Mid price must be positive
                # 3. Exclude options with extremely wide bid-ask spread relative to mid (>100%)
                if bid <= 0.0 or ask <= 0.0 or mid <= 0.0:
                    continue
                if (ask - bid) / mid > 1.0:
                    continue
                    
                records.append({
                    "ticker": ticker_symbol,
                    "spot": spot,
                    "strike": strike,
                    "maturity": T,
                    "expiry_str": expiry_str,
                    "option_type": opt_type,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "implied_vol_mkt": implied_vol,
                    "last_price": last_price,
                    "volume": float(row.get('volume', 0.0) or 0.0),
                    "open_interest": float(row.get('openInterest', 0.0) or 0.0)
                })
                
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    return df
