"""
Data loader for FX Regime Terminal.
Fetches and caches hourly forex data from Yahoo Finance.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Yahoo Finance ticker mapping for forex pairs
PAIR_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
}

SUPPORTED_PAIRS = list(PAIR_TICKERS.keys())

# Cache directory relative to project root
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")


def get_project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_pair_data(pair: str, period_days: int = 730, interval: str = "1h",
                    use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch hourly OHLCV data for a forex pair from Yahoo Finance.

    Args:
        pair: Forex pair code (e.g. 'EURUSD')
        period_days: Number of days of history to fetch (max ~730 for hourly)
        interval: Candle interval ('1h' for hourly)
        use_cache: Whether to use cached data if available and recent

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (UTC)
    """
    if pair not in PAIR_TICKERS:
        raise ValueError(f"Unsupported pair: {pair}. Supported: {SUPPORTED_PAIRS}")

    ticker = PAIR_TICKERS[pair]

    # Check cache first
    if use_cache:
        cached = _load_cache(pair)
        if cached is not None:
            return cached

    # yfinance limits hourly data to ~730 days, fetched in chunks
    # For hourly data, yfinance allows max 730 days but fetches
    # in 60-day chunks internally
    end_date = datetime.utcnow()
    # Yahoo Finance enforces a strict 730-day limit for hourly data.
    # Cap at 729 days to avoid boundary errors from clock skew.
    capped_days = min(period_days, 729)
    start_date = end_date - timedelta(days=capped_days)

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {pair} ({ticker})")

        # Standardise columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "Datetime"

        # Drop rows with NaN prices
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        # Cache the result
        if use_cache:
            _save_cache(pair, df)

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {pair}: {e}")


def fetch_all_pairs(period_days: int = 730, interval: str = "1h",
                    use_cache: bool = True) -> dict:
    """
    Fetch data for all supported forex pairs.

    Returns:
        Dict mapping pair code to DataFrame
    """
    data = {}
    for pair in SUPPORTED_PAIRS:
        try:
            data[pair] = fetch_pair_data(pair, period_days, interval, use_cache)
            print(f"  Loaded {pair}: {len(data[pair])} candles")
        except Exception as e:
            print(f"  WARNING: Failed to load {pair}: {e}")
    return data


def _cache_path(pair: str) -> str:
    """Return the cache file path for a pair."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{pair}_hourly.parquet")


def _save_cache(pair: str, df: pd.DataFrame):
    """Save data to parquet cache."""
    try:
        path = _cache_path(pair)
        df.to_parquet(path)
    except Exception:
        pass  # Cache failures are non-critical


def _load_cache(pair: str, max_age_hours: int = 12) -> pd.DataFrame | None:
    """Load cached data if it exists and is recent enough."""
    path = _cache_path(pair)
    if not os.path.exists(path):
        return None

    # Check file age
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - mtime > timedelta(hours=max_age_hours):
        return None

    try:
        df = pd.read_parquet(path)
        return df
    except Exception:
        return None


if __name__ == "__main__":
    print("FX Regime Terminal — Data Loader")
    print("=" * 40)
    data = fetch_all_pairs(period_days=730)
    for pair, df in data.items():
        print(f"\n{pair}:")
        print(f"  Date range: {df.index[0]} → {df.index[-1]}")
        print(f"  Candles: {len(df)}")
        print(f"  Latest close: {df['Close'].iloc[-1]:.5f}")
