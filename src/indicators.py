"""
Technical indicator helpers for FX Regime Terminal.
Provides RSI, Bollinger Bands, ATR, Stochastic, and MACD calculations.
All functions operate on pandas DataFrames/Series.
"""

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        series: Price series (typically Close)
        period: Lookback period

    Returns:
        RSI values (0-100)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)


def bollinger_bands(series: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> tuple:
    """
    Bollinger Bands.

    Returns:
        (upper_band, middle_band, lower_band)
    """
    middle = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    Args:
        df: DataFrame with High, Low, Close columns
        period: Lookback period

    Returns:
        ATR series
    """
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """
    Stochastic Oscillator (%K and %D).

    Args:
        df: DataFrame with High, Low, Close columns
        k_period: %K lookback period
        d_period: %D smoothing period

    Returns:
        (percent_k, percent_d)
    """
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()

    denom = high_max - low_min
    denom = denom.replace(0, np.nan)

    percent_k = 100 * (df["Close"] - low_min) / denom
    percent_d = percent_k.rolling(window=d_period).mean()

    return percent_k.fillna(50), percent_d.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple:
    """
    MACD (Moving Average Convergence Divergence).

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def swing_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Identify swing highs — highest high in the lookback window.

    Returns:
        Series of swing high values
    """
    return df["High"].rolling(window=lookback).max()


def swing_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Identify swing lows — lowest low in the lookback window.

    Returns:
        Series of swing low values
    """
    return df["Low"].rolling(window=lookback).min()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators and attach them to a copy of the DataFrame.

    Returns:
        DataFrame with original OHLCV + all indicator columns
    """
    result = df.copy()

    # RSI
    result["RSI"] = rsi(df["Close"])

    # Bollinger Bands
    result["BB_upper"], result["BB_mid"], result["BB_lower"] = bollinger_bands(
        df["Close"]
    )

    # ATR
    result["ATR"] = atr(df)
    result["ATR_avg"] = result["ATR"].rolling(window=20).mean()

    # Stochastic
    result["Stoch_K"], result["Stoch_D"] = stochastic(df)

    # MACD
    result["MACD"], result["MACD_signal"], result["MACD_hist"] = macd(df["Close"])

    # Swing levels
    result["SwingHigh"] = swing_high(df)
    result["SwingLow"] = swing_low(df)

    return result


if __name__ == "__main__":
    from src.data_loader import fetch_pair_data

    print("Indicator Calculation Test")
    print("=" * 40)

    df = fetch_pair_data("EURUSD", period_days=30)
    result = compute_all_indicators(df)

    print(f"\nIndicator columns added:")
    indicator_cols = [c for c in result.columns if c not in df.columns]
    for col in indicator_cols:
        valid = result[col].dropna()
        print(f"  {col}: {len(valid)} values, "
              f"range [{valid.min():.4f}, {valid.max():.4f}]")

    print(f"\nLast row:")
    print(result[indicator_cols].iloc[-1])
