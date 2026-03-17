"""
Feature engineering for HMM regime detection.
Computes the 3 core features per candle used to train the HMM.
"""

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HMM training features from OHLCV data.

    Features (per candle):
        1. log_return: log(close_t / close_{t-1})
        2. norm_range: (high - low) / close — normalised candle range
        3. volume_change: percentage change in volume (tick proxy if zero)

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume

    Returns:
        DataFrame with columns: log_return, norm_range, volume_change
        Aligned to original index (first row dropped due to diff).
    """
    features = pd.DataFrame(index=df.index)

    # 1. Log returns (close-to-close)
    features["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 2. Normalised candle range
    features["norm_range"] = (df["High"] - df["Low"]) / df["Close"]

    # 3. Volume change %
    # Use volume if available; if volume is 0/NaN, use tick proxy (norm_range as proxy)
    volume = df["Volume"].copy()
    if volume.sum() == 0 or volume.isna().all():
        # No real volume — use normalised range as a volatility proxy
        features["volume_change"] = features["norm_range"].pct_change()
    else:
        # Replace zeros with NaN to avoid division issues, then ffill
        volume = volume.replace(0, np.nan).ffill().bfill()
        features["volume_change"] = volume.pct_change()

    # Drop first row (NaN from diff/shift) and any remaining NaN rows
    features.dropna(inplace=True)

    # Clip extreme outliers (beyond 5 std) to stabilise HMM training
    for col in features.columns:
        mean = features[col].mean()
        std = features[col].std()
        if std > 0:
            features[col] = features[col].clip(mean - 5 * std, mean + 5 * std)

    return features


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Return the feature matrix as a numpy array ready for HMM training.

    Args:
        df: Raw OHLCV DataFrame

    Returns:
        2D numpy array of shape (n_samples, 3)
    """
    features = compute_features(df)
    return features.values


if __name__ == "__main__":
    from data_loader import fetch_pair_data

    print("Feature Engineering Test")
    print("=" * 40)

    df = fetch_pair_data("EURUSD", period_days=30)
    features = compute_features(df)

    print(f"\nInput candles: {len(df)}")
    print(f"Feature rows:  {len(features)}")
    print(f"\nFeature statistics:")
    print(features.describe())
    print(f"\nSample (last 5 rows):")
    print(features.tail())
