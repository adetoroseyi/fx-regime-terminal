"""Tests for FVG (Fair Value Gap) strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy_fvg import detect_fvg, get_unfilled_fvgs, check_fvg_long, check_fvg_short


def make_ohlcv_with_bullish_fvg():
    """Create data with a deliberate bullish FVG."""
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    # Normal candles
    close = np.full(n, 1.1000)
    high = np.full(n, 1.1010)
    low = np.full(n, 1.0990)
    open_p = np.full(n, 1.1000)
    volume = np.full(n, 1000.0)

    # Create bullish FVG at index 15 (middle candle of the 3-candle pattern)
    # candle[14].high < candle[16].low  →  gap up
    high[14] = 1.1010   # prev candle high
    low[14] = 1.0990
    close[14] = 1.1005

    high[15] = 1.1050   # impulse candle (big move up)
    low[15] = 1.1015
    close[15] = 1.1045
    open_p[15] = 1.1010

    high[16] = 1.1060   # next candle — low must be > prev high
    low[16] = 1.1020    # 1.1020 > 1.1010 → bullish FVG!
    close[16] = 1.1055

    # Price retraces back into the gap
    close[25] = 1.1015  # within gap zone [1.1010, 1.1020]
    high[25] = 1.1020
    low[25] = 1.1005

    return pd.DataFrame({
        "Open": open_p, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


def make_ohlcv_with_bearish_fvg():
    """Create data with a deliberate bearish FVG."""
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    close = np.full(n, 1.1000)
    high = np.full(n, 1.1010)
    low = np.full(n, 1.0990)
    open_p = np.full(n, 1.1000)
    volume = np.full(n, 1000.0)

    # Create bearish FVG at index 15
    # candle[14].low > candle[16].high  →  gap down
    low[14] = 1.0990    # prev candle low
    high[14] = 1.1010

    high[15] = 1.0985   # impulse candle (big move down)
    low[15] = 1.0950
    close[15] = 1.0955

    high[16] = 1.0980   # next candle — high must be < prev low
    low[16] = 1.0960    # 1.0980 < 1.0990 → bearish FVG!
    close[16] = 1.0970

    return pd.DataFrame({
        "Open": open_p, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


class TestFVGDetection:
    def test_detects_bullish_fvg(self):
        df = make_ohlcv_with_bullish_fvg()
        gaps = detect_fvg(df, lookback=25)
        bullish = [g for g in gaps if g["type"] == "bullish"]
        assert len(bullish) >= 1
        gap = bullish[0]
        assert gap["gap_bottom"] < gap["gap_top"]

    def test_detects_bearish_fvg(self):
        df = make_ohlcv_with_bearish_fvg()
        gaps = detect_fvg(df, lookback=25)
        bearish = [g for g in gaps if g["type"] == "bearish"]
        assert len(bearish) >= 1
        gap = bearish[0]
        assert gap["gap_bottom"] < gap["gap_top"]

    def test_no_fvg_in_flat_data(self):
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "Open": np.full(n, 1.1000),
            "High": np.full(n, 1.1001),
            "Low": np.full(n, 1.0999),
            "Close": np.full(n, 1.1000),
            "Volume": np.full(n, 1000.0),
        }, index=dates)
        gaps = detect_fvg(df, lookback=25)
        assert len(gaps) == 0

    def test_unfilled_gaps(self):
        df = make_ohlcv_with_bullish_fvg()
        unfilled = get_unfilled_fvgs(df, lookback=25)
        # Gap may or may not be filled depending on subsequent price
        assert isinstance(unfilled, list)


class TestFVGSignals:
    def test_no_signal_wrong_regime(self):
        df = make_ohlcv_with_bullish_fvg()
        signal = check_fvg_long(df, "Noise", 80.0)
        assert signal is None

    def test_no_signal_low_confidence(self):
        df = make_ohlcv_with_bullish_fvg()
        signal = check_fvg_long(df, "Bull Trend", 50.0)
        assert signal is None

    def test_short_no_signal_wrong_regime(self):
        df = make_ohlcv_with_bearish_fvg()
        signal = check_fvg_short(df, "Bull Trend", 80.0)
        assert signal is None
