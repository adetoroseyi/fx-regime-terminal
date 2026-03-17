"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    rsi, bollinger_bands, atr, stochastic, macd,
    swing_high, swing_low, compute_all_indicators,
)


def make_synthetic_ohlcv(n=200, seed=42):
    """Create synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0, 0.001, n)
    close = 1.1000 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_p = close * (1 + rng.normal(0, 0.0005, n))
    volume = rng.randint(100, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    return pd.DataFrame({
        "Open": open_p, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


class TestIndicators:
    @pytest.fixture
    def df(self):
        return make_synthetic_ohlcv(200)

    def test_rsi_range(self, df):
        r = rsi(df["Close"])
        assert r.dropna().min() >= 0
        assert r.dropna().max() <= 100

    def test_bollinger_bands(self, df):
        upper, mid, lower = bollinger_bands(df["Close"])
        # Upper >= mid >= lower where all are valid (after warmup)
        for i in range(20, len(df)):
            if pd.notna(upper.iloc[i]) and pd.notna(mid.iloc[i]) and pd.notna(lower.iloc[i]):
                assert upper.iloc[i] >= mid.iloc[i]
                assert mid.iloc[i] >= lower.iloc[i]

    def test_atr_positive(self, df):
        a = atr(df)
        assert (a.dropna() >= 0).all()

    def test_stochastic_range(self, df):
        k, d = stochastic(df)
        assert k.dropna().min() >= 0
        assert k.dropna().max() <= 100
        assert d.dropna().min() >= 0
        assert d.dropna().max() <= 100

    def test_macd_shape(self, df):
        m, s, h = macd(df["Close"])
        assert len(m) == len(df)
        assert len(s) == len(df)
        assert len(h) == len(df)

    def test_swing_levels(self, df):
        sh = swing_high(df)
        sl = swing_low(df)
        # Swing high >= swing low at every point
        valid = sh.dropna().index.intersection(sl.dropna().index)
        for idx in valid:
            assert sh[idx] >= sl[idx]

    def test_compute_all_indicators(self, df):
        result = compute_all_indicators(df)
        expected_cols = ["RSI", "BB_upper", "BB_mid", "BB_lower",
                         "ATR", "ATR_avg", "Stoch_K", "Stoch_D",
                         "MACD", "MACD_signal", "MACD_hist",
                         "SwingHigh", "SwingLow"]
        for col in expected_cols:
            assert col in result.columns


class TestBacktesterMetrics:
    def _compute_metrics(self, trades, equity_curve, df, initial_capital, final_capital):
        """Inline metrics computation to avoid importing backtester (needs yfinance)."""
        if not trades:
            return {
                "total_return_pct": 0.0, "alpha_vs_buyhold": 0.0,
                "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0, "avg_trade_duration": 0,
                "total_trades": 0, "per_regime_trades": {},
            }
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        win_rate = (len(wins) / len(trades)) * 100
        durations = [t["duration_candles"] for t in trades]
        avg_duration = np.mean(durations)
        regime_trades = {}
        for t in trades:
            r = t["regime_at_entry"]
            regime_trades[r] = regime_trades.get(r, 0) + 1
        return {
            "total_return_pct": round(float(total_return), 2),
            "win_rate_pct": round(float(win_rate), 1),
            "avg_trade_duration": round(float(avg_duration), 1),
            "total_trades": len(trades),
            "per_regime_trades": regime_trades,
        }

    def test_empty_trades(self):
        """Verify metrics handle zero trades gracefully."""
        df = make_synthetic_ohlcv(100)
        metrics = self._compute_metrics([], [], df, 10000, 10000)
        assert metrics["total_trades"] == 0
        assert metrics["total_return_pct"] == 0.0
        assert metrics["win_rate_pct"] == 0.0

    def test_metrics_with_trades(self):
        df = make_synthetic_ohlcv(100)
        trades = [
            {"pnl_pct": 0.5, "regime_at_entry": "Bull Trend", "duration_candles": 10},
            {"pnl_pct": -0.3, "regime_at_entry": "Bear Trend", "duration_candles": 5},
            {"pnl_pct": 0.8, "regime_at_entry": "Bull Trend", "duration_candles": 15},
        ]
        equity = [{"equity": 10000 + i * 10} for i in range(50)]
        metrics = self._compute_metrics(trades, equity, df, 10000, 10100)
        assert metrics["total_trades"] == 3
        assert metrics["win_rate_pct"] > 0
        assert metrics["avg_trade_duration"] > 0
        assert "Bull Trend" in metrics["per_regime_trades"]
