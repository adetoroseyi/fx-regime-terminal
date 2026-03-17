"""
Value Gap (Fair Value Gap / FVG) Strategy for FX Regime Terminal.

Detects FVGs and generates trade signals when price retraces into them,
filtered by HMM regime confirmation.
"""

import numpy as np
import pandas as pd

from src.indicators import rsi, atr


# Valid regimes for FVG strategy
FVG_LONG_REGIMES = {"Bull Trend", "Value Gap Formation"}
FVG_SHORT_REGIMES = {"Bear Trend", "Value Gap Formation"}
FVG_EXIT_REGIMES = {"Bear Trend", "Noise", "High Volatility Expansion"}
FVG_EXIT_SHORT_REGIMES = {"Bull Trend", "Noise", "High Volatility Expansion"}

MIN_CONFIDENCE = 65.0  # minimum regime confidence %


def detect_fvg(df: pd.DataFrame, lookback: int = 20) -> list:
    """
    Detect all Fair Value Gaps in the data.

    Bullish FVG: candle[i-1].high < candle[i+1].low
      → gap zone = [candle[i-1].high, candle[i+1].low]
      → price left an upward gap (institutions may fill it)

    Bearish FVG: candle[i-1].low > candle[i+1].high
      → gap zone = [candle[i+1].high, candle[i-1].low]
      → price left a downward gap

    Args:
        df: OHLCV DataFrame
        lookback: Only return gaps within the last N candles

    Returns:
        List of dicts with keys:
            type: 'bullish' or 'bearish'
            index: candle index where gap formed (middle candle)
            gap_top: upper edge of gap zone
            gap_bottom: lower edge of gap zone
            filled: whether the gap has been filled by subsequent price action
    """
    gaps = []
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    start = max(1, len(df) - lookback - 1)

    for i in range(start, len(df) - 1):
        # Bullish FVG: previous high < next low (gap up)
        if highs[i - 1] < lows[i + 1]:
            gap = {
                "type": "bullish",
                "index": i,
                "datetime": df.index[i],
                "gap_top": float(lows[i + 1]),
                "gap_bottom": float(highs[i - 1]),
                "filled": False,
            }
            # Check if filled by subsequent candles
            for j in range(i + 2, len(df)):
                if lows[j] <= gap["gap_bottom"]:
                    gap["filled"] = True
                    break
            gaps.append(gap)

        # Bearish FVG: previous low > next high (gap down)
        if lows[i - 1] > highs[i + 1]:
            gap = {
                "type": "bearish",
                "index": i,
                "datetime": df.index[i],
                "gap_top": float(lows[i - 1]),
                "gap_bottom": float(highs[i + 1]),
                "filled": False,
            }
            # Check if filled by subsequent candles
            for j in range(i + 2, len(df)):
                if highs[j] >= gap["gap_top"]:
                    gap["filled"] = True
                    break
            gaps.append(gap)

    return gaps


def get_unfilled_fvgs(df: pd.DataFrame, lookback: int = 20) -> list:
    """Return only unfilled FVGs within the lookback window."""
    all_gaps = detect_fvg(df, lookback)
    return [g for g in all_gaps if not g["filled"]]


def check_fvg_long(df: pd.DataFrame, regime: str, confidence: float,
                   lookback: int = 20) -> dict | None:
    """
    Check for a LONG FVG fill signal.

    Conditions:
      1. Regime is Bull Trend or Value Gap Formation
      2. Regime confidence >= 65%
      3. A bullish FVG exists (unfilled) in last 20 candles
      4. Current price is retracing into the gap (price <= gap_top)
      5. RSI(14) < 55
      6. ATR(14) > its 20-period average

    Returns:
        Signal dict or None
    """
    if regime not in FVG_LONG_REGIMES:
        return None
    if confidence < MIN_CONFIDENCE:
        return None

    unfilled = get_unfilled_fvgs(df, lookback)
    bullish_gaps = [g for g in unfilled if g["type"] == "bullish"]

    if not bullish_gaps:
        return None

    current_price = df["Close"].iloc[-1]
    rsi_val = rsi(df["Close"]).iloc[-1]
    atr_val = atr(df).iloc[-1]
    atr_avg = atr(df).rolling(window=20).mean().iloc[-1]

    if rsi_val >= 55:
        return None
    if atr_val <= atr_avg:
        return None

    # Find the most recent bullish FVG that price is retracing into
    for gap in reversed(bullish_gaps):
        if current_price <= gap["gap_top"]:
            # Calculate TP and SL
            tp = gap["gap_bottom"] + (gap["gap_top"] - gap["gap_bottom"]) + 0.5 * atr_val
            sl = current_price - 1.5 * atr_val

            return {
                "strategy": "FVG",
                "direction": "LONG",
                "pair": None,  # set by caller
                "entry_price": float(current_price),
                "entry_zone_low": float(gap["gap_bottom"]),
                "entry_zone_high": float(gap["gap_top"]),
                "stop_loss": float(sl),
                "take_profit": float(tp),
                "regime": regime,
                "confidence": float(confidence),
                "gap": gap,
                "confirmations": {
                    "regime_valid": True,
                    "fvg_present": True,
                    "price_in_gap": True,
                    "rsi_ok": rsi_val < 55,
                    "atr_above_avg": atr_val > atr_avg,
                },
                "rsi": float(rsi_val),
                "atr": float(atr_val),
            }

    return None


def check_fvg_short(df: pd.DataFrame, regime: str, confidence: float,
                    lookback: int = 20) -> dict | None:
    """
    Check for a SHORT FVG fill signal.

    Conditions:
      1. Regime is Bear Trend or Value Gap Formation
      2. Regime confidence >= 65%
      3. A bearish FVG exists (unfilled) in last 20 candles
      4. Current price is retracing into the gap (price >= gap_bottom)
      5. RSI(14) > 45
      6. ATR(14) > its 20-period average
    """
    if regime not in FVG_SHORT_REGIMES:
        return None
    if confidence < MIN_CONFIDENCE:
        return None

    unfilled = get_unfilled_fvgs(df, lookback)
    bearish_gaps = [g for g in unfilled if g["type"] == "bearish"]

    if not bearish_gaps:
        return None

    current_price = df["Close"].iloc[-1]
    rsi_val = rsi(df["Close"]).iloc[-1]
    atr_val = atr(df).iloc[-1]
    atr_avg = atr(df).rolling(window=20).mean().iloc[-1]

    if rsi_val <= 45:
        return None
    if atr_val <= atr_avg:
        return None

    for gap in reversed(bearish_gaps):
        if current_price >= gap["gap_bottom"]:
            tp = gap["gap_top"] - (gap["gap_top"] - gap["gap_bottom"]) - 0.5 * atr_val
            sl = current_price + 1.5 * atr_val

            return {
                "strategy": "FVG",
                "direction": "SHORT",
                "pair": None,
                "entry_price": float(current_price),
                "entry_zone_low": float(gap["gap_bottom"]),
                "entry_zone_high": float(gap["gap_top"]),
                "stop_loss": float(sl),
                "take_profit": float(tp),
                "regime": regime,
                "confidence": float(confidence),
                "gap": gap,
                "confirmations": {
                    "regime_valid": True,
                    "fvg_present": True,
                    "price_in_gap": True,
                    "rsi_ok": rsi_val > 45,
                    "atr_above_avg": atr_val > atr_avg,
                },
                "rsi": float(rsi_val),
                "atr": float(atr_val),
            }

    return None


def check_fvg_signal(df: pd.DataFrame, regime: str,
                     confidence: float) -> dict | None:
    """
    Check both long and short FVG signals. Returns the first match.
    """
    signal = check_fvg_long(df, regime, confidence)
    if signal:
        return signal
    return check_fvg_short(df, regime, confidence)


def check_fvg_exit(current_regime: str, direction: str) -> bool:
    """
    Check if regime-based exit should trigger.

    For LONG: exit if regime flips to Bear, Noise, or High Vol
    For SHORT: exit if regime flips to Bull, Noise, or High Vol
    """
    if direction == "LONG" and current_regime in FVG_EXIT_REGIMES:
        return True
    if direction == "SHORT" and current_regime in FVG_EXIT_SHORT_REGIMES:
        return True
    return False


if __name__ == "__main__":
    from src.data_loader import fetch_pair_data

    print("FVG Strategy Test")
    print("=" * 40)

    df = fetch_pair_data("EURUSD", period_days=30)
    gaps = detect_fvg(df, lookback=50)

    bullish = [g for g in gaps if g["type"] == "bullish"]
    bearish = [g for g in gaps if g["type"] == "bearish"]
    unfilled = [g for g in gaps if not g["filled"]]

    print(f"\nFVGs detected (last 50 candles):")
    print(f"  Bullish: {len(bullish)}")
    print(f"  Bearish: {len(bearish)}")
    print(f"  Unfilled: {len(unfilled)}")

    for gap in unfilled[-3:]:
        print(f"\n  {gap['type'].upper()} FVG at {gap['datetime']}:")
        print(f"    Gap zone: {gap['gap_bottom']:.5f} — {gap['gap_top']:.5f}")
