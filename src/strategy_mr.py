"""
Mean Reversion Strategy for FX Regime Terminal.

Uses Bollinger Bands, RSI, and Stochastic to identify stretched conditions
in ranging/compressed regimes. Requires 3/4 confirmations to fire.
"""

import numpy as np
import pandas as pd

from src.indicators import rsi, bollinger_bands, atr, stochastic, swing_high, swing_low


# Valid regimes for Mean Reversion
MR_LONG_REGIMES = {"Mean Reversion Range", "Low Volatility Compression"}
MR_SHORT_REGIMES = {"Mean Reversion Range", "Low Volatility Compression"}
MR_EXIT_REGIMES = {"Bull Trend", "Bear Trend"}

MIN_CONFIDENCE = 60.0
MIN_CONFIRMATIONS = 3  # out of 4


def check_mr_long(df: pd.DataFrame, regime: str,
                  confidence: float) -> dict | None:
    """
    Check for a LONG Mean Reversion signal.

    Confirmations (need 3 of 4):
      1. Price below Bollinger Band lower (2 std, 20 period)
      2. RSI(14) < 35
      3. Stochastic %K < 25 and %K crossing above %D
      4. Price within 1.0x ATR of recent swing low (last 10 candles)

    Also requires:
      - Regime: Mean Reversion Range or Low Volatility Compression
      - Regime confidence >= 60%
    """
    if regime not in MR_LONG_REGIMES:
        return None
    if confidence < MIN_CONFIDENCE:
        return None

    current_price = df["Close"].iloc[-1]

    # Compute indicators
    bb_upper, bb_mid, bb_lower = bollinger_bands(df["Close"])
    rsi_val = rsi(df["Close"]).iloc[-1]
    atr_val = atr(df).iloc[-1]
    stoch_k, stoch_d = stochastic(df)
    sw_low = swing_low(df, lookback=10).iloc[-1]

    bb_lower_val = bb_lower.iloc[-1]
    bb_mid_val = bb_mid.iloc[-1]
    stoch_k_val = stoch_k.iloc[-1]
    stoch_d_val = stoch_d.iloc[-1]
    stoch_k_prev = stoch_k.iloc[-2] if len(stoch_k) >= 2 else stoch_k_val

    # Check confirmations
    confirmations = {}

    # 1. Price below BB lower
    confirmations["bb_lower"] = current_price < bb_lower_val

    # 2. RSI < 35
    confirmations["rsi_oversold"] = rsi_val < 35

    # 3. Stochastic %K < 25 and crossing above %D
    stoch_cross_up = stoch_k_val < 25 and stoch_k_val > stoch_d_val and stoch_k_prev <= stoch_d.iloc[-2]
    confirmations["stoch_cross"] = stoch_cross_up

    # 4. Price within 1.0x ATR of swing low
    confirmations["swing_low_proximity"] = abs(current_price - sw_low) <= 1.0 * atr_val

    confirmed_count = sum(confirmations.values())

    if confirmed_count < MIN_CONFIRMATIONS:
        return None

    # Calculate TP and SL
    tp = float(bb_mid_val)
    sl = float(current_price - 1.2 * atr_val)

    return {
        "strategy": "MeanReversion",
        "direction": "LONG",
        "pair": None,
        "entry_price": float(current_price),
        "entry_zone_low": float(current_price - 0.3 * atr_val),
        "entry_zone_high": float(current_price + 0.3 * atr_val),
        "stop_loss": sl,
        "take_profit": tp,
        "regime": regime,
        "confidence": float(confidence),
        "confirmations": {
            "Price below BB lower": confirmations["bb_lower"],
            f"RSI {rsi_val:.1f} < 35": confirmations["rsi_oversold"],
            "Stoch K crossing above D": confirmations["stoch_cross"],
            "Swing low proximity": confirmations["swing_low_proximity"],
        },
        "confirmations_met": confirmed_count,
        "confirmations_required": MIN_CONFIRMATIONS,
        "rsi": float(rsi_val),
        "atr": float(atr_val),
        "bb_mid": float(bb_mid_val),
        "bb_lower": float(bb_lower_val),
        "stoch_k": float(stoch_k_val),
        "stoch_d": float(stoch_d_val),
    }


def check_mr_short(df: pd.DataFrame, regime: str,
                   confidence: float) -> dict | None:
    """
    Check for a SHORT Mean Reversion signal.

    Confirmations (need 3 of 4):
      1. Price above Bollinger Band upper (2 std, 20 period)
      2. RSI(14) > 65
      3. Stochastic %K > 75 and %K crossing below %D
      4. Price within 1.0x ATR of recent swing high (last 10 candles)
    """
    if regime not in MR_SHORT_REGIMES:
        return None
    if confidence < MIN_CONFIDENCE:
        return None

    current_price = df["Close"].iloc[-1]

    bb_upper, bb_mid, bb_lower = bollinger_bands(df["Close"])
    rsi_val = rsi(df["Close"]).iloc[-1]
    atr_val = atr(df).iloc[-1]
    stoch_k, stoch_d = stochastic(df)
    sw_high = swing_high(df, lookback=10).iloc[-1]

    bb_upper_val = bb_upper.iloc[-1]
    bb_mid_val = bb_mid.iloc[-1]
    stoch_k_val = stoch_k.iloc[-1]
    stoch_d_val = stoch_d.iloc[-1]
    stoch_k_prev = stoch_k.iloc[-2] if len(stoch_k) >= 2 else stoch_k_val

    confirmations = {}

    # 1. Price above BB upper
    confirmations["bb_upper"] = current_price > bb_upper_val

    # 2. RSI > 65
    confirmations["rsi_overbought"] = rsi_val > 65

    # 3. Stochastic %K > 75 and crossing below %D
    stoch_cross_down = stoch_k_val > 75 and stoch_k_val < stoch_d_val and stoch_k_prev >= stoch_d.iloc[-2]
    confirmations["stoch_cross"] = stoch_cross_down

    # 4. Price within 1.0x ATR of swing high
    confirmations["swing_high_proximity"] = abs(current_price - sw_high) <= 1.0 * atr_val

    confirmed_count = sum(confirmations.values())

    if confirmed_count < MIN_CONFIRMATIONS:
        return None

    tp = float(bb_mid_val)
    sl = float(current_price + 1.2 * atr_val)

    return {
        "strategy": "MeanReversion",
        "direction": "SHORT",
        "pair": None,
        "entry_price": float(current_price),
        "entry_zone_low": float(current_price - 0.3 * atr_val),
        "entry_zone_high": float(current_price + 0.3 * atr_val),
        "stop_loss": sl,
        "take_profit": tp,
        "regime": regime,
        "confidence": float(confidence),
        "confirmations": {
            "Price above BB upper": confirmations["bb_upper"],
            f"RSI {rsi_val:.1f} > 65": confirmations["rsi_overbought"],
            "Stoch K crossing below D": confirmations["stoch_cross"],
            "Swing high proximity": confirmations["swing_high_proximity"],
        },
        "confirmations_met": confirmed_count,
        "confirmations_required": MIN_CONFIRMATIONS,
        "rsi": float(rsi_val),
        "atr": float(atr_val),
        "bb_mid": float(bb_mid_val),
        "bb_upper": float(bb_upper_val),
        "stoch_k": float(stoch_k_val),
        "stoch_d": float(stoch_d_val),
    }


def check_mr_signal(df: pd.DataFrame, regime: str,
                    confidence: float) -> dict | None:
    """Check both long and short MR signals. Returns the first match."""
    signal = check_mr_long(df, regime, confidence)
    if signal:
        return signal
    return check_mr_short(df, regime, confidence)


def check_mr_exit(current_regime: str) -> bool:
    """
    Check if regime-based exit should trigger for Mean Reversion.
    Exit if regime transitions to any trending regime (Bull or Bear).
    """
    return current_regime in MR_EXIT_REGIMES


if __name__ == "__main__":
    from src.data_loader import fetch_pair_data

    print("Mean Reversion Strategy Test")
    print("=" * 40)

    df = fetch_pair_data("EURUSD", period_days=30)

    # Test with forced regime for demonstration
    for regime in ["Mean Reversion Range", "Low Volatility Compression", "Bull Trend"]:
        signal = check_mr_signal(df, regime, confidence=75.0)
        status = "SIGNAL" if signal else "No signal"
        print(f"\n  Regime: {regime} → {status}")
        if signal:
            print(f"    Direction: {signal['direction']}")
            print(f"    Entry: {signal['entry_price']:.5f}")
            print(f"    TP: {signal['take_profit']:.5f}")
            print(f"    SL: {signal['stop_loss']:.5f}")
            print(f"    Confirmations: {signal['confirmations_met']}/{signal['confirmations_required']}")
            for k, v in signal["confirmations"].items():
                print(f"      {'✅' if v else '❌'} {k}")
