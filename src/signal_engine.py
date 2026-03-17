"""
Forward signal generation engine for FX Regime Terminal.

Fetches latest H1 data, decodes current regimes, checks both strategies,
and outputs signal state for each pair.

Run after daily retraining:
    python src/signal_engine.py
"""

import os
import json
from datetime import datetime, timedelta

from src.data_loader import fetch_pair_data, SUPPORTED_PAIRS
from src.hmm_model import load_model, decode_regime, select_best_n_states, save_model
from src.indicators import compute_all_indicators
from src.strategy_fvg import check_fvg_signal
from src.strategy_mr import check_mr_signal

COOLDOWN_CANDLES = 48


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cooldown() -> dict:
    """Load cooldown state from JSON."""
    path = os.path.join(get_project_root(), "state", "cooldown.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cooldown(cooldown: dict):
    """Save cooldown state to JSON."""
    state_dir = os.path.join(get_project_root(), "state")
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "cooldown.json")
    with open(path, "w") as f:
        json.dump(cooldown, f, indent=2)


def is_in_cooldown(pair: str, cooldown: dict) -> bool:
    """Check if a pair is in cooldown."""
    if pair not in cooldown:
        return False
    cooldown_until = datetime.fromisoformat(cooldown[pair])
    return datetime.utcnow() < cooldown_until


def set_cooldown(pair: str, cooldown: dict):
    """Set cooldown for a pair (48 hours for H1)."""
    cooldown[pair] = (datetime.utcnow() + timedelta(hours=COOLDOWN_CANDLES)).isoformat()


def generate_signals() -> dict:
    """
    Generate current signals for all pairs.

    Returns:
        Dict with signal state per pair, saved to current_signals.json
    """
    print("FX Regime Terminal — Signal Engine")
    print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    cooldown = load_cooldown()
    signals = {}
    signals_fired = 0

    for pair in SUPPORTED_PAIRS:
        print(f"\n--- {pair} ---")
        pair_signal = _process_pair(pair, cooldown)
        signals[pair] = pair_signal

        if pair_signal.get("signal_direction") not in ("HOLD", "AVOID"):
            signals_fired += 1

    # Save signals
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "pairs": signals,
        "signals_fired": signals_fired,
    }

    state_dir = os.path.join(get_project_root(), "state")
    os.makedirs(state_dir, exist_ok=True)
    output_path = os.path.join(state_dir, "current_signals.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save cooldown
    save_cooldown(cooldown)

    print(f"\n{'='*60}")
    print(f"Signals fired: {signals_fired}")
    print(f"Output saved to: {output_path}")

    return output


def _process_pair(pair: str, cooldown: dict) -> dict:
    """Process a single pair and return its signal state."""
    result = {
        "pair": pair,
        "regime": "Unknown",
        "confidence": 0.0,
        "regime_confirmed": False,
        "signal_direction": "AVOID",
        "active_strategy": None,
        "entry_zone": None,
        "stop_loss": None,
        "take_profit": None,
        "confirmations": {},
        "cooldown_active": False,
        "last_updated": datetime.utcnow().isoformat(),
    }

    try:
        # Fetch latest data
        df = fetch_pair_data(pair, period_days=60, use_cache=True)
        if len(df) < 50:
            result["error"] = "Insufficient data"
            return result

        # Load or train model
        try:
            model, regime_map = load_model(pair)
        except FileNotFoundError:
            print(f"  No saved model found, selecting optimal and training...")
            selection = select_best_n_states(df)
            model_result = selection["best_result"]
            model = model_result["model"]
            regime_map = model_result["regime_map"]
            print(f"  Selected n_states={selection['best_n']} (BIC)")
            save_model(model_result, pair)

        # Decode regime
        regime_df = decode_regime(model, df, regime_map)
        current_regime = regime_df["regime"].iloc[-1]
        current_confidence = regime_df["confidence"].iloc[-1]
        is_confirmed = regime_df["confirmed"].iloc[-1]

        result["regime"] = current_regime
        result["confidence"] = round(float(current_confidence), 1)
        result["regime_confirmed"] = bool(is_confirmed)

        print(f"  Regime: {current_regime} ({current_confidence:.1f}%)")
        print(f"  Confirmed: {is_confirmed}")

        # Check cooldown
        if is_in_cooldown(pair, cooldown):
            result["cooldown_active"] = True
            result["signal_direction"] = "HOLD"
            print(f"  Cooldown active until {cooldown[pair]}")
            return result

        # If regime is Noise, avoid trading
        if current_regime == "Noise":
            result["signal_direction"] = "AVOID"
            return result

        # If regime not confirmed, hold
        if not is_confirmed:
            result["signal_direction"] = "HOLD"
            return result

        # Check strategies
        df_ind = compute_all_indicators(df)

        # Try FVG first, then Mean Reversion
        signal = check_fvg_signal(df_ind, current_regime, current_confidence)
        if signal is None:
            signal = check_mr_signal(df_ind, current_regime, current_confidence)

        if signal:
            result["signal_direction"] = signal["direction"]
            result["active_strategy"] = signal["strategy"]
            result["entry_zone"] = {
                "low": signal["entry_zone_low"],
                "high": signal["entry_zone_high"],
            }
            result["entry_price"] = signal["entry_price"]
            result["stop_loss"] = signal["stop_loss"]
            result["take_profit"] = signal["take_profit"]
            result["confirmations"] = signal.get("confirmations", {})
            result["confirmations_met"] = signal.get("confirmations_met")
            result["confirmations_required"] = signal.get("confirmations_required")

            # Set cooldown
            set_cooldown(pair, cooldown)

            print(f"  SIGNAL: {signal['direction']} ({signal['strategy']})")
            print(f"  Entry: {signal['entry_price']:.5f}")
            print(f"  SL: {signal['stop_loss']:.5f} | TP: {signal['take_profit']:.5f}")
        else:
            result["signal_direction"] = "HOLD"
            print(f"  No signal conditions met")

    except Exception as e:
        result["error"] = str(e)
        print(f"  ERROR: {e}")

    return result


if __name__ == "__main__":
    generate_signals()
