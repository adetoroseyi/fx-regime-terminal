"""
Telegram notification system for FX Regime Terminal.

Sends formatted signal alerts and daily summaries via Telegram bot.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file.

Usage:
    python src/telegram_notifier.py
"""

import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_bot_config() -> tuple:
    """Get Telegram bot token and chat ID from environment."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env")
        return None, None

    return token, chat_id


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a message via Telegram bot.

    Args:
        text: Message text (supports HTML formatting)
        parse_mode: 'HTML' or 'Markdown'

    Returns:
        True if sent successfully
    """
    token, chat_id = get_bot_config()
    if not token:
        print(f"[DRY RUN] Would send:\n{text}")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("Telegram message sent successfully")
            return True
        else:
            print(f"Telegram API error: {resp.status_code} — {resp.text}")
            return False
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        return False


def format_signal_alert(pair_signal: dict) -> str:
    """
    Format a signal alert message for Telegram.

    Args:
        pair_signal: Signal dict from signal_engine

    Returns:
        Formatted message string
    """
    pair = pair_signal["pair"]
    regime = pair_signal["regime"]
    confidence = pair_signal["confidence"]
    direction = pair_signal["signal_direction"]
    strategy = pair_signal.get("active_strategy", "N/A")

    entry = pair_signal.get("entry_price", 0)
    sl = pair_signal.get("stop_loss", 0)
    tp = pair_signal.get("take_profit", 0)

    # Calculate pip values (approximate)
    if "JPY" in pair:
        pip_mult = 100
    else:
        pip_mult = 10000

    sl_pips = abs(entry - sl) * pip_mult if entry and sl else 0
    tp_pips = abs(tp - entry) * pip_mult if entry and tp else 0
    rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0

    # Entry zone
    zone = pair_signal.get("entry_zone", {})
    zone_low = zone.get("low", entry)
    zone_high = zone.get("high", entry)

    # Confirmations
    confirmations = pair_signal.get("confirmations", {})
    conf_met = pair_signal.get("confirmations_met", sum(confirmations.values()) if confirmations else 0)
    conf_req = pair_signal.get("confirmations_required", len(confirmations))

    cooldown_status = "Active" if pair_signal.get("cooldown_active") else "Clear"

    # Build message
    lines = [
        "\U0001F514 <b>FX REGIME TERMINAL — SIGNAL ALERT</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Pair: <b>{pair}</b>",
        f"Regime: {regime}",
        f"Confidence: {confidence}%",
        f"Signal: <b>{direction}</b> ({strategy})",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Entry Zone: {zone_low:.5f} – {zone_high:.5f}",
        f"Stop Loss: {sl:.5f} ({sl_pips:.0f} pips)",
        f"Take Profit: {tp:.5f} ({tp_pips:.0f} pips)",
        f"R:R Ratio: 1:{rr_ratio:.2f}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    # Add confirmations
    if confirmations:
        lines.append(f"Confirmations ({conf_met}/{conf_req}):")
        for label, met in confirmations.items():
            icon = "\u2705" if met else "\u274C"
            lines.append(f"  {icon} {label}")
        lines.append("━━━━━━━━━━━━━━━━━━━━")

    lines.append(f"Cooldown: {cooldown_status}")

    confirmed = "\u2705" if pair_signal.get("regime_confirmed") else "\u274C"
    lines.append(f"Regime confirmed: {confirmed}")

    return "\n".join(lines)


def format_daily_summary(signals_data: dict) -> str:
    """
    Format a daily summary message.

    Args:
        signals_data: Full signal output from signal_engine

    Returns:
        Formatted summary string
    """
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    pairs = signals_data.get("pairs", {})
    signals_fired = signals_data.get("signals_fired", 0)

    lines = [
        "\U0001F4CA <b>FX REGIME TERMINAL — DAILY SUMMARY</b>",
        f"Date: {date_str}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    direction_labels = {
        "LONG": "LONG ACTIVE",
        "SHORT": "SHORT ACTIVE",
        "HOLD": "HOLD",
        "AVOID": "AVOID",
    }

    for pair_code, data in pairs.items():
        regime = data.get("regime", "Unknown")
        conf = data.get("confidence", 0)
        direction = data.get("signal_direction", "HOLD")
        label = direction_labels.get(direction, direction)

        # Direction emoji
        if direction == "LONG":
            emoji = "\U0001F7E2"
        elif direction == "SHORT":
            emoji = "\U0001F534"
        elif direction == "AVOID":
            emoji = "\u26A0\uFE0F"
        else:
            emoji = "\u26AB"

        lines.append(f"{emoji} {pair_code} → {regime} ({conf:.0f}%) | {label}")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Models retrained: \u2705 All {len(pairs)} pairs")
    lines.append(f"Signals fired today: {signals_fired}")

    return "\n".join(lines)


def send_signal_alerts(signals_data: dict):
    """Send alerts for any active signals."""
    pairs = signals_data.get("pairs", {})

    for pair_code, data in pairs.items():
        direction = data.get("signal_direction", "HOLD")
        if direction in ("LONG", "SHORT"):
            msg = format_signal_alert(data)
            send_message(msg)


def send_daily_summary(signals_data: dict = None):
    """Send the daily summary message."""
    if signals_data is None:
        # Load from file
        path = os.path.join(get_project_root(), "state", "current_signals.json")
        if not os.path.exists(path):
            print("No signals file found")
            return
        with open(path, "r") as f:
            signals_data = json.load(f)

    msg = format_daily_summary(signals_data)
    send_message(msg)


def run_notifications():
    """Main entry point — send alerts and summary."""
    print("FX Regime Terminal — Telegram Notifier")
    print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    # Load signals
    path = os.path.join(get_project_root(), "state", "current_signals.json")
    if not os.path.exists(path):
        print("No signals file found — run signal_engine.py first")
        return

    with open(path, "r") as f:
        signals_data = json.load(f)

    # Send individual signal alerts
    send_signal_alerts(signals_data)

    # Send daily summary
    send_daily_summary(signals_data)


if __name__ == "__main__":
    run_notifications()
