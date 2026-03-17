"""
Vectorised backtesting engine for FX Regime Terminal.

Walks through historical H1 candles chronologically:
  1. Trains HMM on data up to current candle (no lookahead)
  2. Checks regime confirmation (3-candle lag)
  3. Checks strategy conditions (FVG / Mean Reversion)
  4. Opens/closes simulated positions
  5. Applies cooldown logic

Usage:
    python src/backtester.py --pair EURUSD --strategy both
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from src.data_loader import fetch_pair_data, SUPPORTED_PAIRS
from src.features import compute_features
from src.hmm_model import train_hmm, decode_regime, N_REGIMES
from src.indicators import compute_all_indicators
from src.strategy_fvg import check_fvg_signal, check_fvg_exit
from src.strategy_mr import check_mr_signal, check_mr_exit

warnings.filterwarnings("ignore")

COOLDOWN_CANDLES = 48
MIN_TRAINING_CANDLES = 200  # minimum candles before we start backtesting
RETRAIN_INTERVAL = 500  # retrain HMM every N candles for efficiency


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_backtest(pair: str, strategy: str = "both",
                 period_days: int = 365,
                 initial_capital: float = 10000.0) -> dict:
    """
    Run a full backtest for a pair and strategy combination.

    Args:
        pair: Forex pair code (e.g. 'EURUSD')
        strategy: 'fvg', 'mr', or 'both'
        period_days: Days of historical data to use
        initial_capital: Starting capital for equity tracking

    Returns:
        dict with backtest results and metrics
    """
    print(f"\n{'='*60}")
    print(f"BACKTEST: {pair} | Strategy: {strategy.upper()} | {period_days}d")
    print(f"{'='*60}")

    # Fetch data
    df = fetch_pair_data(pair, period_days=period_days, use_cache=True)
    if len(df) < MIN_TRAINING_CANDLES + 100:
        return {"error": f"Insufficient data: {len(df)} candles"}

    print(f"Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})")

    # Pre-compute indicators for the full dataset
    df_ind = compute_all_indicators(df)

    # State tracking
    trades = []
    equity_curve = []
    position = None  # Current open position
    cooldown_until = 0  # Candle index when cooldown expires
    capital = initial_capital
    regime_counts = {}

    # Train HMM on initial window
    train_end = MIN_TRAINING_CANDLES
    model_result = train_hmm(df.iloc[:train_end])
    model = model_result["model"]
    regime_map = model_result["regime_map"]
    last_train_idx = train_end

    print(f"Initial HMM trained on {train_end} candles")
    print(f"Regime map: {regime_map}")

    # Walk forward through remaining candles
    for i in range(MIN_TRAINING_CANDLES, len(df)):
        current_bar = df.iloc[i]
        current_price = current_bar["Close"]
        current_time = df.index[i]

        # Retrain periodically (avoids retraining every candle for speed)
        if i - last_train_idx >= RETRAIN_INTERVAL:
            try:
                model_result = train_hmm(df.iloc[:i])
                model = model_result["model"]
                regime_map = model_result["regime_map"]
                last_train_idx = i
            except Exception:
                pass  # Keep using old model if retrain fails

        # Decode current regime using data up to this candle
        try:
            regime_df = decode_regime(model, df.iloc[max(0, i - 100):i + 1], regime_map)
            if len(regime_df) == 0:
                continue
            current_regime = regime_df["regime"].iloc[-1]
            current_confidence = regime_df["confidence"].iloc[-1]
            is_confirmed = regime_df["confirmed"].iloc[-1]
        except Exception:
            continue

        # Track regime distribution
        regime_counts[current_regime] = regime_counts.get(current_regime, 0) + 1

        # Check exit conditions for open position
        if position is not None:
            exit_reason = None

            # Stop loss
            if position["direction"] == "LONG" and current_price <= position["stop_loss"]:
                exit_reason = "stop_loss"
            elif position["direction"] == "SHORT" and current_price >= position["stop_loss"]:
                exit_reason = "stop_loss"

            # Take profit
            if position["direction"] == "LONG" and current_price >= position["take_profit"]:
                exit_reason = "take_profit"
            elif position["direction"] == "SHORT" and current_price <= position["take_profit"]:
                exit_reason = "take_profit"

            # Regime-based exit
            if position["strategy"] == "FVG":
                if check_fvg_exit(current_regime, position["direction"]):
                    exit_reason = "regime_exit"
            elif position["strategy"] == "MeanReversion":
                if check_mr_exit(current_regime):
                    exit_reason = "regime_exit"

            if exit_reason:
                # Close position
                if position["direction"] == "LONG":
                    pnl = current_price - position["entry_price"]
                else:
                    pnl = position["entry_price"] - current_price

                # Normalise PnL as percentage of entry
                pnl_pct = (pnl / position["entry_price"]) * 100
                capital *= (1 + pnl_pct / 100)

                trade = {
                    "pair": pair,
                    "strategy": position["strategy"],
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "entry_time": str(position["entry_time"]),
                    "exit_price": float(current_price),
                    "exit_time": str(current_time),
                    "exit_reason": exit_reason,
                    "pnl_pct": float(pnl_pct),
                    "regime_at_entry": position["regime"],
                    "duration_candles": i - position["entry_idx"],
                }
                trades.append(trade)
                position = None
                cooldown_until = i + COOLDOWN_CANDLES

        # Track equity
        equity_curve.append({
            "datetime": str(current_time),
            "equity": float(capital),
            "regime": current_regime,
        })

        # Skip if in cooldown or position open or regime not confirmed
        if position is not None:
            continue
        if i < cooldown_until:
            continue
        if not is_confirmed:
            continue

        # Check for new signals
        # Use a window of data for indicator calculation
        window_start = max(0, i - 100)
        df_window = df_ind.iloc[window_start:i + 1]

        signal = None

        if strategy in ("fvg", "both"):
            signal = check_fvg_signal(df_window, current_regime, current_confidence)

        if signal is None and strategy in ("mr", "both"):
            signal = check_mr_signal(df_window, current_regime, current_confidence)

        if signal is not None:
            position = {
                "strategy": signal["strategy"],
                "direction": signal["direction"],
                "entry_price": float(current_price),
                "entry_time": current_time,
                "entry_idx": i,
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "regime": current_regime,
            }

    # Close any remaining position at last price
    if position is not None:
        last_price = df["Close"].iloc[-1]
        if position["direction"] == "LONG":
            pnl = last_price - position["entry_price"]
        else:
            pnl = position["entry_price"] - last_price
        pnl_pct = (pnl / position["entry_price"]) * 100
        capital *= (1 + pnl_pct / 100)
        trades.append({
            "pair": pair,
            "strategy": position["strategy"],
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "entry_time": str(position["entry_time"]),
            "exit_price": float(last_price),
            "exit_time": str(df.index[-1]),
            "exit_reason": "end_of_data",
            "pnl_pct": float(pnl_pct),
            "regime_at_entry": position["regime"],
            "duration_candles": len(df) - 1 - position["entry_idx"],
        })

    # Calculate metrics
    metrics = _compute_metrics(trades, equity_curve, df, initial_capital, capital)
    metrics["pair"] = pair
    metrics["strategy"] = strategy
    metrics["period_days"] = period_days
    metrics["regime_distribution"] = regime_counts

    # Print summary
    _print_summary(metrics, trades)

    # Save results
    results = {
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve[-500:],  # Last 500 points for chart
    }
    _save_results(results, pair, strategy)

    return results


def _compute_metrics(trades: list, equity_curve: list,
                     df: pd.DataFrame, initial_capital: float,
                     final_capital: float) -> dict:
    """Compute backtest performance metrics."""
    if not trades:
        return {
            "total_return_pct": 0.0,
            "alpha_vs_buyhold": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_trade_duration": 0,
            "total_trades": 0,
            "per_regime_trades": {},
        }

    # Total return
    total_return = ((final_capital - initial_capital) / initial_capital) * 100

    # Buy and hold return
    bh_return = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
    alpha = total_return - bh_return

    # Trade PnLs
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    win_rate = (len(wins) / len(trades)) * 100 if trades else 0

    # Sharpe ratio (annualised, assuming hourly data)
    if len(pnls) > 1:
        pnl_std = np.std(pnls)
        pnl_mean = np.mean(pnls)
        sharpe = (pnl_mean / pnl_std) * np.sqrt(252 * 24) if pnl_std > 0 else 0
    else:
        sharpe = 0

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve]
    if equities:
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak) * 100
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = 0

    # Average trade duration
    durations = [t["duration_candles"] for t in trades]
    avg_duration = np.mean(durations) if durations else 0

    # Per-regime trade count
    regime_trades = {}
    for t in trades:
        r = t["regime_at_entry"]
        regime_trades[r] = regime_trades.get(r, 0) + 1

    return {
        "total_return_pct": round(float(total_return), 2),
        "alpha_vs_buyhold": round(float(alpha), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "win_rate_pct": round(float(win_rate), 1),
        "avg_trade_duration": round(float(avg_duration), 1),
        "total_trades": len(trades),
        "per_regime_trades": regime_trades,
    }


def _print_summary(metrics: dict, trades: list):
    """Print a formatted backtest summary."""
    print(f"\n{'─'*40}")
    print(f"RESULTS SUMMARY")
    print(f"{'─'*40}")
    print(f"  Total Return:     {metrics['total_return_pct']:+.2f}%")
    print(f"  Alpha vs B&H:     {metrics['alpha_vs_buyhold']:+.2f}%")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:         {metrics['win_rate_pct']:.1f}%")
    print(f"  Avg Duration:     {metrics['avg_trade_duration']:.1f} candles")
    print(f"  Total Trades:     {metrics['total_trades']}")
    if metrics.get("per_regime_trades"):
        print(f"\n  Per-regime trades:")
        for regime, count in sorted(metrics["per_regime_trades"].items()):
            print(f"    {regime}: {count}")


def _save_results(results: dict, pair: str, strategy: str):
    """Save backtest results to JSON."""
    results_dir = os.path.join(get_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)

    date_str = datetime.utcnow().strftime("%Y%m%d")
    filename = f"backtest_{pair}_{strategy}_{date_str}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="FX Regime Terminal — Backtester")
    parser.add_argument("--pair", type=str, default="EURUSD",
                        choices=SUPPORTED_PAIRS,
                        help="Forex pair to backtest")
    parser.add_argument("--strategy", type=str, default="both",
                        choices=["fvg", "mr", "both"],
                        help="Strategy to test")
    parser.add_argument("--days", type=int, default=365,
                        help="Days of historical data")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run backtest for all pairs")

    args = parser.parse_args()

    if args.all_pairs:
        all_results = {}
        for pair in SUPPORTED_PAIRS:
            result = run_backtest(pair, args.strategy, args.days, args.capital)
            all_results[pair] = result.get("metrics", {})

        print(f"\n\n{'='*60}")
        print("ALL PAIRS SUMMARY")
        print(f"{'='*60}")
        for pair, m in all_results.items():
            print(f"  {pair}: {m.get('total_return_pct', 0):+.2f}% | "
                  f"Sharpe {m.get('sharpe_ratio', 0):.2f} | "
                  f"Win {m.get('win_rate_pct', 0):.1f}% | "
                  f"{m.get('total_trades', 0)} trades")
    else:
        run_backtest(args.pair, args.strategy, args.days, args.capital)


if __name__ == "__main__":
    main()
