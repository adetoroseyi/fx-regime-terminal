"""
Publish data to dashboard/data/ for Vercel deployment.

Consolidates signals, backtest results, model metadata, and forward
test log into a single directory that the dashboard can serve statically.

Run after retrain + signal + backtest:
    python -m src.publish_dashboard
"""

import os
import json
import glob
from datetime import datetime


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def publish():
    root = get_project_root()
    out_dir = os.path.join(root, "dashboard", "data")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Copy current signals
    signals_src = os.path.join(root, "state", "current_signals.json")
    if os.path.exists(signals_src):
        with open(signals_src) as f:
            signals = json.load(f)
        with open(os.path.join(out_dir, "signals.json"), "w") as f:
            json.dump(signals, f, indent=2, default=str)
        print(f"  Published signals.json")

    # 2. Consolidate backtest results
    results_dir = os.path.join(root, "results")
    backtest_data = []
    if os.path.isdir(results_dir):
        for fpath in sorted(glob.glob(os.path.join(results_dir, "backtest_*.json"))):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                backtest_data.append(data)
            except Exception:
                pass

    with open(os.path.join(out_dir, "backtests.json"), "w") as f:
        json.dump(backtest_data, f, indent=2, default=str)
    print(f"  Published backtests.json ({len(backtest_data)} results)")

    # 3. Consolidate model metadata
    models_dir = os.path.join(root, "models")
    model_meta = {}
    if os.path.isdir(models_dir):
        for fpath in glob.glob(os.path.join(models_dir, "*_meta.json")):
            try:
                with open(fpath) as f:
                    meta = json.load(f)
                pair = meta.get("pair", os.path.basename(fpath).split("_")[0])
                model_meta[pair] = meta
            except Exception:
                pass

    with open(os.path.join(out_dir, "models.json"), "w") as f:
        json.dump(model_meta, f, indent=2, default=str)
    print(f"  Published models.json ({len(model_meta)} pairs)")

    # 4. Read retrain log (last 30 entries)
    log_path = os.path.join(root, "logs", "retrain_log.csv")
    retrain_history = []
    if os.path.exists(log_path):
        import csv
        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            retrain_history = rows[-30:]  # last 30 entries

    with open(os.path.join(out_dir, "retrain_history.json"), "w") as f:
        json.dump(retrain_history, f, indent=2, default=str)
    print(f"  Published retrain_history.json ({len(retrain_history)} entries)")

    # 5. Forward test log — append current signals as a snapshot
    ft_path = os.path.join(out_dir, "forward_tests.json")
    forward_tests = []
    if os.path.exists(ft_path):
        try:
            with open(ft_path) as f:
                forward_tests = json.load(f)
        except Exception:
            forward_tests = []

    # Add today's snapshot if signals exist
    if os.path.exists(signals_src):
        with open(signals_src) as f:
            current = json.load(f)
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "pairs": {},
        }
        for pair, data in current.get("pairs", {}).items():
            snapshot["pairs"][pair] = {
                "regime": data.get("regime"),
                "confidence": data.get("confidence"),
                "signal_direction": data.get("signal_direction"),
                "regime_confirmed": data.get("regime_confirmed"),
                "entry_price": data.get("entry_price"),
                "stop_loss": data.get("stop_loss"),
                "take_profit": data.get("take_profit"),
            }
        forward_tests.append(snapshot)
        # Keep last 90 days of snapshots
        forward_tests = forward_tests[-90:]

    with open(ft_path, "w") as f:
        json.dump(forward_tests, f, indent=2, default=str)
    print(f"  Published forward_tests.json ({len(forward_tests)} snapshots)")

    print(f"\nDashboard data published to: {out_dir}")


if __name__ == "__main__":
    print("FX Regime Terminal — Publishing Dashboard Data")
    print("=" * 50)
    publish()
