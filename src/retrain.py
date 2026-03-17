"""
Daily retrain entry point for FX Regime Terminal.
Retrains HMM models for all forex pairs and logs results.

Run daily at 00:05 UTC via GitHub Actions:
    python src/retrain.py
"""

import os
import csv
import sys
from datetime import datetime

from src.data_loader import fetch_pair_data, SUPPORTED_PAIRS
from src.hmm_model import train_hmm, save_model, N_REGIMES


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def retrain_all(period_days: int = 730) -> list:
    """
    Retrain HMM models for all supported pairs.

    Returns:
        List of result dicts for logging
    """
    results = []
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    print(f"FX Regime Terminal — Model Retraining")
    print(f"Date: {date_str}")
    print(f"Pairs: {SUPPORTED_PAIRS}")
    print(f"{'='*60}")

    for pair in SUPPORTED_PAIRS:
        print(f"\n--- Retraining {pair} ---")
        result = {"date": date_str, "pair": pair}

        try:
            # Fetch fresh data
            df = fetch_pair_data(pair, period_days=period_days, use_cache=False)
            result["n_samples"] = len(df)
            print(f"  Data: {len(df)} candles")

            # Train HMM
            model_result = train_hmm(df, n_states=N_REGIMES)
            result["score"] = round(model_result["score"], 2)
            result["bic"] = round(model_result["bic"], 2)
            result["aic"] = round(model_result["aic"], 2)
            result["converged"] = model_result["model"].monitor_.converged
            result["n_iter"] = model_result["model"].monitor_.n_iter

            # Regime distribution
            regime_counts = {}
            for state, label in model_result["regime_map"].items():
                count = int((model_result["states"] == state).sum())
                regime_counts[label] = count
            result["regime_counts"] = str(regime_counts)

            # Save model
            save_model(model_result, pair)
            result["status"] = "OK"

            print(f"  Score: {result['score']}")
            print(f"  BIC: {result['bic']}  AIC: {result['aic']}")
            print(f"  Converged: {result['converged']} ({result['n_iter']} iterations)")
            print(f"  Regimes: {regime_counts}")
            print(f"  Model saved ✓")

        except Exception as e:
            result["status"] = f"FAILED: {str(e)}"
            result["n_samples"] = 0
            result["score"] = None
            result["converged"] = False
            result["regime_counts"] = ""
            print(f"  ERROR: {e}")

        results.append(result)

    # Log results
    _log_results(results)

    # Summary
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{'='*60}")
    print(f"Retraining complete: {ok}/{len(results)} pairs succeeded")

    return results


def _log_results(results: list):
    """Append retrain results to CSV log."""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "retrain_log.csv")

    fieldnames = ["date", "pair", "n_samples", "score", "bic", "aic",
                  "converged", "n_iter", "regime_counts", "status"]

    write_header = not os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    retrain_all()
