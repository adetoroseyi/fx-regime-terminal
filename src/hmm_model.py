"""
HMM Regime Detection Model for FX Regime Terminal.
Uses GaussianHMM from hmmlearn to classify market regimes.

7 regimes are trained, then auto-labelled based on cluster characteristics:
  0: Bull Trend
  1: Bear Trend
  2: High Volatility Expansion
  3: Low Volatility Compression
  4: Mean Reversion Range
  5: Value Gap Formation
  6: Noise / Choppy
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM

from src.features import compute_features

# Regime label constants
REGIME_LABELS = {
    "bull_trend": "Bull Trend",
    "bear_trend": "Bear Trend",
    "high_vol": "High Volatility Expansion",
    "low_vol": "Low Volatility Compression",
    "mean_reversion": "Mean Reversion Range",
    "value_gap": "Value Gap Formation",
    "noise": "Noise",
}

N_REGIMES = 7
CONFIRMATION_LAG = 3  # candles required to confirm regime


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_hmm(df: pd.DataFrame, n_states: int = N_REGIMES,
              n_iter: int = 200, random_state: int = 42) -> dict:
    """
    Train a GaussianHMM on OHLCV data.

    Args:
        df: OHLCV DataFrame
        n_states: Number of hidden states (regimes)
        n_iter: Maximum EM iterations
        random_state: Random seed for reproducibility

    Returns:
        dict with keys:
            model: trained GaussianHMM
            features: feature DataFrame used
            states: decoded state sequence (Viterbi)
            posteriors: posterior probabilities per state
            regime_map: mapping from state index to regime label
            score: log-likelihood of the model
    """
    features = compute_features(df)
    X = features.values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            verbose=False,
        )
        model.fit(X)

    # Decode most likely state sequence (Viterbi)
    states = model.predict(X)

    # Posterior probabilities
    posteriors = model.predict_proba(X)

    # Auto-label regimes based on cluster characteristics
    regime_map = _auto_label_regimes(model, features, states)

    return {
        "model": model,
        "features": features,
        "states": states,
        "posteriors": posteriors,
        "regime_map": regime_map,
        "score": model.score(X),
    }


def _auto_label_regimes(model: GaussianHMM, features: pd.DataFrame,
                        states: np.ndarray) -> dict:
    """
    Auto-label regime states based on their statistical characteristics.

    Uses the mean of each feature cluster to assign semantic labels:
      - log_return sign/magnitude → trend direction
      - norm_range magnitude → volatility level
      - Combined patterns → special regimes (FVG, MR, Noise)

    Returns:
        dict mapping state_index → regime label string
    """
    n_states = model.n_components
    means = model.means_  # shape: (n_states, n_features)
    # means columns: [log_return, norm_range, volume_change]

    # Compute per-state statistics
    state_stats = []
    for i in range(n_states):
        mask = states == i
        state_features = features.values[mask]
        count = mask.sum()

        stats = {
            "state": i,
            "count": count,
            "mean_return": means[i, 0],
            "mean_range": means[i, 1],
            "mean_vol_change": means[i, 2],
            "return_std": state_features[:, 0].std() if count > 1 else 0,
            "range_std": state_features[:, 1].std() if count > 1 else 0,
        }
        state_stats.append(stats)

    stats_df = pd.DataFrame(state_stats)

    # Scoring system to assign labels
    # Sort by different criteria and assign labels greedily
    assigned = {}
    used_labels = set()

    def assign(state_idx, label):
        if state_idx not in assigned and label not in used_labels:
            assigned[state_idx] = label
            used_labels.add(label)
            return True
        return False

    # 1. Bull Trend: highest positive mean return
    bull_candidates = stats_df.sort_values("mean_return", ascending=False)
    for _, row in bull_candidates.iterrows():
        if row["mean_return"] > 0:
            if assign(int(row["state"]), REGIME_LABELS["bull_trend"]):
                break

    # 2. Bear Trend: most negative mean return
    bear_candidates = stats_df.sort_values("mean_return", ascending=True)
    for _, row in bear_candidates.iterrows():
        if row["mean_return"] < 0:
            if assign(int(row["state"]), REGIME_LABELS["bear_trend"]):
                break

    # 3. High Volatility Expansion: highest mean range
    hv_candidates = stats_df.sort_values("mean_range", ascending=False)
    for _, row in hv_candidates.iterrows():
        if assign(int(row["state"]), REGIME_LABELS["high_vol"]):
            break

    # 4. Low Volatility Compression: lowest mean range
    lv_candidates = stats_df.sort_values("mean_range", ascending=True)
    for _, row in lv_candidates.iterrows():
        if assign(int(row["state"]), REGIME_LABELS["low_vol"]):
            break

    # 5. Value Gap Formation: high volume change + high range (imbalance)
    stats_df["fvg_score"] = (
        stats_df["mean_vol_change"].abs() * 0.5 + stats_df["mean_range"] * 0.5
    )
    fvg_candidates = stats_df.sort_values("fvg_score", ascending=False)
    for _, row in fvg_candidates.iterrows():
        if assign(int(row["state"]), REGIME_LABELS["value_gap"]):
            break

    # 6. Mean Reversion Range: low return std + moderate range (oscillating)
    stats_df["mr_score"] = -stats_df["return_std"] + stats_df["mean_range"] * 0.3
    mr_candidates = stats_df.sort_values("mr_score", ascending=False)
    for _, row in mr_candidates.iterrows():
        if assign(int(row["state"]), REGIME_LABELS["mean_reversion"]):
            break

    # 7. Noise: whatever is left
    for i in range(n_states):
        if i not in assigned:
            assign(i, REGIME_LABELS["noise"])

    return assigned


def decode_regime(model: GaussianHMM, df: pd.DataFrame,
                  regime_map: dict) -> pd.DataFrame:
    """
    Decode regimes for new data using a trained model.

    Args:
        model: Trained GaussianHMM
        df: OHLCV DataFrame
        regime_map: State index → label mapping

    Returns:
        DataFrame with columns:
            state: raw state index
            regime: human-readable regime label
            confidence: posterior probability of assigned state (0-100%)
            confirmed: whether regime has been confirmed (3-candle lag)
    """
    features = compute_features(df)
    X = features.values

    states = model.predict(X)
    posteriors = model.predict_proba(X)

    result = pd.DataFrame(index=features.index)
    result["state"] = states
    result["regime"] = [regime_map.get(s, "Unknown") for s in states]
    result["confidence"] = [posteriors[i, s] * 100 for i, s in enumerate(states)]

    # Apply confirmation lag (hysteresis)
    result["confirmed"] = _apply_confirmation_lag(states, lag=CONFIRMATION_LAG)

    return result


def _apply_confirmation_lag(states: np.ndarray, lag: int = 3) -> np.ndarray:
    """
    A regime is only 'confirmed' if it has persisted for `lag` consecutive candles.

    Returns:
        Boolean array — True if regime at position i has been the same
        for the last `lag` candles (inclusive).
    """
    n = len(states)
    confirmed = np.zeros(n, dtype=bool)

    for i in range(lag - 1, n):
        window = states[i - lag + 1: i + 1]
        if np.all(window == states[i]):
            confirmed[i] = True

    return confirmed


def get_regime_transition_matrix(model: GaussianHMM, regime_map: dict) -> pd.DataFrame:
    """
    Return the regime transition probability matrix with human-readable labels.
    """
    transmat = model.transmat_
    labels = [regime_map.get(i, f"State {i}") for i in range(len(transmat))]
    return pd.DataFrame(transmat, index=labels, columns=labels)


def save_model(result: dict, pair: str, directory: str = None):
    """Save trained HMM model and metadata to disk."""
    if directory is None:
        directory = os.path.join(get_project_root(), "models")
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, f"{pair}_hmm.pkl")
    meta_path = os.path.join(directory, f"{pair}_meta.json")

    joblib.dump(result["model"], model_path)

    meta = {
        "pair": pair,
        "n_states": result["model"].n_components,
        "score": float(result["score"]),
        "regime_map": {str(k): v for k, v in result["regime_map"].items()},
        "n_samples": len(result["features"]),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_model(pair: str, directory: str = None) -> tuple:
    """
    Load a trained HMM model and its metadata.

    Returns:
        (model, regime_map)
    """
    if directory is None:
        directory = os.path.join(get_project_root(), "models")

    model_path = os.path.join(directory, f"{pair}_hmm.pkl")
    meta_path = os.path.join(directory, f"{pair}_meta.json")

    model = joblib.load(model_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    regime_map = {int(k): v for k, v in meta["regime_map"].items()}

    return model, regime_map


if __name__ == "__main__":
    from src.data_loader import fetch_pair_data

    print("HMM Model Training Test")
    print("=" * 40)

    pair = "EURUSD"
    print(f"\nFetching data for {pair}...")
    df = fetch_pair_data(pair, period_days=60)
    print(f"Loaded {len(df)} candles")

    print(f"\nTraining HMM with {N_REGIMES} states...")
    result = train_hmm(df)

    print(f"Model score (log-likelihood): {result['score']:.2f}")
    print(f"\nRegime mapping:")
    for state, label in sorted(result["regime_map"].items()):
        count = (result["states"] == state).sum()
        pct = count / len(result["states"]) * 100
        print(f"  State {state}: {label} ({count} candles, {pct:.1f}%)")

    print(f"\nTransition matrix:")
    print(get_regime_transition_matrix(result["model"], result["regime_map"]).round(3))

    # Test decode
    regime_df = decode_regime(result["model"], df, result["regime_map"])
    confirmed = regime_df["confirmed"].sum()
    print(f"\nConfirmed regime candles: {confirmed}/{len(regime_df)}")

    # Save and reload test
    save_model(result, pair)
    model2, rmap2 = load_model(pair)
    print(f"\nSave/load test: OK (model has {model2.n_components} states)")
