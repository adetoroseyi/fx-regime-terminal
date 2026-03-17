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

    # Try with increasing covariance regularization to handle
    # non-positive-definite covariance matrices during EM.
    # Falls back to diagonal covariance if full covariance fails entirely.
    attempts = [
        ("full", 1e-3),
        ("full", 1e-2),
        ("full", 1e-1),
        ("diag", 1e-3),
    ]
    last_error = None

    for cov_type, min_covar in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type=cov_type,
                    n_iter=n_iter,
                    random_state=random_state,
                    min_covar=min_covar,
                    verbose=False,
                )
                model.fit(X)

            # Decode most likely state sequence (Viterbi)
            states = model.predict(X)

            # Posterior probabilities
            posteriors = model.predict_proba(X)

            # Auto-label regimes based on cluster characteristics
            regime_map = _auto_label_regimes(model, features, states)

            log_likelihood = model.score(X)
            n_samples = X.shape[0]
            n_free_params = _count_free_params(model)
            bic = -2 * log_likelihood * n_samples + n_free_params * np.log(n_samples)
            aic = -2 * log_likelihood * n_samples + 2 * n_free_params

            return {
                "model": model,
                "features": features,
                "states": states,
                "posteriors": posteriors,
                "regime_map": regime_map,
                "score": log_likelihood,
                "bic": bic,
                "aic": aic,
                "n_free_params": n_free_params,
            }
        except ValueError as e:
            if "positive-definite" in str(e):
                last_error = e
                continue
            raise

    raise ValueError(
        f"HMM training failed after all attempts: {last_error}"
    )


def _count_free_params(model: GaussianHMM) -> int:
    """
    Count the number of free parameters in a fitted GaussianHMM.

    This is needed for BIC/AIC computation.
    """
    n = model.n_components
    n_features = model.means_.shape[1]

    # Transition matrix: each row sums to 1, so (n-1) free per row
    n_transition = n * (n - 1)

    # Start probabilities: n-1 free
    n_start = n - 1

    # Means: n_states * n_features
    n_means = n * n_features

    # Covariances depend on type
    if model.covariance_type == "full":
        # Each state: symmetric matrix with n_features*(n_features+1)/2 free params
        n_cov = n * n_features * (n_features + 1) // 2
    elif model.covariance_type == "diag":
        n_cov = n * n_features
    elif model.covariance_type == "spherical":
        n_cov = n
    elif model.covariance_type == "tied":
        n_cov = n_features * (n_features + 1) // 2
    else:
        n_cov = n * n_features  # fallback

    return n_transition + n_start + n_means + n_cov


def select_best_n_states(df: pd.DataFrame, state_range: range = None,
                         n_iter: int = 200, random_state: int = 42,
                         criterion: str = "bic") -> dict:
    """
    Sweep over different n_states values and select the best model
    using BIC or AIC.

    Args:
        df: OHLCV DataFrame
        state_range: Range of n_states to try (default: 3-10)
        n_iter: Max EM iterations per model
        random_state: Random seed
        criterion: 'bic' or 'aic' (lower is better)

    Returns:
        dict with keys:
            best_n: optimal number of states
            best_result: full train_hmm result dict for best model
            sweep: list of {n_states, score, bic, aic, converged} dicts
    """
    if state_range is None:
        state_range = range(3, 11)

    sweep = []

    for n_states in state_range:
        try:
            result = train_hmm(df, n_states=n_states, n_iter=n_iter,
                               random_state=random_state)
            sweep.append({
                "n_states": n_states,
                "score": result["score"],
                "bic": result["bic"],
                "aic": result["aic"],
                "converged": result["model"].monitor_.converged,
                "result": result,
            })
        except Exception as e:
            sweep.append({
                "n_states": n_states,
                "score": None,
                "bic": float("inf"),
                "aic": float("inf"),
                "converged": False,
                "error": str(e),
            })

    # Select best by criterion
    valid = [s for s in sweep if s["score"] is not None]
    if not valid:
        raise ValueError("All model fits failed during sweep")

    best = min(valid, key=lambda s: s[criterion])

    return {
        "best_n": best["n_states"],
        "best_result": best["result"],
        "sweep": [{k: v for k, v in s.items() if k != "result"} for s in sweep],
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

    # Remaining states: assign as Noise (may be multiple with variable n_states)
    for i in range(n_states):
        if i not in assigned:
            assigned[i] = REGIME_LABELS["noise"]

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
        "covariance_type": result["model"].covariance_type,
        "score": float(result["score"]),
        "bic": float(result["bic"]),
        "aic": float(result["aic"]),
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
    print(f"BIC: {result['bic']:.2f}  AIC: {result['aic']:.2f}")
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
