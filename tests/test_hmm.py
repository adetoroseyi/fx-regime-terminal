"""Tests for HMM regime detection model."""

import numpy as np
import pandas as pd
import pytest

from src.features import compute_features, get_feature_matrix
from src.hmm_model import (
    train_hmm, decode_regime, _apply_confirmation_lag,
    save_model, load_model, get_regime_transition_matrix,
    select_best_n_states,
    N_REGIMES, REGIME_LABELS,
)


def make_synthetic_ohlcv(n=500, seed=42):
    """Create synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)

    # Generate a random walk for close prices
    returns = rng.normal(0, 0.001, n)
    close = 1.1000 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    high = close * (1 + rng.uniform(0, 0.002, n))
    low = close * (1 - rng.uniform(0, 0.002, n))
    open_p = close * (1 + rng.normal(0, 0.0005, n))
    volume = rng.randint(100, 10000, n).astype(float)

    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    return pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


class TestFeatures:
    def test_compute_features_shape(self):
        df = make_synthetic_ohlcv(100)
        features = compute_features(df)
        assert features.shape[1] == 3
        assert len(features) <= len(df)
        assert len(features) >= len(df) - 5  # some rows dropped for NaN

    def test_feature_columns(self):
        df = make_synthetic_ohlcv(100)
        features = compute_features(df)
        assert list(features.columns) == ["log_return", "norm_range", "volume_change"]

    def test_no_nans(self):
        df = make_synthetic_ohlcv(200)
        features = compute_features(df)
        assert not features.isna().any().any()

    def test_feature_matrix(self):
        df = make_synthetic_ohlcv(100)
        X = get_feature_matrix(df)
        assert X.ndim == 2
        assert X.shape[1] == 3


class TestHMMModel:
    @pytest.fixture
    def trained_model(self):
        df = make_synthetic_ohlcv(500)
        return train_hmm(df, n_states=N_REGIMES)

    def test_train_returns_dict(self, trained_model):
        assert "model" in trained_model
        assert "states" in trained_model
        assert "posteriors" in trained_model
        assert "regime_map" in trained_model
        assert "score" in trained_model
        assert "bic" in trained_model
        assert "aic" in trained_model
        assert "n_free_params" in trained_model

    def test_n_states(self, trained_model):
        assert trained_model["model"].n_components == N_REGIMES

    def test_regime_map_has_all_states(self, trained_model):
        regime_map = trained_model["regime_map"]
        assert len(regime_map) == N_REGIMES
        for i in range(N_REGIMES):
            assert i in regime_map

    def test_regime_labels_are_known(self, trained_model):
        known_labels = set(REGIME_LABELS.values())
        for label in trained_model["regime_map"].values():
            assert label in known_labels

    def test_states_shape(self, trained_model):
        states = trained_model["states"]
        features = trained_model["features"]
        assert len(states) == len(features)

    def test_posteriors_shape(self, trained_model):
        posteriors = trained_model["posteriors"]
        features = trained_model["features"]
        assert posteriors.shape == (len(features), N_REGIMES)

    def test_posteriors_sum_to_one(self, trained_model):
        posteriors = trained_model["posteriors"]
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_decode_regime(self, trained_model):
        df = make_synthetic_ohlcv(500)
        regime_df = decode_regime(
            trained_model["model"], df, trained_model["regime_map"]
        )
        assert "state" in regime_df.columns
        assert "regime" in regime_df.columns
        assert "confidence" in regime_df.columns
        assert "confirmed" in regime_df.columns

    def test_transition_matrix(self, trained_model):
        tmat = get_regime_transition_matrix(
            trained_model["model"], trained_model["regime_map"]
        )
        assert tmat.shape == (N_REGIMES, N_REGIMES)
        # Rows should sum to ~1
        np.testing.assert_allclose(tmat.values.sum(axis=1), 1.0, atol=1e-6)


class TestConfirmationLag:
    def test_basic_confirmation(self):
        states = np.array([0, 0, 0, 0, 1, 1, 1])
        confirmed = _apply_confirmation_lag(states, lag=3)
        assert not confirmed[0]
        assert not confirmed[1]
        assert confirmed[2]  # 3 consecutive 0s
        assert confirmed[3]  # 4 consecutive 0s
        assert not confirmed[4]
        assert not confirmed[5]
        assert confirmed[6]  # 3 consecutive 1s

    def test_no_confirmation_on_switch(self):
        states = np.array([0, 1, 0, 1, 0, 1, 0])
        confirmed = _apply_confirmation_lag(states, lag=3)
        assert not any(confirmed)

    def test_all_same(self):
        states = np.array([2, 2, 2, 2, 2])
        confirmed = _apply_confirmation_lag(states, lag=3)
        assert not confirmed[0]
        assert not confirmed[1]
        assert confirmed[2]
        assert confirmed[3]
        assert confirmed[4]


class TestVariableStates:
    """Tests for variable n_states (non-default regime counts)."""

    def test_train_with_fewer_states(self):
        df = make_synthetic_ohlcv(500)
        result = train_hmm(df, n_states=4)
        assert result["model"].n_components == 4
        assert len(result["regime_map"]) == 4
        assert len(result["states"]) == len(result["features"])

    def test_train_with_more_states(self):
        df = make_synthetic_ohlcv(500)
        result = train_hmm(df, n_states=9)
        assert result["model"].n_components == 9
        assert len(result["regime_map"]) == 9
        # Extra states beyond 7 should all be labeled Noise
        noise_count = sum(1 for v in result["regime_map"].values()
                          if v == "Noise")
        assert noise_count >= 3  # at least 9 - 6 = 3 Noise states

    def test_all_labels_from_known_set(self):
        """With any n_states, all labels must come from REGIME_LABELS."""
        known = set(REGIME_LABELS.values())
        df = make_synthetic_ohlcv(500)
        for n in [3, 5, 7, 9]:
            result = train_hmm(df, n_states=n)
            for label in result["regime_map"].values():
                assert label in known, f"n={n}: unknown label '{label}'"

    def test_bic_decreases_then_increases(self):
        """BIC should not monotonically decrease — it penalizes complexity."""
        df = make_synthetic_ohlcv(1000, seed=123)
        bics = []
        for n in range(3, 9):
            result = train_hmm(df, n_states=n)
            bics.append(result["bic"])
        # Not all differences should be negative (i.e. BIC shouldn't
        # always decrease). If it does, the penalty isn't working.
        diffs = [bics[i+1] - bics[i] for i in range(len(bics) - 1)]
        assert not all(d < 0 for d in diffs), (
            f"BIC monotonically decreased, penalty may be broken: {bics}"
        )


class TestSelectBestNStates:
    def test_returns_best_n(self):
        df = make_synthetic_ohlcv(500)
        result = select_best_n_states(df, state_range=range(3, 6))
        assert "best_n" in result
        assert result["best_n"] in range(3, 6)
        assert "best_result" in result
        assert result["best_result"]["model"].n_components == result["best_n"]

    def test_sweep_has_all_states(self):
        df = make_synthetic_ohlcv(500)
        result = select_best_n_states(df, state_range=range(3, 6))
        sweep_ns = {s["n_states"] for s in result["sweep"]}
        assert sweep_ns == {3, 4, 5}

    def test_best_has_lowest_bic(self):
        df = make_synthetic_ohlcv(500)
        result = select_best_n_states(df, state_range=range(3, 6),
                                      criterion="bic")
        valid_bics = {s["n_states"]: s["bic"]
                      for s in result["sweep"] if s["score"] is not None}
        best_bic = valid_bics[result["best_n"]]
        assert best_bic == min(valid_bics.values())


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        df = make_synthetic_ohlcv(300)
        result = train_hmm(df)
        save_model(result, "TEST", directory=str(tmp_path))

        model, regime_map = load_model("TEST", directory=str(tmp_path))
        assert model.n_components == N_REGIMES
        assert len(regime_map) == N_REGIMES

    def test_save_and_load_variable_states(self, tmp_path):
        """Model saved with non-default n_states loads correctly."""
        df = make_synthetic_ohlcv(300)
        result = train_hmm(df, n_states=4)
        save_model(result, "TEST4", directory=str(tmp_path))

        model, regime_map = load_model("TEST4", directory=str(tmp_path))
        assert model.n_components == 4
        assert len(regime_map) == 4
