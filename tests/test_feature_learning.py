"""Fast tests for feature analysis and representation helpers."""

from __future__ import annotations

import numpy as np
import pytest

import feature_learning as F


def test_target_correlations_rank_informative_feature_first():
    X = np.array(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    out = F.target_correlations(X, y, ["signal", "noise"], "toy")

    assert out.iloc[0]["feature"] == "signal"
    assert out.iloc[0]["abs_target_corr"] == pytest.approx(1.0)


def test_high_correlation_pairs_finds_duplicate_columns():
    X = np.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
            [3.0, 3.0, 0.0],
            [4.0, 4.0, 1.0],
        ]
    )

    pairs = F.high_correlation_pairs(X, ["a", "b", "c"], "toy", threshold=0.99)

    assert len(pairs) == 1
    assert pairs.iloc[0]["feature_a"] == "a"
    assert pairs.iloc[0]["feature_b"] == "b"


def test_outlier_rates_iqr_flags_extreme_column():
    X = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [100.0, 0.0],
        ]
    )

    out = F.outlier_rates_iqr(X, ["spiky", "constant"], "toy")

    spiky = out[out["feature"] == "spiky"].iloc[0]
    constant = out[out["feature"] == "constant"].iloc[0]
    assert spiky["outlier_rate_iqr"] == pytest.approx(0.25)
    assert constant["outlier_rate_iqr"] == pytest.approx(0.0)


def test_stratified_cap_indices_keeps_both_classes():
    y = np.array([0] * 90 + [1] * 10)

    idx = F.stratified_cap_indices(y, max_rows=20, seed=7)

    assert len(idx) == 20
    assert set(np.unique(y[idx])) == {0, 1}


def test_autoencoder_embeddings_have_requested_shape():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(40, 6)).astype(np.float32)
    X_test = rng.normal(size=(12, 6)).astype(np.float32)

    z_train, z_test = F.autoencoder_embeddings(
        X_train,
        X_test,
        latent_dim=3,
        max_train_rows=30,
        epochs=1,
        batch_size=16,
        seed=7,
    )

    assert z_train.shape == (40, 3)
    assert z_test.shape == (12, 3)
