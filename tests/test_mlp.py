"""Tests for src/train_mlp.py — the deterministic, unit-testable pieces.

We don't assert exact trained accuracy (stochastic); we check the maths and that
the train/predict round-trip produces well-formed outputs.
"""

from __future__ import annotations

import numpy as np
import torch

import train_mlp as M


def test_mlp_forward_shape():
    model = M.MLP(in_dim=10, n_classes=3)
    out = model(torch.zeros(4, 10))
    assert out.shape == (4, 3)


def test_class_weights_favour_rare_classes():
    # counts: class0=80, class1=15, class2=5  -> rarer class, larger weight
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    w = M.compute_class_weights(y, n_classes=3)
    assert w[2] > w[1] > w[0]
    # balanced formula: weight_c = N / (K * count_c)
    assert np.isclose(w[0], 100 / (3 * 80))
    assert np.isclose(w[2], 100 / (3 * 5))


def test_class_weights_handle_absent_class_without_div0():
    y = np.array([0, 0, 1, 1])  # class 2 absent
    w = M.compute_class_weights(y, n_classes=3)
    assert np.isfinite(w).all()


def test_train_and_predict_roundtrip_on_synthetic_data():
    rng = np.random.default_rng(0)
    # two linearly separable-ish blobs
    X = np.vstack([rng.normal(-1, 0.5, (100, 8)),
                   rng.normal(+1, 0.5, (100, 8))]).astype(np.float32)
    y = np.array([0] * 100 + [1] * 100)
    device = torch.device("cpu")
    model = M.train_model(X, y, n_classes=2, device=device)
    y_pred, proba = M.predict(model, X, device=device)
    assert y_pred.shape == (200,)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # should learn *something* on separable data (well above chance)
    assert (y_pred == y).mean() > 0.75
