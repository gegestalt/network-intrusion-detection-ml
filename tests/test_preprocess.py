"""Tests for src/preprocess.py — shapes, label encoding, and the two
properties that are easy to get subtly wrong: **no leakage** and
**unseen-category tolerance**."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data as D
import preprocess as P
from conftest import needs_data


def _dummy_frame(n: int = 2) -> pd.DataFrame:
    """A minimal valid feature frame (all 41 columns) for logic-only tests."""
    row = {c: 0.0 for c in D.NUMERIC_COLS}
    row |= {"protocol_type": "tcp", "service": "http", "flag": "SF"}
    return pd.DataFrame([row] * n)


# --- pure-logic tests ------------------------------------------------------- #
def test_build_preprocessor_is_unfitted_and_typed():
    pre = P.build_preprocessor(D.CATEGORICAL_COLS, D.NUMERIC_COLS)
    # get_feature_names_out before fit should fail — it's genuinely unfitted.
    with pytest.raises(Exception):
        pre.get_feature_names_out()


def test_prepare_fails_loud_on_label_outside_class_order():
    train = _dummy_frame(2)
    test = _dummy_frame(1)
    y_tr = pd.Series(["normal", "attack"])
    y_te = pd.Series(["normal"])
    with pytest.raises(ValueError, match="absent from class_order"):
        P.prepare(train, test, D.CATEGORICAL_COLS, D.NUMERIC_COLS,
                  y_tr, y_te, class_order=["normal"], scheme="binary")


# --- data-dependent tests --------------------------------------------------- #
@needs_data
@pytest.mark.parametrize("scheme,n_classes", [("binary", 2), ("multiclass", 5)])
def test_shapes_and_labels(scheme, n_classes):
    d = P.prepare_nsl_kdd(scheme)
    assert d.X_train.shape == (125_973, 122)
    assert d.X_test.shape == (22_544, 122)
    assert len(d.classes) == n_classes
    assert d.n_features == len(d.feature_names) == 122
    # labels are integers in [0, n_classes)
    assert set(np.unique(d.y_train)).issubset(range(n_classes))


@needs_data
def test_no_nans_produced():
    d = P.prepare_nsl_kdd("multiclass")
    assert not np.isnan(d.X_train).any()
    assert not np.isnan(d.X_test).any()


@needs_data
def test_no_leakage_scaler_fitted_on_train_only():
    """The fitted scaler's means must equal the TRAIN numeric means, proving the
    transformer never saw the test set during fit."""
    d = P.prepare_nsl_kdd("binary")
    scaler = d.preprocessor.named_transformers_["num"]
    train_means = D.load_nsl_kdd("train")[D.NUMERIC_COLS].mean().to_numpy()
    assert np.allclose(scaler.mean_, train_means, rtol=1e-6, atol=1e-6)


@needs_data
def test_handle_unknown_service_becomes_all_zero_block():
    """An unseen test-set category must encode as an all-zero one-hot block, not
    crash (this is what handle_unknown='ignore' buys us)."""
    train = D.load_nsl_kdd("train")
    pre = P.build_preprocessor(D.CATEGORICAL_COLS, D.NUMERIC_COLS)
    pre.fit(train)

    fake = train.iloc[[0]].copy()
    fake["service"] = "__never_seen_service__"
    out = pre.transform(fake)  # must not raise

    names = list(pre.get_feature_names_out())
    svc_cols = [i for i, n in enumerate(names) if n.startswith("service_")]
    assert svc_cols, "expected service_* one-hot columns"
    assert out[0, svc_cols].sum() == 0.0  # unknown -> all zeros
    # sanity: a known categorical (flag) is still one-hot active
    flag_cols = [i for i, n in enumerate(names) if n.startswith("flag_")]
    assert out[0, flag_cols].sum() == 1.0
