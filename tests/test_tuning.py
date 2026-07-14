"""Tests for src/tuning.py — the fine-tuning protocol guardrails.

The point is not to freeze exact NSL-KDD scores. The point is to make sure the
selection machinery is deterministic, validation-only, and metric-driven.
"""

from __future__ import annotations

import numpy as np
import pytest

import tuning as T


def test_validation_split_is_disjoint_deterministic_and_stratified():
    y = np.array([0] * 50 + [1] * 50)
    s1 = T.make_validation_split(y, validation_size=0.2, seed=7)
    s2 = T.make_validation_split(y, validation_size=0.2, seed=7)

    assert np.array_equal(s1.train_idx, s2.train_idx)
    assert np.array_equal(s1.val_idx, s2.val_idx)
    assert set(s1.train_idx).isdisjoint(set(s1.val_idx))
    assert set(s1.train_idx) | set(s1.val_idx) == set(range(len(y)))
    assert y[s1.val_idx].mean() == pytest.approx(0.5)


def test_select_best_candidate_keeps_first_row_on_tie():
    rows = [
        {"name": "simple", "macro_f1": 0.8},
        {"name": "complex", "macro_f1": 0.8},
        {"name": "bad", "macro_f1": 0.4},
    ]

    assert T.select_best_candidate(rows, "macro_f1")["name"] == "simple"


def test_select_best_candidate_rejects_empty_rows():
    with pytest.raises(ValueError, match="empty"):
        T.select_best_candidate([], "macro_f1")


def test_threshold_sweep_chooses_validation_threshold_for_f2():
    y_val = np.array([0, 0, 1, 1])
    scores = np.array([0.10, 0.40, 0.35, 0.90])
    out = T.threshold_sweep(
        y_val,
        scores,
        thresholds=np.array([0.30, 0.50]),
        beta=2.0,
    )

    # threshold=0.30 catches both positives, accepting one false positive.
    # F2 values recall more heavily, so this beats threshold=0.50.
    assert out["best"]["threshold"] == pytest.approx(0.30)
    assert out["best"]["score"] > out["rows"][1]["score"]


def test_metric_row_reports_macro_and_per_class_values():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    row = T.metric_row(y_true, y_pred, ["normal", "R2L", "U2R"])

    assert row["accuracy"] == pytest.approx(3 / 5)
    assert row["recall:normal"] == pytest.approx(0.5)
    assert row["recall:R2L"] == pytest.approx(1.0)
    assert row["recall:U2R"] == pytest.approx(0.0)
    assert row["support:U2R"] == 1
    assert row["macro_f1"] < row["weighted_f1"]


def test_rare_recall_mean_uses_named_rare_classes():
    metric = {
        "recall:normal": 0.9,
        "recall:R2L": 0.2,
        "recall:U2R": 0.6,
    }

    assert T.rare_recall_mean(metric, ["R2L", "U2R"]) == pytest.approx(0.4)


def test_rare_recall_mean_rejects_empty_class_list():
    with pytest.raises(ValueError, match="must not be empty"):
        T.rare_recall_mean({}, [])
