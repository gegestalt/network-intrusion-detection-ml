"""Tests for src/evaluate.py — metric maths verified against hand computation.

Worked example (binary, classes=['normal','attack'], attack=positive=1):
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
  Confusion: normal -> {0:1, 1:1}, attack -> {1:2}
  accuracy = 3/4 = 0.75
  attack : precision 2/3, recall 2/2=1.0, f1 = 2*(2/3*1)/(2/3+1) = 0.8
  normal : precision 1/1=1.0, recall 1/2=0.5, f1 = 2*(1*0.5)/1.5 = 0.6667
  macro_f1 = (0.8 + 0.6667)/2 = 0.7333
"""

from __future__ import annotations

import numpy as np
import pytest

import evaluate as E

CLASSES = ["normal", "attack"]
Y_TRUE = np.array([0, 0, 1, 1])
Y_PRED = np.array([0, 1, 1, 1])


def test_accuracy():
    m = E.compute_metrics(Y_TRUE, Y_PRED, CLASSES)
    assert m["accuracy"] == pytest.approx(0.75)


def test_per_class_precision_recall_f1():
    m = E.compute_metrics(Y_TRUE, Y_PRED, CLASSES)["per_class"]
    assert m["attack"]["recall"] == pytest.approx(1.0)
    assert m["attack"]["precision"] == pytest.approx(2 / 3)
    assert m["attack"]["f1"] == pytest.approx(0.8)
    assert m["normal"]["recall"] == pytest.approx(0.5)
    assert m["normal"]["f1"] == pytest.approx(2 / 3, abs=1e-4)


def test_macro_f1_weights_classes_equally():
    m = E.compute_metrics(Y_TRUE, Y_PRED, CLASSES)
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2, abs=1e-4)


def test_support_counts():
    m = E.compute_metrics(Y_TRUE, Y_PRED, CLASSES)["per_class"]
    assert m["normal"]["support"] == 2
    assert m["attack"]["support"] == 2


def test_roc_and_pr_auc_present_only_for_binary_with_scores():
    y_score = np.array([0.1, 0.6, 0.8, 0.9])  # perfectly ranks positives highest
    m = E.compute_metrics(Y_TRUE, Y_PRED, CLASSES, y_score=y_score)
    assert m["roc_auc"] == pytest.approx(1.0)  # all positives score above negatives
    assert 0.0 <= m["pr_auc"] <= 1.0
    # without scores -> keys absent
    m2 = E.compute_metrics(Y_TRUE, Y_PRED, CLASSES)
    assert "roc_auc" not in m2


def test_no_auc_for_multiclass_even_with_scores():
    classes = ["a", "b", "c"]
    yt = np.array([0, 1, 2, 1])
    yp = np.array([0, 1, 2, 0])
    m = E.compute_metrics(yt, yp, classes)
    assert "roc_auc" not in m
    assert set(m["per_class"]) == {"a", "b", "c"}


def test_macro_recall_is_mean_of_per_class_recall():
    m = E.compute_metrics(Y_TRUE, Y_PRED, CLASSES)
    pc = m["per_class"]
    expected = np.mean([pc[c]["recall"] for c in CLASSES])
    assert m["macro_recall"] == pytest.approx(expected)
