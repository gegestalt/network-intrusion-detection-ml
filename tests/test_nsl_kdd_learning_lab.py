"""Tests for the first-dataset NSL-KDD learning lab."""

from __future__ import annotations

import numpy as np

import nsl_kdd_learning_lab as L


def test_concept_matrix_marks_cnn_as_not_applied_to_tabular_rows():
    matrix = L.concept_matrix()
    by_concept = dict(zip(matrix["concept"], matrix["status"], strict=True))

    assert by_concept["Logistic Regression variants"] == "APPLIED"
    assert by_concept["Feature-group ablation"] == "APPLIED"
    assert by_concept["CNN / pooling"] == "NOT_APPLIED"


def test_feature_group_masks_cover_expected_nsl_groups():
    names = [
        "protocol_type_tcp",
        "service_http",
        "flag_SF",
        "src_bytes",
        "dst_bytes",
        "count",
        "same_srv_rate",
        "dst_host_count",
        "logged_in",
    ]
    groups = L.feature_group_masks(names)

    assert groups["categorical_one_hot"] == [0, 1, 2]
    assert groups["byte_volume"] == [3, 4]
    assert groups["connection_count_rates"] == [5, 6]
    assert groups["dst_host_behavior"] == [7]
    assert groups["content_login_shell"] == [8]


def test_threshold_tuning_moves_toward_attack_recall_when_beta_is_high():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.10, 0.40, 0.35, 0.90])

    threshold_f1, _ = L.tune_threshold(y, scores, beta=1.0)
    threshold_f2, _ = L.tune_threshold(y, scores, beta=2.0)

    assert threshold_f2 <= threshold_f1
