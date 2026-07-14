"""Tests for the visual comparison dashboard inputs."""

from __future__ import annotations

import pandas as pd

import visual_comparison as V


def test_metric_summary_combines_model_and_feature_group_rows():
    lab = pd.DataFrame(
        [
            {
                "dataset": "nsl_kdd",
                "level": "Level 1",
                "family": "logistic_regression",
                "method": "LogReg",
                "variant": "balanced",
                "accuracy": 0.7,
                "balanced_accuracy": 0.72,
                "macro_f1": 0.71,
                "weighted_f1": 0.73,
                "precision_attack": 0.8,
                "attack_recall": 0.6,
                "normal_recall": 0.84,
                "mcc": 0.4,
                "fp_per_10k_benign": 100.0,
                "fn_per_10k_attack": 4000.0,
            }
        ]
    )
    groups = pd.DataFrame(
        [
            {
                "dataset": "nsl_kdd",
                "level": "Feature groups",
                "family": "feature_ablation",
                "method": "LogReg",
                "variant": "numeric_only",
                "accuracy": 0.65,
                "balanced_accuracy": 0.66,
                "macro_f1": 0.64,
                "weighted_f1": 0.65,
                "precision_attack": 0.75,
                "attack_recall": 0.55,
                "normal_recall": 0.77,
                "mcc": 0.3,
                "fp_per_10k_benign": 120.0,
                "fn_per_10k_attack": 4500.0,
            }
        ]
    )

    summary = V.make_metric_summary(lab, groups)

    assert len(summary) == 2
    assert set(summary["source"]) == {"method_lab", "feature_group_ablation"}
    assert {"macro_f1", "precision_attack", "fp_per_10k_benign"} <= set(summary.columns)
