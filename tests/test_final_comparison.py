"""Tests for the consolidated final comparison layer."""

from __future__ import annotations

import pandas as pd

import final_comparison as F


def test_coverage_matrix_names_completed_and_blocked_tracks():
    coverage = F.build_coverage_matrix()
    statuses = set(coverage["status"])
    areas = set(coverage["experiment_area"])

    assert {"PROVEN", "PARTIAL", "BLOCKED", "MISSING"} <= statuses
    assert "Threshold tuning" in areas
    assert "SOC workload simulation" in areas
    assert "CICIoT2023 raw CSV" in areas
    assert "TON_IoT" in areas
    assert "Graph ML" in areas


def test_build_comparison_uses_common_schema():
    comparison = F.build_comparison()

    assert list(comparison.columns) == F.COMMON_COLUMNS
    assert len(comparison) > 0
    assert comparison["track"].notna().any()
    assert pd.api.types.is_numeric_dtype(comparison["macro_f1"])
    assert {"dataset", "task", "track", "method", "macro_f1"} <= set(comparison.columns)


def test_render_report_contains_referee_sections():
    comparison = F.build_comparison()
    coverage = F.build_coverage_matrix()
    report = F.render_report(comparison, coverage)

    assert "Coverage Matrix" in report
    assert "Best NSL-KDD Binary Rows By Macro-F1" in report
    assert "Best NSL-KDD Binary Rows By Attack Recall" in report
    assert "Lowest Operational Alert Burden" in report
    assert "CICIoT2023 dev rows are useful" in report
