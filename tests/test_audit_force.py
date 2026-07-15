"""Tests for the prompt-coverage referee audit."""

from __future__ import annotations

import audit_force as A


def test_snapshot_sees_current_project_evidence():
    state = A.snapshot()

    assert state["test_file_count"] >= 1
    assert state["figure_count"] >= 1
    assert state["has_threshold_sweep"] is True
    assert state["has_ciciot2023_mapper"] is True


def test_audit_items_include_proven_partial_and_missing_tracks():
    items = A.build_audit_items(A.snapshot())
    statuses = {item.status for item in items}
    by_area = {item.area: item for item in items}

    assert {"PROVEN", "PARTIAL", "MISSING"} <= statuses
    assert by_area["NSL-KDD controlled baseline"].status == "PROVEN"
    assert by_area["Artificial neuron and activation foundations"].status in {"PROVEN", "MISSING"}
    assert by_area["Controlled neural architecture ablations"].status in {"PROVEN", "MISSING"}
    assert by_area["CICIoT2023 primary modern dataset"].status in {"PARTIAL", "PROVEN"}
    assert by_area["Unsupervised anomaly detection"].status in {"PROVEN", "MISSING"}
    assert by_area["Semi-supervised learning"].status in {"PROVEN", "MISSING"}
    assert by_area["Clustering and representation studies"].status in {"PARTIAL", "MISSING"}
    assert by_area["Temporal and sequence detection"].status == "MISSING"
    assert by_area["Graph machine learning"].status == "MISSING"


def test_report_blocks_unproven_big_lab_claims():
    report = A.render_report()

    assert "Claims I Would Block" in report
    assert "Any claim that TON_IoT or CSE-CIC-IDS2018 has been evaluated." in report
    assert "true temporal sequence detection" in report
    assert "neural" in report.lower()
    assert "Requested Lab Matrix" in report
