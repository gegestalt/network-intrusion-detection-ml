"""Tests for CICIoT2023 Phase-1 helpers.

These are synthetic and fast; the full CICIoT2023 CSV release is not required
for CI or local smoke tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

import ciciot2023 as C


@pytest.mark.parametrize(
    "label,category",
    [
        ("BenignTraffic", "Benign"),
        ("DDoS-UDP_Flood", "DDoS"),
        ("DoS-SYN_Flood", "DoS"),
        ("Recon-PortScan", "Recon"),
        ("VulnerabilityScan", "Recon"),
        ("SqlInjection", "Web-based"),
        ("CommandInjection", "Web-based"),
        ("DictionaryBruteForce", "Brute Force"),
        ("DNS_Spoofing", "Spoofing"),
        ("Mirai-greeth_flood", "Mirai"),
    ],
)
def test_attack_category_maps_known_label_variants(label, category):
    assert C.attack_category(label) == category


def test_attack_category_raises_on_unknown_label():
    with pytest.raises(KeyError, match="Unmapped CICIoT2023 label"):
        C.attack_category("brand-new-attack")


def test_add_label_columns_normalizes_binary_and_category():
    df = pd.DataFrame(
        {
            "flow_duration": [1.0, 2.0],
            "Label": ["BenignTraffic", "DDoS-UDP_Flood"],
        }
    )

    out = C.add_label_columns(df)

    assert out["label"].tolist() == ["BenignTraffic", "DDoS-UDP_Flood"]
    assert out["attack_category"].tolist() == ["Benign", "DDoS"]
    assert out["binary_label"].tolist() == [0, 1]


def test_load_csv_sample_reads_multiple_files_and_limits_rows(tmp_path):
    d = tmp_path / "CSV"
    d.mkdir()
    pd.DataFrame(
        {
            "flow_duration": [1, 2, 3],
            "Label": ["BenignTraffic", "DDoS-UDP_Flood", "DDoS-TCP_Flood"],
        }
    ).to_csv(d / "part1.csv", index=False)
    pd.DataFrame(
        {
            "flow_duration": [4, 5, 6],
            "Label": ["SqlInjection", "DNS_Spoofing", "Mirai-udpplain"],
        }
    ).to_csv(d / "part2.csv", index=False)

    sample = C.load_csv_sample(d, max_rows_per_file=2)

    assert len(sample) == 4
    assert set(sample["source_file"]) == {"part1.csv", "part2.csv"}
    assert set(sample["attack_category"]) == {
        "Benign",
        "DDoS",
        "Web-based",
        "Spoofing",
    }


def test_load_csv_sample_fails_loud_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No CICIoT2023 CSV files"):
        C.load_csv_sample(tmp_path / "missing")


def test_phase1_summary_counts_labels_and_categories():
    df = pd.DataFrame(
        {
            "flow_duration": [1, 2, 3, 4],
            "Label": ["BenignTraffic", "DDoS-UDP_Flood", "DDoS-TCP_Flood", "XSS"],
        }
    )

    summary = C.phase1_summary(df)

    assert summary["rows"] == 4
    assert summary["n_fine_labels"] == 4
    assert summary["category_counts"]["Benign"] == 1
    assert summary["category_counts"]["DDoS"] == 2
    assert summary["category_counts"]["Web-based"] == 1
