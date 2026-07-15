"""Tests for CICIoT2023 Phase-1 helpers.

These are synthetic and fast; the full CICIoT2023 CSV release is not required
for CI or local smoke tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ciciot2023 as C

_HAS_PARQUET = (C.DATA_DIR / C.PARQUET_SPLITS["train"]).exists()
needs_parquet = pytest.mark.skipif(
    not _HAS_PARQUET,
    reason="CICIoT2023 parquet dev sample not present (see data/ciciot2023/SOURCE.md)",
)


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


# --- parquet dev sample: canonical loader + scale-only preprocessing --------- #
@needs_parquet
def test_load_parquet_canonical_label_columns():
    df = C.load_parquet("test")
    for col in ("label", "attack_category", "binary_label"):
        assert col in df.columns
    feats = C.feature_columns(df)
    # label/provenance columns must not leak into features
    assert not ({"label", "Label", "attack_class", "attack_category",
                 "binary_label"} & set(feats))
    assert len(feats) == 39  # all-numeric CICIoT2023 features


@needs_parquet
def test_every_parquet_fine_label_maps_to_a_category():
    # The schema/leakage guard: no fine label may be silently unmapped.
    df = C.load_parquet("train")
    assert df["attack_category"].notna().all()
    assert set(df["attack_category"]).issubset(set(C.CATEGORY_ORDER))


@needs_parquet
@pytest.mark.parametrize("level,n_classes", [("binary", 2), ("category", 8)])
def test_prepare_shapes_labels_and_no_leakage(level, n_classes):
    tr, te = C.load_parquet("train"), C.load_parquet("test")
    p = C.prepare(tr, te, level=level)
    assert p.X_train.shape[1] == 39 == len(p.feature_names)
    assert len(p.classes) == n_classes
    assert set(np.unique(p.y_train)).issubset(range(n_classes))
    assert not np.isnan(p.X_train).any() and not np.isnan(p.X_test).any()
    # NO LEAKAGE: scaler statistics come from TRAIN only
    feats = p.feature_names
    train_means = (tr[feats].replace([np.inf, -np.inf], np.nan)
                   .fillna(tr[feats].median()).mean().to_numpy())
    assert np.allclose(p.scaler.mean_, train_means, rtol=1e-6, atol=1e-6)


@needs_parquet
def test_prepare_binary_is_benign_zero_attack_one():
    tr, te = C.load_parquet("train"), C.load_parquet("test")
    p = C.prepare(tr, te, level="binary")
    # benign rows -> 0
    benign_mask = (tr["attack_category"] == "Benign").to_numpy()
    assert set(p.y_train[benign_mask]) == {0}


def test_prepare_rejects_unknown_level():
    df = pd.DataFrame({"f": [1.0, 2.0], "label": ["BenignTraffic", "DDoS-UDP_Flood"],
                       "attack_category": ["Benign", "DDoS"], "binary_label": [0, 1]})
    with pytest.raises(ValueError, match="level must be one of"):
        C.prepare(df, df, level="nonsense")
