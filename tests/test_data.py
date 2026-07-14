"""Tests for src/data.py — schema integrity and the attack-family label map."""

from __future__ import annotations

import pandas as pd
import pytest

import data as D
from conftest import needs_data


# --- pure-logic tests (no downloaded data needed) --------------------------- #
def test_column_schema_sizes():
    assert len(D.FEATURE_NAMES) == 41
    assert len(D.COLUMN_NAMES) == 43  # 41 features + label + difficulty
    assert D.COLUMN_NAMES[-2:] == ["label", "difficulty"]


def test_feature_groupings_are_consistent():
    # categorical + numeric partition the 41 features, no overlap, no stragglers.
    assert set(D.CATEGORICAL_COLS).isdisjoint(D.NUMERIC_COLS)
    assert set(D.CATEGORICAL_COLS) | set(D.NUMERIC_COLS) == set(D.FEATURE_NAMES)
    assert set(D.BINARY_COLS).issubset(D.NUMERIC_COLS)


def test_family_map_values_are_known_families():
    assert set(D.ATTACK_FAMILY_MAP.values()) == set(D.FAMILY_ORDER)


def test_add_label_columns_maps_and_derives():
    df = pd.DataFrame({"label": ["normal", "neptune", "satan", "guess_passwd",
                                 "buffer_overflow"]})
    out = D.add_label_columns(df)
    assert out["attack_family"].tolist() == ["normal", "DoS", "Probe", "R2L", "U2R"]
    assert out["binary_label"].tolist() == [0, 1, 1, 1, 1]  # 1 = attack


def test_add_label_columns_raises_on_unmapped_label():
    df = pd.DataFrame({"label": ["normal", "totally_new_attack"]})
    with pytest.raises(KeyError):
        D.add_label_columns(df)


# --- data-dependent tests --------------------------------------------------- #
@needs_data
@pytest.mark.parametrize("split,rows", [("train", 125_973), ("test", 22_544),
                                        ("test-21", 11_850), ("train-20", 25_192)])
def test_official_row_counts(split, rows):
    assert len(D.load_nsl_kdd(split)) == rows


@needs_data
def test_all_labels_in_both_splits_are_mapped():
    # The whole point of the fail-loud map: every label in train AND test resolves.
    for split in ("train", "test"):
        df = D.load_nsl_kdd(split)  # would raise if any label were unmapped
        assert df["attack_family"].notna().all()
