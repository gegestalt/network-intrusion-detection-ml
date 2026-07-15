"""Tests for the NSL-KDD attribute-by-attribute comparison layer."""

from __future__ import annotations

import pandas as pd

import attribute_comparison as A
from conftest import needs_data


def test_encoded_features_map_back_to_raw_attributes():
    assert A.encoded_to_attribute("service_http") == "service"
    assert A.encoded_to_attribute("flag_SF") == "flag"
    assert A.encoded_to_attribute("protocol_type_tcp") == "protocol_type"
    assert A.encoded_to_attribute("same_srv_rate") == "same_srv_rate"


def test_attribute_flags_and_recommendations_are_actionable():
    row = pd.Series(
        {
            "train_missing_rate": 0.0,
            "test_missing_rate": 0.0,
            "raw_type": "numeric",
            "unseen_test_categories": 0,
            "train_unique": 1,
            "train_test_shift_std": 0.0,
            "outlier_rate_max": 0.0,
            "redundant_pair_count": 0,
            "target_corr_max_abs": 0.0,
            "class_separation": 0.0,
            "importance_consensus": 0.0,
        }
    )

    flags = A.attribute_flags(row)
    row["audit_flags"] = flags

    assert "constant_or_near_constant" in flags
    assert "Drop or keep only for schema compatibility" in A.attribute_recommendation(row)


@needs_data
def test_attribute_audit_has_raw_and_encoded_coverage():
    attr, encoded = A.attribute_audit()

    assert len(attr) == 41
    assert len(encoded) == 122
    assert set(A.ATTR_COLUMNS) == set(attr.columns)
    assert attr["rank_by_consensus"].min() == 1
    assert attr["attribute"].is_unique
    assert {"service", "flag", "src_bytes"} <= set(attr["attribute"])
