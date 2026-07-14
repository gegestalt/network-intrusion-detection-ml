"""Tests for row-level outlier datapoint mapping."""

from __future__ import annotations

import pandas as pd

import data as D
import outlier_datapoints as O


def _base_frame(duration_values: list[float]) -> pd.DataFrame:
    rows = []
    for duration in duration_values:
        row = {name: 0 for name in D.FEATURE_NAMES}
        row.update({"protocol_type": "tcp", "service": "http", "flag": "SF"})
        row["duration"] = duration
        row["label"] = "normal"
        row["difficulty"] = 0
        rows.append(row)
    return D.add_label_columns(pd.DataFrame(rows))


def test_score_split_identifies_actual_outlier_row_and_feature():
    train = _base_frame([0, 0, 0, 0])
    test = _base_frame([0, 10])
    ref = O.iqr_reference(train)

    rows, long_df = O.score_split(test, "test", ref)
    outlier_row = rows.loc[rows["row_index"] == 1].iloc[0]

    assert outlier_row["outlier_feature_count"] >= 1
    assert "duration" in outlier_row["top_outlier_features"]
    assert {"split", "row_index", "feature", "abs_robust_z"} <= set(long_df.columns)
    assert "duration" in set(long_df["feature"])


def test_feature_frequency_counts_row_feature_hits():
    long_df = pd.DataFrame(
        [
            {"split": "test", "row_index": 0, "feature": "duration", "abs_robust_z": 2.0},
            {"split": "test", "row_index": 1, "feature": "duration", "abs_robust_z": 3.0},
            {"split": "test", "row_index": 1, "feature": "src_bytes", "abs_robust_z": 4.0},
        ]
    )

    freq = O.feature_frequency(long_df)
    duration = freq[freq["feature"] == "duration"].iloc[0]

    assert duration["outlier_rows"] == 2
    assert duration["mean_abs_robust_z"] == 2.5
