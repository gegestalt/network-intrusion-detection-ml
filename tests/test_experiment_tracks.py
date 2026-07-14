"""Fast tests for the newer experimental-lab tracks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import anomaly_detection as A
import ciciot2023_raw_audit as CRAW
import dataset_catalog as DC
import online_learning as OL
import semi_supervised as SS
import soc_simulation as SOC
import threshold_ablation as TH


def test_threshold_predictions_and_fp_per_10k():
    scores = np.array([0.1, 0.7, 0.8, 0.2])
    pred = TH.predictions_from_threshold(scores, 0.5)
    assert pred.tolist() == [0, 1, 1, 0]

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    assert TH.false_positives_per_10k_benign(y_true, y_pred) == pytest.approx(5000.0)


def test_anomaly_quantile_threshold_flags_high_scores():
    train_scores = np.array([1.0, 2.0, 3.0, 4.0])
    test_scores = np.array([1.5, 3.5, 5.0])
    pred, threshold = A.scores_to_predictions(train_scores, test_scores, 0.75)

    assert threshold == pytest.approx(3.25)
    assert pred.tolist() == [0, 1, 1]


def test_family_recalls_treat_normal_as_normal_recall():
    y_true = np.array([0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    fam = pd.Series(["normal", "DoS", "R2L", "U2R"])

    recalls = A.family_recalls(y_true, y_pred, fam)

    assert recalls["normal"] == pytest.approx(1.0)
    assert recalls["DoS"] == pytest.approx(1.0)
    assert recalls["R2L"] == pytest.approx(0.0)
    assert recalls["U2R"] == pytest.approx(1.0)


def test_label_budget_indices_are_stratified_and_deterministic():
    y = np.array([0] * 100 + [1] * 50)
    a = SS.label_budget_indices(y, 0.10, seed=7)
    b = SS.label_budget_indices(y, 0.10, seed=7)

    assert np.array_equal(a, b)
    assert (y[a] == 0).sum() == 10
    assert (y[a] == 1).sum() == 5


def test_label_budget_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="fraction"):
        SS.label_budget_indices(np.array([0, 1]), 0)


def test_soc_simulation_translates_rates_to_daily_alerts():
    scenario = SOC.SocScenario(daily_flows=1_000_000, malicious_rate=0.005)
    out = SOC.simulate_row(normal_recall=0.99, attack_recall=0.80, scenario=scenario)

    assert out["false_alerts_per_day"] == pytest.approx(9_950)
    assert out["true_alerts_per_day"] == pytest.approx(4_000)
    assert out["missed_attacks_per_day"] == pytest.approx(1_000)


def test_ciciot_raw_blocked_report_names_missing_csvs():
    text = CRAW.render_blocked()

    assert "blocked" in text.lower()
    assert "data/ciciot2023/CSV" in text


def test_dataset_catalog_has_blocked_ton_and_cse_entries(tmp_path):
    assert not DC.has_files(tmp_path, "*.csv")
    (tmp_path / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert DC.has_files(tmp_path, "*.csv")

    names = {entry.name: entry for entry in DC.build_entries()}
    assert "TON_IoT" in names
    assert "CSE-CIC-IDS2018" in names
    assert names["TON_IoT"].current_status in {"available", "blocked_missing_local_files"}


def test_online_chunks_cover_all_rows():
    chunks = OL.iter_chunks(25, 10)

    assert chunks == [slice(0, 10), slice(10, 20), slice(20, 25)]
    covered = []
    for s in chunks:
        covered.extend(range(s.start, s.stop))
    assert covered == list(range(25))
