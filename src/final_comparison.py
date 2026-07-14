"""Consolidated comparison report across implemented experiment tracks.

The individual experiment scripts answer local questions. This script pulls
their saved artifacts into one comparison layer so the project can answer:

* Which track is best under macro-F1?
* Which track gives highest attack recall?
* Which track has the lowest operational alert burden?
* Which experiment families are complete, partial, blocked, or missing?

Run:
    .venv/bin/python src/final_comparison.py
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

import data as D

RESULTS = D.REPO_ROOT / "results"
DOCS = D.REPO_ROOT / "docs"
OUT_CSV = RESULTS / "final_comparison.csv"
COVERAGE_CSV = RESULTS / "experiment_coverage_matrix.csv"
REPORT_PATH = RESULTS / "final_comparison.md"


COMMON_COLUMNS = [
    "dataset",
    "task",
    "track",
    "method",
    "variant",
    "comparison_family",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "attack_recall",
    "normal_recall",
    "rare_recall",
    "mcc",
    "features",
    "false_alerts_per_day",
    "total_alerts_per_day",
    "missed_attacks_per_day",
    "operational_precision",
    "source_file",
    "limitation",
]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMON_COLUMNS)


def _coerce_common(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty_frame()
    out = pd.DataFrame(rows)
    for col in COMMON_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[COMMON_COLUMNS]


def _float(value: object) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return np.nan


def _extract_first_float(text: object) -> float:
    hit = re.search(r"[-+]?\d*\.?\d+", str(text))
    return float(hit.group(0)) if hit else np.nan


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def parse_reference_track() -> pd.DataFrame:
    """Parse binary/multiclass reference-track markdown tables."""
    path = RESULTS / "reference_track.md"
    if not path.exists():
        return _empty_frame()
    rows: list[dict[str, object]] = []
    scheme = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            scheme = line.replace("## ", "").strip()
        if not line.startswith("|") or "---" in line or "Model" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 4 or scheme is None:
            continue
        rare = cells[3]
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": scheme,
                "track": "supervised_reference",
                "method": cells[0],
                "variant": "balanced where supported",
                "comparison_family": "classical_ml",
                "accuracy": _float(cells[1]),
                "macro_f1": _float(cells[2]),
                "attack_recall": _extract_first_float(rare) if scheme == "binary" else np.nan,
                "rare_recall": rare,
                "source_file": "results/reference_track.md",
                "limitation": "single seed for several reference models",
            }
        )
    return _coerce_common(rows)


def parse_phase3_and_mlp_metrics() -> pd.DataFrame:
    """Parse Phase 3 and Phase 4 summary rows from metrics.md."""
    path = RESULTS / "metrics.md"
    if not path.exists():
        return _empty_frame()
    rows: list[dict[str, object]] = []
    mode = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Model | Task | Test set |"):
            mode = "phase3"
            continue
        if line.startswith("## Per-class"):
            mode = None
            continue
        if line.startswith("| Variant | Task |"):
            mode = "phase4"
            continue
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if mode == "phase3" and len(cells) >= 9:
            rows.append(
                {
                    "dataset": "nsl_kdd",
                    "task": cells[1],
                    "track": "phase3_tree_boosting",
                    "method": cells[0],
                    "variant": cells[2],
                    "comparison_family": "tree_boosting",
                    "accuracy": _float(cells[3]),
                    "macro_f1": _float(cells[4]),
                    "weighted_f1": _float(cells[5]),
                    "source_file": "results/metrics.md",
                    "limitation": "only RF/LightGBM in this artifact",
                }
            )
        elif mode == "phase4" and len(cells) >= 5:
            rows.append(
                {
                    "dataset": "nsl_kdd",
                    "task": cells[1],
                    "track": "phase4_mlp_weighting",
                    "method": cells[0],
                    "variant": "weighted/unweighted",
                    "comparison_family": "neural_mlp",
                    "accuracy": _float(cells[2]),
                    "macro_f1": _float(cells[3]),
                    "attack_recall": _extract_first_float(cells[4]) if cells[1] == "binary" else np.nan,
                    "rare_recall": cells[4],
                    "source_file": "results/metrics.md",
                    "limitation": "single run; stability reported separately",
                }
            )
    return _coerce_common(rows)


def load_threshold_ablation() -> pd.DataFrame:
    df = _read_csv(RESULTS / "threshold_ablation.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": "binary",
                "track": "threshold_tuning",
                "method": r["model"],
                "variant": r["threshold_name"],
                "comparison_family": "threshold_policy",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "source_file": "results/threshold_ablation.csv",
                "limitation": "threshold selected on one validation split",
            }
        )
    return _coerce_common(rows)


def load_feature_learning() -> pd.DataFrame:
    df = _read_csv(RESULTS / "feature_learning_results.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": r["dataset"],
                "task": "binary",
                "track": "feature_learning",
                "method": "LogisticRegression",
                "variant": r["representation"],
                "comparison_family": "feature_representation",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "mcc": r["mcc"],
                "features": r["n_features"],
                "source_file": "results/feature_learning_results.csv",
                "limitation": "CICIoT2023 rows are dev-sample only when dataset=ciciot2023_dev",
            }
        )
    return _coerce_common(rows)


def load_neural_ablation() -> pd.DataFrame:
    df = _read_csv(RESULTS / "neural_ablation.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": "binary",
                "track": "neural_ablation",
                "method": "MLP",
                "variant": r["config"],
                "comparison_family": "neural_mlp_ablation",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "mcc": r["mcc"],
                "features": r["param_count"],
                "source_file": "results/neural_ablation.csv",
                "limitation": "bounded train subset; single seed",
            }
        )
    return _coerce_common(rows)


def load_anomaly() -> pd.DataFrame:
    df = _read_csv(RESULTS / "anomaly_detection.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": "binary",
                "track": "normal_only_anomaly",
                "method": r["model"],
                "variant": f"q={r['normal_quantile']:.2f}",
                "comparison_family": "unsupervised_anomaly",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "rare_recall": f"DoS {r['DoS_recall']:.3f}; Probe {r['Probe_recall']:.3f}; R2L {r['R2L_recall']:.3f}; U2R {r['U2R_recall']:.3f}",
                "source_file": "results/anomaly_detection.csv",
                "limitation": "normal-only training; threshold from normal quantiles",
            }
        )
    return _coerce_common(rows)


def load_semi_supervised() -> pd.DataFrame:
    df = _read_csv(RESULTS / "semi_supervised.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": "binary",
                "track": "semi_supervised",
                "method": r["method"],
                "variant": f"{r['label_fraction']:.2f} labels",
                "comparison_family": "label_budget",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "features": r["labelled_rows"],
                "source_file": "results/semi_supervised.csv",
                "limitation": "binary only; self-training only",
            }
        )
    return _coerce_common(rows)


def load_online() -> pd.DataFrame:
    df = _read_csv(RESULTS / "online_learning.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": "binary",
                "track": "online_learning_proxy",
                "method": r["model"],
                "variant": f"chunk={int(r['chunk_size'])}",
                "comparison_family": "online_partial_fit",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "source_file": "results/online_learning.csv",
                "limitation": "NSL-KDD file order is not chronological drift",
            }
        )
    return _coerce_common(rows)


def load_soc() -> pd.DataFrame:
    df = _read_csv(RESULTS / "soc_simulation.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": "binary",
                "track": "soc_simulation",
                "method": r["model"],
                "variant": r["threshold_name"],
                "comparison_family": "operational_policy",
                "macro_f1": r["macro_f1"],
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "false_alerts_per_day": r["false_alerts_per_day"],
                "total_alerts_per_day": r["total_alerts_per_day"],
                "missed_attacks_per_day": r["missed_attacks_per_day"],
                "operational_precision": r["operational_precision"],
                "source_file": "results/soc_simulation.csv",
                "limitation": "fixed scenario: 1M flows/day and 0.5% malicious",
            }
        )
    return _coerce_common(rows)


def load_nsl_kdd_learning_lab() -> pd.DataFrame:
    df = _read_csv(RESULTS / "nsl_kdd_learning_lab.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": "binary",
                "track": "first_dataset_learning_lab",
                "method": r["method"],
                "variant": r["variant"],
                "comparison_family": r["family"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r.get("weighted_f1", np.nan),
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "mcc": r.get("mcc", np.nan),
                "source_file": "results/nsl_kdd_learning_lab.csv",
                "limitation": "inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation",
            }
        )
    return _coerce_common(rows)


def load_nsl_kdd_feature_groups() -> pd.DataFrame:
    df = _read_csv(RESULTS / "nsl_kdd_feature_group_ablation.csv")
    if df.empty:
        return _empty_frame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "nsl_kdd",
                "task": "binary",
                "track": "first_dataset_feature_ablation",
                "method": r["method"],
                "variant": r["variant"],
                "comparison_family": "feature_group_ablation",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r.get("weighted_f1", np.nan),
                "attack_recall": r["attack_recall"],
                "normal_recall": r["normal_recall"],
                "mcc": r.get("mcc", np.nan),
                "features": r["encoded_feature_count"],
                "source_file": "results/nsl_kdd_feature_group_ablation.csv",
                "limitation": "feature groups tested with balanced Logistic Regression only",
            }
        )
    return _coerce_common(rows)


def build_comparison() -> pd.DataFrame:
    frames = [
        parse_reference_track(),
        parse_phase3_and_mlp_metrics(),
        load_nsl_kdd_learning_lab(),
        load_nsl_kdd_feature_groups(),
        load_threshold_ablation(),
        load_feature_learning(),
        load_neural_ablation(),
        load_anomaly(),
        load_semi_supervised(),
        load_online(),
        load_soc(),
    ]
    out = pd.concat(frames, ignore_index=True)
    numeric = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "attack_recall",
        "normal_recall",
        "mcc",
        "features",
        "false_alerts_per_day",
        "total_alerts_per_day",
        "missed_attacks_per_day",
        "operational_precision",
    ]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_coverage_matrix() -> pd.DataFrame:
    rows = [
        ("Raw schema and preprocessing", "PROVEN", "NSL-KDD", "src/data.py, src/preprocess.py", "CICIoT raw CSV absent"),
        ("Classical supervised reference", "PROVEN", "NSL-KDD", "results/reference_track.md", "single seed for some models"),
        ("Tree/boosting official split", "PROVEN", "NSL-KDD", "results/metrics.md", "RF/LightGBM only"),
        ("MLP weighted/unweighted", "PROVEN", "NSL-KDD", "results/metrics.md, results/stability.md", "more ablations now separate"),
        ("Feature engineering/learning", "PROVEN", "NSL-KDD, CICIoT2023 dev", "results/feature_learning.md", "not full raw CICIoT"),
        ("Attribute-by-attribute audit", "PROVEN", "NSL-KDD", "results/attribute_comparison_nsl_kdd.md", "binary target attribution only"),
        ("Row-level outlier datapoints", "PROVEN", "NSL-KDD", "results/nsl_kdd_outlier_datapoints.md", "numeric IQR outliers only; outlier is not automatically attack"),
        ("First-dataset learning lab", "PROVEN", "NSL-KDD", "results/nsl_kdd_learning_lab.md", "CNN/pooling intentionally not applied to isolated tabular rows"),
        ("Neural foundations/ablations", "PROVEN", "NSL-KDD", "results/neural_foundations.md, results/neural_ablation.md", "bounded single-seed MLP ablations"),
        ("Threshold tuning", "PROVEN", "NSL-KDD", "results/threshold_ablation.md", "binary only"),
        ("SOC workload simulation", "PROVEN", "NSL-KDD", "results/soc_simulation.md", "one traffic scenario"),
        ("Normal-only anomaly", "PROVEN", "NSL-KDD", "results/anomaly_detection.md", "not modern zero-day split"),
        ("Semi-supervised label budget", "PROVEN", "NSL-KDD", "results/semi_supervised.md", "binary and self-training only"),
        ("Online learning", "PARTIAL", "NSL-KDD", "results/online_learning.md", "not true chronological drift"),
        ("CICIoT2023 dev", "PARTIAL", "CICIoT2023", "results/ciciot2023_quality.md", "random dev sample only"),
        ("CICIoT2023 raw CSV", "BLOCKED", "CICIoT2023", "results/ciciot2023_raw_audit.md", "data/ciciot2023/CSV is empty"),
        ("TON_IoT", "BLOCKED", "TON_IoT", "docs/datasets/catalog.md", "local files absent"),
        ("CSE-CIC-IDS2018", "BLOCKED", "CSE-CIC-IDS2018", "docs/datasets/catalog.md", "local files absent"),
        ("Temporal CNN/RNN/LSTM/GRU", "MISSING", "needs timestamped/sequence data", "docs/deep_learning_taxonomy.md", "invalid on isolated NSL-KDD rows"),
        ("Graph ML", "MISSING", "needs IP/device graph data", "docs/audits/experimental_lab_prompt_audit.md", "no graph representation"),
        ("Calibration/explainability", "PARTIAL", "NSL-KDD", "feature importance exists", "Brier/ECE/SHAP not done"),
    ]
    return pd.DataFrame(rows, columns=["experiment_area", "status", "dataset_scope", "evidence", "limitation"])


def _top(df: pd.DataFrame, sort_col: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    subset = df.dropna(subset=[sort_col]).copy()
    return subset.sort_values(sort_col, ascending=ascending).head(n)


def render_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if df.empty:
        return ["_No rows available._", ""]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for col in cols:
            value = r[col]
            if isinstance(value, float):
                if np.isnan(value):
                    cells.append("")
                elif abs(value) >= 100:
                    cells.append(f"{value:,.0f}")
                else:
                    cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_report(comparison: pd.DataFrame, coverage: pd.DataFrame) -> str:
    binary = comparison[(comparison["dataset"] == "nsl_kdd") & (comparison["task"] == "binary")]
    macro = _top(binary, "macro_f1", n=12)
    recall = _top(binary, "attack_recall", n=12)
    operational = _top(
        comparison[comparison["track"] == "soc_simulation"],
        "total_alerts_per_day",
        n=12,
        ascending=True,
    )
    by_family = (
        binary.dropna(subset=["macro_f1"])
        .sort_values("macro_f1", ascending=False)
        .groupby("comparison_family")
        .head(1)
        .sort_values("macro_f1", ascending=False)
    )
    lines = [
        "# Final Consolidated Comparison",
        "",
        "This report consolidates the saved experiment artifacts. It is meant to stop "
        "the project from becoming a folder of unrelated tables.",
        "",
        "## Coverage Matrix",
        "",
    ]
    lines += render_table(coverage, ["experiment_area", "status", "dataset_scope", "evidence", "limitation"])
    lines += [
        "## Best NSL-KDD Binary Rows By Macro-F1",
        "",
    ]
    lines += render_table(
        macro,
        ["track", "method", "variant", "comparison_family", "macro_f1", "attack_recall", "normal_recall", "limitation"],
    )
    lines += [
        "## Best NSL-KDD Binary Rows By Attack Recall",
        "",
    ]
    lines += render_table(
        recall,
        ["track", "method", "variant", "comparison_family", "attack_recall", "macro_f1", "normal_recall", "limitation"],
    )
    lines += [
        "## Best Row Per Experiment Family",
        "",
    ]
    lines += render_table(
        by_family,
        ["comparison_family", "track", "method", "variant", "macro_f1", "attack_recall", "limitation"],
    )
    lines += [
        "## Lowest Operational Alert Burden",
        "",
    ]
    lines += render_table(
        operational,
        [
            "method",
            "variant",
            "macro_f1",
            "attack_recall",
            "false_alerts_per_day",
            "missed_attacks_per_day",
            "total_alerts_per_day",
            "operational_precision",
        ],
    )
    lines += [
        "## Referee Interpretation",
        "",
        "- The highest macro-F1 rows are not necessarily the lowest-alert operational rows.",
        "- Threshold and SOC rows answer a different question than classifier macro-F1 rows.",
        "- Anomaly detection is valuable as a different learning paradigm, not because it automatically beats supervised learning.",
        "- Feature-learning and neural-ablation rows are bounded experiments; they need seed stability before strong ranking claims.",
        "- CICIoT2023 dev rows are useful but cannot be sold as full raw CICIoT2023 results.",
        "",
    ]
    return "\n".join(lines)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison()
    coverage = build_coverage_matrix()
    comparison.to_csv(OUT_CSV, index=False)
    coverage.to_csv(COVERAGE_CSV, index=False)
    REPORT_PATH.write_text(render_report(comparison, coverage), encoding="utf-8")
    return comparison, coverage


def main() -> int:
    comparison, coverage = run()
    print(f"comparison rows: {len(comparison)}")
    print(f"coverage rows: {len(coverage)}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {COVERAGE_CSV}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
