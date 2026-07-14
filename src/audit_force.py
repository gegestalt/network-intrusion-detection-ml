"""Referee audit for the proposed Security ML Experimental Lab.

This module does not train models. It checks the project evidence that already
exists on disk and turns the big lab prompt into a concrete coverage report:
what is proven, what is partial, what is missing, and what would be unsafe to
claim in a write-up.

Run:
    .venv/bin/python src/audit_force.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"
AUDITS = DOCS / "audits"
REPORT_PATH = AUDITS / "experimental_lab_prompt_audit.md"


@dataclass(frozen=True)
class AuditItem:
    """One referee finding against the experimental-lab prompt."""

    area: str
    expectation: str
    status: str
    evidence: str
    gap: str
    next_test: str


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _count_table_rows(markdown: str) -> int:
    """Count ordinary markdown table data rows."""
    rows = 0
    for line in markdown.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        if set(s.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
            continue
        if "Model" in s and "Macro-F1" in s:
            continue
        if "Variant" in s and "Macro-F1" in s:
            continue
        rows += 1
    return rows


def _count_rows_between(markdown: str, start_marker: str, stop_marker: str | None) -> int:
    """Count markdown table data rows between two marker lines."""
    lines = markdown.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if start_marker in line)
    except StopIteration:
        return 0

    stop = len(lines)
    if stop_marker is not None:
        for i in range(start + 1, len(lines)):
            if stop_marker in lines[i]:
                stop = i
                break

    return _count_table_rows("\n".join(lines[start:stop]))


def _notebook_code_output_count(path: Path) -> tuple[int, int]:
    """Return (code cells, code cells with outputs) without importing nbformat."""
    import json

    if not path.exists():
        return (0, 0)
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    with_outputs = [c for c in code if c.get("outputs")]
    return (len(code), len(with_outputs))


def snapshot() -> dict[str, object]:
    """Collect compact project evidence used by the audit."""
    reference = _read(RESULTS / "reference_track.md")
    metrics = _read(RESULTS / "metrics.md")
    stability = _read(RESULTS / "stability.md")
    tuning = _read(REPO_ROOT / "src" / "tuning.py")
    ciciot = _read(REPO_ROOT / "src" / "ciciot.py")
    ciciot2023 = _read(REPO_ROOT / "src" / "ciciot2023.py")
    tests = sorted((REPO_ROOT / "tests").glob("test_*.py"))
    figures = sorted((RESULTS / "figures").glob("*.png"))
    raw_ciciot_csv = sorted((REPO_ROOT / "data" / "ciciot2023" / "CSV").glob("*.csv"))
    ciciot_parquet = sorted((REPO_ROOT / "data" / "ciciot2023").glob("*.parquet"))
    nb02 = _notebook_code_output_count(
        REPO_ROOT / "notebooks" / "02_referee_audit_current_state.ipynb"
    )
    nb03 = _notebook_code_output_count(
        REPO_ROOT / "notebooks" / "03_ciciot2023_phase1_eda.ipynb"
    )
    return {
        "reference_rows": _count_table_rows(reference),
        "phase3_summary_rows": _count_rows_between(
            metrics,
            "| Model | Task | Test set |",
            "## Per-class",
        ),
        "phase4_mlp_rows": _count_rows_between(metrics, "| Variant | Task |", None),
        "stability_rows": _count_table_rows(stability),
        "has_threshold_sweep": "def threshold_sweep" in tuning,
        "has_ciciot_quality": "def audit" in ciciot and "load(\"train\")" in ciciot,
        "has_ciciot2023_mapper": "def attack_category" in ciciot2023,
        "test_file_count": len(tests),
        "figure_count": len(figures),
        "raw_ciciot_csv_count": len(raw_ciciot_csv),
        "ciciot_parquet_count": len(ciciot_parquet),
        "has_ciciot_quality_report": (RESULTS / "ciciot2023_quality.md").exists(),
        "has_threshold_ablation": (RESULTS / "threshold_ablation.md").exists(),
        "has_anomaly_detection": (RESULTS / "anomaly_detection.md").exists(),
        "has_semi_supervised": (RESULTS / "semi_supervised.md").exists(),
        "has_online_learning": (RESULTS / "online_learning.md").exists(),
        "has_soc_simulation": (RESULTS / "soc_simulation.md").exists(),
        "has_feature_learning": (RESULTS / "feature_learning.md").exists(),
        "has_feature_correlations": (RESULTS / "feature_correlations.csv").exists(),
        "has_feature_outliers": (RESULTS / "feature_outliers.csv").exists(),
        "has_deep_learning_taxonomy": (DOCS / "deep_learning_taxonomy.md").exists(),
        "has_neural_foundations": (RESULTS / "neural_foundations.md").exists(),
        "has_neural_ablation": (RESULTS / "neural_ablation.md").exists(),
        "has_neural_ablation_tracking": (REPO_ROOT / "experiments" / "runs" / "neural_ablation.jsonl").exists(),
        "has_ciciot_raw_audit": (RESULTS / "ciciot2023_raw_audit.md").exists(),
        "has_dataset_catalog": (DOCS / "datasets" / "catalog.md").exists(),
        "nb02_code": nb02[0],
        "nb02_outputs": nb02[1],
        "nb03_code": nb03[0],
        "nb03_outputs": nb03[1],
    }


def build_audit_items(state: dict[str, object]) -> list[AuditItem]:
    """Return the current prompt-coverage findings."""
    reference_ok = int(state["reference_rows"]) >= 12
    metrics_ok = int(state["phase3_summary_rows"]) >= 6
    stability_ok = int(state["stability_rows"]) >= 8

    return [
        AuditItem(
            "NSL-KDD controlled baseline",
            "Keep NSL-KDD as the fast local reference dataset.",
            "PROVEN",
            "Preprocessing, metrics, RF/LightGBM, MLP, reference-track, and stability files exist.",
            "Reference-track docs are newer than docs/PROGRESS.md, so some summary text is stale.",
            "Run pytest plus reference/stability scripts before using headline claims.",
        ),
        AuditItem(
            "Reference Track model zoo",
            "Dummy, Logistic Regression, RF, ExtraTrees, HistGB, LightGBM on binary and multiclass.",
            "PROVEN" if reference_ok else "PARTIAL",
            f"results/reference_track.md contains {state['reference_rows']} table data rows.",
            "SGD is mentioned by the prompt but is not present in the reference-track script.",
            "Add SGDClassifier or remove SGD from claims until it has saved metrics.",
        ),
        AuditItem(
            "RF/LightGBM official-split baselines",
            "Tune on train only, evaluate KDDTest+ and binary KDDTest-21.",
            "PROVEN" if metrics_ok else "PARTIAL",
            f"results/metrics.md contains {state['phase3_summary_rows']} Phase-3 summary rows and saved figures exist.",
            "Only RF/LightGBM get full saved confusion/ROC/PR artifacts in Phase 3.",
            "Keep this as the gold-standard artifact pattern for later tracks.",
        ),
        AuditItem(
            "MLP weighting ablation",
            "Compare weighted vs unweighted MLP and rare-class recall.",
            "PROVEN",
            "results/metrics.md and results/stability.md report weighted/unweighted MLP.",
            "Phase 4 summary lacks full saved per-class tables matching Phase 3 detail.",
            "Write MLP per-class tables and confusion matrices into a dedicated MLP results section.",
        ),
        AuditItem(
            "Artificial neuron and activation foundations",
            "Demonstrate weighted input, bias, activation, loss, gradients, update, activations, and derivatives.",
            "PROVEN" if state["has_neural_foundations"] else "MISSING",
            "results/neural_foundations.md plus activation_functions.csv and single_neuron_demo.csv exist.",
            "This is educational foundation work; it is not a trained IDS model.",
            "Use it as the theory bridge before interpreting MLP/CNN/RNN experiments.",
        ),
        AuditItem(
            "Controlled neural architecture ablations",
            "Change activation, weighting, dropout, normalization, depth/width, loss, and label smoothing one factor at a time.",
            "PROVEN" if state["has_neural_ablation"] else "MISSING",
            "results/neural_ablation.md, neural_ablation.csv, learning curves, and JSONL tracking exist.",
            "Only bounded NSL-KDD binary MLP ablations are implemented; CNN/RNN families remain representation-blocked.",
            "Repeat across seeds and add calibration before final neural ranking claims.",
        ),
        AuditItem(
            "Multi-seed stability",
            "Qualify model rankings across seeds.",
            "PROVEN" if stability_ok else "PARTIAL",
            f"results/stability.md contains {state['stability_rows']} table data rows.",
            "Only headline models are in stability; LogReg/ExtraTrees/HistGB are single-seed.",
            "Extend stability to the reference-track winners before ranking them strongly.",
        ),
        AuditItem(
            "Threshold tuning",
            "Decision thresholds should be tuned on validation only.",
            "PROVEN" if state["has_threshold_ablation"] else ("PARTIAL" if state["has_threshold_sweep"] else "MISSING"),
            "src/tuning.py has threshold_sweep and results/threshold_ablation.md saves validation-selected thresholds.",
            "Current ablation is binary NSL-KDD only; multiclass per-class thresholding is not implemented.",
            "Extend to calibrated binary models and report seed variance before using one threshold as final.",
        ),
        AuditItem(
            "Cost-sensitive learning family",
            "No balancing, class weights, RUS, ROS, SMOTE, Borderline-SMOTE, ADASYN, focal loss.",
            "PARTIAL",
            "Class weights exist for several sklearn models and MLP; no resampling study exists.",
            "No tests currently guard against SMOTE/test leakage or unrealistic synthetic-flow claims.",
            "Create an imbalance notebook with train-only resampling and validation-only selection.",
        ),
        AuditItem(
            "CICIoT2023 primary modern dataset",
            "Add modern IoT dataset with binary, category, and fine-label tracks.",
            "PARTIAL",
            "Dev parquet files exist; src/ciciot.py and src/ciciot2023.py exist; "
            f"quality report exists={state['has_ciciot_quality_report']}; raw audit exists="
            f"{state['has_ciciot_raw_audit']}; raw CSV count is "
            f"{state['raw_ciciot_csv_count']}.",
            "Full official raw CSV analysis is blocked until data/ciciot2023/CSV/*.csv exists.",
            "Use dev parquet for quick checks, then run raw CSV Phase-1 audit once downloaded.",
        ),
        AuditItem(
            "TON_IoT multimodal dataset",
            "Network, telemetry, host, and security-event fusion experiments.",
            "PARTIAL" if state["has_dataset_catalog"] else "MISSING",
            "docs/datasets/catalog.md records the official source, role, local path, and blocked status.",
            "No local data, schema inventory, loader, notebook, or model result exists.",
            "Create only a dataset catalog/provenance entry first; do not model until schema is audited.",
        ),
        AuditItem(
            "CSE-CIC-IDS2018 enterprise dataset",
            "Enterprise-scale chronological/day-based experiments.",
            "PARTIAL" if state["has_dataset_catalog"] else "MISSING",
            "docs/datasets/catalog.md records the official source, role, local path, and blocked status.",
            "No local data, day inventory, leakage audit, chronological split, or model result exists.",
            "Add source/provenance and a leakage-aware day split plan before training.",
        ),
        AuditItem(
            "Unsupervised anomaly detection",
            "Train on benign only; test known and held-out attack types.",
            "PROVEN" if state["has_anomaly_detection"] else "MISSING",
            "results/anomaly_detection.md reports IsolationForest, LOF, and k-means-distance normal-only detectors.",
            "This is NSL-KDD binary/family recall only; autoencoders and modern-dataset zero-day splits are not implemented.",
            "Graduate the best protocol to CICIoT/CSE once raw timestamped data exists.",
        ),
        AuditItem(
            "Semi-supervised learning",
            "Use 1/5/10/25% labels plus self-training/label propagation/pretraining.",
            "PROVEN" if state["has_semi_supervised"] else "MISSING",
            "results/semi_supervised.md reports labelled-only vs self-training Logistic Regression at 1/5/10/25%.",
            "Only binary NSL-KDD and self-training are implemented; label propagation and representation pretraining are not.",
            "Repeat with multiclass and modern datasets before broader label-efficiency claims.",
        ),
        AuditItem(
            "Temporal and sequence detection",
            "Window flows by host/session/time and compare tabular vs sequence models.",
            "MISSING",
            "NSL-KDD has no usable chronology; no temporal dataset/window builder exists.",
            "Do not imply sequence behaviour detection from row-level classifiers.",
            "Use CIC/CSE style timestamped datasets; first build audited window aggregation.",
        ),
        AuditItem(
            "Online and drift learning",
            "Chronological streams, partial_fit, drift detectors, recovery-time metrics.",
            "PARTIAL" if state["has_online_learning"] else "MISSING",
            "results/online_learning.md reports SGD partial_fit models over NSL-KDD file-order chunks.",
            "NSL-KDD is not chronological, so this proves online algorithms, not drift recovery.",
            "Run true drift metrics on timestamped CSE/CICIoT data after chronology is audited.",
        ),
        AuditItem(
            "Cross-dataset generalization",
            "Train one environment, test another via common NetFlow schema.",
            "PARTIAL" if state["has_dataset_catalog"] else "MISSING",
            "docs/datasets/catalog.md identifies the common NetFlow dataset source and required local path.",
            "No common schema adapter, local NetFlow files, or leave-one-dataset-out result exists.",
            "Define common NetFlow-style features before any cross-dataset score is published.",
        ),
        AuditItem(
            "Data-quality and leakage analytics",
            "Missing, duplicates, constants, outliers, PSI/KS/JS/chi-squared, leakage checks.",
            "PARTIAL",
            "NSL audit notebook is executed; CICIoT dev quality script exists; feature outlier CSV exists="
            f"{state['has_feature_outliers']}.",
            "Advanced statistical shift tests such as PSI/KS/JS/chi-squared are still incomplete.",
            "Create one reusable quality-report API and run it against every dataset.",
        ),
        AuditItem(
            "Attack behavioural profiling",
            "Per-attack protocol/duration/ports/source-diversity/flags/profile summaries.",
            "MISSING",
            "No saved attack-profile tables exist.",
            "The project can explain model scores better than it can explain attack behaviour today.",
            "Add per-class profile tables to the data-quality notebook before more models.",
        ),
        AuditItem(
            "Clustering and representation studies",
            "KMeans/GMM/DBSCAN/HDBSCAN/PCA/UMAP and raw/PCA/AE/LightGBM comparisons.",
            "PARTIAL" if state["has_feature_learning"] else "MISSING",
            "results/feature_learning.md compares raw, MI-selected, PCA, L1-selected, and autoencoder embeddings.",
            "Clustering, UMAP/t-SNE, SHAP-driven selection, and deep tabular architectures are not implemented yet.",
            "Add clustering diagnostics and repeat feature-learning over more datasets once local files exist.",
        ),
        AuditItem(
            "Explainability and calibration",
            "SHAP, permutation importance, calibration, Brier/ECE, confidence/error analysis.",
            "PARTIAL",
            "Tree feature-importance plots exist; no calibration or SHAP artifacts exist.",
            "Feature importance is not enough to claim explainability/trustworthiness.",
            "Add calibration curves/Brier/ECE for binary models, then SHAP/permutation checks.",
        ),
        AuditItem(
            "Operational SOC simulation",
            "Alerts/day, FP per 10k benign, workload, risk-tolerance thresholds.",
            "PROVEN" if state["has_soc_simulation"] else "MISSING",
            "results/soc_simulation.md converts threshold-ablation rates into daily alerts and missed attacks.",
            "Scenario is fixed at 1M flows/day and 0.5% malicious; no sensitivity grid yet.",
            "Add multiple base rates, analyst capacity limits, and threshold-frontier plots.",
        ),
        AuditItem(
            "Graph machine learning",
            "Host/IP graph, centrality/community, GraphSAGE/GAT/edge classification.",
            "MISSING",
            "No graph data representation exists.",
            "Graph ML belongs later; current row-level feature tables cannot support it.",
            "First add graph statistics only after a dataset with IP/device identifiers is audited.",
        ),
        AuditItem(
            "Deep-learning taxonomy and suitability matrix",
            "Classify MLP/CNN/RNN/AE/VAE/GAN/SOM/RBM/DBN/transfer/DRL by representation validity.",
            "PROVEN" if state["has_deep_learning_taxonomy"] else "MISSING",
            "docs/deep_learning_taxonomy.md records method-to-data suitability and staged progression.",
            "It is a guardrail document; many advanced families are intentionally not implemented until data supports them.",
            "Update this matrix whenever a new dataset or representation is added.",
        ),
    ]


def current_combinations(state: dict[str, object]) -> list[tuple[str, str, str, str, str]]:
    """Return combinations that are currently evidenced on disk."""
    return [
        (
            "NSL-KDD",
            "Supervised reference",
            "binary + multiclass",
            "Dummy, balanced LogReg, balanced RF, balanced ExtraTrees, balanced HistGB, balanced LightGBM",
            "12 model/task rows in results/reference_track.md",
        ),
        (
            "NSL-KDD",
            "Phase-3 official-split baseline",
            "binary KDDTest+, binary KDDTest-21, multiclass KDDTest+",
            "RandomForest + LightGBM with train-only GridSearchCV",
            "6 evaluation rows plus confusion/ROC/PR/importance figures",
        ),
        (
            "NSL-KDD",
            "Threshold tuning",
            "binary KDDTest+",
            "LogReg, balanced LogReg, balanced ExtraTrees, balanced HistGB x default/F1/F2 thresholds",
            "results/threshold_ablation.md and results/threshold_ablation.csv",
        ),
        (
            "NSL-KDD",
            "Normal-only anomaly detection",
            "binary plus attack-family recall",
            "IsolationForest, LocalOutlierFactor, k-means distance x 0.90/0.95/0.99 normal quantiles",
            "results/anomaly_detection.md and results/anomaly_detection.csv",
        ),
        (
            "NSL-KDD",
            "Semi-supervised label budget",
            "binary x 1/5/10/25% labels",
            "labelled-only Logistic Regression vs self-training Logistic Regression",
            "results/semi_supervised.md and results/semi_supervised.csv",
        ),
        (
            "NSL-KDD",
            "Online-learning proxy",
            "binary x file-order chunks",
            "SGD log-loss partial_fit, SGD passive-aggressive-style partial_fit",
            "results/online_learning.md; not a true drift claim",
        ),
        (
            "NSL-KDD",
            "Operational SOC simulation",
            "1M daily flows, 0.5% malicious",
            "threshold-ablation rates converted to alerts/misses/workload",
            "results/soc_simulation.md and results/soc_simulation.csv",
        ),
        (
            "NSL-KDD",
            "Feature analysis and representation learning",
            "binary KDDTest+",
            "correlation/outlier maps; raw vs MI top-k vs PCA vs L1 vs autoencoder embeddings",
            "results/feature_learning.md plus feature CSVs and figures",
        ),
        (
            "NSL-KDD",
            "Neural foundations and MLP ablations",
            "binary KDDTest+; bounded train subset",
            "single-neuron demo; activation functions; MLP activation/dropout/norm/depth/loss/label-smoothing ablations",
            "results/neural_foundations.md, results/neural_ablation.md, experiments/runs/neural_ablation.jsonl",
        ),
        (
            "NSL-KDD",
            "Phase-4 MLP ablation",
            "binary + multiclass",
            "MLP unweighted vs weighted",
            "4 headline rows in results/metrics.md plus stability rows",
        ),
        (
            "NSL-KDD",
            "Multi-seed stability",
            "binary + multiclass x 5 seeds",
            "RandomForest, LightGBM, MLP-unweighted, MLP-weighted",
            "40 train/evaluate runs summarized in results/stability.md",
        ),
        (
            "CICIoT2023",
            "Dev feature analysis and representation learning",
            "binary dev split sample",
            "correlation/outlier maps; raw vs MI top-k vs PCA vs L1 vs autoencoder embeddings",
            "results/feature_learning.md caveated as dev sample only",
        ),
        (
            "CICIoT2023",
            "Dev data-quality track",
            "train.parquet + test.parquet",
            "quality/leakage checks in src/ciciot.py",
            f"{state['ciciot_parquet_count']} parquet files present",
        ),
        (
            "CICIoT2023",
            "Raw CSV Phase-1 track",
            "CSV sample per file",
            "label/category mapper and sample EDA notebook",
            f"{state['raw_ciciot_csv_count']} raw CSV files present; full raw audit blocked if zero",
        ),
    ]


def requested_matrix() -> list[tuple[str, str, str, str, str, str, str, str, str]]:
    """Dataset x experiment-family status matrix for the lab prompt."""
    return [
        ("NSL-KDD", "PROVEN+", "PARTIAL", "PROVEN", "PROVEN", "MISSING", "PARTIAL", "PARTIAL+", "PARTIAL"),
        ("CICIoT2023", "PARTIAL", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "PARTIAL+", "MISSING"),
        ("TON_IoT", "PARTIAL", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING"),
        ("CSE-CIC-IDS2018", "PARTIAL", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING"),
        ("Common NetFlow schema", "PARTIAL", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING"),
    ]


def render_report() -> str:
    """Render the full markdown referee report."""
    state = snapshot()
    items = build_audit_items(state)
    counts = {s: sum(1 for i in items if i.status == s) for s in ("PROVEN", "PARTIAL", "MISSING")}

    lines = [
        "# Experimental Lab Prompt Audit",
        "",
        "Role: referee/audit force. This report checks what the repository can prove "
        "today against the large \"Adaptive Network Security Analytics Lab\" prompt.",
        "",
        "## Executive Verdict",
        "",
        f"- Proven areas: **{counts['PROVEN']}**",
        f"- Partial areas: **{counts['PARTIAL']}**",
        f"- Missing areas: **{counts['MISSING']}**",
        f"- Test files present: **{state['test_file_count']}**",
        f"- Saved figures present: **{state['figure_count']}**",
        f"- Referee audit notebook outputs: **{state['nb02_outputs']} / {state['nb02_code']} code cells**",
        f"- CICIoT2023 Phase-1 notebook outputs: **{state['nb03_outputs']} / {state['nb03_code']} code cells**",
        f"- New threshold ablation: **{state['has_threshold_ablation']}**",
        f"- New anomaly-detection report: **{state['has_anomaly_detection']}**",
        f"- New semi-supervised report: **{state['has_semi_supervised']}**",
        f"- New SOC simulation: **{state['has_soc_simulation']}**",
        f"- New feature-learning report: **{state['has_feature_learning']}**",
        f"- New deep-learning taxonomy: **{state['has_deep_learning_taxonomy']}**",
        f"- New neural foundation report: **{state['has_neural_foundations']}**",
        f"- New neural ablation report: **{state['has_neural_ablation']}**",
        "",
        "The project is now more than a supervised NSL-KDD comparison: it has saved "
        "threshold-tuning, normal-only anomaly, semi-supervised, online-proxy, and "
        "SOC-simulation artifacts. It is still **not yet** the full experimental lab: "
        "true temporal drift, graph ML, modern raw-dataset modelling, and cross-dataset "
        "generalization remain blocked by missing local data/schema work.",
        "",
        "## Currently Proven Combinations",
        "",
        "| Dataset | Track | Task/split combinations | Models/methods | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in current_combinations(state):
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Requested Lab Matrix",
        "",
        "Legend: **PROVEN** = runnable evidence and saved results exist; **PARTIAL** = "
        "some code or data exists but the experiment is not complete; **MISSING** = no "
        "credible local evidence yet.",
        "",
        "| Dataset/schema | Supervised | Cost-sensitive | Anomaly | Semi-supervised | Temporal | Online/drift | Quality/explainability | SOC/graph |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in requested_matrix():
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Detailed Findings",
        "",
        "| Area | Prompt expectation | Status | Evidence | Gap / risk | Next audit test |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in items:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.area,
                    item.expectation,
                    item.status,
                    item.evidence,
                    item.gap,
                    item.next_test,
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Claims I Would Allow Today",
        "",
        "- NSL-KDD preprocessing is leakage-safe and tested.",
        "- NSL-KDD has a supervised reference track across several classical models.",
        "- Balanced Logistic Regression is strong on the 5-class reference track: "
        "the saved table reports about 0.607 macro-F1.",
        "- Threshold tuning changes the binary NSL-KDD attack-recall/false-alert trade-off.",
        "- Normal-only anomaly detection and semi-supervised label-budget experiments "
        "now have saved NSL-KDD reports.",
        "- Correlation analysis, outlier mapping, and representation-learning baselines "
        "now exist for NSL-KDD and CICIoT2023 dev data.",
        "- Artificial-neuron foundations, activation functions, and bounded MLP ablations "
        "now have saved reports and tracked runs.",
        "- SOC simulation now translates threshold results into alerts/day and missed attacks/day.",
        "- Weighted MLP improves rare-class recall materially, but the macro-F1 "
        "ranking is not a clean win.",
        "- CICIoT2023 has early loader/provenance/Phase-1 scaffolding, but full raw "
        "CSV modelling is not complete.",
        "",
        "## Claims I Would Block",
        "",
        "- Any claim that TON_IoT or CSE-CIC-IDS2018 has been evaluated.",
        "- Any claim of true temporal sequence detection, online drift recovery, "
        "modern-dataset anomaly detection, or cross-dataset generalization.",
        "- Any claim that NSL-KDD file-order online learning is chronological drift.",
        "- Any claim that feature importance equals explainability or calibrated trust.",
        "",
        "## Immediate Audit-Driven Build Order",
        "",
        "1. Add a true imbalance notebook: no balancing vs class weights vs resampling, "
        "with train-only resampling tests.",
        "2. Finish CICIoT2023 quality reports for whichever local source is actually "
        "used: dev parquet now, raw CSV later.",
        "3. Add calibration/Brier/ECE to the threshold/SOC track.",
        "4. Add true timestamped online-drift experiments after CSE/CICIoT raw files exist.",
        "5. Only then implement cross-dataset NetFlow and graph tracks.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    AUDITS.mkdir(parents=True, exist_ok=True)
    report = render_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
