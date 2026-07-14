"""NSL-KDD binary threshold-tuning ablation.

The threshold is selected on a validation slice of the training set only. The
official test set is touched exactly once at the end, after the threshold choice
is frozen.

Run:
    .venv/bin/python src/threshold_ablation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import data as D
import evaluate as E
import preprocess as P
import tuning as T

RESULTS = D.REPO_ROOT / "results"
CSV_PATH = RESULTS / "threshold_ablation.csv"
REPORT_PATH = RESULTS / "threshold_ablation.md"


@dataclass(frozen=True)
class ThresholdObjective:
    """Validation objective for a binary threshold sweep."""

    name: str
    beta: float


OBJECTIVES = [
    ThresholdObjective("F1", 1.0),
    ThresholdObjective("F2_attack_recall_weighted", 2.0),
]


def build_models(seed: int = D.RANDOM_STATE) -> dict[str, object]:
    """Fast binary classifiers that expose positive-class probabilities."""
    return {
        "LogReg": LogisticRegression(max_iter=2000),
        "LogReg_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
        ),
        "ExtraTrees_balanced": ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGB_balanced": HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
        ),
    }


def predictions_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert positive-class scores into binary predictions."""
    return (np.asarray(scores) >= threshold).astype(np.int64)


def false_positives_per_10k_benign(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Operational false-positive rate: false alerts per 10k benign flows."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    benign = y_true == 0
    if benign.sum() == 0:
        return 0.0
    fp = int(((y_pred == 1) & benign).sum())
    return float(fp / benign.sum() * 10_000)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Compact metric row for threshold tables."""
    metrics = E.compute_metrics(y_true, y_pred, classes)
    p, r, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "normal_recall": float(r[0]),
        "attack_precision": float(p[1]),
        "attack_recall": float(r[1]),
        "fp_per_10k_benign": false_positives_per_10k_benign(y_true, y_pred),
    }


def run() -> pd.DataFrame:
    """Run the full ablation and return the result table."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    prep = P.prepare_nsl_kdd("binary")
    split = T.make_validation_split(prep.y_train, validation_size=0.2, seed=D.RANDOM_STATE)

    rows: list[dict[str, object]] = []
    for model_name, model in build_models().items():
        model.fit(prep.X_train[split.train_idx], prep.y_train[split.train_idx])
        val_scores = model.predict_proba(prep.X_train[split.val_idx])[:, 1]

        candidates = [("default_0.50", 0.5, "fixed")]
        for objective in OBJECTIVES:
            sweep = T.threshold_sweep(
                prep.y_train[split.val_idx],
                val_scores,
                beta=objective.beta,
            )
            candidates.append(
                (
                    f"validation_{objective.name}",
                    float(sweep["best"]["threshold"]),
                    f"validation F-beta beta={objective.beta}",
                )
            )

        # Refit final model on all training rows after the threshold is selected
        # from validation. Test labels/features are still unseen.
        final_model = build_models()[model_name]
        final_model.fit(prep.X_train, prep.y_train)
        test_scores = final_model.predict_proba(prep.X_test)[:, 1]

        for threshold_name, threshold, selection_rule in candidates:
            y_pred = predictions_from_threshold(test_scores, threshold)
            row = {
                "model": model_name,
                "threshold_name": threshold_name,
                "threshold": threshold,
                "selection_rule": selection_rule,
            }
            row.update(metric_bundle(prep.y_test, y_pred, prep.classes))
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False)
    REPORT_PATH.write_text(render_markdown(out), encoding="utf-8")
    return out


def render_markdown(rows: pd.DataFrame) -> str:
    """Render the saved threshold-ablation report."""
    lines = [
        "# NSL-KDD — Threshold Tuning Ablation",
        "",
        "Thresholds are selected on a stratified validation split from the training set. "
        "The official KDDTest+ set is used only after the threshold is fixed.",
        "",
        "| Model | Threshold rule | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | Normal recall | FP / 10k benign |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['model']} | {r['threshold_name']} | {r['threshold']:.2f} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['attack_precision']:.4f} | {r['attack_recall']:.4f} | "
            f"{r['normal_recall']:.4f} | {r['fp_per_10k_benign']:.1f} |"
        )
    lines += [
        "",
        "## Reading this table",
        "",
        "Lowering the threshold usually raises attack recall but also creates more false "
        "alerts from benign flows. The operational column, FP / 10k benign, is the "
        "bridge from ML score to SOC workload.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    rows = run()
    print(rows.to_string(index=False))
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
