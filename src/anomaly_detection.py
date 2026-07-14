"""Normal-only anomaly detection on NSL-KDD.

These models train only on normal training traffic. They never see attack rows
while fitting or selecting their operating threshold. The test set is then scored
as benign vs attack, with per-family attack recall as the zero-day-style audit.

Run:
    .venv/bin/python src/anomaly_detection.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import data as D
import evaluate as E
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
CSV_PATH = RESULTS / "anomaly_detection.csv"
REPORT_PATH = RESULTS / "anomaly_detection.md"


@dataclass(frozen=True)
class AnomalyScores:
    """Higher score means more anomalous."""

    train_normal: np.ndarray
    test: np.ndarray


def _sample_rows(X: np.ndarray, max_rows: int, seed: int = D.RANDOM_STATE) -> np.ndarray:
    """Deterministically cap expensive anomaly training."""
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[np.sort(idx)]


def scores_to_predictions(
    train_normal_scores: np.ndarray,
    test_scores: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, float]:
    """Threshold anomaly scores using a normal-only training quantile."""
    threshold = float(np.quantile(train_normal_scores, quantile))
    return (test_scores >= threshold).astype(np.int64), threshold


def family_recalls(y_true_binary: np.ndarray, y_pred: np.ndarray, test_families: pd.Series) -> dict[str, float]:
    """Attack-detection recall by NSL-KDD family."""
    out: dict[str, float] = {}
    for family in D.FAMILY_ORDER:
        mask = test_families.to_numpy() == family
        if not mask.any():
            out[family] = 0.0
            continue
        if family == "normal":
            out[family] = float((y_pred[mask] == 0).mean())
        else:
            out[family] = float((y_pred[mask] == 1).mean())
    return out


def isolation_forest_scores(X_normal: np.ndarray, X_test: np.ndarray) -> AnomalyScores:
    model = IsolationForest(
        n_estimators=150,
        max_samples=min(10_000, len(X_normal)),
        random_state=D.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_normal)
    return AnomalyScores(
        train_normal=-model.score_samples(X_normal),
        test=-model.score_samples(X_test),
    )


def lof_scores(X_normal: np.ndarray, X_test: np.ndarray) -> AnomalyScores:
    model = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1)
    model.fit(X_normal)
    return AnomalyScores(
        train_normal=-model.score_samples(X_normal),
        test=-model.score_samples(X_test),
    )


def kmeans_distance_scores(X_normal: np.ndarray, X_test: np.ndarray) -> AnomalyScores:
    model = MiniBatchKMeans(
        n_clusters=16,
        random_state=D.RANDOM_STATE,
        batch_size=4096,
        n_init="auto",
    )
    model.fit(X_normal)
    train_dist = model.transform(X_normal).min(axis=1)
    test_dist = model.transform(X_test).min(axis=1)
    return AnomalyScores(train_normal=train_dist, test=test_dist)


def run() -> pd.DataFrame:
    """Run normal-only anomaly detection and save result artifacts."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    prep = P.prepare_nsl_kdd("binary")
    raw_test = D.load_nsl_kdd("test")
    X_normal = _sample_rows(prep.X_train[prep.y_train == 0], max_rows=20_000)

    score_builders = {
        "IsolationForest": isolation_forest_scores,
        "LocalOutlierFactor": lof_scores,
        "KMeans_distance": kmeans_distance_scores,
    }
    quantiles = [0.90, 0.95, 0.99]
    rows: list[dict[str, object]] = []

    for model_name, build_scores in score_builders.items():
        scores = build_scores(X_normal, prep.X_test)
        for q in quantiles:
            y_pred, threshold = scores_to_predictions(scores.train_normal, scores.test, q)
            metrics = E.compute_metrics(prep.y_test, y_pred, prep.classes)
            recalls = family_recalls(prep.y_test, y_pred, raw_test["attack_family"])
            rows.append(
                {
                    "model": model_name,
                    "normal_quantile": q,
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "normal_recall": metrics["per_class"]["normal"]["recall"],
                    "attack_recall": metrics["per_class"]["attack"]["recall"],
                    "DoS_recall": recalls["DoS"],
                    "Probe_recall": recalls["Probe"],
                    "R2L_recall": recalls["R2L"],
                    "U2R_recall": recalls["U2R"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False)
    REPORT_PATH.write_text(render_markdown(out), encoding="utf-8")
    return out


def render_markdown(rows: pd.DataFrame) -> str:
    lines = [
        "# NSL-KDD — Normal-Only Anomaly Detection",
        "",
        "Models are trained on normal training traffic only. Thresholds are chosen from "
        "normal-only training score quantiles, so attack labels do not tune the detector.",
        "",
        "| Model | Normal quantile | Accuracy | Macro-F1 | Normal recall | Attack recall | DoS | Probe | R2L | U2R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['model']} | {r['normal_quantile']:.2f} | {r['accuracy']:.4f} | "
            f"{r['macro_f1']:.4f} | {r['normal_recall']:.4f} | "
            f"{r['attack_recall']:.4f} | {r['DoS_recall']:.4f} | "
            f"{r['Probe_recall']:.4f} | {r['R2L_recall']:.4f} | "
            f"{r['U2R_recall']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This is not expected to beat supervised classifiers on known NSL-KDD labels. "
        "Its purpose is different: measure whether a normal-behaviour model catches "
        "attack families without being trained on attack examples.",
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
