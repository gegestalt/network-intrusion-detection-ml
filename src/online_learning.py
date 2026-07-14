"""Online-learning proxy experiment on NSL-KDD.

NSL-KDD is not a real chronological stream, so this is an online *algorithm*
test, not a drift claim. It processes the training rows in file order using
partial_fit and evaluates on the official test split.

Run:
    .venv/bin/python src/online_learning.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight

import data as D
import evaluate as E
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
CSV_PATH = RESULTS / "online_learning.csv"
REPORT_PATH = RESULTS / "online_learning.md"


def iter_chunks(n_rows: int, chunk_size: int) -> list[slice]:
    """Return deterministic stream chunks."""
    return [slice(start, min(start + chunk_size, n_rows)) for start in range(0, n_rows, chunk_size)]


def class_weight_dict(y_train: np.ndarray) -> dict[int, float]:
    """Compute fixed balanced class weights for partial_fit models."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def build_models(class_weight: dict[int, float], seed: int = D.RANDOM_STATE) -> dict[str, object]:
    return {
        "SGD_log_loss_balanced": SGDClassifier(
            loss="log_loss",
            class_weight=class_weight,
            random_state=seed,
            alpha=1e-4,
        ),
        "SGD_passive_aggressive_balanced": SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=1.0,
            random_state=seed,
            class_weight=class_weight,
        ),
    }


def run(chunk_size: int = 10_000) -> pd.DataFrame:
    """Train online models over chunks and save final test metrics."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    prep = P.prepare_nsl_kdd("binary")
    classes = np.arange(len(prep.classes))
    weights = class_weight_dict(prep.y_train)
    rows: list[dict[str, object]] = []

    for name, model in build_models(weights).items():
        chunks_seen = 0
        for chunk in iter_chunks(len(prep.y_train), chunk_size):
            model.partial_fit(prep.X_train[chunk], prep.y_train[chunk], classes=classes)
            chunks_seen += 1

        y_pred = model.predict(prep.X_test)
        m = E.compute_metrics(prep.y_test, y_pred, prep.classes)
        rows.append(
            {
                "model": name,
                "chunk_size": chunk_size,
                "chunks_seen": chunks_seen,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "normal_recall": m["per_class"]["normal"]["recall"],
                "attack_recall": m["per_class"]["attack"]["recall"],
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False)
    REPORT_PATH.write_text(render_markdown(out), encoding="utf-8")
    return out


def render_markdown(rows: pd.DataFrame) -> str:
    lines = [
        "# NSL-KDD — Online-Learning Proxy",
        "",
        "This is an online algorithm test using `partial_fit` over NSL-KDD training "
        "rows in file order. Because NSL-KDD has no reliable chronology, this is "
        "**not** evidence of drift recovery.",
        "",
        "| Model | Chunk size | Chunks seen | Accuracy | Macro-F1 | Normal recall | Attack recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['chunk_size']):,} | {int(r['chunks_seen'])} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['normal_recall']:.4f} | {r['attack_recall']:.4f} |"
        )
    lines += [
        "",
        "## Next real drift test",
        "",
        "Use a timestamped dataset such as CSE-CIC-IDS2018 or CICIoT2023 raw flows, "
        "then evaluate pre-drift, post-drift, recovery time, latency, and memory.",
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
