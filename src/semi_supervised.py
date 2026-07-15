"""Semi-supervised label-budget experiment on NSL-KDD.

We pretend only a small fraction of training labels are available. The
comparison is deliberately simple and auditable:

* supervised Logistic Regression trained only on labelled rows
* SelfTrainingClassifier using labelled rows plus unlabelled training rows

Run:
    .venv/bin/python src/semi_supervised.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

import data as D
import evaluate as E
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
CSV_PATH = RESULTS / "semi_supervised.csv"
REPORT_PATH = RESULTS / "semi_supervised.md"


def label_budget_indices(
    y: np.ndarray,
    fraction: float,
    seed: int = D.RANDOM_STATE,
    min_per_class: int = 1,
) -> np.ndarray:
    """Return deterministic stratified labelled-row indices."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        n = max(min_per_class, int(round(len(cls_idx) * fraction)))
        n = min(n, len(cls_idx))
        chosen.append(rng.choice(cls_idx, size=n, replace=False))
    return np.sort(np.concatenate(chosen))


def make_base_model() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def run() -> pd.DataFrame:
    """Run label-budget ablations and save result artifacts."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    prep = P.prepare_nsl_kdd("binary")
    budgets = [0.01, 0.05, 0.10, 0.25]
    rows: list[dict[str, object]] = []

    for budget in budgets:
        labelled_idx = label_budget_indices(prep.y_train, budget)

        supervised = make_base_model()
        supervised.fit(prep.X_train[labelled_idx], prep.y_train[labelled_idx])
        y_pred = supervised.predict(prep.X_test)
        m = E.compute_metrics(prep.y_test, y_pred, prep.classes)
        rows.append(
            {
                "method": "labelled_only_logreg",
                "label_fraction": budget,
                "labelled_rows": len(labelled_idx),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "normal_recall": m["per_class"]["normal"]["recall"],
                "attack_recall": m["per_class"]["attack"]["recall"],
            }
        )

        semi_labels = np.full_like(prep.y_train, fill_value=-1)
        semi_labels[labelled_idx] = prep.y_train[labelled_idx]
        self_training = SelfTrainingClassifier(
            estimator=make_base_model(),
            threshold=0.80,
            max_iter=10,
            verbose=False,
        )
        self_training.fit(prep.X_train, semi_labels)
        y_pred = self_training.predict(prep.X_test)
        m = E.compute_metrics(prep.y_test, y_pred, prep.classes)
        rows.append(
            {
                "method": "self_training_logreg",
                "label_fraction": budget,
                "labelled_rows": len(labelled_idx),
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
        "# NSL-KDD — Semi-Supervised Label-Budget Experiment",
        "",
        "Only a stratified fraction of training labels is revealed. Self-training may "
        "help when pseudo-labels are reliable, but it can also amplify early mistakes.",
        "",
        "| Method | Label fraction | Labelled rows | Accuracy | Macro-F1 | Normal recall | Attack recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['method']} | {r['label_fraction']:.2f} | {int(r['labelled_rows']):,} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['normal_recall']:.4f} | {r['attack_recall']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This track measures label efficiency. If self-training underperforms, that is "
        "a valid finding: pseudo-labels are not automatically trustworthy in shifted "
        "security data.",
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
