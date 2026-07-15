"""CICIoT2023 supervised baselines (Stages 0-1 + tree/boosting).

The controlled supervised yardstick for CICIoT2023, mirroring the NSL-KDD
Reference Track. Runs the model zoo at two label levels (binary benign/attack and
the 8-category level) on the parquet dev sample via the consolidated
``ciciot2023.prepare``. Train is stratified-subsampled for dev speed; the full
official test split is used untouched.

Stage 0 (Dummy) is the floor; Stage 1 (balanced Logistic Regression) is the
transparent baseline; RF/HistGB/LightGBM (balanced) probe whether complexity or
imbalance handling drives results — the lab's recurring question.

Run:  .venv/bin/python src/ciciot2023_baselines.py  (writes results/ciciot2023_baselines.{md,csv})
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

import ciciot2023 as C
import data as D
import evaluate as E

RESULTS = D.REPO_ROOT / "results"
SEED = D.RANDOM_STATE
SUBSAMPLE_TRAIN = 200_000
LEVELS = ("binary", "category")


def build_models() -> dict:
    return {
        "Dummy (stratified)": DummyClassifier(strategy="stratified",
                                              random_state=SEED),
        "LogReg (balanced)": LogisticRegression(
            class_weight="balanced", max_iter=1000, n_jobs=-1),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "HistGB (balanced)": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
        "LightGBM (balanced)": LGBMClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED,
            n_jobs=1, num_threads=1, verbose=-1),
    }


def _subsample(X: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y,
                                    random_state=SEED)
    return Xs, ys


def main() -> int:
    tr, te = C.load_parquet("train"), C.load_parquet("test")
    rows: list[dict] = []
    md = ["# CICIoT2023 — supervised baselines (dev parquet)", "",
          f"Train subsampled to {SUBSAMPLE_TRAIN:,} (stratified); full test split. "
          "`class_weight='balanced'` where supported. Random split → "
          "in-distribution (caveat vs NSL-KDD's official shift).", ""]

    for level in LEVELS:
        prep = C.prepare(tr, te, level=level)
        X_train, y_train = _subsample(prep.X_train, prep.y_train, SUBSAMPLE_TRAIN)
        classes = prep.classes
        print(f"\n=== level: {level}  ({len(classes)} classes, "
              f"train={len(y_train):,}, test={len(prep.y_test):,}) ===")

        md += [f"## {level}", "",
               "| Model | Accuracy | Balanced-acc | Macro-F1 | "
               + ("ROC-AUC | PR-AUC | " if level == "binary" else "")
               + "Rare-class recall |",
               "| --- | ---: | ---: | ---: | "
               + ("---: | ---: | " if level == "binary" else "") + "--- |"]

        for name, model in build_models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(prep.X_test)
            y_score = (model.predict_proba(prep.X_test)[:, 1]
                       if level == "binary" and hasattr(model, "predict_proba")
                       else None)
            m = E.compute_metrics(prep.y_test, y_pred, classes, y_score=y_score)
            bal = balanced_accuracy_score(prep.y_test, y_pred)
            pc = m["per_class"]
            if level == "binary":
                rare = f"attack {pc['Attack']['recall']:.3f}"
            else:  # highlight the two smallest attack categories
                rare = (f"Web-based {pc['Web-based']['recall']:.3f} / "
                        f"Brute Force {pc['Brute Force']['recall']:.3f}")

            rows.append({"level": level, "model": name,
                         "accuracy": m["accuracy"], "balanced_acc": bal,
                         "macro_f1": m["macro_f1"],
                         "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
                         "rare": rare})
            extra = (f"{m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
                     if level == "binary" else "")
            md.append(f"| {name} | {m['accuracy']:.4f} | {bal:.4f} | "
                      f"{m['macro_f1']:.4f} | {extra}{rare} |")
            print(f"  {name:26s} macroF1={m['macro_f1']:.4f} bal_acc={bal:.4f}")
        md.append("")

    (RESULTS / "ciciot2023_baselines.md").write_text("\n".join(md) + "\n",
                                                     encoding="utf-8")
    pd.DataFrame(rows).to_csv(RESULTS / "ciciot2023_baselines.csv", index=False)
    print(f"\nWrote {RESULTS / 'ciciot2023_baselines.md'} (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
