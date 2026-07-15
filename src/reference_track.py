"""Reference Track — unified supervised benchmark on NSL-KDD.

The controlled baseline for the whole lab: a zoo of classical classifiers on the
same leakage-safe preprocessing, so every later paradigm (anomaly detection,
semi-supervised, temporal, online, cross-dataset) has a fair supervised yardstick.

All models use ``class_weight='balanced'`` where supported — imbalance handling
matters as much as model family here, and a balanced Logistic Regression is a
famously strong, cheap baseline on this dataset. Dummy is the floor.

Run:  .venv/bin/python src/reference_track.py   (writes results/reference_track.md)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

import data as D
import evaluate as E
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
SEED = D.RANDOM_STATE


def build_models() -> dict:
    """The reference zoo (balanced where supported)."""
    return {
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
        "LogReg (balanced)": LogisticRegression(
            class_weight="balanced", max_iter=2000, n_jobs=-1),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "ExtraTrees (balanced)": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "HistGradientBoosting (balanced)": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
        "LightGBM (balanced)": LGBMClassifier(
            n_estimators=400, num_leaves=31, class_weight="balanced",
            random_state=SEED, n_jobs=1, num_threads=1, verbose=-1),
    }


def run() -> int:
    rows: list[dict] = []
    for scheme in ("binary", "multiclass"):
        prep = P.prepare_nsl_kdd(scheme)
        for name, model in build_models().items():
            model.fit(prep.X_train, prep.y_train)
            y_pred = model.predict(prep.X_test)
            m = E.compute_metrics(prep.y_test, y_pred, prep.classes)
            pc = m["per_class"]
            rare = (f"attack {pc['attack']['recall']:.3f}" if scheme == "binary"
                    else f"R2L {pc['R2L']['recall']:.3f} / U2R {pc['U2R']['recall']:.3f}")
            rows.append({"model": name, "scheme": scheme,
                         "acc": m["accuracy"], "mf1": m["macro_f1"], "rare": rare})
            print(f"  {scheme:10s} {name:32s} macroF1={m['macro_f1']:.4f}")

    lines = ["# NSL-KDD — Reference Track (supervised benchmark)", "",
             "`class_weight='balanced'` where supported. Single seed; see "
             "`stability.md` for seed variance on the headline models.", ""]
    for scheme in ("binary", "multiclass"):
        lines += [f"## {scheme}", "",
                  "| Model | Accuracy | Macro-F1 | Rare-class recall |",
                  "| --- | ---: | ---: | --- |"]
        for r in [r for r in rows if r["scheme"] == scheme]:
            lines.append(f"| {r['model']} | {r['acc']:.4f} | {r['mf1']:.4f} | "
                         f"{r['rare']} |")
        lines.append("")
    (RESULTS / "reference_track.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    print(f"\nWrote {RESULTS / 'reference_track.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
