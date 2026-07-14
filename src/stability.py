"""Multi-seed stability study for the NSL-KDD models.

A single run is a hypothesis, not a finding — MLP-on-MPS is non-deterministic and
single-thread vs parallel LightGBM shifts results. This script trains each model
over several seeds (on CPU for reproducibility) and reports **mean +/- std** for
macro-F1, accuracy, and rare-class recall, so every headline claim is qualified.

Run:  .venv/bin/python src/stability.py   (writes results/stability.md; ~10 min)
"""

from __future__ import annotations

import os

# Single-threaded OpenMP: stable LightGBM, reproducible timing (see train_baselines).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

import data as D
import evaluate as E
import preprocess as P
import train_mlp as T

SEEDS = [0, 1, 2, 3, 4]
RESULTS = D.REPO_ROOT / "results"
DEVICE = torch.device("cpu")  # CPU is far more reproducible than MPS here


def _rare_recall(m: dict, scheme: str) -> dict[str, float]:
    pc = m["per_class"]
    if scheme == "binary":
        return {"attack_recall": pc["attack"]["recall"]}
    return {"R2L_recall": pc["R2L"]["recall"], "U2R_recall": pc["U2R"]["recall"]}


def _fit_eval(model_name: str, prep: P.PreparedData, scheme: str,
              seed: int) -> dict:
    """Train one model at one seed; return its metric bundle."""
    if model_name == "RandomForest":
        md = None if scheme == "binary" else 25
        clf = RandomForestClassifier(n_estimators=300, max_depth=md,
                                     random_state=seed, n_jobs=1)
        clf.fit(prep.X_train, prep.y_train)
        y_pred = clf.predict(prep.X_test)
        y_score = clf.predict_proba(prep.X_test)[:, 1] if scheme == "binary" else None
    elif model_name == "LightGBM":
        clf = LGBMClassifier(n_estimators=400, num_leaves=31, random_state=seed,
                             n_jobs=1, num_threads=1, verbose=-1, force_col_wise=True)
        clf.fit(prep.X_train, prep.y_train)
        y_pred = clf.predict(prep.X_test)
        y_score = clf.predict_proba(prep.X_test)[:, 1] if scheme == "binary" else None
    elif model_name in ("MLP-unweighted", "MLP-weighted"):
        nC = len(prep.classes)
        cw = (T.compute_class_weights(prep.y_train, nC)
              if model_name == "MLP-weighted" else None)
        clf = T.train_model(prep.X_train, prep.y_train, nC,
                            class_weight=cw, device=DEVICE, seed=seed)
        y_pred, proba = T.predict(clf, prep.X_test, device=DEVICE)
        y_score = proba[:, 1] if scheme == "binary" else None
    else:
        raise ValueError(model_name)
    return E.compute_metrics(prep.y_test, y_pred, prep.classes, y_score=y_score)


def main() -> int:
    models = ["RandomForest", "LightGBM", "MLP-unweighted", "MLP-weighted"]
    # rows[(scheme, model)] = {metric_name: [values across seeds]}
    agg: dict[tuple[str, str], dict[str, list[float]]] = {}

    for scheme in ("binary", "multiclass"):
        prep = P.prepare_nsl_kdd(scheme)
        for model_name in models:
            acc, mf1 = [], []
            rares: dict[str, list[float]] = {}
            for seed in SEEDS:
                m = _fit_eval(model_name, prep, scheme, seed)
                acc.append(m["accuracy"]); mf1.append(m["macro_f1"])
                for k, v in _rare_recall(m, scheme).items():
                    rares.setdefault(k, []).append(v)
                print(f"  {scheme:10s} {model_name:15s} seed={seed} "
                      f"macroF1={m['macro_f1']:.4f}")
            agg[(scheme, model_name)] = {"accuracy": acc, "macro_f1": mf1, **rares}

    # --- render results/stability.md --------------------------------------- #
    def ms(xs: list[float]) -> str:
        return f"{np.mean(xs):.4f} ± {np.std(xs):.4f}"

    lines = ["# NSL-KDD — multi-seed stability (5 seeds, CPU)", "",
             f"Seeds {SEEDS}. Mean ± std over seeds. CPU for reproducibility. "
             "Preprocessing is deterministic; only model init/splits vary.", ""]
    for scheme in ("binary", "multiclass"):
        rare_cols = (["attack_recall"] if scheme == "binary"
                     else ["R2L_recall", "U2R_recall"])
        header = "| Model | Accuracy | Macro-F1 | " + \
                 " | ".join(rare_cols) + " |"
        sep = "| --- | ---: | ---: | " + " | ".join(["---:"] * len(rare_cols)) + " |"
        lines += [f"## {scheme}", "", header, sep]
        for model_name in models:
            a = agg[(scheme, model_name)]
            cells = [ms(a["accuracy"]), ms(a["macro_f1"])] + [ms(a[c]) for c in rare_cols]
            lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Takeaway",
              "Report these mean ± std, not single-run numbers. Where std is a "
              "large fraction of the gap between two models, the ranking is **not** "
              "statistically meaningful on one run — a lesson the MLP-on-MPS "
              "swing taught us directly."]
    (RESULTS / "stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS / 'stability.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
