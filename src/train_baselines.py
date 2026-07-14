"""Phase 3 — Random Forest + LightGBM baselines on NSL-KDD.

For both label schemes (binary, 5-class) we:
  1. run a *light* CV hyperparameter search on the TRAIN set only (f1_macro),
  2. refit the best config on full train,
  3. evaluate on the official KDDTest+ (and KDDTest-21 for the binary task),
  4. save confusion matrices, ROC/PR curves (binary), and feature-importance
     plots to results/figures/, and a consolidated results/metrics.md.

LightGBM is our single gradient-boosting library (histogram-based, fast, scales
to the big NetFlow datasets later). Trees use default class weighting here —
explicit imbalance handling is the MLP phase's job, so the baseline stays honest.

Run:  .venv/bin/python src/train_baselines.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

# ColumnTransformer yields a bare numpy array; LightGBM's sklearn wrapper warns
# about missing feature names on predict. Cosmetic — silence for clean output.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Nested parallelism (GridSearchCV n_jobs=-1 over an estimator with n_jobs=-1)
# makes joblib warn it can't propagate config to workers. Harmless — silence it.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import data as D
import evaluate as E
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
FIG = RESULTS / "figures"
RANDOM_STATE = D.RANDOM_STATE

# Light search grids (baseline, not the star — a couple of combos each).
GRIDS = {
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": [300], "max_depth": [None, 25]},
    ),
    "LightGBM": (
        LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        {"n_estimators": [400], "num_leaves": [31, 63]},
    ),
}


def tune_and_fit(name, X_train, y_train):
    """Light CV search on train only; return (best_estimator, best_params)."""
    estimator, grid = GRIDS[name]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"    {name}: best {search.best_params_}  "
          f"(cv f1_macro={search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_


def evaluate_on(model, data, y_true, X, classes, scheme, name, tag):
    """Score ``model`` on features ``X`` vs ``y_true``; save plots; return row."""
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1] if len(classes) == 2 else None
    m = E.compute_metrics(y_true, y_pred, classes, y_score=y_score)

    slug = f"p3_nslkdd_{name.lower()}_{scheme}_{tag}"
    E.plot_confusion_matrices(
        y_true, y_pred, classes,
        f"NSL-KDD {name} — {scheme} ({tag})", FIG / f"{slug}_confusion.png")
    if y_score is not None:
        E.plot_roc_pr(y_true, y_score,
                      f"NSL-KDD {name} — {scheme} ({tag})", FIG / f"{slug}_roc_pr.png")
    return m


def render_metrics_md(rows: list[dict], per_class_blocks: list[str]) -> str:
    lines = ["# NSL-KDD — Phase 3 baseline results (RF vs LightGBM)", "",
             "Official split; metrics on the held-out test set. Leading with "
             "**macro-F1** and per-class recall (accuracy is inflated by the "
             "easy majority classes).", "",
             "| Model | Task | Test set | Accuracy | Macro-F1 | Weighted-F1 | "
             "ROC-AUC | PR-AUC | Best params |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        roc = f"{r['roc_auc']:.4f}" if "roc_auc" in r else "—"
        pr = f"{r['pr_auc']:.4f}" if "pr_auc" in r else "—"
        lines.append(
            f"| {r['model']} | {r['scheme']} | {r['tag']} | {r['accuracy']:.4f} | "
            f"{r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | {roc} | {pr} | "
            f"`{r['params']}` |")
    lines += ["", "## Per-class breakdown (KDDTest+)", ""]
    lines += per_class_blocks
    return "\n".join(lines) + "\n"


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    per_class_blocks: list[str] = []

    for scheme in ("binary", "multiclass"):
        print(f"\n=== scheme: {scheme} ===")
        prep = P.prepare_nsl_kdd(scheme)
        classes = prep.classes

        for name in GRIDS:
            print(f"  [{name}]")
            model, params = tune_and_fit(name, prep.X_train, prep.y_train)

            # Main evaluation on the full official test set.
            m = evaluate_on(model, prep, prep.y_test, prep.X_test,
                            classes, scheme, name, "test")
            m.update(model=name, scheme=scheme, tag="KDDTest+", params=params)
            rows.append(m)
            print(f"    KDDTest+  macro-F1={m['macro_f1']:.4f}  "
                  f"acc={m['accuracy']:.4f}")

            per_class_blocks.append(
                f"**{name} — {scheme}**\n\n{E.format_per_class_table(m)}\n")

            # Feature importance (both tree models expose it).
            E.plot_feature_importance(
                model.feature_importances_.astype(float), prep.feature_names,
                f"NSL-KDD {name} — top features ({scheme})",
                FIG / f"p3_nslkdd_{name.lower()}_{scheme}_importance.png")

            # Binary: also evaluate on the hard KDDTest-21 subset.
            if scheme == "binary":
                prep21 = P.prepare_nsl_kdd("binary", test_split="test-21")
                m21 = evaluate_on(model, prep21, prep21.y_test, prep21.X_test,
                                  classes, scheme, name, "test21")
                m21.update(model=name, scheme=scheme, tag="KDDTest-21",
                           params=params)
                rows.append(m21)
                print(f"    KDDTest-21 macro-F1={m21['macro_f1']:.4f}  "
                      f"acc={m21['accuracy']:.4f}")

    md = render_metrics_md(rows, per_class_blocks)
    (RESULTS / "metrics.md").write_text(md, encoding="utf-8")
    print(f"\nWrote {RESULTS / 'metrics.md'} and figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
