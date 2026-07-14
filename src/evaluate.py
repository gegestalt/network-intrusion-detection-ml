"""Evaluation utilities: metrics + plots, shared by every model and dataset.

Kept deliberately model-agnostic — it takes ``y_true``/``y_pred`` (and optional
scores), never a model — so Random Forest, LightGBM, and the PyTorch MLP are all
scored by the exact same code. This is what makes the cross-model comparison
honest.

Headline philosophy (see nids-conventions skill): lead with **macro-F1** and
**per-class recall**, not accuracy; show confusion matrices raw *and*
row-normalized; for binary tasks report **ROC-AUC and PR-AUC** (PR-AUC is more
informative under heavy class imbalance).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    classes: list[str],
                    y_score: np.ndarray | None = None) -> dict:
    """Compute the standard metric bundle.

    Parameters
    ----------
    y_true, y_pred : integer-encoded labels in ``range(len(classes))``.
    classes : label names, index i <-> integer i.
    y_score : optional. For a **binary** task, the probability of the positive
        (index-1) class; enables ROC-AUC and PR-AUC.

    Returns
    -------
    dict with: ``accuracy``; ``macro_precision/recall/f1``; ``weighted_f1``;
    ``per_class`` (name -> {precision, recall, f1, support}); and, when binary
    scores are given, ``roc_auc`` and ``pr_auc``.
    """
    labels = list(range(len(classes)))
    acc = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class = {
        classes[i]: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f1[i]), "support": int(support[i])}
        for i in labels
    }

    out = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }

    if y_score is not None and len(classes) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))

    return out


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                            classes: list[str], title: str, path: Path) -> None:
    """Save a two-panel figure: raw counts and row-normalized (recall)."""
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(6 + 1.4 * len(classes), 5))
    for ax, mat, fmt, sub in ((axes[0], cm, "d", "counts"),
                              (axes[1], cm_norm, ".2f", "row-normalized (recall)")):
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    vmin=0, vmax=(cm.max() if fmt == "d" else 1.0))
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(sub)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray,
                title: str, path: Path) -> None:
    """Binary ROC and Precision-Recall curves side by side."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC curve"); axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color="darkorange", label=f"PR-AUC = {pr_auc:.3f}")
    baseline = y_true.mean()
    axes[1].axhline(baseline, ls="--", color="grey", lw=1,
                    label=f"baseline = {baseline:.3f}")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall curve"); axes[1].legend(loc="lower left")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, feature_names: list[str],
                            title: str, path: Path, top_n: int = 20) -> None:
    """Horizontal bar chart of the top-N most important features."""
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order][::-1]
    vals = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
    ax.barh(names, vals, color="#4C72B0")
    ax.set_xlabel("importance"); ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_per_class_table(metrics: dict) -> str:
    """Markdown table of per-class precision/recall/F1/support."""
    lines = ["| class | precision | recall | f1 | support |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, m in metrics["per_class"].items():
        lines.append(f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['f1']:.4f} | {m['support']:,} |")
    return "\n".join(lines)
