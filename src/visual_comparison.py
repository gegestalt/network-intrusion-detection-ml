"""Visual comparison dashboard for the NSL-KDD experiment lab.

The project now produces many tables. This script turns the important ones into
comparison figures so the work is readable at a glance.

Run:
    .venv/bin/python src/visual_comparison.py
"""

from __future__ import annotations

from pathlib import Path
import os
import textwrap

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import data as D

RESULTS = D.REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
REPORT_PATH = RESULTS / "visual_comparison_dashboard.md"
SUMMARY_CSV = RESULTS / "visual_comparison_metric_summary.csv"

FIGURE_PATHS = {
    "metric_heatmap": FIGURES / "nsl_metric_heatmap.png",
    "macro_attack_scatter": FIGURES / "nsl_macro_attack_scatter.png",
    "feature_group_bars": FIGURES / "nsl_feature_group_ablation_bars.png",
    "bootstrap_ci": FIGURES / "nsl_bootstrap_confidence_intervals.png",
    "attribute_importance": FIGURES / "nsl_attribute_consensus_top20.png",
    "threshold_tradeoff": FIGURES / "nsl_threshold_tradeoff.png",
    "coverage_status": FIGURES / "nsl_concept_coverage_status.png",
    "outlier_distribution": FIGURES / "nsl_outlier_score_distribution.png",
    "outlier_top_rows": FIGURES / "nsl_top_outlier_datapoints.png",
    "outlier_heatmap": FIGURES / "nsl_top_outlier_feature_heatmap.png",
    "outlier_feature_frequency": FIGURES / "nsl_outlier_feature_frequency.png",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def model_label(df: pd.DataFrame) -> pd.Series:
    return df["method"].astype(str) + "\n" + df["variant"].astype(str)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "lab": _read(RESULTS / "nsl_kdd_learning_lab.csv"),
        "groups": _read(RESULTS / "nsl_kdd_feature_group_ablation.csv"),
        "bootstrap": _read(RESULTS / "nsl_kdd_bootstrap_ci.csv"),
        "attributes": _read(RESULTS / "attribute_comparison_nsl_kdd.csv"),
        "thresholds": _read(RESULTS / "threshold_ablation.csv"),
        "matrix": _read(RESULTS / "nsl_kdd_concept_application_matrix.csv"),
        "final": _read(RESULTS / "final_comparison.csv"),
    }


def make_metric_summary(lab: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_attack",
        "attack_recall",
        "normal_recall",
        "mcc",
        "fp_per_10k_benign",
        "fn_per_10k_attack",
    ]
    frames = []
    if not lab.empty:
        use = lab.copy()
        use["source"] = "method_lab"
        use["comparison_label"] = model_label(use)
        frames.append(use)
    if not groups.empty:
        use = groups.copy()
        use["source"] = "feature_group_ablation"
        use["comparison_label"] = use["variant"].astype(str)
        frames.append(use)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    keep = ["dataset", "source", "level", "family", "method", "variant", "comparison_label", *metric_cols]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]


def plot_metric_heatmap(lab: pd.DataFrame) -> None:
    if lab.empty:
        return
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_attack",
        "attack_recall",
        "normal_recall",
        "mcc",
    ]
    top = lab.sort_values("macro_f1", ascending=False).head(18).copy()
    top["label"] = model_label(top)
    heat = top.set_index("label")[metrics]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(heat))))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis", linewidths=0.4, ax=ax)
    ax.set_title("NSL-KDD model comparison: core metrics", fontweight="bold")
    ax.set_xlabel("metric")
    ax.set_ylabel("model / variant")
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["metric_heatmap"], dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_macro_attack_scatter(lab: pd.DataFrame) -> None:
    if lab.empty:
        return
    df = lab.copy()
    df["label"] = df["method"].astype(str)

    # Dummy baselines are important evidence, but if plotted on the same axis
    # they force 80% of the competitive model region into a tiny corner. Keep
    # them in the annotation and focus the chart on models worth comparing.
    focus = df[df["family"] != "naive_baseline"].copy()
    if focus.empty:
        focus = df.copy()

    x_pad = max(0.01, (focus["attack_recall"].max() - focus["attack_recall"].min()) * 0.18)
    y_pad = max(0.01, (focus["macro_f1"].max() - focus["macro_f1"].min()) * 0.18)
    x_min = max(0.0, focus["attack_recall"].min() - x_pad)
    x_max = min(1.0, focus["attack_recall"].max() + x_pad)
    y_min = max(0.0, focus["macro_f1"].min() - y_pad)
    y_max = min(1.0, focus["macro_f1"].max() + y_pad)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=focus,
        x="attack_recall",
        y="macro_f1",
        hue="family",
        size="fp_per_10k_benign",
        sizes=(90, 430),
        alpha=0.86,
        edgecolor="white",
        linewidth=0.7,
        ax=ax,
    )
    offsets = [(6, 6), (6, -12), (-72, 6), (-80, -12), (8, 18), (-86, 18), (10, -24), (-92, -24)]
    for n, (_, row) in enumerate(focus.sort_values("macro_f1", ascending=False).head(8).iterrows()):
        dx, dy = offsets[n % len(offsets)]
        ax.annotate(
            row["method"],
            (row["attack_recall"], row["macro_f1"]),
            fontsize=9,
            xytext=(dx, dy),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.72},
        )
    ax.set_title("NSL-KDD tradeoff: macro-F1 vs attack recall", fontweight="bold")
    ax.set_xlabel("attack recall")
    ax.set_ylabel("macro-F1")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)
    naive = df[df["family"] == "naive_baseline"]
    if not naive.empty:
        summary = ", ".join(
            f"{r['variant']}: F1={r['macro_f1']:.3f}, recall={r['attack_recall']:.3f}"
            for _, r in naive.sort_values("macro_f1", ascending=False).iterrows()
        )
        ax.text(
            0.01,
            0.01,
            f"Naive baselines excluded from axis scaling: {summary}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB", "alpha": 0.86},
        )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, title=None)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["macro_attack_scatter"], dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_groups(groups: pd.DataFrame) -> None:
    if groups.empty:
        return
    df = groups.sort_values("macro_f1", ascending=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * len(df))))
    ax.barh(y - 0.16, df["macro_f1"], height=0.32, label="macro-F1", color="#4C72B0")
    ax.barh(y + 0.16, df["attack_recall"], height=0.32, label="attack recall", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(df["variant"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title("NSL-KDD feature-group ablation", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["feature_group_bars"], dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    if bootstrap.empty:
        return
    df = bootstrap.copy()
    df["label"] = df["model"] + "\n" + df["metric"]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(df))))
    lower = df["mean"] - df["ci_low"]
    upper = df["ci_high"] - df["mean"]
    ax.errorbar(df["mean"], y, xerr=[lower, upper], fmt="o", color="#4C72B0", ecolor="#333333", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("score with 95% bootstrap interval")
    ax.set_title("NSL-KDD bootstrap uncertainty", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["bootstrap_ci"], dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_attribute_importance(attributes: pd.DataFrame) -> None:
    if attributes.empty:
        return
    top = attributes.sort_values("importance_consensus", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = top["raw_type"].map({"categorical": "#4C72B0", "numeric": "#55A868", "binary_numeric": "#C44E52"}).fillna("#8172B2")
    ax.barh(top["attribute"], top["importance_consensus"], color=colors)
    ax.set_xlabel("consensus importance")
    ax.set_title("NSL-KDD top raw attributes by consensus signal", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["attribute_importance"], dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(thresholds: pd.DataFrame) -> None:
    if thresholds.empty:
        return
    df = thresholds.copy()
    df["label"] = df["model"].astype(str) + "\n" + df["threshold_name"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="normal_recall",
        y="attack_recall",
        hue="model",
        size="macro_f1",
        sizes=(60, 300),
        alpha=0.85,
        ax=ax,
    )
    for _, row in df.sort_values("macro_f1", ascending=False).head(8).iterrows():
        ax.annotate(row["threshold_name"], (row["normal_recall"], row["attack_recall"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_title("Threshold policy tradeoff: normal recall vs attack recall", fontweight="bold")
    ax.set_xlabel("normal recall")
    ax.set_ylabel("attack recall")
    ax.set_xlim(max(0.0, df["normal_recall"].min() - 0.03), 1.0)
    ax.set_ylim(max(0.0, df["attack_recall"].min() - 0.03), 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["threshold_tradeoff"], dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(matrix: pd.DataFrame) -> None:
    if matrix.empty:
        return
    counts = matrix["status"].value_counts().reindex(["APPLIED", "NOT_APPLIED"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#55A868", "#C44E52"])
    ax.set_ylabel("concept count")
    ax.set_title("First-dataset concept coverage", fontweight="bold")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.05, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["coverage_status"], dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_report(summary: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# Visual Comparison Dashboard",
        "",
        "This dashboard exists because tables alone do not make the experiment work visible.",
        "",
        "## Generated Figures",
        "",
    ]
    descriptions = {
        "metric_heatmap": "Core model metrics across the strongest NSL-KDD model rows.",
        "macro_attack_scatter": "Macro-F1 versus attack recall; point size shows benign false-positive burden.",
        "feature_group_bars": "Feature-group ablation showing which information families carry signal.",
        "bootstrap_ci": "Bootstrap uncertainty for selected model/metric pairs.",
        "attribute_importance": "Top raw attributes by consensus signal across correlation, MI, LR, and RF signals.",
        "threshold_tradeoff": "How threshold policy moves normal recall against attack recall.",
        "coverage_status": "Which first-dataset requested concepts are applied versus intentionally not applied.",
        "outlier_distribution": "Distribution of row-level outlier feature counts by normal versus attack.",
        "outlier_top_rows": "Actual split/row IDs with the most numeric outlier features.",
        "outlier_heatmap": "Which features caused the top outlier rows to be flagged.",
        "outlier_feature_frequency": "Numeric features most often responsible for row-level outlier flags.",
    }
    for key, path in FIGURE_PATHS.items():
        if path.exists():
            lines.append(f"- `{path.relative_to(D.REPO_ROOT)}`: {descriptions[key]}")
    lines += [
        "",
        "## Metric Coverage",
        "",
        f"- Metric summary rows: **{len(summary)}**",
        f"- NSL-KDD model-lab rows: **{len(inputs['lab'])}**",
        f"- NSL-KDD feature-group rows: **{len(inputs['groups'])}**",
        f"- Attribute-audit rows: **{len(inputs['attributes'])}**",
        "",
        "## Referee Reading",
        "",
        "- Do not use one metric alone. Macro-F1, attack recall, precision, and false-positive burden tell different stories.",
        "- The scatter plot is usually more honest than a sorted leaderboard because it shows tradeoffs.",
        "- Feature-group bars show what information families are doing, not merely which final model won.",
        "- Bootstrap intervals show that tiny score differences should not be overclaimed.",
        "",
    ]
    if not summary.empty:
        top = summary[summary["source"] == "method_lab"].sort_values("macro_f1", ascending=False).head(5)
        cols = ["comparison_label", "macro_f1", "attack_recall", "precision_attack", "fp_per_10k_benign"]
        lines += ["## Top Method Rows", "", "| model | macro_f1 | attack_recall | precision_attack | fp_per_10k_benign |", "| --- | ---: | ---: | ---: | ---: |"]
        for _, row in top.iterrows():
            label = str(row["comparison_label"]).replace("\n", " / ")
            label = "<br>".join(textwrap.wrap(label, width=40))
            lines.append(
                f"| {label} | {row['macro_f1']:.4f} | {row['attack_recall']:.4f} | "
                f"{row['precision_attack']:.4f} | {row['fp_per_10k_benign']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run() -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    summary = make_metric_summary(inputs["lab"], inputs["groups"])
    if not summary.empty:
        summary.to_csv(SUMMARY_CSV, index=False)

    plot_metric_heatmap(inputs["lab"])
    plot_macro_attack_scatter(inputs["lab"])
    plot_feature_groups(inputs["groups"])
    plot_bootstrap(inputs["bootstrap"])
    plot_attribute_importance(inputs["attributes"])
    plot_threshold_tradeoff(inputs["thresholds"])
    plot_coverage(inputs["matrix"])
    REPORT_PATH.write_text(render_report(summary, inputs), encoding="utf-8")
    return summary


def main() -> int:
    summary = run()
    print(f"visual metric rows: {len(summary)}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {REPORT_PATH}")
    for path in FIGURE_PATHS.values():
        if path.exists():
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
