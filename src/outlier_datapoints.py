"""Row-level outlier mapping for NSL-KDD.

Feature-level outlier rates answer "which columns are spiky?" This script
answers the row-level question: which actual records are outliers, which
features made them outliers, and what labels/families do they belong to?

Run:
    .venv/bin/python src/outlier_datapoints.py
"""

from __future__ import annotations

from pathlib import Path
import os

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

DATAPOINT_CSV = RESULTS / "nsl_kdd_outlier_datapoints.csv"
FEATURE_CSV = RESULTS / "nsl_kdd_outlier_feature_frequency.csv"
LONG_CSV = RESULTS / "nsl_kdd_top_outlier_feature_matrix.csv"
REPORT_PATH = RESULTS / "nsl_kdd_outlier_datapoints.md"

PLOT_DISTRIBUTION = FIGURES / "nsl_outlier_score_distribution.png"
PLOT_TOP_ROWS = FIGURES / "nsl_top_outlier_datapoints.png"
PLOT_HEATMAP = FIGURES / "nsl_top_outlier_feature_heatmap.png"
PLOT_FEATURE_FREQ = FIGURES / "nsl_outlier_feature_frequency.png"


def iqr_reference(train: pd.DataFrame) -> pd.DataFrame:
    """Fit IQR fences on train numeric features only."""
    rows = []
    for feature in D.NUMERIC_COLS:
        values = train[feature].astype(float)
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        median = float(values.median())
        if iqr == 0:
            lower = q1
            upper = q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        rows.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "median": median,
                "lower_fence": lower,
                "upper_fence": upper,
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def score_split(df: pd.DataFrame, split: str, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each row and return row summary plus long outlier-feature matrix."""
    numeric = df[D.NUMERIC_COLS].astype(float)
    lower = ref.loc[D.NUMERIC_COLS, "lower_fence"].to_numpy()
    upper = ref.loc[D.NUMERIC_COLS, "upper_fence"].to_numpy()
    med = ref.loc[D.NUMERIC_COLS, "median"].to_numpy()
    iqr = ref.loc[D.NUMERIC_COLS, "iqr"].to_numpy()
    iqr_safe = np.where(iqr == 0, 1.0, iqr)
    values = numeric.to_numpy()
    mask = (values < lower) | (values > upper)
    robust_z = np.abs((values - med) / iqr_safe)
    robust_z[:, iqr == 0] = np.abs(values[:, iqr == 0] - med[iqr == 0])

    rows = []
    long_rows = []
    for row_pos in range(len(df)):
        feature_idx = np.flatnonzero(mask[row_pos])
        if len(feature_idx):
            order = feature_idx[np.argsort(robust_z[row_pos, feature_idx])[::-1]]
            top_features = [D.NUMERIC_COLS[i] for i in order[:8]]
            max_robust_z = float(np.nanmax(robust_z[row_pos, feature_idx]))
        else:
            top_features = []
            max_robust_z = 0.0

        rows.append(
            {
                "dataset": "nsl_kdd",
                "split": split,
                "row_index": row_pos,
                "label": df.iloc[row_pos]["label"],
                "binary_label": int(df.iloc[row_pos]["binary_label"]),
                "attack_family": df.iloc[row_pos]["attack_family"],
                "outlier_feature_count": int(len(feature_idx)),
                "outlier_feature_fraction": float(len(feature_idx) / len(D.NUMERIC_COLS)),
                "max_abs_robust_z": max_robust_z,
                "top_outlier_features": ", ".join(top_features),
            }
        )
        for i in feature_idx:
            long_rows.append(
                {
                    "dataset": "nsl_kdd",
                    "split": split,
                    "row_index": row_pos,
                    "label": df.iloc[row_pos]["label"],
                    "attack_family": df.iloc[row_pos]["attack_family"],
                    "feature": D.NUMERIC_COLS[i],
                    "value": float(values[row_pos, i]),
                    "lower_fence": float(lower[i]),
                    "upper_fence": float(upper[i]),
                    "abs_robust_z": float(robust_z[row_pos, i]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(long_rows)


def feature_frequency(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["split", "feature", "outlier_rows", "mean_abs_robust_z"])
    return (
        long_df.groupby(["split", "feature"])
        .agg(outlier_rows=("row_index", "count"), mean_abs_robust_z=("abs_robust_z", "mean"))
        .reset_index()
        .sort_values(["split", "outlier_rows"], ascending=[True, False])
    )


def plot_distribution(rows: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    use = rows.copy()
    use["class"] = np.where(use["binary_label"] == 1, "attack", "normal")
    sns.histplot(
        data=use,
        x="outlier_feature_count",
        hue="class",
        bins=range(0, int(use["outlier_feature_count"].max()) + 2),
        element="step",
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_title("NSL-KDD row-level outlier-feature count distribution", fontweight="bold")
    ax.set_xlabel("number of numeric features outside train-fitted IQR fences")
    ax.set_ylabel("density")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DISTRIBUTION, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_top_rows(rows: pd.DataFrame, top_n: int = 30) -> None:
    top = rows.sort_values(["outlier_feature_count", "max_abs_robust_z"], ascending=False).head(top_n).iloc[::-1]
    labels = top.apply(lambda r: f"{r['split']}#{int(r['row_index'])} {r['attack_family']}", axis=1)
    colors = top["binary_label"].map({0: "#4C72B0", 1: "#C44E52"})
    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(top))))
    ax.barh(labels, top["outlier_feature_count"], color=colors)
    ax.set_xlabel("outlier feature count")
    ax.set_title("Top NSL-KDD rows by number of outlier features", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_TOP_ROWS, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows: pd.DataFrame, long_df: pd.DataFrame, top_n: int = 40) -> None:
    if long_df.empty:
        return
    top = rows.sort_values(["outlier_feature_count", "max_abs_robust_z"], ascending=False).head(top_n)
    key = top[["split", "row_index"]].copy()
    key["row_key"] = key["split"] + "#" + key["row_index"].astype(str)
    use = long_df.merge(key, on=["split", "row_index"], how="inner")
    most_common = use["feature"].value_counts().head(25).index.tolist()
    use = use[use["feature"].isin(most_common)]
    matrix = (
        use.pivot_table(index="row_key", columns="feature", values="abs_robust_z", aggfunc="max", fill_value=0.0)
        .reindex(key["row_key"])
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(13, max(7, 0.22 * len(matrix))))
    sns.heatmap(np.log1p(matrix), cmap="magma", ax=ax, cbar_kws={"label": "log(1 + abs robust z)"})
    ax.set_title("Top outlier rows: which features triggered the outlier score", fontweight="bold")
    ax.set_xlabel("numeric feature")
    ax.set_ylabel("split#row_index")
    fig.tight_layout()
    fig.savefig(PLOT_HEATMAP, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_frequency(freq: pd.DataFrame, top_n: int = 20) -> None:
    if freq.empty:
        return
    top_features = freq.groupby("feature")["outlier_rows"].sum().sort_values(ascending=False).head(top_n).index
    use = freq[freq["feature"].isin(top_features)]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(top_features))))
    sns.barplot(data=use, y="feature", x="outlier_rows", hue="split", ax=ax)
    ax.set_title("Numeric features most often responsible for row outliers", fontweight="bold")
    ax.set_xlabel("outlier row-feature hits")
    ax.set_ylabel("feature")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_FEATURE_FREQ, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_report(rows: pd.DataFrame, freq: pd.DataFrame) -> str:
    top = rows.sort_values(["outlier_feature_count", "max_abs_robust_z"], ascending=False).head(25)
    lines = [
        "# NSL-KDD Row-Level Outlier Datapoints",
        "",
        "IQR fences are fitted on KDDTrain+ numeric features only, then applied to train and KDDTest+.",
        "This identifies actual rows whose numeric values sit outside the training distribution.",
        "",
        "## Outputs",
        "",
        f"- Row summary: `results/{DATAPOINT_CSV.name}`",
        f"- Feature-frequency table: `results/{FEATURE_CSV.name}`",
        f"- Long row-feature matrix: `results/{LONG_CSV.name}`",
        f"- Distribution plot: `results/figures/{PLOT_DISTRIBUTION.name}`",
        f"- Top-row plot: `results/figures/{PLOT_TOP_ROWS.name}`",
        f"- Row-feature heatmap: `results/figures/{PLOT_HEATMAP.name}`",
        f"- Feature-frequency plot: `results/figures/{PLOT_FEATURE_FREQ.name}`",
        "",
        "## Top Outlier Datapoints",
        "",
        "| split | row_index | label | family | outlier_feature_count | max_abs_robust_z | top_outlier_features |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['split']} | {int(row['row_index'])} | {row['label']} | {row['attack_family']} | "
            f"{int(row['outlier_feature_count'])} | {row['max_abs_robust_z']:.2f} | {row['top_outlier_features']} |"
        )
    lines += [
        "",
        "## Most Frequent Outlier Features",
        "",
        "| split | feature | outlier_rows | mean_abs_robust_z |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in freq.groupby("split").head(12).iterrows():
        lines.append(
            f"| {row['split']} | {row['feature']} | {int(row['outlier_rows'])} | {row['mean_abs_robust_z']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- Outlier datapoints are not automatically bad data. In intrusion detection, rare/extreme rows can be the actual attack signal.",
        "- The key question is whether outlier rows cluster in certain labels, families, or features.",
        "- Since fences are fitted on train only, test rows are judged against the learned training distribution rather than their own distribution.",
        "",
    ]
    return "\n".join(lines)


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    train = D.load_nsl_kdd("train")
    test = D.load_nsl_kdd("test")
    ref = iqr_reference(train)
    train_rows, train_long = score_split(train, "train", ref)
    test_rows, test_long = score_split(test, "test", ref)
    rows = pd.concat([train_rows, test_rows], ignore_index=True)
    long_df = pd.concat([train_long, test_long], ignore_index=True)
    freq = feature_frequency(long_df)
    rows.to_csv(DATAPOINT_CSV, index=False)
    freq.to_csv(FEATURE_CSV, index=False)
    long_df.to_csv(LONG_CSV, index=False)
    plot_distribution(rows)
    plot_top_rows(rows)
    plot_heatmap(rows, long_df)
    plot_feature_frequency(freq)
    REPORT_PATH.write_text(render_report(rows, freq), encoding="utf-8")
    return rows, freq, long_df


def main() -> int:
    rows, freq, long_df = run()
    print(f"row summaries: {len(rows)}")
    print(f"row-feature outlier hits: {len(long_df)}")
    print(f"feature frequency rows: {len(freq)}")
    print("top outlier datapoints")
    print(rows.sort_values(["outlier_feature_count", "max_abs_robust_z"], ascending=False).head(10).to_string(index=False))
    print(f"Wrote {DATAPOINT_CSV}")
    print(f"Wrote {FEATURE_CSV}")
    print(f"Wrote {LONG_CSV}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
