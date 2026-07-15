"""Correlation, outlier, feature-selection, and representation-learning studies.

This script creates the feature-analysis layer of the lab:

* target correlation ranking
* highly correlated feature-pair map
* IQR outlier-rate map
* representation comparison:
  - raw features + Logistic Regression
  - mutual-information top-k features + Logistic Regression
  - PCA features + Logistic Regression
  - L1-selected features + Logistic Regression
  - small autoencoder embedding + Logistic Regression

NSL-KDD runs by default. CICIoT2023 dev parquet runs automatically if present,
using a bounded stratified sample so the experiment is laptop-safe.

Run:
    .venv/bin/python src/feature_learning.py
"""

from __future__ import annotations

from dataclasses import dataclass
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
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

import ciciot2023 as ciciot
import data as D
import evaluate as E
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
REPORT_PATH = RESULTS / "feature_learning.md"
CORR_CSV = RESULTS / "feature_correlations.csv"
PAIR_CSV = RESULTS / "feature_correlation_pairs.csv"
OUTLIER_CSV = RESULTS / "feature_outliers.csv"
LEARNING_CSV = RESULTS / "feature_learning_results.csv"


@dataclass(frozen=True)
class DatasetBundle:
    """Numeric train/test matrices plus names and binary labels."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    classes: list[str]
    caveat: str


def stratified_cap_indices(y: np.ndarray, max_rows: int, seed: int = D.RANDOM_STATE) -> np.ndarray:
    """Return a deterministic stratified subset up to max_rows."""
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        n = max(1, int(round(max_rows * len(cls_idx) / len(y))))
        n = min(n, len(cls_idx))
        parts.append(rng.choice(cls_idx, size=n, replace=False))
    out = np.sort(np.concatenate(parts))
    if len(out) > max_rows:
        out = np.sort(rng.choice(out, size=max_rows, replace=False))
    return out


def load_nsl_kdd_binary() -> DatasetBundle:
    prep = P.prepare_nsl_kdd("binary")
    return DatasetBundle(
        name="nsl_kdd",
        X_train=prep.X_train,
        X_test=prep.X_test,
        y_train=prep.y_train,
        y_test=prep.y_test,
        feature_names=prep.feature_names,
        classes=prep.classes,
        caveat="Official NSL-KDD split; old synthetic benchmark with strong train/test shift.",
    )


def load_ciciot_dev_binary(max_train: int = 120_000, max_test: int = 40_000) -> DatasetBundle | None:
    """Load the CICIoT2023 dev parquet if present; otherwise skip gracefully."""
    try:
        train = ciciot.load_parquet("train")
        test = ciciot.load_parquet("test")
    except FileNotFoundError:
        return None

    feature_names = ciciot.feature_columns(train)
    # consolidated loader: `binary_label` is the 0/1 target (`label` is now the
    # fine attack name).
    train_idx = stratified_cap_indices(train["binary_label"].to_numpy(), max_train)
    test_idx = stratified_cap_indices(test["binary_label"].to_numpy(), max_test)

    X_train_raw = train.iloc[train_idx][feature_names].to_numpy(dtype=np.float32)
    X_test_raw = test.iloc[test_idx][feature_names].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return DatasetBundle(
        name="ciciot2023_dev",
        X_train=X_train,
        X_test=X_test,
        y_train=train.iloc[train_idx]["binary_label"].to_numpy(dtype=np.int64),
        y_test=test.iloc[test_idx]["binary_label"].to_numpy(dtype=np.int64),
        feature_names=feature_names,
        classes=["benign", "attack"],
        caveat="Downsampled random dev split; not the full official raw CSV release.",
    )


def target_correlations(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    dataset: str,
) -> pd.DataFrame:
    """Point-biserial-style feature/target correlations."""
    y_float = y.astype(float)
    y_std = float(y_float.std())
    rows: list[dict[str, object]] = []
    for i, name in enumerate(feature_names):
        col = X[:, i].astype(float)
        if col.std() == 0 or y_std == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, y_float)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        rows.append(
            {
                "dataset": dataset,
                "feature": name,
                "target_corr": corr,
                "abs_target_corr": abs(corr),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_target_corr", ascending=False)


def high_correlation_pairs(
    X: np.ndarray,
    feature_names: list[str],
    dataset: str,
    threshold: float = 0.95,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    """Find strongly correlated feature pairs."""
    idx = stratified_cap_indices(np.zeros(len(X), dtype=int), max_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X[idx], rowvar=False)
    corr = np.nan_to_num(corr)
    rows: list[dict[str, object]] = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            value = float(corr[i, j])
            if abs(value) >= threshold:
                rows.append(
                    {
                        "dataset": dataset,
                        "feature_a": feature_names[i],
                        "feature_b": feature_names[j],
                        "corr": value,
                        "abs_corr": abs(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False) if rows else pd.DataFrame(
        columns=["dataset", "feature_a", "feature_b", "corr", "abs_corr"]
    )


def outlier_rates_iqr(
    X: np.ndarray,
    feature_names: list[str],
    dataset: str,
) -> pd.DataFrame:
    """Per-feature IQR outlier rates on training data."""
    q1 = np.quantile(X, 0.25, axis=0)
    q3 = np.quantile(X, 0.75, axis=0)
    iqr = q3 - q1
    rows: list[dict[str, object]] = []
    for i, name in enumerate(feature_names):
        if iqr[i] == 0:
            rate = 0.0
            lower = float(q1[i])
            upper = float(q3[i])
        else:
            lower = float(q1[i] - 1.5 * iqr[i])
            upper = float(q3[i] + 1.5 * iqr[i])
            rate = float(((X[:, i] < lower) | (X[:, i] > upper)).mean())
        rows.append(
            {
                "dataset": dataset,
                "feature": name,
                "outlier_rate_iqr": rate,
                "q1": float(q1[i]),
                "q3": float(q3[i]),
                "lower_fence": lower,
                "upper_fence": upper,
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_rate_iqr", ascending=False)


def fit_logreg(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    model.fit(X_train, y_train)
    return model


def evaluate_representation(
    dataset: DatasetBundle,
    representation: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
) -> dict[str, object]:
    model = fit_logreg(X_train, dataset.y_train)
    y_pred = model.predict(X_test)
    metrics = E.compute_metrics(dataset.y_test, y_pred, dataset.classes)
    return {
        "dataset": dataset.name,
        "representation": representation,
        "n_features": n_features,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "mcc": float(matthews_corrcoef(dataset.y_test, y_pred)),
        "normal_recall": metrics["per_class"][dataset.classes[0]]["recall"],
        "attack_recall": metrics["per_class"][dataset.classes[1]]["recall"],
    }


def select_l1_features(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Return selected column indices from an L1 logistic model."""
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        C=0.05,
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    selected = np.flatnonzero(np.abs(model.coef_[0]) > 1e-8)
    if len(selected) == 0:
        selected = np.arange(X_train.shape[1])
    return selected


class AutoEncoder(torch.nn.Module):
    """Small tabular autoencoder for learned numeric embeddings."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden = min(128, max(32, input_dim))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def autoencoder_embeddings(
    X_train: np.ndarray,
    X_test: np.ndarray,
    latent_dim: int = 16,
    max_train_rows: int = 25_000,
    epochs: int = 6,
    batch_size: int = 512,
    seed: int = D.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a tiny autoencoder on train only and return latent embeddings."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_idx = (
        np.arange(len(X_train))
        if len(X_train) <= max_train_rows
        else np.sort(rng.choice(len(X_train), size=max_train_rows, replace=False))
    )
    X_fit = X_train[train_idx].astype(np.float32)
    model = AutoEncoder(input_dim=X_train.shape[1], latent_dim=min(latent_dim, X_train.shape[1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    tensor = torch.from_numpy(X_fit)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tensor))
        for start in range(0, len(tensor), batch_size):
            batch = tensor[order[start:start + batch_size]]
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        z_train = model.encoder(torch.from_numpy(X_train.astype(np.float32))).numpy()
        z_test = model.encoder(torch.from_numpy(X_test.astype(np.float32))).numpy()
    return z_train, z_test


def representation_results(dataset: DatasetBundle, k: int = 30) -> pd.DataFrame:
    """Compare raw, selected, statistical, and learned representations."""
    rows: list[dict[str, object]] = []
    k = min(k, dataset.X_train.shape[1])

    rows.append(
        evaluate_representation(
            dataset,
            "raw_all_features",
            dataset.X_train,
            dataset.X_test,
            dataset.X_train.shape[1],
        )
    )

    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_mi = selector.fit_transform(dataset.X_train, dataset.y_train)
    X_test_mi = selector.transform(dataset.X_test)
    rows.append(evaluate_representation(dataset, f"mutual_info_top_{k}", X_train_mi, X_test_mi, k))

    pca = PCA(n_components=k, random_state=D.RANDOM_STATE)
    X_train_pca = pca.fit_transform(dataset.X_train)
    X_test_pca = pca.transform(dataset.X_test)
    rows.append(
        {
            **evaluate_representation(dataset, f"pca_{k}", X_train_pca, X_test_pca, k),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
        }
    )

    selected = select_l1_features(dataset.X_train, dataset.y_train)
    rows.append(
        evaluate_representation(
            dataset,
            "l1_selected",
            dataset.X_train[:, selected],
            dataset.X_test[:, selected],
            len(selected),
        )
    )

    z_train, z_test = autoencoder_embeddings(dataset.X_train, dataset.X_test, latent_dim=min(16, k))
    rows.append(
        evaluate_representation(
            dataset,
            "autoencoder_embedding_16",
            z_train,
            z_test,
            z_train.shape[1],
        )
    )

    return pd.DataFrame(rows)


def plot_top_bars(df: pd.DataFrame, value_col: str, title: str, path: Path, top_n: int = 20) -> None:
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top[value_col], color="#4C72B0")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(value_col)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def available_datasets() -> list[DatasetBundle]:
    datasets = [load_nsl_kdd_binary()]
    ciciot_dev = load_ciciot_dev_binary()
    if ciciot_dev is not None:
        datasets.append(ciciot_dev)
    return datasets


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    corr_frames = []
    pair_frames = []
    outlier_frames = []
    learning_frames = []
    caveats = {}

    for dataset in available_datasets():
        caveats[dataset.name] = dataset.caveat
        corr = target_correlations(dataset.X_train, dataset.y_train, dataset.feature_names, dataset.name)
        pairs = high_correlation_pairs(dataset.X_train, dataset.feature_names, dataset.name)
        outliers = outlier_rates_iqr(dataset.X_train, dataset.feature_names, dataset.name)
        learning = representation_results(dataset)

        corr_frames.append(corr)
        pair_frames.append(pairs)
        outlier_frames.append(outliers)
        learning_frames.append(learning)

        plot_top_bars(
            corr,
            "abs_target_corr",
            f"{dataset.name}: strongest feature-target correlations",
            FIGURES / f"feature_corr_{dataset.name}.png",
        )
        plot_top_bars(
            outliers,
            "outlier_rate_iqr",
            f"{dataset.name}: highest IQR outlier rates",
            FIGURES / f"feature_outliers_{dataset.name}.png",
        )

    correlations = pd.concat(corr_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    outliers = pd.concat(outlier_frames, ignore_index=True)
    learning = pd.concat(learning_frames, ignore_index=True)

    correlations.to_csv(CORR_CSV, index=False)
    pairs.to_csv(PAIR_CSV, index=False)
    outliers.to_csv(OUTLIER_CSV, index=False)
    learning.to_csv(LEARNING_CSV, index=False)
    REPORT_PATH.write_text(render_markdown(correlations, pairs, outliers, learning, caveats), encoding="utf-8")
    return correlations, pairs, outliers, learning


def render_markdown(
    correlations: pd.DataFrame,
    pairs: pd.DataFrame,
    outliers: pd.DataFrame,
    learning: pd.DataFrame,
    caveats: dict[str, str],
) -> str:
    lines = [
        "# Feature Analysis and Representation Learning",
        "",
        "This report covers correlation analysis, outlier mapping, feature selection, "
        "and learned representations. Every transform is fit on training data only.",
        "",
        "## Dataset Caveats",
        "",
    ]
    for dataset, caveat in caveats.items():
        lines.append(f"- **{dataset}**: {caveat}")

    lines += [
        "",
        "## Representation Results",
        "",
        "| Dataset | Representation | Features | Accuracy | Macro-F1 | MCC | Normal/benign recall | Attack recall | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, r in learning.iterrows():
        notes = ""
        if "explained_variance" in r and not pd.isna(r.get("explained_variance")):
            notes = f"explained variance {r['explained_variance']:.3f}"
        lines.append(
            f"| {r['dataset']} | {r['representation']} | {int(r['n_features'])} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['mcc']:.4f} | "
            f"{r['normal_recall']:.4f} | {r['attack_recall']:.4f} | {notes} |"
        )

    lines += [
        "",
        "## Top Feature-Target Correlations",
        "",
        "| Dataset | Feature | Correlation | |corr| |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, r in correlations.groupby("dataset").head(10).iterrows():
        lines.append(
            f"| {r['dataset']} | {r['feature']} | {r['target_corr']:.4f} | {r['abs_target_corr']:.4f} |"
        )

    lines += [
        "",
        "## Highest IQR Outlier Rates",
        "",
        "| Dataset | Feature | Outlier rate |",
        "| --- | --- | ---: |",
    ]
    for _, r in outliers.groupby("dataset").head(10).iterrows():
        lines.append(f"| {r['dataset']} | {r['feature']} | {r['outlier_rate_iqr']:.4f} |")

    lines += [
        "",
        "## High-Correlation Feature Pairs",
        "",
        "| Dataset | Feature A | Feature B | Correlation |",
        "| --- | --- | --- | ---: |",
    ]
    if pairs.empty:
        lines.append("| all | none above threshold | none | 0.0000 |")
    else:
        for _, r in pairs.groupby("dataset").head(15).iterrows():
            lines.append(
                f"| {r['dataset']} | {r['feature_a']} | {r['feature_b']} | {r['corr']:.4f} |"
            )

    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- Correlation is not causation; it is a triage tool for feature inspection.",
        "- Outlier flags are not automatically bad rows; in security data, rare values can be the signal.",
        "- PCA and autoencoder embeddings are feature-learning baselines, not proof that deep learning is better.",
        "- A representation only matters if it beats or clarifies a strong simple baseline.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    correlations, pairs, outliers, learning = run()
    print("representation results")
    print(learning.to_string(index=False))
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {CORR_CSV}, {PAIR_CSV}, {OUTLIER_CSV}, {LEARNING_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
