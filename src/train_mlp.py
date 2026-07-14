"""Phase 4 — PyTorch MLP on NSL-KDD, with a class-weighting ablation.

Same preprocessed data and same evaluation protocol as the Phase 3 trees, so the
comparison is apples-to-apples. The headline experiment: train the MLP **with**
and **without** inverse-frequency class weights and measure the effect on the
rare classes (attack recall for binary; R2L/U2R recall for 5-class).

Architecture: 122 -> 128 -> 64 -> n_classes, ReLU + dropout, Adam, early stopping
on validation macro-F1. Trains on Apple-Silicon MPS when available.

Run:  .venv/bin/python src/train_mlp.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn

import data as D
import evaluate as E
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
FIG = RESULTS / "figures"
SEED = D.RANDOM_STATE

BATCH = 512
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class MLP(nn.Module):
    """2 hidden layers with ReLU + dropout; outputs one logit per class."""

    def __init__(self, in_dim: int, n_classes: int,
                 hidden: tuple[int, int] = (128, 64), dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights (sklearn 'balanced'): n / (K * count_c)."""
    counts = np.bincount(y, minlength=n_classes)
    weights = len(y) / (n_classes * np.clip(counts, 1, None))
    return weights.astype(np.float32)


def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool):
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


def train_model(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                class_weight: np.ndarray | None = None,
                device: torch.device | None = None,
                verbose: bool = False) -> MLP:
    """Train with early stopping on a stratified validation split."""
    device = device or get_device()
    set_seed()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

    model = MLP(X_train.shape[1], n_classes).to(device)
    weight_t = (torch.as_tensor(class_weight, dtype=torch.float32).to(device)
                if class_weight is not None else None)
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = _loader(X_tr, y_tr, shuffle=True)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(device)

    best_f1, best_state, since_improve = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # validation macro-F1 for early stopping
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).argmax(1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if val_f1 > best_f1 + 1e-4:
            best_f1, since_improve = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improve += 1
        if verbose:
            print(f"      epoch {epoch:3d}  val_macroF1={val_f1:.4f}")
        if since_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: MLP, X: np.ndarray,
            device: torch.device | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, probabilities) for features ``X``."""
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32).to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba.argmax(1), proba


def _append_section(md_path: Path, marker: str, block: str) -> None:
    """Append (or replace) a marked section in a markdown file, idempotently."""
    existing = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n"
    md_path.write_text(existing + "\n" + block, encoding="utf-8")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")
    rows: list[dict] = []

    for scheme in ("binary", "multiclass"):
        print(f"\n=== scheme: {scheme} ===")
        prep = P.prepare_nsl_kdd(scheme)
        classes = prep.classes
        n_classes = len(classes)

        for weighted in (False, True):
            tag = "weighted" if weighted else "unweighted"
            print(f"  [MLP {tag}]")
            cw = compute_class_weights(prep.y_train, n_classes) if weighted else None
            model = train_model(prep.X_train, prep.y_train, n_classes,
                                class_weight=cw, device=device)
            y_pred, proba = predict(model, prep.X_test, device=device)
            y_score = proba[:, 1] if n_classes == 2 else None
            m = E.compute_metrics(prep.y_test, y_pred, classes, y_score=y_score)
            m.update(model=f"MLP ({tag})", scheme=scheme, tag=tag)
            rows.append(m)

            slug = f"p4_nslkdd_mlp_{tag}_{scheme}"
            E.plot_confusion_matrices(
                prep.y_test, y_pred, classes,
                f"NSL-KDD MLP ({tag}) — {scheme}", FIG / f"{slug}_confusion.png")
            print(f"    macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")

    # --- comparison table focused on rare-class recall --------------------- #
    lines = ["# NSL-KDD — Phase 4: MLP + class-weighting ablation", "",
             "Same preprocessing and evaluation as Phase 3. The point of this "
             "phase is the **with vs without class-weighting** contrast on the "
             "rare classes.", "",
             "| Variant | Task | Accuracy | Macro-F1 | Rare-class recall |",
             "| --- | --- | ---: | ---: | --- |"]
    for r in rows:
        pc = r["per_class"]
        if r["scheme"] == "binary":
            rare = f"attack {pc['attack']['recall']:.3f}"
        else:
            rare = f"R2L {pc['R2L']['recall']:.3f} / U2R {pc['U2R']['recall']:.3f}"
        lines.append(f"| {r['model']} | {r['scheme']} | {r['accuracy']:.4f} | "
                     f"{r['macro_f1']:.4f} | {rare} |")
    _append_section(RESULTS / "metrics.md",
                    "# NSL-KDD — Phase 4", "\n".join(lines) + "\n")
    print(f"\nAppended Phase 4 results to {RESULTS / 'metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
