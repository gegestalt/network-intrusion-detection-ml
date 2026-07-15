# Learning note 04 — CICIoT2023 supervised baselines

*Companion for the CICIoT2023 baselines (Stages 0-1 + trees/boosting). First
modern-dataset results in the lab.*

---

## Setup
Parquet dev sample (train subsampled to 200k stratified; full 268k test),
**scale-only** preprocessing (all 39 features numeric), `class_weight='balanced'`
everywhere it's supported. Two label levels: binary (benign/attack) and the
8-category level. **Random split → in-distribution**, so scores are higher than
NSL-KDD's shift-heavy official split (a protocol difference, not model quality).

## Headline (8-category)

| Model | Macro-F1 | Balanced-acc | Web-based rec | Brute Force rec |
|---|---:|---:|---:|---:|
| RandomForest | **0.736** | 0.721 | 0.325 | 0.354 |
| LightGBM | 0.724 | 0.765 | 0.594 | 0.532 |
| HistGB | 0.702 | **0.763** | **0.627** | **0.597** |
| LogReg (balanced) | 0.569 | 0.629 | 0.462 | 0.404 |
| Dummy | 0.125 | 0.125 | 0.02 | 0.01 |

## The lesson: your metric picks your winner
- **RandomForest wins macro-F1** (0.736) — but has the **worst rare-class recall**
  (Web-based 0.33, Brute Force 0.35). Its macro-F1 lead comes from nailing the
  big categories (DDoS/DoS/Recon), not the rare ones.
- **HistGB wins balanced-accuracy and rare-class recall** (Web-based 0.63, Brute
  Force 0.60) while trailing slightly on macro-F1. If your operational priority is
  *catching the rare attacks*, HistGB is the better model here — the leaderboard
  metric hides that.
- This is the same tension as NSL-KDD, one level up: **there is no single "best"
  model — it depends on whether you optimise aggregate F1, class-balanced
  accuracy, or rare-class recall.** State the objective before naming a winner.

## Contrast with NSL-KDD
On NSL-KDD (small, shifted), balanced LogReg/HistGB *beat* the trees on macro-F1.
On CICIoT2023 (larger, cleaner, in-distribution) the **trees pull ahead on
aggregate** — but HistGB still leads on the rare classes. So "imbalance handling +
model choice > complexity" holds for *rare-class* performance, while raw
tree power helps most when the data is plentiful and the split is easy.

## Limitations / next
- In-distribution random split flatters everyone; a device- or time-based split
  would be harder and more honest (future stage).
- Dev subsample (200k) — confirm at full scale later.
- Next stages: MLP + activation ablations, imbalance arsenal (SMOTE/focal/
  thresholds), then benign-only anomaly detection with held-out attack categories.
