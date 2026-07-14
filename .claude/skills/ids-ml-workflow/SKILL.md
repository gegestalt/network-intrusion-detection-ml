---
name: ids-ml-workflow
description: >-
  Conventions for this NSL-KDD network-intrusion-detection ML project. Load
  whenever working on NSL-KDD data, building/using the preprocessing pipeline,
  training or evaluating intrusion-detection models (Random Forest, gradient
  boosting, MLP), producing confusion matrices / per-class F1, or writing up
  results. Encodes the dataset schema, preprocessing rules, the official
  train/test evaluation protocol (no random re-splitting), and the results
  documentation style. Use it to keep every phase reproducible and consistent.
---

# NSL-KDD Intrusion-Detection ML Workflow

Project conventions for a research-grade intrusion-detection ML project on the
**NSL-KDD** dataset. Follow these so that data handling, evaluation, and
reporting stay consistent and reproducible across sessions and phases.

## When to use
- Loading or preprocessing NSL-KDD (`KDDTrain+.txt`, `KDDTest+.txt`).
- Training/evaluating any model (RandomForest, XGBoost/LightGBM, MLP).
- Building confusion matrices, per-class metrics, ROC/PR curves.
- Writing results into `results/` or the README.

## Golden rules (do not violate)
1. **Use the official split.** Train on `KDDTrain+`, test on `KDDTest+`.
   **Never** concatenate and random-split, and never fit any transformer,
   scaler, encoder, resampler (SMOTE), or hyperparameter search on the test set.
   The official test set intentionally contains attack *types* unseen in
   training — this is what makes the benchmark honest and hard. Report on it as-is.
2. **Fit on train, transform on test.** All preprocessing state (scaler stats,
   one-hot vocab, feature selection) is learned from train only. `service` has
   categories in test that are absent in train → `OneHotEncoder(handle_unknown="ignore")`.
3. **Reproducibility.** Set `RANDOM_STATE = 42` everywhere (numpy, sklearn,
   torch, and torch CUDA/MPS). Log library versions into results.
4. **Lead with macro-F1 and per-class recall, not accuracy.** The data is highly
   imbalanced; accuracy is misleading (see below).

## Dataset schema
43 columns: 41 features + `label` + `difficulty` (last column; drop for modeling,
keep for optional difficulty-stratified analysis). Column order is fixed — define
once in `src/` as `COLUMN_NAMES` and reuse. Files are headerless, comma-separated.

- **3 categorical features:** `protocol_type` (3 vals), `service` (~70 vals),
  `flag` (11 vals).
- **Numeric features:** the remaining 38 (mix of counts, rates in [0,1], bytes).
- **4 binary-valued** numerics (`land`, `logged_in`, `root_shell`,
  `su_attempted`, `is_host_login`, `is_guest_login`) — leave as 0/1, scaling is
  harmless but do not one-hot them.

## Labeling schemes (build both)
The raw `label` is a specific attack name (e.g. `neptune`, `satan`) or `normal`.
- **Binary:** `normal` → 0, everything else → `attack` (1).
- **5-class:** map each attack name to its family via `ATTACK_FAMILY_MAP`:
  - **DoS:** neptune, smurf, back, teardrop, pod, land, apache2, udpstorm,
    processtable, mailbomb, worm
  - **Probe:** satan, ipsweep, nmap, portsweep, mscan, saint
  - **R2L:** guess_passwd, ftp_write, imap, phf, multihop, warezmaster,
    warezclient, spy, xlock, xsnoop, snmpguess, snmpgetattack, httptunnel,
    sendmail, named
  - **U2R:** buffer_overflow, loadmodule, rootkit, perl, sqlattack, xterm, ps
  - **normal:** normal
  Any label not in the map must raise, not silently drop — the test set adds new
  attack names and each must be assigned a family deliberately.

## Preprocessing pipeline (`src/`)
Build with `ColumnTransformer` inside a `Pipeline` so it is fit-once, serialize-able:
- Categorical → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Numeric → `StandardScaler()` (tree models are scale-invariant, but a single
  shared pipeline keeps RF and MLP strictly apples-to-apples).
- Expose reusable functions: `load_nsl_kdd()`, `build_preprocessor()`,
  `make_labels(scheme)`, `get_feature_names()`. No preprocessing logic inside
  notebooks — notebooks import from `src/`.

## Evaluation protocol (identical for every model)
Report all of the following on `KDDTest+`:
- **accuracy**, **macro precision/recall/F1**, **weighted F1**.
- **Per-class precision / recall / F1** (full `classification_report`). U2R and
  R2L are the classes that matter — they are rare and operationally the most
  dangerous (privilege escalation, remote-to-local). A model can score 99% on
  DoS and be useless on U2R; always show the per-class breakdown.
- **Confusion matrix** — both raw counts and row-normalized (recall) versions.
- **Binary task also:** ROC-AUC and PR-AUC (PR-AUC is more informative under
  imbalance). Pick/operating-threshold discussion where relevant.
- **Hyperparameter search:** cross-validate on the **training set only**
  (`StratifiedKFold`), scoring on `f1_macro`. Never touch test during tuning.
- **Cost framing:** in intrusion detection a false negative (missed attack)
  usually costs more than a false positive. Note recall on attack classes
  prominently and mention the FN/FP trade-off.

## Results & documentation style
- **Figures:** save to `results/figures/` as `<phase>_<what>.png`, 150+ DPI,
  titled, axis-labelled, colorblind-safe palette. E.g.
  `p1_class_distribution.png`, `p3_rf_confusion_binary.png`,
  `p3_rf_feature_importance.png`, `p4_mlp_vs_rf.png`.
- **Metrics:** append machine-comparable tables to `results/metrics.md`. One
  markdown table per (model × task), plus a consolidated comparison table.
  Columns: Model, Task, Accuracy, Macro-F1, Weighted-F1, and per-class recall
  for the 5-class runs. Keep numbers to 4 decimals.
- **Interpretation:** every notable result gets a plain-English,
  security-analyst-framed explanation (what it means for a SOC, not just the
  number). Tie feature-importance findings to attack behaviour (e.g. high
  `src_bytes`/`count`/`same_srv_rate` → DoS/Probe signatures).
- **Honesty:** always state NSL-KDD's limitations near results — it is a 1999-era
  KDD'99 derivative, so absolute numbers do not transfer to modern traffic;
  the value is the methodology and relative model comparison.

## Reproduce
Run from repo root inside `.venv`:
`src/download_data.py` → Phase 1 notebook → `src/preprocess.py` →
`src/train_rf.py` → `src/train_mlp.py` → `src/compare.py`. Every script is
deterministic under `RANDOM_STATE = 42`.
