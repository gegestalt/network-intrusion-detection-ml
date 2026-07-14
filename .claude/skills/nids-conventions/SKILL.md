---
name: nids-conventions
description: >-
  Conventions for this multi-dataset network-intrusion-detection ML study
  (NSL-KDD, UNSW-NB15, CICIDS2017). Load whenever loading/preprocessing any of
  these datasets, building the pipeline, training/evaluating models (RandomForest,
  LightGBM, MLP), producing confusion matrices / per-class F1, or writing
  up results. Encodes dataset layouts, preprocessing rules, the evaluation
  protocol (official splits where they exist; per-class + macro F1; confusion
  matrices to results/), the deliverable-first working order, and the results
  documentation style. Keep it current and commit changes when conventions move.
---

# NIDS ML study — project conventions

Research-grade intrusion-detection study. **NSL-KDD** (1999) is the historical
baseline (standalone; its 41-feature space differs from the others). The modern
core is the **NetFlow-V2 standardized family** — **NF-UNSW-NB15-v2** (2015),
**NF-ToN-IoT-v2** (2020), **NF-CSE-CIC-IDS2018-v2** (2018) — which all share one
**43-feature NetFlow schema**. That shared schema enables *true cross-dataset
transfer* (train on A, test on B), the real "detection goes stale as attacks
evolve" experiment. The raw UNSW-NB15 / CICIDS2017 CSVs are **retired** in favour
of their NetFlow-V2 versions (their `SOURCE.md` provenance is kept; migration is
documented in the README). `NF-UQ-NIDS-v2` (merged) is future work.

**Learning is the primary goal; timeline is secondary.**

## Working rules (standing)
1. **Deliverable-first.** Finish **NSL-KDD end-to-end** (EDA → preprocessing → RF
   + boosting → MLP → results figures → README results section, committed &
   pushed) *before* any UNSW-NB15 or CICIDS2017 modelling. One finished pipeline
   is the portfolio piece; three half-built ones are nothing. All three still
   ship — this only sequences them.
2. **Teach before each significant modelling choice** (encoding, imbalance
   handling, hyperparameter ranges, early stopping): 2-3 sentences on the options
   and why this one, so every decision is interview-defensible.
3. **Never fit on test.** No transformer, scaler, encoder, resampler (SMOTE), or
   hyperparameter search ever sees the test set. Fit on train, transform on test.
4. **Reproducibility.** `RANDOM_STATE = 42` everywhere (numpy, sklearn, torch,
   torch MPS). Pinned deps in `requirements.txt`.
5. **Lead with macro-F1 + per-class recall, not accuracy** — every dataset here
   is imbalanced; accuracy flatters trivial majority predictors.
6. **LEARNING MODE (teach-then-do).** For each new component: (a) explain the
   concept + plan first; (b) for **high-learning-value** code — preprocessing
   transforms, model training loops, evaluation functions, and *especially* the
   PyTorch MLP — **scaffold signatures + docstrings + TODOs and let the USER write
   the implementation**; review their code like a strict senior engineer (bugs,
   style, better idioms) but **do not rewrite it unless asked**; (c) the assistant
   fully writes low-learning-value plumbing (download/path/plotting boilerplate);
   (d) after each phase, **quiz the user 3-5 questions** and grade honestly.
7. **v1.0 snapshot.** When NSL-KDD is fully done (models, results, README
   section), **tag `v1.0`** with the repo presentable at that tag. After that the
   project is open-ended.
8. **Big-data scaling is curriculum, not an obstacle.** For ToN-IoT (~16.9M) /
   CSE-CIC-IDS2018 (~18.9M) flows: develop on stratified samples (rare classes
   preserved in full), and **teach the scaling techniques as first-class
   material** — chunked reading, dtype downcasting, memory profiling, and
   when/why to reach for each.
9. **TDD.** The project is test-driven. New high-value components get a `pytest`
   test (ideally written *before* the implementation) covering the invariants
   that matter — schema, leakage (fit-on-train-only), unseen-category tolerance,
   fail-loud on bad input. Tests live in `tests/`; data-dependent ones use the
   `needs_data` marker so `pytest` still passes on a fresh clone. Keep the suite
   green before every commit/push.

## Data layout
Downloaded + verified by `src/download_data.py` (row/col checks, SHA-256 to
`data/<dataset>/SOURCE.md`). Bulk data is gitignored; provenance is committed.

```
data/
  nsl_kdd/     KDDTrain+.txt KDDTest+.txt KDDTest-21.txt KDDTrain+_20Percent.txt
  unsw_nb15/   UNSW_NB15_training-set.csv  UNSW_NB15_testing-set.csv
  cicids2017/  <8 CICFlowMeter CSVs>
```

## Dataset specifics

### NSL-KDD  (official split; 125,973 / 22,544)
- 43 cols: 41 features + `label` + `difficulty`. **Drop `difficulty`** for
  modelling (it's "how many original learners missed this row" — metadata/leak).
- Categorical: `protocol_type`, `service` (~70), `flag`. Numeric: 38.
  `num_outbound_cmds` is constant (zero-variance) — harmless, flagged.
- Families via `ATTACK_FAMILY_MAP` in `src/data.py`: normal, DoS, Probe, R2L, U2R.
- Also evaluate on `KDDTest-21` (hard subset). Rare class: **U2R** (52 train).

### UNSW-NB15  (official partition; 175,341 / 82,332)
- 45 cols: `id` + 42 features + `attack_cat` + `label`. **Drop `id`** (leak).
- Categorical: `proto`, `service`, `state`. Numeric: the rest.
- Binary = `label` (0/1). Multiclass = `attack_cat` (10: Normal + Fuzzers,
  Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms).
- Mirror (Mireu-Lab) has train/test filenames **swapped** — roles assigned by
  row count in the downloader; on disk names are canonical. Rare class: **Worms**.

### CICIDS2017  (NO official split; 2,830,743 flows, 79 cols)
- **Clean first:** strip header whitespace (`' Label'`→`Label`); fix mojibake
  (`Web Attack � …`→`Web Attack - …`); replace `Inf`→`NaN` then drop NaN rows
  (<0.1%); drop socket-identifier leak columns if present (Flow ID, IPs,
  timestamp). Treat `Destination Port` as numeric but note leak risk.
- Binary = BENIGN vs attack. Multiclass = grouped families: DoS (Hulk/GoldenEye/
  slowloris/Slowhttptest), DDoS, PortScan, BruteForce (FTP/SSH-Patator),
  WebAttack (BruteForce/XSS/SQLi), Bot, Infiltration, Heartbleed, + Normal.
- **Split (primary): stratified 70/30** (fixed seed), stratified on the fine
  label so every class is in both sets. This is *in-distribution* (easier than
  NSL/UNSW official splits) — state this caveat by every CICIDS result.
- **Split (secondary): temporal.** CICIDS was captured over 5 consecutive
  weekdays; also train on earlier days / test on later days and contrast with the
  stratified split. The stratified-vs-temporal gap is the "detection goes stale
  as attacks evolve" story in one table. If runtime on the sample is
  unreasonable, skip it and record that under limitations instead. (Caveat:
  several attack classes are confined to a single day, so a temporal split leaves
  them test-only — expected, and part of the point.)
- **Dev on a stratified 20-30% sample** (preserve rare classes — Heartbleed=11,
  Infiltration=36 — in FULL); scale up only if training time is reasonable.
  Rare classes: Heartbleed, Infiltration.

## Preprocessing pipeline (`src/`)
`ColumnTransformer` inside a `Pipeline`, fit-once and serializable:
- Categorical → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
  (test sets contain unseen categories — ignore encodes them as all-zeros).
- Numeric → `StandardScaler()` (trees are scale-invariant, but one shared
  pipeline keeps RF/boosting/MLP strictly apples-to-apples).
- Two label schemes per dataset: **binary** and **multiclass**.

## Models (approved scope)
- **Trees:** RandomForest + **LightGBM**. One gradient-boosting library only —
  LightGBM over XGBoost for its histogram-based speed and low memory footprint,
  which matters when we scale to CICIDS's 2.8M flows on a laptop. Sensible
  defaults + at most a **light** CV search — it's a strong baseline, not the star;
  no extensive tuning detour.
- **MLP (PyTorch):** 2-3 hidden layers, dropout, early stopping, MPS. On the MLP,
  **handle imbalance explicitly via class weights** and report rare-class recall
  (U2R/R2L; Worms; Heartbleed/Infiltration) **with vs without** weighting — the
  contrast is a required result, with a teach-up before implementing.

## Evaluation protocol (identical for every model & dataset)
Report on the held-out test set (official where it exists; constructed for CICIDS):
- accuracy, **macro P/R/F1**, weighted F1, and the **full per-class report**
  (rare classes — U2R/R2L, Worms, Heartbleed/Infiltration — are the point).
- **Confusion matrix**: raw counts *and* row-normalized (recall) versions.
- Binary also: **ROC-AUC + PR-AUC** (PR-AUC more informative under imbalance).
- **Hyperparameter search: `StratifiedKFold` on the training set only**, scoring
  `f1_macro`.
- **Cost framing:** a false negative (missed attack) usually costs more than a
  false positive; report attack-class recall prominently.

## Results & documentation style
- Figures → `results/figures/` as `<phase>_<dataset>_<what>.png`, ≥150 DPI,
  titled, labelled, colorblind-safe. E.g. `p3_nslkdd_rf_confusion_binary.png`.
- Metrics → `results/metrics.md`: one table per (dataset × model × task) plus a
  consolidated cross-dataset comparison. 4 decimals.
- Every notable result gets a plain-English, **SOC-analyst** interpretation.
- Always state limitations near results (dataset age, dataset shift, why
  real-world deployment differs).

## Reproduce
Inside `.venv` from repo root: `src/download_data.py` → Phase-1 EDA notebook →
`src/preprocess.py` → `src/train_rf.py` → `src/train_mlp.py` → `src/compare.py`.
Deterministic under `RANDOM_STATE = 42`.
