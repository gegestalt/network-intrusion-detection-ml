# Project progress report

*Snapshot of where the project stands, the reasoning behind each choice, and —
importantly — an honest list of the current problems in our implementation,
dataset, and approach. Updated as we go.*

Last updated: after Phase 1 (EDA), before Phase 2 (preprocessing).

---

## 1. Status at a glance

| Phase | What | State |
|---|---|---|
| 0 | Repo, venv, pinned deps, git, **verified multi-dataset download** | ✅ done |
| 1 | EDA on NSL-KDD (5 figures + interpretation) | ✅ done |
| 2 | Preprocessing pipeline (NSL-KDD) | ▶️ building now |
| 3 | RF + LightGBM baseline (NSL-KDD) | ⏳ next |
| 4 | MLP comparison (NSL-KDD) | ⏳ |
| — | **NSL-KDD deliverable checkpoint** (README results + push) | ⏳ |
| 5+ | Generalize → UNSW-NB15 → CICIDS2017 → synthesis | ⏳ |

Working rule in force: **finish NSL-KDD end-to-end before touching UNSW/CICIDS.**
All three still ship; this only sequences them.

## 2. What physically exists in the repo

```
src/            download_data.py (3-dataset, verified), data.py (NSL schema+loaders)
notebooks/      01_eda.ipynb (executed, figures embedded)
results/figures 5 Phase-1 figures
data/           nsl_kdd/ unsw_nb15/ cicids2017/  (bulk gitignored; SOURCE.md kept)
docs/learning/  00_foundations.md (plain-language explainer)
.claude/skills/ nids-conventions/  (our conventions, machine-loadable)
requirements.txt, README (scaffold), .gitignore
```

## 3. Datasets acquired & verified

| Dataset | Rows (train/test) | Features | Split type | Rare class |
|---|---|---|---|---|
| NSL-KDD | 125,973 / 22,544 | 41 | official (+ hard KDDTest-21) | U2R (52) |
| UNSW-NB15 | 175,341 / 82,332 | 42 | official | Worms (130) |
| CICIDS2017 | 2,830,743 total | 78 | none (we build one) | Heartbleed (11) |

All row/column counts verified on download; SHA-256 hashes in `data/*/SOURCE.md`.

## 4. Decisions approved so far

Encoding **one-hot** (unseen categories → all-zeros), **StandardScaler** numerics,
**official splits** (no random re-split), **macro-F1 + per-class recall** as
headline metrics, **LightGBM** as the single boosting library (light tuning only),
**class weights on the MLP** with a with/without rare-class comparison, and a
**secondary temporal split** experiment on CICIDS. Full list + rationale is in
the chat log and encoded in the `nids-conventions` skill.

---

## 5. Current problems — honest list (this is the important part)

### 5a. Problems in our **implementation**
1. **No automated tests yet.** `data.py`/`download_data.py` have smoke tests but no
   unit tests. For a portfolio piece, a few `pytest` checks (schema, label map,
   leakage guard) would raise the bar. *Planned: add after the pipeline exists.*
2. **`num_outbound_cmds` is dead weight.** It's constant (all zeros) in NSL-KDD; it
   survives into the model as a useless feature. Harmless but untidy. *We flag it;
   could drop it explicitly.*
3. **High dimensionality from one-hot `service`.** ~70 service values become ~70
   columns. Fine for trees; for the MLP it inflates the input layer and most
   columns are almost always zero (sparse). *Acceptable, but worth naming.*
4. **Notebook is built via a script, not authored interactively.** Reproducible,
   but a reviewer opening Jupyter sees generated cells. *Trade-off we chose for
   determinism.*

### 5b. Problems in the **datasets**
1. **NSL-KDD is old and synthetic.** It derives from KDD'99, whose traffic was
   *simulated* in 1998. The attacks (and "normal" behaviour) don't resemble modern
   networks. Absolute scores here do **not** transfer to production.
2. **Extreme imbalance may make some classes unlearnable.** 52 U2R training
   examples (and 11 Heartbleed in CICIDS) may simply be too few for any model to
   learn reliably. We will likely see near-zero recall there — and that's an
   honest finding, not a bug.
3. **CICIDS2017 is messy.** Header whitespace, mojibake labels, NaN/Inf values,
   duplicate flows, and columns that can *leak* the label (source/dest IP,
   timestamp). We have a cleaning plan, but cleaning choices affect results.
4. **The three datasets don't share a feature space.** NSL-KDD (41 features),
   UNSW-NB15 (42), CICIDS2017 (78) measure *different things*. (See 5c.1.)

### 5c. Problems in our **approach** (the subtle, interview-critical ones)
1. **"Cross-dataset transfer" in the literal sense is not possible here.** You
   cannot train a model on NSL-KDD and test it on CICIDS2017 — the input columns
   are completely different, so the model has nothing to apply. The **honest**
   version of the "detection goes stale" story is told two ways instead:
   (a) *within* each dataset, the official train→test shift already hurts (unseen
   attack types); and (b) the **CICIDS temporal split** (train on earlier days,
   test on later days) is *real* time-based generalisation within one feature
   space — that is the cleanest "staleness" experiment we have. We should describe
   the cross-dataset angle as **"same methodology, three eras, contrast the
   difficulty each poses,"** not literal model transfer.
2. **Different split types make absolute numbers non-comparable across datasets.**
   NSL/UNSW use hard official splits; CICIDS uses an easier stratified split. So
   CICIDS will look "better" for reasons of *protocol*, not model quality. Only
   *within-dataset* and *relative* comparisons are clean; we must caveat every
   cross-dataset table.
3. **We report on a single fixed split, not cross-validated test performance.**
   The official test set is one sample; a single number has variance. We mitigate
   by cross-validating *during tuning* (on train), but the headline test score is
   still one draw. Standard for these benchmarks, but worth stating.
4. **Imbalance handling is currently planned only for the MLP.** Trees also
   support `class_weight`; we may extend the with/without study to them for
   consistency. *Open item.*

None of these are blockers — they're the kind of limitations a strong write-up
*names explicitly* rather than hides. Several become bullet points in the final
README's "Limitations" section.

## 6. Immediate next step
Build `src/preprocess.py` (the one-hot + scale `ColumnTransformer`), verify it on
NSL-KDD (column counts after encoding, class balances for both label schemes,
proof the leakage-safe encoder handles unseen test services), then pause with
learning note 01.
