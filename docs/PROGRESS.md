# Project progress report

*Snapshot of where the project stands, the reasoning behind each choice, and the
current problems worth naming honestly. Updated after the NSL-KDD Phase 3/4
rerun and runtime-stability fix.*

Last updated: after NSL-KDD Phase 4 rerun.

---

## 1. Status at a glance

| Phase | What | State |
|---|---|---|
| 0 | Repo, venv, pinned deps, git, verified downloads | done |
| 1 | EDA on NSL-KDD | done |
| 2 | Leakage-safe preprocessing | done |
| 3 | Random Forest + LightGBM baselines | done |
| 4 | MLP with/without class weighting | done |
| Audit | Referee notebook for current-state review | drafted |
| Next | Stabilize docs, repeat/extend experiments, then modern datasets | in progress |

Working rule in force: **finish NSL-KDD end-to-end before expanding the scope.**
One complete, reproducible pipeline is the portfolio asset.

## 2. What physically exists in the repo

```
src/            download_data.py · data.py · preprocess.py · evaluate.py
                train_baselines.py · train_mlp.py
tests/          pytest coverage for schema, preprocessing, metrics, and MLP pieces
notebooks/      01_eda.ipynb · 02_referee_audit_current_state.ipynb
results/        metrics.md · figures/
docs/learning/  plain-language notes for the completed NSL-KDD phases
data/           dataset files are gitignored; SOURCE.md provenance is committed
```

## 3. Current NSL-KDD result summary

The latest stable run uses the official `KDDTest+` split and writes full details
to `results/metrics.md`.

| Model | Task | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Random Forest | binary | 0.777 | 0.776 |
| LightGBM | binary | 0.785 | 0.785 |
| Random Forest | 5-class | 0.743 | 0.504 |
| LightGBM | 5-class | 0.561 | 0.281 |
| MLP unweighted | binary | 0.811 | 0.810 |
| MLP weighted | binary | 0.798 | 0.798 |
| MLP unweighted | 5-class | 0.748 | 0.563 |
| MLP weighted | 5-class | 0.776 | 0.561 |

Interpretation: the MLP currently gives the best 5-class macro-F1, but the
weighted MLP is best understood as a rare-class recall trade-off rather than a
clear macro-F1 winner. It improves R2L recall from 0.013 to 0.122 and U2R recall
from 0.269 to 0.537, while macro-F1 stays about flat.

## 4. Runtime fix

The reported crash was a native LightGBM/OpenMP failure on macOS, not an ordinary
Python exception. `src/train_baselines.py` now keeps LightGBM and GridSearchCV
threading conservative (`NIDS_N_JOBS=1` by default, LightGBM `num_threads=1`) and
sets a local Matplotlib cache directory. This avoids nested parallelism, which is
the likely trigger for the `libomp.dylib` / `lib_lightgbm.dylib` segfault.

## 5. Current problems — honest list

1. **Single-seed MLP result.** The MLP conclusion needs repeated seeds before it
   should be sold as a stable model ranking.
2. **Class weighting only tested on the MLP.** For a fairer imbalance study, add
   class-weighted/sample-weighted tree baselines too.
3. **Phase 4 stores only headline/rare recall.** Save full per-class MLP
   precision/recall/F1 tables like Phase 3 does.
4. **NSL-KDD is old and synthetic.** The project value is the methodology and
   failure analysis, not production-ready absolute scores.
5. **Rare classes may remain fundamentally under-specified.** U2R has only 52
   training examples; near-zero recall from some models is an honest benchmark
   finding, not automatically a code bug.

## 6. Immediate next step

Keep NSL-KDD reproducible and documented, then run multi-seed MLP and
class-weighted tree ablations. After that, move to the modern standardized
NetFlow-V2 datasets for the larger generalization story.
