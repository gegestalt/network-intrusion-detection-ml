# Learning note 02 — Baselines: Random Forest & LightGBM

*Companion for Phase 3. What the two models are, what we measured, and — most
importantly — what the numbers reveal about the problem.*

---

## 1. The two models, in plain terms

**Random Forest (RF).** Grow hundreds of decision trees, each on a random
subset of the data and features, then let them vote. One tree overfits; a
*forest* of decorrelated trees averages out the noise. Needs almost no tuning
and is a rock-solid tabular baseline.

**LightGBM.** Also trees, but built **sequentially** — each new tree is trained
to correct the errors the previous ones made (this is *gradient boosting*).
Usually the strongest classical model on tabular data. We picked it (over
XGBoost) for speed and low memory, which will matter on the millions-of-rows
NetFlow datasets later. Per our rules it got only a *light* hyperparameter search
— it's a baseline, not the star.

Both were tuned with **3-fold cross-validation on the training set only** (never
the test set), scoring macro-F1.

## 2. Headline results (official KDDTest+)

| Model | Task | Accuracy | Macro-F1 |
|---|---|---:|---:|
| RandomForest | binary | 0.777 | 0.776 |
| LightGBM | binary | 0.785 | 0.785 |
| RandomForest | 5-class | 0.743 | **0.504** |
| LightGBM | 5-class | 0.561 | 0.281 |

Three things here are worth more than the numbers themselves.

### Finding A — the generalization gap is huge (and that's the point)
During cross-validation *on the training set*, both models scored **~0.99**
macro-F1. On the official test set they dropped to **~0.78** (binary). That ~20-
point fall is not a bug — it's the **unseen attack types** in the test set (17
attack labels never appear in training). This is the single most important lesson
of the whole project: **a model can look near-perfect on data resembling its
training set and still miss a fifth of real, novel attacks.** A random train/test
split would have hidden this entirely.

### Finding B — high ROC-AUC can hide bad recall
The binary models have a beautiful **ROC-AUC of 0.96**… yet **attack recall is
only ~0.63** (they miss ~37% of attacks). How can both be true? ROC-AUC measures
whether the model can *rank* attacks above normals given the right threshold; at
the *default* 0.5 threshold the model is still too conservative. Lesson: a single
headline metric lies. In a SOC you'd tune the threshold to trade some false
alarms for catching more attacks — which is exactly what PR curves and threshold
tuning (a later phase) are for.

### Finding C — the rare classes are effectively invisible
The 5-class confusion matrix is the money shot:

| true class | recall (RF) | where the misses go |
|---|---:|---|
| normal | 0.97 | — |
| DoS | 0.77 | mostly normal |
| Probe | 0.60 | normal |
| **R2L** | **0.05** | **95% → predicted "normal"** |
| **U2R** | **0.06** | **91% → predicted "normal"** |

R2L (remote-to-local) and U2R (privilege escalation) — the *most dangerous*
attacks — are almost entirely classified as benign. With 995 and 52 training
examples respectively, and almost no separating signal in the traffic features,
the models have nothing to learn from. **Macro-F1 (0.50) exposes this; accuracy
(0.74) hides it** — because R2L+U2R are a tiny slice of the rows, getting them
100% wrong barely dents accuracy. This is *why* we lead with macro-F1.

### Finding D — hardness is real: KDDTest-21
On the **KDDTest-21** subset (the records every classifier in the original study
found hard), binary macro-F1 falls further to **~0.55**. Same model, harder
sample — a reminder that benchmark numbers are only meaningful next to the exact
test set they came from.

## 3. RF vs LightGBM — who won?
Roughly tied on binary (LightGBM +0.9pt). But on the 5-class task **RF beat
LightGBM on macro-F1 (0.504 vs 0.281)**. Why? In the latest stable run, LightGBM
mostly collapsed the harder minority families: Probe recall fell to 0.09, R2L to
0.001, and U2R to zero. RF was more conservative and recovered more of the
non-normal classes. Takeaway: "the fancier model" doesn't automatically win,
*especially* on rare classes — you have to look at the per-class breakdown, not
the leaderboard.

## 4. What the feature-importance plots showed
Both models leaned on the traffic-statistics features we flagged in EDA —
`src_bytes`, `dst_host_*` service-rate features, `flag`/`serror_rate` — the DoS
and Probe fingerprints. That the models "agree" with domain intuition is a good
sanity check that they learned signal, not noise.

## 5. Teach-back — can you explain these?
1. Our models scored 0.99 in cross-validation but 0.78 on test. What causes that
   gap here, and why is it a *feature* of this benchmark rather than a failure?
2. ROC-AUC is 0.96 but attack recall is 0.63. How are both true at once?
3. Accuracy for the 5-class RF is 0.74 but macro-F1 is 0.50. Which do we report,
   and why does the gap exist?
4. LightGBM had a higher single-number score on binary but lost on 5-class
   macro-F1. What did looking at *per-class* metrics reveal that the headline hid?

*(Answers, brief: 1 — unseen attack types in test; a random split would hide it,
so it's honest novelty. 2 — ROC-AUC measures ranking ability across thresholds;
at the default 0.5 threshold the model is too conservative, so recall is low. 3 —
macro-F1, because it weights the rare, dangerous R2L/U2R classes equally, which
accuracy drowns out. 4 — LightGBM mostly missed Probe/R2L/U2R in the 5-class
setting; per-class metrics exposed that the binary "win" did not transfer to the
harder task.)*

---
*Next (Phase 4): a PyTorch MLP on the identical data — can a neural net beat the
trees, especially on the rare classes? And what does explicit class-weighting buy
us there?*
