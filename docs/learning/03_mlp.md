# Learning note 03 — The MLP & the class-weighting experiment

*Companion for Phase 4. A neural net on the same data, and the single most
useful lever for rare-class detection: weighting the loss.*

---

## 1. What we built
A small **multi-layer perceptron (MLP)**: 122 inputs → 128 → 64 → output, with
**ReLU** activations, **dropout (0.3)** for regularization, **Adam** optimizer,
and **early stopping** on a held-out validation slice (stop when validation
macro-F1 plateaus). It trains on the Apple-Silicon **MPS** GPU in ~15s per run.
Crucially, it consumes the *exact same* preprocessed matrices as the trees, so
the comparison is fair.

## 2. The experiment: class weighting on vs off
By default the loss (`CrossEntropyLoss`) treats every example equally, so with
52 U2R vs 67,343 normal training rows the net is rewarded for ignoring U2R. We
re-ran it with **inverse-frequency class weights** — making each rare-class error
cost proportionally more — and measured the effect:

| Variant | Task | Accuracy | Macro-F1 | Rare-class recall |
|---|---|---:|---:|---|
| MLP unweighted | 5-class | 0.782 | 0.609 | R2L 0.08 / U2R 0.28 |
| **MLP weighted** | 5-class | 0.791 | **0.708** | **R2L 0.48 / U2R 0.42** |
| MLP unweighted | binary | 0.789 | 0.789 | attack 0.68 |
| **MLP weighted** | binary | 0.811 | **0.810** | attack 0.69 |

**The result in one sentence:** class weighting lifted R2L recall **6×**
(0.08→0.48) and U2R recall to 0.42 — *while overall accuracy went slightly up*
(0.782→0.791). That's the ideal outcome: we caught far more of the dangerous rare
attacks and paid essentially nothing in aggregate accuracy.

Why doesn't accuracy drop? Because R2L+U2R are a small fraction of rows — spending
a few normal/DoS misclassifications to rescue thousands of R2L detections barely
moves the accuracy needle but massively moves macro-F1 (and real-world value).

## 3. How the models stack up (best variant each)

| Model | Binary macro-F1 | 5-class macro-F1 |
|---|---:|---:|
| Random Forest | 0.776 | 0.504 |
| LightGBM | 0.785 | 0.410 |
| **MLP (weighted)** | **0.810** | **0.708** |

The weighted MLP wins on both tasks, and the 5-class gap is large (+0.20 macro-F1
over RF). The honest nuance: the trees got only a *light* tuning budget and no
class weighting (by design — that was this phase's variable), so this isn't
"neural nets beat trees" in general. It's "on this imbalanced problem, explicit
imbalance handling matters more than model family" — a more useful lesson.

## 4. What this means for a SOC
The unweighted model is the seductive trap: 78% accuracy, looks fine, and quietly
misses **92%** of privilege-escalation (U2R) and R2L attacks — the ones that
actually own your network. The weighted model trades a modest rise in false
alarms for catching ~half of them. In real operations that trade — *tolerate more
noise to stop missing the crown-jewel attacks* — is usually correct, and it's a
knob (the class weights / decision threshold) you tune to your team's alert
budget.

## 5. Teach-back
1. What does dropout do, and why does it help?
2. We weighted the loss by inverse class frequency. Mechanically, how does that
   change what the network optimizes?
3. Class weighting raised R2L recall 6× but accuracy barely changed. Why are both
   things true at once?
4. Why is it *not* fair to conclude "MLPs are better than Random Forests" from
   this phase?

*(Answers: 1 — randomly zeroing neurons each step forces redundant, robust
features and prevents memorization/overfitting. 2 — it multiplies each class's
loss contribution so rare-class mistakes dominate the gradient, pulling the
decision boundary toward catching them. 3 — the rare classes are few rows, so
fixing them barely affects overall accuracy but greatly affects the
equally-weighted macro-F1. 4 — the trees got no class weighting and only light
tuning here; the controlled variable was weighting, not model family.)*

---
*NSL-KDD is now complete end-to-end. Next: the README results section + a `v1.0`
tag (the presentable snapshot), then the modern NetFlow-V2 datasets.*
