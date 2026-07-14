# Learning note 03 — The MLP & the class-weighting experiment

*Companion for Phase 4. A neural net on the same data, and the single most
important lever for rare-class detection: weighting the loss.*

---

## 1. What we built
A small **multi-layer perceptron (MLP)**: 122 inputs → 128 → 64 → output, with
**ReLU** activations, **dropout (0.3)** for regularization, **Adam** optimizer,
and **early stopping** on a held-out validation slice (stop when validation
macro-F1 plateaus). It can train on CPU or Apple-Silicon **MPS** depending on the
machine/runtime available.
Crucially, it consumes the *exact same* preprocessed matrices as the trees, so
the comparison is fair.

## 2. The experiment: class weighting on vs off
By default the loss (`CrossEntropyLoss`) treats every example equally, so with
52 U2R vs 67,343 normal training rows the net is rewarded for ignoring U2R. We
re-ran it with **inverse-frequency class weights** — making each rare-class error
cost proportionally more — and measured the effect:

| Variant | Task | Accuracy | Macro-F1 | Rare-class recall |
|---|---|---:|---:|---|
| MLP unweighted | 5-class | 0.748 | **0.563** | R2L 0.013 / U2R 0.269 |
| MLP weighted | 5-class | **0.776** | 0.561 | **R2L 0.122 / U2R 0.537** |
| MLP unweighted | binary | **0.811** | **0.810** | attack 0.688 |
| MLP weighted | binary | 0.798 | 0.798 | **attack 0.700** |

**The result in one sentence:** class weighting did what it is supposed to do for
rare recall — **R2L 0.013→0.122** and **U2R 0.269→0.537** — but it did **not**
improve overall macro-F1 in this latest stable run. That makes the conclusion
more nuanced: weighting is a SOC trade-off knob, not a guaranteed free upgrade.

Why doesn't macro-F1 automatically rise? Because class weighting moves the
decision boundary. It may rescue rare attacks while also creating new mistakes in
normal/DoS/Probe. That trade can still be worthwhile operationally, but the
metric has to account for all classes, not only the two rare ones.

## 3. How the models stack up (best variant each)

| Model | Binary macro-F1 | 5-class macro-F1 |
|---|---:|---:|
| Random Forest | 0.776 | 0.504 |
| LightGBM | 0.785 | 0.281 |
| MLP (unweighted) | **0.810** | **0.563** |
| MLP (weighted) | 0.798 | 0.561 |

The MLP family currently wins on the 5-class macro-F1 score, but the gap over RF
is modest and the weighted variant does not beat the unweighted one on macro-F1.
The honest nuance: the trees got only a *light* tuning budget and no class
weighting (by design — that was this phase's variable), so this is not "neural
nets beat trees" in general. It is "on this imbalanced problem, model choice,
weighting, and threshold trade-offs must be evaluated per class."

## 4. What this means for a SOC
The unweighted model is the seductive trap: 75% accuracy, looks fine, and quietly
misses almost all R2L attacks. The weighted model catches more R2L and roughly
half of U2R, but the global macro-F1 stays about flat. In real operations that
trade — *tolerate different errors to stop missing crown-jewel attacks* — can be
correct, but it needs threshold tuning, validation, and analyst-capacity
constraints before anyone calls it "better."

## 5. Teach-back
1. What does dropout do, and why does it help?
2. We weighted the loss by inverse class frequency. Mechanically, how does that
   change what the network optimizes?
3. Class weighting raised rare-class recall but did not raise macro-F1. Why can
   both things be true at once?
4. Why is it *not* fair to conclude "MLPs are better than Random Forests" from
   this phase?

*(Answers: 1 — randomly zeroing neurons each step forces redundant, robust
features and prevents memorization/overfitting. 2 — it multiplies each class's
loss contribution so rare-class mistakes dominate the gradient, pulling the
decision boundary toward catching them. 3 — the rare classes improved, but the
model also changed its mistakes on other classes, so the average F1 stayed about
flat. 4 — the trees got no class weighting and only light tuning here; the
controlled variable was weighting, not model family.)*

---
*NSL-KDD is now complete end-to-end. Next: the README results section + a `v1.0`
tag (the presentable snapshot), then the modern NetFlow-V2 datasets.*
