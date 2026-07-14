# Learning note 00 — Foundations: what we've built so far

*Plain-language companion for Phases 0-1 (project setup, data, exploration).
Written to be understandable without an ML background, with the theory that
matters for interviews. Read top to bottom; the glossary at the end is a
cheat-sheet.*

---

## 1. The problem in one paragraph

A **Network Intrusion Detection System (NIDS)** watches network traffic and
tries to flag connections that are attacks (a port scan, a denial-of-service
flood, someone brute-forcing a login) versus normal activity. Classic NIDS use
hand-written *rules* ("if >100 half-open connections in 2 seconds, alert"). Rules
are precise but brittle — they miss anything nobody wrote a rule for, and
attackers change tactics constantly. **Machine learning (ML)** flips this: instead
of us writing rules, we show an algorithm thousands of labelled examples
("this connection was an attack, this one wasn't") and it *learns* the patterns
itself. That's what this project builds and tests.

Why this matters to a SOC analyst: it's the same job you already do (triage
alerts, separate signal from noise), but we're measuring how well a model can do
the first-pass triage — and, crucially, *where it fails*, because a model that
misses the dangerous-but-rare attacks is worse than useless.

## 2. The data: three datasets from three eras

We deliberately use three public benchmark datasets instead of one. Each is a big
table: **rows = network connections, columns = measured features** (bytes sent,
protocol, connection flags, error rates…), plus a **label** column saying what it
was.

| Dataset | Year | Why we included it |
|---|---|---|
| **NSL-KDD** | 1999-era | The classic teaching benchmark. Small, clean, well-understood. Our *first, safe* deliverable. |
| **UNSW-NB15** | 2015 | A modern successor — newer attack types, different balance of normal vs attack. |
| **CICIDS2017** | 2017 | Built from *real* captured traffic. Big (2.8 million connections), realistic, messy. |

The cross-dataset idea is the strong part: a model trained on 1999 attacks, when
tested on 2017 traffic, struggles — exactly like a SOC's detection rules going
stale as attackers evolve. We're going to *measure* that staleness.

## 3. What we set up (Phase 0) and why each piece matters

- **A virtual environment (`.venv`) with pinned versions.** A venv is an isolated
  Python sandbox so this project's libraries don't collide with anything else on
  your Mac. "Pinned" means we recorded the *exact* version of every library
  (`requirements.txt`). Theory: **reproducibility** — a portfolio project that
  someone else (an admissions reviewer) can't re-run is worth much less. Pinning
  guarantees they get the same result you did.
- **Git with small, labelled commits.** Version control. Each commit is a labelled
  checkpoint. If something breaks we can see exactly what changed. Reviewers read
  your commit history to judge how you work.
- **A programmatic, *verified* download script.** Rather than "download these
  files by hand," a script fetches them and **checks they're correct** (right
  number of rows and columns, and records a SHA-256 fingerprint of each file).
  Theory: **data integrity** — if a download is truncated or a mirror serves the
  wrong file, every downstream result is silently wrong. We fail loudly instead.
- **A "skill" file (`nids-conventions`).** A written record of our conventions
  (how we split data, which metrics we report) so the project stays consistent.

## 4. Phase 1 — Exploration (EDA), and what it told us

**EDA = Exploratory Data Analysis:** looking at the data with plots *before*
modelling, so you understand it and don't get fooled later. Three findings
matter, and each has a lesson behind it.

### Finding A — the classes are wildly imbalanced
In NSL-KDD's training data, "normal" and "DoS" attacks are huge, but **U2R**
(user-to-root, i.e. privilege escalation) is **0.04%** — 52 examples out of
125,973. R2L (remote-to-local) is 0.8%.

> **Theory — why imbalance is dangerous.** Imagine a model that just says
> "not U2R" every single time. It would be ~99.96% *accurate* — and catch zero
> privilege-escalation attacks. **Accuracy is a liar on imbalanced data.** This
> is why we will judge models on **per-class recall** (of the real U2R attacks,
> what fraction did we catch?) and **macro-F1** (which averages performance across
> classes equally, so the tiny classes actually count). This single idea is the
> backbone of the whole project.

### Finding B — the official train/test split is *deliberately hard*
Normally you'd shuffle your data and randomly cut off, say, 30% for testing. We
do **not** do that here. NSL-KDD and UNSW-NB15 ship an *official* train/test
split where the test set contains **attack types the model never saw in
training**. In NSL-KDD, R2L attacks jump from 0.8% of training to **12.8%** of
test.

> **Theory — why this is more honest.** A random split makes train and test look
> identical, so scores come out flatteringly high — but that's not the real world.
> In a real SOC, *tomorrow's* attacks aren't in *yesterday's* labelled data. The
> official split simulates that novelty, so our numbers mean something. A golden
> rule falls out of this: **never let the model or any preprocessing step "see"
> the test data during training** — that's called *leakage*, and it fakes good
> results.

### Finding C — some features separate attacks cleanly, others don't
Our plots showed that DoS/Probe attacks light up obvious features (a connection
with `same_srv_rate ≈ 1` and a high `count` = hammering one service repeatedly =
a flood or scan). But R2L/U2R attacks look almost like normal sessions at the
traffic level — they only reveal themselves in rare "content" features (failed
logins, shell access).

> **Theory — the ceiling this sets.** No model can conjure signal that isn't in
> the data. We can already predict, before training anything, that DoS/Probe
> detection will be strong and R2L/U2R will be hard. Being able to *anticipate*
> where a model will fail (and why) is exactly what separates an analyst who
> understands ML from one who just runs `.fit()`.

We saved five figures to `results/figures/` (class distribution, top attack
types, protocol/flag behaviour, a correlation heatmap, and feature-separation
plots), each with a security interpretation in the notebook.

## 5. Mini-glossary (interview-ready)

- **Feature** — one measured column (e.g. `src_bytes`). The model's input.
- **Label** — the answer we're trying to predict (attack? which kind?).
- **Class** — one possible label value (normal, DoS, U2R…).
- **Class imbalance** — some classes have far more examples than others.
- **Accuracy** — % of predictions correct overall. Misleading under imbalance.
- **Recall** (per class) — of the real X's, what fraction did we catch? The
  security-critical metric (missing an attack = a false negative).
- **Precision** (per class) — of the things we *called* X, what fraction really
  were? (False alarms hurt precision.)
- **F1** — the balance of precision and recall (their harmonic mean).
- **Macro-F1** — F1 averaged equally over classes, so rare classes count fully.
- **Train/test split** — data the model learns from vs data we grade it on.
- **Leakage** — accidentally letting test information into training; fakes good
  scores.
- **Reproducibility** — anyone can re-run and get the same numbers.

## 6. Teach-back — can you explain these?

If you can answer these out loud, Phase 0-1 has landed:
1. Why is 99% accuracy potentially a *terrible* result on NSL-KDD?
2. Why do we use the official train/test split instead of shuffling and
   re-splitting ourselves?
3. Before training anything, why do we already expect U2R to be hard to detect?

*(Answers: 1 — a model can hit 99% by ignoring the rare, dangerous classes;
accuracy hides that. 2 — the official split hides *new* attack types in the test
set, mimicking real-world novelty and preventing over-optimistic scores. 3 — the
separating signal for U2R barely exists in the recorded features, so no model can
reliably find it.)*

---
*Next note (01) will cover Phase 2: turning this raw data into clean numbers a
model can actually consume — the preprocessing pipeline.*
