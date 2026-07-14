# Adaptive Network Security Analytics Lab — Roadmap

*Branch: `security-ml-lab`. `main` holds the stable NSL-KDD `v1.0` deliverable.*

This project graduates from an NSL-KDD model comparison into a **security ML
experimental lab** where each phase teaches a *different data-science problem*.
The point is breadth of method, not another 0.5% on one metric.

## Guiding research question
> Which modelling strategy — supervised classification, normal-only anomaly
> detection, temporal modelling, or online learning — generalizes most reliably
> across changing network environments while keeping an operationally acceptable
> false-positive rate?

## Standing principles (carried from `main`)
- **NSL-KDD is the controlled baseline.** Every new method runs here first (it's
  small and fast); only methods that work graduate to the modern datasets.
- **Deliverable-first & one-thing-at-a-time.** Land a phase end-to-end before
  starting the next; don't half-build ten notebooks.
- **Rigor:** macro-F1 + per-class recall over accuracy; leakage-safe pipelines;
  **≥3-seed mean ± std before any strong claim** (a single run is a hypothesis).
- **Each dataset gets a *distinct* method/technique/eval angle** (not one pipeline
  copy-pasted).
- **TDD** for reusable components; commit/push per phase.

## Findings so far (NSL-KDD Reference Track)
- Balanced **Logistic Regression ≈ 0.61 macro-F1** — imbalance handling and
  decision thresholds move the result *as much as* changing model family.
- **MLP macro-F1 is high-variance** (0.577 ± 0.046); class weighting reliably
  rescues rare-class recall (U2R 0.51 ± 0.06) without significantly changing
  macro-F1. (`results/stability.md`)
- LightGBM collapses minority classes under deterministic single-thread (0.28).

## Datasets
| Dataset | Role | Why |
| --- | --- | --- |
| **NSL-KDD** (1999) | controlled baseline | fast local iteration; every method starts here |
| **CICIoT2023** | primary modern supervised | 33 attacks / 7 categories / 105-device IoT; 3 label levels (binary → 7-class → 33-class) |
| **TON_IoT** | multimodal | network + IoT/IIoT telemetry + host traces + events → SOC/EDR-style fusion |
| **CSE-CIC-IDS2018** | enterprise-scale | 420 machines, ~80 features, day-based splits → drift & scalability |
| **NetFlow-v2 family** | cross-dataset schema | shared 43-feature space → leave-one-dataset-out transfer |

## Tracks (experiment families)
- **Reference Track (supervised).** Dummy, LogReg, RF, ExtraTrees, HistGB,
  LightGBM, MLP; weighted/unweighted; threshold optimization.
- **A — Cost-sensitive / imbalance.** none / class-weight / RUS / ROS / SMOTE /
  Borderline-SMOTE / ADASYN / focal loss / balanced batches / per-class
  thresholds. Track macro-F1, balanced-acc, MCC, PR-AUC, per-class P/R, and
  **FP per 10k benign**. Research Q: does synthetic oversampling *genuinely* help
  rare-attack recall, or just flatter validation? (SMOTE on flows is suspect.)
- **B — Unsupervised anomaly detection.** Train on benign only: IsolationForest,
  LOF, One-Class SVM, robust covariance, GMM, k-means, AE / DAE / VAE, Deep SVDD.
  Two tests: (1) known attacks, (2) **attack types excluded from training**
  (zero-day proxy — the valuable one).
- **C — Semi-supervised.** Reveal 1/5/10/25% of labels; self-training,
  pseudo-labelling, label propagation/spreading, AE/contrastive pretraining →
  fine-tune, teacher–student. Q: how much labelled security data is actually needed?
- **D — Temporal / sequence.** Window by src/dst IP, pair, device, session, and
  5s/30s/5m intervals; features (flow_count, unique_dst_ports,
  failed_conn_ratio, syn/rst counts, dst entropy, interarrival, …). Compare
  aggregated-tabular+LightGBM vs LSTM / GRU / TCN / Transformer / seq-AE.
  Flow classification → **behaviour detection**.
- **E — Online / continual.** Chronological stream: SGD partial_fit,
  Passive-Aggressive, Hoeffding Tree, Adaptive RF, online NB; ADWIN /
  Page-Hinkley drift detection; sliding-window retrain. Record perf before/after
  drift, recovery time, memory, latency, throughput.
- **Cross-dataset generalization.** Leave-one-dataset-out on the shared NetFlow
  schema; domain adaptation, distribution shift, dataset bias, transfer.

## Cross-cutting analytics
- **Data-quality & leakage report** per dataset: missing/duplicate/constant
  features, outliers, imbalance, skew, train/test shift (KS, PSI, JS-divergence,
  χ², mutual information, effect sizes), suspicious identifiers, target leakage.
- **Attack behavioural profiling** per class: protocols, duration, packet-rate,
  dst-port dist, source diversity, byte/packet asymmetry, TCP flags, interarrival.
- **Clustering** (KMeans/GMM/DBSCAN/HDBSCAN/agglomerative) + PCA/UMAP/t-SNE:
  do attacks form clusters? mislabels? benign sub-profiles? overlap?
- **Feature-engineering study:** original vs reduced (MI/ANOVA/L1/RFE/perm/SHAP)
  vs **operational** (NetFlow/IPFIX/Zeek-available only) vs **learned**
  (AE/DAE/contrastive/TabNet/transformer embeddings). Key: does representation
  learning beat a strong classical baseline?
- **Graph ML (later):** IP/host nodes, flow edges; degree/centrality/community,
  GraphSAGE / GAT / edge-classification / temporal graph nets.
- **Explainability & calibration:** SHAP, permutation importance, PDP,
  calibration curve, Brier, ECE, confidence dist, per-family/per-device error
  analysis, seed-stability. Do explanations make security sense?
- **Operational SOC simulation:** 1M daily flows, 0.5% malicious, 0.2% FP rate →
  alerts/day, detections, misses, analyst workload, alert reduction vs threshold.
  The highest-macro-F1 model may not be the best *operational* model.

## Notebook plan
```
00_dataset_catalog            07_semi_supervised_learning
01_data_quality_and_leakage   08_temporal_sequence_detection
02_referee_audit_current_state 09_online_learning_and_drift
03_supervised_benchmark       10_cross_dataset_generalization
04_imbalance_and_cost_sensitive 11_explainability_and_calibration
05_feature_selection_and_repr 12_graph_network_analytics
06_unsupervised_anomaly       13_operational_soc_simulation
```

## Implementation order (highest learning value first)
1. **CICIoT2023 loader + data-quality audit** (nb 00–01)
2. **Unified supervised benchmark** — expand the Reference Track (nb 03)
3. **Normal-only anomaly detection** (nb 06)
4. **Chronological online-learning + drift** (nb 09)
5. **Cross-dataset NetFlow generalization** (nb 10)

Then A/C/D, feature-engineering, explainability, SOC sim, graph — as capacity allows.

## Definition of done (per phase)
Runs end-to-end on NSL-KDD first; reusable code in `src/` with tests; results +
figures saved; a `docs/learning/NN_*.md` plain-language note; committed & pushed.
