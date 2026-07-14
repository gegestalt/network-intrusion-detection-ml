# Experimental Lab Prompt Audit

Role: referee/audit force. This report checks what the repository can prove today against the large "Adaptive Network Security Analytics Lab" prompt.

## Executive Verdict

- Proven areas: **12**
- Partial areas: **9**
- Missing areas: **3**
- Test files present: **15**
- Saved figures present: **43**
- Referee audit notebook outputs: **28 / 28 code cells**
- CICIoT2023 Phase-1 notebook outputs: **8 / 8 code cells**
- New threshold ablation: **True**
- New anomaly-detection report: **True**
- New semi-supervised report: **True**
- New SOC simulation: **True**
- New feature-learning report: **True**
- New deep-learning taxonomy: **True**
- New neural foundation report: **True**
- New neural ablation report: **True**

The project is now more than a supervised NSL-KDD comparison: it has saved threshold-tuning, normal-only anomaly, semi-supervised, online-proxy, and SOC-simulation artifacts. It is still **not yet** the full experimental lab: true temporal drift, graph ML, modern raw-dataset modelling, and cross-dataset generalization remain blocked by missing local data/schema work.

## Currently Proven Combinations

| Dataset | Track | Task/split combinations | Models/methods | Evidence |
| --- | --- | --- | --- | --- |
| NSL-KDD | Supervised reference | binary + multiclass | Dummy, balanced LogReg, balanced RF, balanced ExtraTrees, balanced HistGB, balanced LightGBM | 12 model/task rows in results/reference_track.md |
| NSL-KDD | Phase-3 official-split baseline | binary KDDTest+, binary KDDTest-21, multiclass KDDTest+ | RandomForest + LightGBM with train-only GridSearchCV | 6 evaluation rows plus confusion/ROC/PR/importance figures |
| NSL-KDD | Threshold tuning | binary KDDTest+ | LogReg, balanced LogReg, balanced ExtraTrees, balanced HistGB x default/F1/F2 thresholds | results/threshold_ablation.md and results/threshold_ablation.csv |
| NSL-KDD | Normal-only anomaly detection | binary plus attack-family recall | IsolationForest, LocalOutlierFactor, k-means distance x 0.90/0.95/0.99 normal quantiles | results/anomaly_detection.md and results/anomaly_detection.csv |
| NSL-KDD | Semi-supervised label budget | binary x 1/5/10/25% labels | labelled-only Logistic Regression vs self-training Logistic Regression | results/semi_supervised.md and results/semi_supervised.csv |
| NSL-KDD | Online-learning proxy | binary x file-order chunks | SGD log-loss partial_fit, SGD passive-aggressive-style partial_fit | results/online_learning.md; not a true drift claim |
| NSL-KDD | Operational SOC simulation | 1M daily flows, 0.5% malicious | threshold-ablation rates converted to alerts/misses/workload | results/soc_simulation.md and results/soc_simulation.csv |
| NSL-KDD | Feature analysis and representation learning | binary KDDTest+ | correlation/outlier maps; raw vs MI top-k vs PCA vs L1 vs autoencoder embeddings | results/feature_learning.md plus feature CSVs and figures |
| NSL-KDD | Neural foundations and MLP ablations | binary KDDTest+; bounded train subset | single-neuron demo; activation functions; MLP activation/dropout/norm/depth/loss/label-smoothing ablations | results/neural_foundations.md, results/neural_ablation.md, experiments/runs/neural_ablation.jsonl |
| NSL-KDD | Phase-4 MLP ablation | binary + multiclass | MLP unweighted vs weighted | 4 headline rows in results/metrics.md plus stability rows |
| NSL-KDD | Multi-seed stability | binary + multiclass x 5 seeds | RandomForest, LightGBM, MLP-unweighted, MLP-weighted | 40 train/evaluate runs summarized in results/stability.md |
| CICIoT2023 | Dev feature analysis and representation learning | binary dev split sample | correlation/outlier maps; raw vs MI top-k vs PCA vs L1 vs autoencoder embeddings | results/feature_learning.md caveated as dev sample only |
| CICIoT2023 | Dev data-quality track | train.parquet + test.parquet | quality/leakage checks in src/ciciot.py | 2 parquet files present |
| CICIoT2023 | Raw CSV Phase-1 track | CSV sample per file | label/category mapper and sample EDA notebook | 0 raw CSV files present; full raw audit blocked if zero |

## Requested Lab Matrix

Legend: **PROVEN** = runnable evidence and saved results exist; **PARTIAL** = some code or data exists but the experiment is not complete; **MISSING** = no credible local evidence yet.

| Dataset/schema | Supervised | Cost-sensitive | Anomaly | Semi-supervised | Temporal | Online/drift | Quality/explainability | SOC/graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | PROVEN+ | PARTIAL | PROVEN | PROVEN | MISSING | PARTIAL | PARTIAL+ | PARTIAL |
| CICIoT2023 | PARTIAL | MISSING | MISSING | MISSING | MISSING | MISSING | PARTIAL+ | MISSING |
| TON_IoT | PARTIAL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| CSE-CIC-IDS2018 | PARTIAL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| Common NetFlow schema | PARTIAL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

## Detailed Findings

| Area | Prompt expectation | Status | Evidence | Gap / risk | Next audit test |
| --- | --- | --- | --- | --- | --- |
| NSL-KDD controlled baseline | Keep NSL-KDD as the fast local reference dataset. | PROVEN | Preprocessing, metrics, RF/LightGBM, MLP, reference-track, and stability files exist. | Reference-track docs are newer than docs/PROGRESS.md, so some summary text is stale. | Run pytest plus reference/stability scripts before using headline claims. |
| Reference Track model zoo | Dummy, Logistic Regression, RF, ExtraTrees, HistGB, LightGBM on binary and multiclass. | PROVEN | results/reference_track.md contains 12 table data rows. | SGD is mentioned by the prompt but is not present in the reference-track script. | Add SGDClassifier or remove SGD from claims until it has saved metrics. |
| RF/LightGBM official-split baselines | Tune on train only, evaluate KDDTest+ and binary KDDTest-21. | PROVEN | results/metrics.md contains 6 Phase-3 summary rows and saved figures exist. | Only RF/LightGBM get full saved confusion/ROC/PR artifacts in Phase 3. | Keep this as the gold-standard artifact pattern for later tracks. |
| MLP weighting ablation | Compare weighted vs unweighted MLP and rare-class recall. | PROVEN | results/metrics.md and results/stability.md report weighted/unweighted MLP. | Phase 4 summary lacks full saved per-class tables matching Phase 3 detail. | Write MLP per-class tables and confusion matrices into a dedicated MLP results section. |
| Artificial neuron and activation foundations | Demonstrate weighted input, bias, activation, loss, gradients, update, activations, and derivatives. | PROVEN | results/neural_foundations.md plus activation_functions.csv and single_neuron_demo.csv exist. | This is educational foundation work; it is not a trained IDS model. | Use it as the theory bridge before interpreting MLP/CNN/RNN experiments. |
| Controlled neural architecture ablations | Change activation, weighting, dropout, normalization, depth/width, loss, and label smoothing one factor at a time. | PROVEN | results/neural_ablation.md, neural_ablation.csv, learning curves, and JSONL tracking exist. | Only bounded NSL-KDD binary MLP ablations are implemented; CNN/RNN families remain representation-blocked. | Repeat across seeds and add calibration before final neural ranking claims. |
| Multi-seed stability | Qualify model rankings across seeds. | PROVEN | results/stability.md contains 8 table data rows. | Only headline models are in stability; LogReg/ExtraTrees/HistGB are single-seed. | Extend stability to the reference-track winners before ranking them strongly. |
| Threshold tuning | Decision thresholds should be tuned on validation only. | PROVEN | src/tuning.py has threshold_sweep and results/threshold_ablation.md saves validation-selected thresholds. | Current ablation is binary NSL-KDD only; multiclass per-class thresholding is not implemented. | Extend to calibrated binary models and report seed variance before using one threshold as final. |
| Cost-sensitive learning family | No balancing, class weights, RUS, ROS, SMOTE, Borderline-SMOTE, ADASYN, focal loss. | PARTIAL | Class weights exist for several sklearn models and MLP; no resampling study exists. | No tests currently guard against SMOTE/test leakage or unrealistic synthetic-flow claims. | Create an imbalance notebook with train-only resampling and validation-only selection. |
| CICIoT2023 primary modern dataset | Add modern IoT dataset with binary, category, and fine-label tracks. | PARTIAL | Dev parquet files exist; src/ciciot.py and src/ciciot2023.py exist; quality report exists=True; raw audit exists=True; raw CSV count is 0. | Full official raw CSV analysis is blocked until data/ciciot2023/CSV/*.csv exists. | Use dev parquet for quick checks, then run raw CSV Phase-1 audit once downloaded. |
| TON_IoT multimodal dataset | Network, telemetry, host, and security-event fusion experiments. | PARTIAL | docs/datasets/catalog.md records the official source, role, local path, and blocked status. | No local data, schema inventory, loader, notebook, or model result exists. | Create only a dataset catalog/provenance entry first; do not model until schema is audited. |
| CSE-CIC-IDS2018 enterprise dataset | Enterprise-scale chronological/day-based experiments. | PARTIAL | docs/datasets/catalog.md records the official source, role, local path, and blocked status. | No local data, day inventory, leakage audit, chronological split, or model result exists. | Add source/provenance and a leakage-aware day split plan before training. |
| Unsupervised anomaly detection | Train on benign only; test known and held-out attack types. | PROVEN | results/anomaly_detection.md reports IsolationForest, LOF, and k-means-distance normal-only detectors. | This is NSL-KDD binary/family recall only; autoencoders and modern-dataset zero-day splits are not implemented. | Graduate the best protocol to CICIoT/CSE once raw timestamped data exists. |
| Semi-supervised learning | Use 1/5/10/25% labels plus self-training/label propagation/pretraining. | PROVEN | results/semi_supervised.md reports labelled-only vs self-training Logistic Regression at 1/5/10/25%. | Only binary NSL-KDD and self-training are implemented; label propagation and representation pretraining are not. | Repeat with multiclass and modern datasets before broader label-efficiency claims. |
| Temporal and sequence detection | Window flows by host/session/time and compare tabular vs sequence models. | MISSING | NSL-KDD has no usable chronology; no temporal dataset/window builder exists. | Do not imply sequence behaviour detection from row-level classifiers. | Use CIC/CSE style timestamped datasets; first build audited window aggregation. |
| Online and drift learning | Chronological streams, partial_fit, drift detectors, recovery-time metrics. | PARTIAL | results/online_learning.md reports SGD partial_fit models over NSL-KDD file-order chunks. | NSL-KDD is not chronological, so this proves online algorithms, not drift recovery. | Run true drift metrics on timestamped CSE/CICIoT data after chronology is audited. |
| Cross-dataset generalization | Train one environment, test another via common NetFlow schema. | PARTIAL | docs/datasets/catalog.md identifies the common NetFlow dataset source and required local path. | No common schema adapter, local NetFlow files, or leave-one-dataset-out result exists. | Define common NetFlow-style features before any cross-dataset score is published. |
| Data-quality and leakage analytics | Missing, duplicates, constants, outliers, PSI/KS/JS/chi-squared, leakage checks. | PARTIAL | NSL audit notebook is executed; CICIoT dev quality script exists; feature outlier CSV exists=True. | Advanced statistical shift tests such as PSI/KS/JS/chi-squared are still incomplete. | Create one reusable quality-report API and run it against every dataset. |
| Attack behavioural profiling | Per-attack protocol/duration/ports/source-diversity/flags/profile summaries. | MISSING | No saved attack-profile tables exist. | The project can explain model scores better than it can explain attack behaviour today. | Add per-class profile tables to the data-quality notebook before more models. |
| Clustering and representation studies | KMeans/GMM/DBSCAN/HDBSCAN/PCA/UMAP and raw/PCA/AE/LightGBM comparisons. | PARTIAL | results/feature_learning.md compares raw, MI-selected, PCA, L1-selected, and autoencoder embeddings. | Clustering, UMAP/t-SNE, SHAP-driven selection, and deep tabular architectures are not implemented yet. | Add clustering diagnostics and repeat feature-learning over more datasets once local files exist. |
| Explainability and calibration | SHAP, permutation importance, calibration, Brier/ECE, confidence/error analysis. | PARTIAL | Tree feature-importance plots exist; no calibration or SHAP artifacts exist. | Feature importance is not enough to claim explainability/trustworthiness. | Add calibration curves/Brier/ECE for binary models, then SHAP/permutation checks. |
| Operational SOC simulation | Alerts/day, FP per 10k benign, workload, risk-tolerance thresholds. | PROVEN | results/soc_simulation.md converts threshold-ablation rates into daily alerts and missed attacks. | Scenario is fixed at 1M flows/day and 0.5% malicious; no sensitivity grid yet. | Add multiple base rates, analyst capacity limits, and threshold-frontier plots. |
| Graph machine learning | Host/IP graph, centrality/community, GraphSAGE/GAT/edge classification. | MISSING | No graph data representation exists. | Graph ML belongs later; current row-level feature tables cannot support it. | First add graph statistics only after a dataset with IP/device identifiers is audited. |
| Deep-learning taxonomy and suitability matrix | Classify MLP/CNN/RNN/AE/VAE/GAN/SOM/RBM/DBN/transfer/DRL by representation validity. | PROVEN | docs/deep_learning_taxonomy.md records method-to-data suitability and staged progression. | It is a guardrail document; many advanced families are intentionally not implemented until data supports them. | Update this matrix whenever a new dataset or representation is added. |

## Claims I Would Allow Today

- NSL-KDD preprocessing is leakage-safe and tested.
- NSL-KDD has a supervised reference track across several classical models.
- Balanced Logistic Regression is strong on the 5-class reference track: the saved table reports about 0.607 macro-F1.
- Threshold tuning changes the binary NSL-KDD attack-recall/false-alert trade-off.
- Normal-only anomaly detection and semi-supervised label-budget experiments now have saved NSL-KDD reports.
- Correlation analysis, outlier mapping, and representation-learning baselines now exist for NSL-KDD and CICIoT2023 dev data.
- Artificial-neuron foundations, activation functions, and bounded MLP ablations now have saved reports and tracked runs.
- SOC simulation now translates threshold results into alerts/day and missed attacks/day.
- Weighted MLP improves rare-class recall materially, but the macro-F1 ranking is not a clean win.
- CICIoT2023 has early loader/provenance/Phase-1 scaffolding, but full raw CSV modelling is not complete.

## Claims I Would Block

- Any claim that TON_IoT or CSE-CIC-IDS2018 has been evaluated.
- Any claim of true temporal sequence detection, online drift recovery, modern-dataset anomaly detection, or cross-dataset generalization.
- Any claim that NSL-KDD file-order online learning is chronological drift.
- Any claim that feature importance equals explainability or calibrated trust.

## Immediate Audit-Driven Build Order

1. Add a true imbalance notebook: no balancing vs class weights vs resampling, with train-only resampling tests.
2. Finish CICIoT2023 quality reports for whichever local source is actually used: dev parquet now, raw CSV later.
3. Add calibration/Brier/ECE to the threshold/SOC track.
4. Add true timestamped online-drift experiments after CSE/CICIoT raw files exist.
5. Only then implement cross-dataset NetFlow and graph tracks.
