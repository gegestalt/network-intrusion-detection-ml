# CICIoT2023 dataset track

## Recommendation

Use **CICIoT2023** as the next Phase-1 dataset after NSL-KDD.

## Evidence

The official UNB/CIC page describes CICIoT2023 as a real-time IoT attack dataset
for large-scale attacks. It reports:

- 105 IoT devices in the topology.
- 33 attacks.
- 7 attack categories: DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, and
  Mirai.
- CSV flow features, PCAP files, example notebook, and supplementary tooling.

Source: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>

## Why it diversifies the project

NSL-KDD is old, compact, and useful for teaching official-split generalization.
CICIoT2023 changes the problem:

1. **Modern IoT context.** The traffic is built around IoT devices, attackers,
   and victims rather than 1990s enterprise simulation.
2. **Broader attack taxonomy.** The 33 fine labels and 7 families give us a more
   realistic multiclass imbalance problem.
3. **Scale.** The dataset is large enough that sampling, chunking, and runtime
   discipline become part of the engineering story.
4. **Portfolio value.** It lets the project contrast "historical benchmark" vs
   "modern IoT benchmark" without pretending the two feature spaces are directly
   interchangeable.

## Phase-1 questions

The Phase-1 EDA notebook should answer:

- How imbalanced are the fine labels and the 7 attack categories?
- Which categories are rare enough to dominate macro-F1?
- Are there missing values, infinities, or constant columns?
- Which features have extreme scale or suspiciously deterministic values?
- Is a stratified sample sufficient for fast modelling, or do rare classes
  require full-data handling?

## Modelling plan after Phase 1

Use the lessons from the NSL-KDD referee audit:

1. Start with Dummy, Logistic Regression, balanced Logistic Regression,
   RandomForest, LightGBM, and MLP.
2. Lead with macro-F1, per-class recall, and confusion matrices.
3. Add class weighting and threshold tuning as explicit ablations.
4. Use sample-based development first, then scale up.
5. Keep tuning on train/validation only.
