# Deep-Learning Taxonomy and Suitability Matrix

This document maps model families to dataset representations. It is a methodological guardrail: an architecture is allowed only when the data representation supports its assumptions.

## Suitability Matrix

| Method | NSL-KDD tabular | CICIoT2023 dev tabular | Raw CICIoT sequences | TON_IoT multimodal | CSE-CIC temporal | Role |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | strong baseline | strong baseline | aggregates only | per modality | aggregates/chronological | scientific baseline |
| MLP | valid | valid | aggregates/windows | per modality/fusion | aggregates/windows | modern tabular baseline |
| CNN | invalid unless feature ordering is justified | invalid unless feature ordering is justified | valid for packet/window sequences | possible for telemetry windows | valid for traffic matrices/windows | representation-dependent |
| RNN/LSTM/GRU | no temporal value on isolated rows | no temporal value on random dev rows | valid for ordered sequences | valid for logs/telemetry timelines | valid for day/time streams | temporal candidate |
| BiLSTM | invalid for real-time row detection | invalid for random dev rows | offline sequence analysis only | offline timeline analysis only | offline chronological analysis only | offline model unless future context is allowed |
| Autoencoder | valid representation/anomaly | valid representation/anomaly | valid | valid | valid | unsupervised representation/anomaly |
| Denoising AE | valid robustness test | valid robustness test | valid masking/noise test | valid missing telemetry test | valid noisy-flow test | robustness representation |
| VAE | educational/probabilistic baseline | possible | possible | possible | possible | probabilistic representation |
| SOM | valid exploratory clustering | valid exploratory clustering | embeddings/windows | embeddings/per modality | embeddings/windows | educational/exploratory |
| RBM/DBN | historical only | historical only | historical only | historical only | historical only | controlled historical comparison |
| GAN | tabular GAN only; train partition only | tabular GAN only; train partition only | sequence GAN possible | modality-specific possible | time-series/tabular possible | augmentation with strict validation |
| CNN-LSTM | invalid without sequences | invalid on random rows | valid for local+long temporal patterns | valid for telemetry sequences | valid for flow windows | hybrid temporal candidate |
| AE-SVM | valid | valid | valid on embeddings | valid on embeddings | valid on embeddings | hybrid representation baseline |
| Transfer learning | source/target must be defined | dev only; not full raw release | valid after schema alignment | valid after modality alignment | valid after feature alignment | domain adaptation candidate |
| Deep RL | not for static classification | not for static classification | only threshold/response environment | triage/response environment | threshold/drift/response environment | sequential decision only |

## Required Experimental Progression

- Stage 1 - raw dataset construction and validation
- Stage 2 - EDA and leakage audit
- Stage 3 - dummy/statistical baselines
- Stage 4 - Logistic Regression
- Stage 5 - tree and boosting baselines
- Stage 6 - artificial neuron demonstration
- Stage 7 - MLP ablations
- Stage 8 - activation-function experiments
- Stage 9 - unsupervised representation learning
- Stage 10 - benign-only anomaly detection
- Stage 11 - temporal or packet representation construction
- Stage 12 - CNN experiments
- Stage 13 - RNN/LSTM/BiLSTM/GRU experiments
- Stage 14 - CNN-LSTM hybrids
- Stage 15 - GAN-based augmentation
- Stage 16 - AE-SVM and representation-transfer
- Stage 17 - RBM/DBN historical comparisons
- Stage 18 - cross-dataset transfer learning
- Stage 19 - DRL for sequential security decisions
- Stage 20 - robustness, explainability, and operational analysis

## Current Execution Status

- Stages 1-5: substantially implemented for NSL-KDD; partially for CICIoT2023 dev.
- Stage 6: implemented by `src/neural_foundations.py`.
- Stage 7-8: implemented first-pass by `src/neural_ablation.py`.
- Stage 9: first-pass autoencoder embeddings implemented by `src/feature_learning.py`.
- Stage 10: implemented for NSL-KDD by `src/anomaly_detection.py`.
- Stages 11-19: blocked until sequence/timestamp/raw/multimodal data exists or a valid environment is defined.

CNN/RNN/GAN/DRL work must be added only after the representation requirements in this matrix are satisfied.
