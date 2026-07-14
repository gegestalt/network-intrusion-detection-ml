# NSL-KDD — Online-Learning Proxy

This is an online algorithm test using `partial_fit` over NSL-KDD training rows in file order. Because NSL-KDD has no reliable chronology, this is **not** evidence of drift recovery.

| Model | Chunk size | Chunks seen | Accuracy | Macro-F1 | Normal recall | Attack recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGD_log_loss_balanced | 10,000 | 13 | 0.7644 | 0.7642 | 0.9240 | 0.6437 |
| SGD_passive_aggressive_balanced | 10,000 | 13 | 0.7368 | 0.7358 | 0.9248 | 0.5945 |

## Next real drift test

Use a timestamped dataset such as CSE-CIC-IDS2018 or CICIoT2023 raw flows, then evaluate pre-drift, post-drift, recovery time, latency, and memory.
