# CICIoT2023 (dev sample) — data-quality & leakage audit

All features numeric (flag/protocol indicators already 0/1) → preprocessing is scale-only. This dev sample uses a **random** train/test split, so it is *in-distribution* (unlike NSL-KDD's official shift-heavy split) — CICIoT2023 scores will look higher for reasons of protocol; caveat every result.

> No socket-identifier columns (IP/port/timestamp) are present, so the classic CICFlowMeter host/time **leakage risk is already avoided** here.

### train  (1,073,851 rows x 39 features)

- missing cells: **0**
- infinite cells: **0**
- fully duplicated rows: **84,370** (7.86%)
- constant features: **0** 
- near-constant (>99.9% one value): **2** ['Telnet', 'SMTP']

- label-level consistency (binary vs category): **OK**

### test  (268,463 rows x 39 features)

- missing cells: **0**
- infinite cells: **0**
- fully duplicated rows: **9,434** (3.51%)
- constant features: **0** 
- near-constant (>99.9% one value): **2** ['Telnet', 'SMTP']

- label-level consistency (binary vs category): **OK**

## Label distribution (train)

| Level | #classes | head |
| --- | ---: | --- |
| binary `label` | 2 | attack 85.1% |
| category `attack_class` | 8 | DDoS 361,842, Recon 161,877, Benign 159,990, DoS 159,925, … |
| fine `Label` | 32 | 32 attack types |

